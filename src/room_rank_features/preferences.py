"""Point-in-time traveler preferences from confirmed, previously booked rooms."""
from __future__ import annotations

import numpy as np
import pandas as pd

BED_COLUMNS = ("total_beds", "count_king_beds", "count_queen_beds", "count_twin_beds")
PREFERENCE_FLAGS = ("free_breakfast", "free_parking", "free_wifi", "is_refundable")
BED_PREFERENCE_FEATURES = ("history_bed_config_match_rate", "history_total_beds_abs_diff")
PREFERENCE_FEATURES = (
    "history_booking_count", "history_has_booking", "history_days_since_last_booking",
    "history_last_bed_config_match", "history_bed_config_match_rate",
    "history_total_beds_abs_diff",
    *(f"history_{name}_match_rate" for name in PREFERENCE_FLAGS),
)


class TravelerPreferenceTransformer:
    """Return (dense pandas history features, search group-size array).

    fit loads selected booked outcomes (label=1, at most one per search).
    It learns no timeless target averages: every transform performs strict
    as-of lookups. Source bookings must occur on/after their own search and
    become available strictly before the search being scored. Equal-time and
    current-search outcomes are excluded. Future registered events cannot
    affect earlier rows. update_history registers newly confirmed outcomes.

    Source BK_TIMESTAMP clock digits are interpreted as US Pacific local time
    by default, despite the supplied dataset's misleading Z suffix. Set
    booking_time_is_utc=True for genuinely UTC timestamps. Search timestamps
    are UTC. Set available_time_column to an actual arrival timestamp when
    available; otherwise availability is booking time plus availability_delay.

    Match rates use a fixed symmetric Beta prior: (matches + smoothing/2) /
    (observations + smoothing). Default smoothing=2 means Beta(1,1). These
    are candidate-attribute match rates, not candidate booking probabilities.
    No-history match rates are 0.5; recency, last-match and bed-distance are NaN.
    All four structured bed values, including zero, are used as supplied.
    """

    def __init__(
        self, *, traveler_column="traveler_id", group_column="inference_id",
        search_time_column="timestamp", booking_time_column="BK_TIMESTAMP",
        booking_timezone="America/Los_Angeles", booking_time_is_utc=False,
        available_time_column=None, availability_delay="0s", smoothing=2.0,
    ):
        names = [traveler_column, group_column, search_time_column, booking_time_column]
        if available_time_column is not None:
            names.append(available_time_column)
        if not all(isinstance(n, str) and n for n in names) or len(set(names)) != len(names):
            raise ValueError("Metadata column names must be distinct nonempty strings.")
        if set(names) & {*BED_COLUMNS, *PREFERENCE_FLAGS, "label", "margin", "rate_type"}:
            raise ValueError("Metadata columns must not shadow feature or target columns.")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be positive and finite.")
        delay = pd.Timedelta(availability_delay)
        if pd.isna(delay) or delay < pd.Timedelta(0):
            raise ValueError("availability_delay must be nonnegative.")
        self.traveler_column = traveler_column
        self.group_column = group_column
        self.search_time_column = search_time_column
        self.booking_time_column = booking_time_column
        self.booking_timezone = booking_timezone
        self.booking_time_is_utc = booking_time_is_utc
        self.available_time_column = available_time_column
        self.availability_delay = delay
        self.smoothing = float(smoothing)
        self.feature_names_ = PREFERENCE_FEATURES
        self.output_columns_ = PREFERENCE_FEATURES
        self._fitted = False

    @staticmethod
    def _frame(df, required):
        if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
            raise ValueError("Supply a pandas DataFrame with unique column names.")
        if missing := set(required) - set(df.columns):
            raise ValueError(f"Missing columns: {sorted(missing)}")

    @staticmethod
    def _utc(values):
        return pd.to_datetime(values, utc=True, format="mixed", errors="raise").astype("datetime64[ns, UTC]")

    @staticmethod
    def _numeric(df):
        values = df[[*BED_COLUMNS, *PREFERENCE_FLAGS]].apply(pd.to_numeric, errors="raise").astype(float)
        if np.isinf(values.to_numpy()).any():
            raise ValueError("Structured features must not contain infinity.")
        for flag in PREFERENCE_FLAGS:
            if not values[flag].dropna().isin([0, 1]).all():
                raise ValueError(f"{flag} must contain 0/1 or missing values.")
        return values

    def _extract(self, df):
        names = [self.group_column, self.traveler_column, self.search_time_column,
                 self.booking_time_column, "label", *BED_COLUMNS, *PREFERENCE_FLAGS]
        if self.available_time_column:
            names.append(self.available_time_column)
        self._frame(df, names)
        if df.label.isna().any() or not df.label.isin([0, 1]).all():
            raise ValueError("History labels must be numeric 0/1.")
        selected = df.loc[df.label.eq(1), names].reset_index(drop=True)
        if selected[self.group_column].isna().any():
            raise ValueError("Booked outcomes require a nonmissing search ID.")
        if selected[self.group_column].duplicated().any():
            raise ValueError("Supply at most one selected booked outcome per search; clean_data prepares this input.")
        records = self._numeric(selected)
        records["source_id"] = selected[self.group_column].astype(object)
        records["traveler"] = selected[self.traveler_column].astype(object)
        records["searched"] = self._utc(selected[self.search_time_column])
        booking = self._utc(selected[self.booking_time_column])
        if not self.booking_time_is_utc:
            booking = booking.dt.tz_localize(None).dt.tz_localize(
                self.booking_timezone, ambiguous="raise", nonexistent="raise",
            ).dt.tz_convert("UTC")
        records["booked"] = booking
        if self.available_time_column:
            arrived = self._utc(selected[self.available_time_column])
            available = booking.where(booking.ge(arrived), arrived)
            available = available.where(booking.notna() & arrived.notna())
        else:
            available = booking
        records["available"] = (available + self.availability_delay).astype("datetime64[ns, UTC]")
        return records

    def _install(self, records):
        records = records.drop_duplicates().reset_index(drop=True)
        if records.source_id.duplicated().any():
            raise ValueError("Conflicting historical outcomes for the same search ID.")
        missing = records[["traveler", "searched", "booked", "available", *BED_COLUMNS]].isna().any(axis=1)
        contradictory = records.booked.lt(records.searched)
        events = records.loc[~missing & ~contradictory].copy()
        events["_tie_id"] = events.source_id.map(repr)
        events = events.sort_values(["available", "booked", "_tie_id"], kind="stable")
        self.audit_ = {
            "registered_booked_searches": len(records), "usable_history_events": len(events),
            "missing_history_fields": int(missing.sum()),
            "booking_before_source_search": int(contradictory.sum()),
        }
        self._records = records
        self._source_metadata = records.set_index("source_id")[["traveler", "searched"]]
        self._configs = pd.MultiIndex.from_frame(events[list(BED_COLUMNS)]).unique()
        if events.empty:
            self._events = events
            self._config_events = events
            self._fitted = True
            return self
        events["config"] = self._configs.get_indexer(pd.MultiIndex.from_frame(events[list(BED_COLUMNS)]))
        grouped = events.groupby("traveler", sort=False)
        events["count"] = grouped.cumcount() + 1
        newest = events.booked.eq(grouped.booked.cummax())
        last_position = pd.Series(np.where(newest, np.arange(len(events)), np.nan), index=events.index)
        last_position = last_position.groupby(events.traveler, sort=False).ffill().to_numpy(dtype=int)
        events["last_booked"] = events.booked.array.take(last_position)
        events["last_config"] = events.config.to_numpy()[last_position]
        medians = grouped.total_beds.expanding().median().reset_index(level=0, drop=True)
        events["median_beds"] = medians.reindex(events.index)
        for flag in PREFERENCE_FLAGS:
            events[f"{flag}_yes"] = events[flag].fillna(0).groupby(events.traveler, sort=False).cumsum()
            events[f"{flag}_n"] = events[flag].notna().astype(int).groupby(events.traveler, sort=False).cumsum()
        config_events = events[["traveler", "available", "config"]].copy()
        config_events["config_count"] = config_events.groupby(["traveler", "config"], sort=False).cumcount() + 1
        # Original event index is no longer needed after expanding statistics.
        self._events = events.reset_index(drop=True)
        self._config_events = config_events.reset_index(drop=True)
        self._fitted = True
        return self

    def fit(self, history_df):
        """Register selected outcomes; loading events does not expose future labels."""
        self._fitted = False
        return self._install(self._extract(history_df))

    def update_history(self, history_df):
        """Register more outcomes; identical repeats are idempotent, conflicts raise."""
        if not self._fitted:
            raise ValueError("Call fit first.")
        return self._install(pd.concat([self._records, self._extract(history_df)], ignore_index=True))

    def fit_transform(self, training_df):
        return self.fit(training_df).transform(training_df)

    def transform(self, df):
        """Construct past-only features; never reads label or booking-time columns."""
        if not self._fitted:
            raise ValueError("Call fit first.")
        self._frame(df, [self.group_column, self.traveler_column, self.search_time_column,
                         *BED_COLUMNS, *PREFERENCE_FLAGS])
        if df[self.group_column].isna().any():
            raise ValueError("Search IDs must be nonmissing.")
        times = self._utc(df[self.search_time_column]).reset_index(drop=True)
        if times.isna().any():
            raise ValueError("Search times must be nonmissing.")
        codes, ids = pd.factorize(df[self.group_column], sort=False)
        starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1] if len(df) else np.array([], dtype=int)
        if len(starts) != len(ids):
            raise ValueError("Rows for each search must be contiguous; sort features and labels together.")
        groups = np.diff(np.r_[starts, len(df)]).astype(np.int32)
        traveler_codes = pd.factorize(df[self.traveler_column], sort=False)[0]
        if len(df) and (not np.array_equal(traveler_codes, np.repeat(traveler_codes[starts], groups))
                        or not np.array_equal(times.array.asi8, np.repeat(times.array.asi8[starts], groups))):
            raise ValueError("Each search must have one traveler and one search timestamp.")
        numeric = self._numeric(df)
        queries = pd.DataFrame({
            "source_id": df[self.group_column].iloc[starts].to_numpy(),
            "traveler": df[self.traveler_column].iloc[starts].astype(object).to_numpy(),
            "searched": times.iloc[starts].array, "query": np.arange(len(starts)),
        })
        known = queries.source_id.isin(self._source_metadata.index)
        if known.any():
            previous = self._source_metadata.reindex(queries.loc[known, "source_id"])
            current = queries.loc[known]
            same_traveler = (previous.traveler.to_numpy() == current.traveler.to_numpy()) | (
                previous.traveler.isna().to_numpy() & current.traveler.isna().to_numpy())
            if not same_traveler.all() or not np.array_equal(previous.searched.array.asi8, current.searched.array.asi8):
                raise ValueError("A registered search's traveler/time changed; its own outcome must never become history.")
        result = np.full((len(df), len(PREFERENCE_FEATURES)), np.nan, dtype=np.float32)
        result[:, :2] = 0
        result[:, 4] = .5
        result[:, 6:] = .5
        if df.empty or self._events.empty:
            return pd.DataFrame(result, columns=PREFERENCE_FEATURES, index=df.index), groups

        snapshot_cols = ["traveler", "available", "last_booked", "source_id", "count", "last_config", "median_beds",
                         *(f"{flag}_{suffix}" for flag in PREFERENCE_FLAGS for suffix in ("yes", "n"))]
        snapshots = pd.merge_asof(
            queries.sort_values("searched", kind="stable"),
            self._events[snapshot_cols].rename(columns={"source_id": "history_source_id"}),
            by="traveler", left_on="searched", right_on="available",
            allow_exact_matches=False, direction="backward",
        ).sort_values("query")
        matched = snapshots.available.notna()
        if snapshots.loc[matched, "history_source_id"].eq(snapshots.loc[matched, "source_id"]).any():
            raise ValueError("Current search outcome was encountered in its history.")
        counts = snapshots["count"].fillna(0).to_numpy()[codes]
        result[:, 0] = counts
        result[:, 1] = counts > 0
        result[:, 2] = ((snapshots.searched - snapshots.last_booked).dt.total_seconds() / 86400).to_numpy()[codes]
        configs = self._configs.get_indexer(pd.MultiIndex.from_frame(numeric[list(BED_COLUMNS)]))
        has_beds = numeric[list(BED_COLUMNS)].notna().all(axis=1).to_numpy()
        last = snapshots.last_config.to_numpy()[codes]
        result[:, 3] = np.where((counts > 0) & has_beds, configs == last, np.nan)
        result[:, 5] = np.abs(numeric.total_beds.to_numpy() - snapshots.median_beds.to_numpy()[codes])

        # Lookup once per distinct search/configuration, then expand positionally.
        pairs = pd.DataFrame({"query": codes, "config": configs})
        pair_codes, unique_pairs = pd.factorize(pd.MultiIndex.from_frame(pairs), sort=False)
        lookup = unique_pairs.to_frame(index=False)
        lookup.columns = ["query", "config"]
        lookup["pair"] = np.arange(len(lookup))
        lookup = lookup.merge(queries[["query", "traveler", "searched"]], on="query", validate="many_to_one")
        matched_configs = pd.merge_asof(
            lookup.sort_values("searched", kind="stable"), self._config_events,
            by=["traveler", "config"], left_on="searched", right_on="available",
            direction="backward", allow_exact_matches=False,
        ).sort_values("pair")
        matches = matched_configs.config_count.fillna(0).to_numpy()[pair_codes]
        result[:, 4] = np.where(has_beds, (matches + self.smoothing / 2) / (counts + self.smoothing), np.nan)
        for index, flag in enumerate(PREFERENCE_FLAGS, 6):
            yes = snapshots[f"{flag}_yes"].fillna(0).to_numpy()[codes]
            n = snapshots[f"{flag}_n"].fillna(0).to_numpy()[codes]
            value = numeric[flag].to_numpy()
            equal = np.where(value == 1, yes, n - yes)
            result[:, index] = np.where(np.isnan(value), np.nan, (equal + self.smoothing / 2) / (n + self.smoothing))
        return pd.DataFrame(result, columns=PREFERENCE_FEATURES, index=df.index), groups

    def get_feature_names_out(self):
        return np.asarray(PREFERENCE_FEATURES, dtype=object)
