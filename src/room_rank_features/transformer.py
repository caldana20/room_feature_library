"""Training-fitted, label-independent pandas features with a portable JSON state."""

from __future__ import annotations

import json
from importlib.resources import files
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .parser import FAMILIES, normalize_description, parse_description, room_text_without_beds
from .preferences import BED_PREFERENCE_FEATURES, TravelerPreferenceTransformer


BASE_SCENARIOS = {
    "R": "Commercial baseline",
    "P14": "Structured room model",
    "P14_L": "Structured + length",
    "P14_SWL": "Structured + full text",
}
HISTORY_SCENARIOS = {f"{name}_H2": name for name in BASE_SCENARIOS}
SCENARIOS = {**BASE_SCENARIOS,
             **{name: BASE_SCENARIOS[base] + " + two bed-preference features"
                for name, base in HISTORY_SCENARIOS.items()}}
BASE_FEATURES = (
    "rate_to_market_ratio", "policy_violations", "elegible_for_loyalty",
    "free_breakfast", "free_parking", "free_wifi", "free_dinner",
    "is_refundable",
)
# Historical scenario keys stay stable; P14 now contains 12 model inputs.
STRUCTURED_FEATURES = (
    *BASE_FEATURES, "total_beds",
    "count_king_beds", "count_queen_beds", "count_twin_beds",
)
EXCLUDED_COLUMNS = ("margin", "rate_type")
PARSED_FEATURES = tuple(
    col for family, cols in FAMILIES.items() if family != "format_metadata" for col in cols
)
TOKEN_PATTERN = r"(?u)\b\w+\b"
PARSER_VERSION = "structured-beds-room-parser-v3"


class RoomFeatureTransformer:
    """Construct four original scenarios and their four _H2 history variants.

    ``fit`` learns full-text vocabulary exclusively from training rows.
    It never reads booking labels, margin or rate_type. ``transform`` and
    ``fit_transform`` return (predictor DataFrame, group-size array).
    ``add_features`` appends text features to the supplied frame after dropping
    margin and rate_type. Neither method mutates, filters, or sorts the input.

    Model inputs use ordinary float32 pandas columns, including words. Group
    sizes contain one count per contiguous search, in input order. Interleaved
    searches raise: order the input and its labels together before transforming.
    ``feature_names_`` and ``output_columns_`` list predictors only.
    _H2 variants require an explicitly fitted TravelerPreferenceTransformer
    supplied through ``history=`` and append only BED_PREFERENCE_FEATURES.
    Register outcomes on that separate store; room fitting never updates it.
    """

    def __init__(
        self,
        scenario: str = "P14_SWL",
        *,
        min_df: int = 30,
        max_features: int = 1200,
        text_column: str = "room_description",
        group_column: str = "inference_id",
        history: TravelerPreferenceTransformer | None = None,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose one of {tuple(SCENARIOS)}.")
        for name, value in (("min_df", min_df), ("max_features", max_features)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(text_column, str) or not isinstance(group_column, str):
            raise TypeError("text_column and group_column must be strings.")
        if not text_column or not group_column or text_column == group_column:
            raise ValueError("text_column and group_column must be distinct, nonempty names.")
        reserved = {*STRUCTURED_FEATURES, *PARSED_FEATURES, *BED_PREFERENCE_FEATURES, "text_length", *EXCLUDED_COLUMNS}
        if group_column in reserved:
            raise ValueError("group_column must not name a model feature or excluded column.")
        if text_column in EXCLUDED_COLUMNS:
            raise ValueError("text_column must not name an excluded column.")
        self.scenario = scenario
        self.base_scenario = HISTORY_SCENARIOS.get(scenario, scenario)
        self.uses_history = scenario in HISTORY_SCENARIOS
        if self.uses_history:
            if not isinstance(history, TravelerPreferenceTransformer):
                raise ValueError("History scenarios require a TravelerPreferenceTransformer via history=.")
            if history.group_column != group_column:
                raise ValueError("Room and history group_column must match.")
        elif history is not None:
            raise ValueError("Use an _H2 scenario when supplying history=.")
        self.history = history
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.text_column = text_column
        self.group_column = group_column
        self._fitted = False

    @property
    def _base_columns(self):
        return BASE_FEATURES if self.base_scenario == "R" else STRUCTURED_FEATURES

    @property
    def _description_columns(self):
        if self.base_scenario == "P14_L":
            return ("text_length",)
        if self.base_scenario == "P14_SWL":
            return (*PARSED_FEATURES, "text_length")
        return ()

    def _require_fitted(self):
        if not self._fitted:
            raise ValueError("Call fit(training_df), load(path), or from_report() first.")

    @staticmethod
    def _check_frame(df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")
        if not df.columns.is_unique:
            raise ValueError("Input column names must be unique.")

    @staticmethod
    def _require_columns(df, names):
        missing = [name for name in names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _check_model_input(self, df):
        self._check_frame(df)
        self._require_columns(df, (*self._base_columns, self.group_column))
        if df[self.group_column].isna().any():
            raise ValueError(f"{self.group_column} must be nonmissing.")
        if self._description_columns:
            self._require_columns(df, [self.text_column])
        if self.uses_history:
            if not self.history._fitted:
                raise ValueError("Fit the history store before fitting or transforming an _H2 scenario.")
            if self.history.group_column != self.group_column:
                raise ValueError("Room and history group_column must match.")

    def _texts(self, df):
        series = df[self.text_column]
        raw = series.astype(object).where(series.notna(), "")
        codes, unique = pd.factorize(raw, sort=False)
        if not all(isinstance(x, str) for x in unique):
            raise TypeError(f"{self.text_column} must contain strings or missing values.")
        clean = np.asarray([room_text_without_beds(x) for x in unique], dtype=object)
        return pd.Series(clean[codes], index=series.index, name=series.name)

    def _vectorizer(self, vocabulary=None):
        return CountVectorizer(
            binary=True, ngram_range=(1, 2), min_df=self.min_df,
            max_features=self.max_features, token_pattern=TOKEN_PATTERN,
            dtype=np.float32, vocabulary=vocabulary,
        )

    def _numeric_values(self, series):
        try:
            numeric = pd.to_numeric(series, errors="raise")
            if np.iscomplexobj(numeric):
                raise ValueError("complex numbers are unsupported")
            with np.errstate(over="ignore", invalid="ignore"):
                values = numeric.to_numpy(dtype=np.float32, na_value=np.nan)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{series.name} must contain numeric values or missing values.") from exc
        if np.isinf(values).any():
            raise ValueError(f"{series.name} contains infinity or values outside the float32 range.")
        return values

    def fit(self, training_df: pd.DataFrame):
        """Learn vocabulary; each search-description pair counts once.

        Every scenario requires nonmissing group IDs. Repeated descriptions
        in one search count once for the
        full-text vocabulary. No category mapping is learned.
        """
        # An unsuccessful refit must not leave an old fitted state usable.
        self._fitted = False
        self._check_model_input(training_df)
        if training_df.empty:
            raise ValueError("Cannot fit on an empty training DataFrame.")
        for name in self._base_columns:
            self._numeric_values(training_df[name])
        vocabulary = []
        if self._description_columns:
            self._texts(training_df)
        if self.base_scenario == "P14_SWL":
            # Ignore bed wording even when counting distinct training documents.
            corpus = pd.DataFrame({self.group_column: training_df[self.group_column],
                                   self.text_column: self._texts(training_df)}).drop_duplicates()
            try:
                v = self._vectorizer().fit(self._texts(corpus))
            except ValueError as exc:
                raise ValueError(
                    "Cannot fit a word vocabulary. Supply enough nonempty training "
                    "search-description pairs, or lower min_df explicitly for a small dataset. "
                    f"Details: {exc}"
                ) from exc
            vocabulary = v.get_feature_names_out().tolist()
        self._install_state(vocabulary)
        return self

    def _install_state(self, vocabulary):
        # Retain empty categorical metadata for existing LightGBM callers.
        self.rate_categories_ = ()
        self.vocabulary_ = tuple(vocabulary)
        self.word_feature_mapping_ = {f"word_{i}": term for i, term in enumerate(vocabulary)}
        self.feature_names_ = (
            *self._base_columns, *self._description_columns, *self.word_feature_mapping_,
            *(BED_PREFERENCE_FEATURES if self.uses_history else ()),
        )
        if self.group_column in self.feature_names_:
            raise ValueError("group_column conflicts with a generated model feature.")
        self.output_columns_ = self.feature_names_
        self.categorical_features_ = ()
        self.categorical_indices_ = ()
        self._word_vectorizer = (
            self._vectorizer({term: i for i, term in enumerate(vocabulary)})
            if vocabulary else None
        )
        self._fitted = True

    def _derived_features(self, df):
        if not self._description_columns:
            return pd.DataFrame(index=df.index)
        self._require_columns(df, [self.text_column])
        texts = self._texts(df)
        if self.base_scenario == "P14_L":
            lengths = {t: len(normalize_description(t)) for t in texts.unique()}
            return pd.DataFrame(
                {"text_length": texts.map(lengths).to_numpy(dtype=np.float32)},
                index=df.index,
            )
        # Parse each distinct description once per call. Positional expansion
        # preserves arbitrary/duplicate/MultiIndex row labels without joins.
        codes, unique = pd.factorize(texts, sort=False)
        lookup = pd.DataFrame(
            [parse_description(t) for t in unique], columns=self._description_columns,
            dtype=np.float32,
        )
        parsed = pd.DataFrame(
            lookup.to_numpy()[codes], columns=self._description_columns, index=df.index,
        )
        if len(df):
            word_values = self._word_vectorizer.transform(texts).toarray()
        else:
            word_values = np.empty((0, len(self.vocabulary_)), dtype=np.float32)
        words = pd.DataFrame(
            word_values, index=df.index, columns=list(self.word_feature_mapping_),
            dtype=np.float32,
        )
        return pd.concat([parsed, words], axis=1)

    def _history_features(self, df, groups):
        if not self.uses_history:
            return pd.DataFrame(index=df.index)
        frame, history_groups = self.history.transform(df)
        if not np.array_equal(groups, history_groups) or not frame.index.equals(df.index):
            raise ValueError("Room and history features must have identical rows and search groups.")
        return frame.loc[:, list(BED_PREFERENCE_FEATURES)]

    def _group_sizes(self, df):
        """Count contiguous groups without sorting rows or coercing their IDs."""
        if df.empty:
            return np.empty(0, dtype=np.int32)
        codes, unique = pd.factorize(df[self.group_column], sort=False)
        starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1]
        if len(starts) != len(unique):
            raise ValueError(
                f"Rows for each {self.group_column} must be contiguous. "
                f"Sort the input DataFrame by {self.group_column!r} before "
                "transforming, then take labels from that same ordered DataFrame."
            )
        sizes = np.diff(np.r_[starts, len(df)])
        if sizes.max() > np.iinfo(np.int32).max:
            raise ValueError("A group size exceeds the supported int32 range.")
        return sizes.astype(np.int32)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (dense float32 predictor DataFrame, group sizes) for these rows.

        Each search must occupy one contiguous block; blocks need not be
        alphabetically sorted. Row order and index are preserved. Group sizes
        are int32 counts in block order and sum to the number of rows.
        No group column, margin, rate_type, labels or raw text is returned in X.
        Validation uses its own groups and the training-fitted vocabulary.
        """
        self._require_fitted()
        self._check_model_input(df)
        groups = self._group_sizes(df)
        values = {name: self._numeric_values(df[name]) for name in self._base_columns}
        base = pd.DataFrame(values, index=df.index)
        return pd.concat([base, self._derived_features(df), self._history_features(df, groups)], axis=1), groups

    def fit_transform(self, training_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Fit on training rows and return (predictor DataFrame, group sizes)."""
        return self.fit(training_df).transform(training_df)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop excluded columns and append text features to a copy of df.

        Margin and rate_type are omitted in every scenario. Other raw inputs,
        including the group, are preserved. R/P14 add no text columns. Use
        transform for model inputs. Existing derived columns raise.
        """
        self._require_fitted()
        self._check_model_input(df)
        generated = (*self._description_columns, *self.word_feature_mapping_,
                     *(BED_PREFERENCE_FEATURES if self.uses_history else ()))
        collisions = [name for name in generated if name in df.columns]
        if collisions:
            raise ValueError(f"Generated feature columns already exist: {collisions}")
        original = df.drop(columns=list(EXCLUDED_COLUMNS), errors="ignore").copy(deep=True)
        history = self._history_features(df, self._group_sizes(df)) if self.uses_history else pd.DataFrame(index=df.index)
        return pd.concat([original, self._derived_features(df), history], axis=1)

    def to_csr(
        self, df: pd.DataFrame, *, return_groups: bool = False,
    ) -> sparse.csr_matrix | tuple[sparse.csr_matrix, np.ndarray]:
        """Return a predictor-only CSR matrix, optionally with group sizes.

        Legacy opt-in utility; the notebook and pandas workflow do not use it.
        Missing parsed/numeric values stay NaN. Zero word values mean absence;
        retain LightGBM's zero_as_missing=False default. The same contiguous
        group requirement as transform applies. return_groups=True returns
        (matrix, groups) without constructing the features twice.
        """
        frame, groups = self.transform(df)
        matrix = sparse.csr_matrix(frame.to_numpy(dtype=np.float32))
        return (matrix, groups) if return_groups else matrix

    def get_feature_names_out(self) -> np.ndarray:
        """Return predictor names, also listed in output_columns_."""
        self._require_fitted()
        return np.asarray(self.feature_names_, dtype=object)

    def _history_config(self):
        if not self.uses_history:
            return None
        config = {key: getattr(self.history, key) for key in (
            "traveler_column", "group_column", "search_time_column", "booking_time_column",
            "booking_timezone", "booking_time_is_utc", "available_time_column", "smoothing",
        )}
        config["availability_delay"] = str(self.history.availability_delay)
        return config

    def save(self, path: str | Path):
        """Save schema/vocabulary and history settings; no history records or labels."""
        self._require_fitted()
        state = {
            "schema_version": 6, "parser_version": PARSER_VERSION,
            "scenario": self.scenario, "min_df": self.min_df, "max_features": self.max_features,
            "text_column": self.text_column, "group_column": self.group_column,
            "vocabulary": list(self.vocabulary_),
            "feature_names": list(self.feature_names_),
            "output_columns": list(self.output_columns_),
            "history_features": list(BED_PREFERENCE_FEATURES) if self.uses_history else [],
            "history_config": self._history_config(),
        }
        Path(path).write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, history=None):
        """Restore schema/vocabulary; _H2 states require a separately rebuilt history store."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("schema_version") not in (4, 5, 6) or state.get("parser_version") != PARSER_VERSION:
            raise ValueError(
                "Unsupported feature state or parser version. Version 4 removes margin "
                "and rate_type; refit the transformer "
                "and its matching model."
            )
        obj = cls(history=history, **{k: state[k] for k in (
            "scenario", "min_df", "max_features", "text_column", "group_column",
        )})
        if state["schema_version"] < 6 and obj.uses_history:
            raise ValueError("Historical schemas do not support _H2 scenarios.")
        if state["schema_version"] == 6:
            expected_history = list(BED_PREFERENCE_FEATURES) if obj.uses_history else []
            if state.get("history_features") != expected_history or state.get("history_config") != obj._history_config():
                raise ValueError("Saved history feature selection/settings do not match the supplied history store.")
        if obj.uses_history and not obj.history._fitted:
            raise ValueError("Fit the history store before loading an _H2 transformer.")
        obj._validate_state(state["vocabulary"])
        obj._install_state(state["vocabulary"])
        if list(obj.feature_names_) != state["feature_names"]:
            raise ValueError("Saved feature names do not match this scenario/parser.")
        # Schema 4 used a group column but the exact same predictor values.
        # Validate that legacy schema before adapting it to the tuple API.
        expected_output = list(obj.output_columns_)
        if state["schema_version"] == 4:
            expected_output.append(obj.group_column)
        if expected_output != state["output_columns"]:
            raise ValueError("Saved output columns do not match the group/feature schema.")
        return obj

    def _validate_state(self, vocabulary):
        if not isinstance(vocabulary, list) or not all(isinstance(x, str) for x in vocabulary):
            raise ValueError("vocabulary must be a list of strings.")
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("vocabulary contains duplicate entries.")
        if self.base_scenario == "P14_SWL" and not vocabulary:
            raise ValueError("P14_SWL requires a nonempty vocabulary.")
        if self.base_scenario != "P14_SWL" and vocabulary:
            raise ValueError("This scenario has no word features.")
        if len(vocabulary) > self.max_features:
            raise ValueError("Vocabulary is larger than max_features.")

    @classmethod
    def from_report(cls, scenario: str = "P14_SWL"):
        """Restore the unchanged baseline; other historical models need refitting."""
        if scenario in ("P14", "P14_L", "P14_SWL") or scenario in HISTORY_SCENARIOS:
            raise ValueError("The historical report uses a different feature schema. "
                             "Fit version 4 on training data and retrain the matching model.")
        metadata = json.loads(
            files("room_rank_features").joinpath("data/report_preprocessing.json").read_text(encoding="utf-8")
        )
        obj = cls(scenario)
        obj._validate_state([])
        obj._install_state([])
        if list(obj.feature_names_) != metadata["feature_names"][scenario]:
            raise ValueError("Bundled report feature order does not match the transformer.")
        return obj
