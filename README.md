# Room feature library — 4.3.1

Python feature transformations for hotel-room ranking. The library provides
`RoomFeatureTransformer` and `TravelerPreferenceTransformer`. Room transformations
return `(X, groups)`: a dense float32 pandas DataFrame and a separate int32 array
with one candidate count per contiguous search.

This repository contains the source needed to build version **4.3.1**, its package
metadata, and the bundled compatibility resource. Cleaning, splitting, model
training and ranking evaluation belong to the calling application or notebook;
`clean_data`, `split_data`, `fit_ranker` and `evaluate_cohorts` are not in this library.

## Build the package on Linux

The package requires Python **3.10 or newer**. The commands and container example
below use **Python 3.12**. Install Python 3.12, its `venv` support, and Git through
your Linux distribution or use the container build below.

```bash
git clone https://github.com/caldana20/room_feature_library.git
cd room_feature_library

python3.12 -m venv .venv-build
. .venv-build/bin/activate
python -m pip install --upgrade pip build
python -m build
```

`python -m build` uses the setuptools backend in `pyproject.toml`. It creates a
source distribution and builds a wheel from that source distribution:

```text
dist/room_rank_features-4.3.1.tar.gz
dist/room_rank_features-4.3.1-py3-none-any.whl
```

The library itself is pure Python; there is no C/C++ compilation step. The
`py3-none-any` wheel is platform independent. Its dependencies—NumPy, pandas,
SciPy and scikit-learn—include platform-specific components and must be installed
for the target Linux architecture. The build does not need the dataset, notebooks,
experiment outputs, LightGBM, AWS credentials or a running EKS cluster.

The expected source layout is:

```text
README.md
pyproject.toml
MANIFEST.in
src/room_rank_features/
    __init__.py
    parser.py
    preferences.py
    transformer.py
    data/report_preprocessing.json
```

Keep the JSON resource in both the source distribution and wheel: the legacy
`from_report("R")` compatibility method reads it. It is not a deployed model or a
substitute for fitting/loading the preprocessing state for your own ranker.

## Install and verify the wheel

Install into your application environment. `--only-binary=:all:` requires wheels
for dependencies too, so a missing compatible dependency wheel produces an error
instead of unexpectedly compiling scientific packages in the container.

```bash
python -m pip install --only-binary=:all: dist/room_rank_features-4.3.1-py3-none-any.whl
python -m pip check
python - <<'PY'
import importlib.util
from importlib.resources import files
import room_rank_features
from room_rank_features import RoomFeatureTransformer

assert room_rank_features.__version__ == "4.3.1"
assert importlib.util.find_spec("room_rank_features.experiments") is None
assert files("room_rank_features").joinpath("data/report_preprocessing.json").is_file()
assert len(RoomFeatureTransformer.from_report("R").feature_names_) == 8
print("room-rank-features", room_rank_features.__version__, "installed successfully")
PY
```

The dependency ranges are declared in `pyproject.toml`. Resolve and pin the
dependency versions in your application's normal release process; the library's
version alone does not pin the complete application environment.

## Build a Linux image for EKS

Install the wheel while building your application's container image, before
deploying that image to EKS. The library needs no EKS-specific compilation or AWS
SDK. Do not copy a macOS virtual environment into a Linux image.

Save the following as `Dockerfile` in the repository root. This example builds a
library base image; add your application's code and its `CMD` or `ENTRYPOINT` to
make it a running service.

```dockerfile
FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m pip install --no-cache-dir build
COPY pyproject.toml README.md MANIFEST.in ./
COPY src/ ./src/
RUN python -m build --wheel --outdir /wheels

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /wheels/ /wheels/
RUN python -m pip install --no-cache-dir --only-binary=:all: \
        /wheels/room_rank_features-4.3.1-py3-none-any.whl \
    && python -m pip check \
    && rm -rf /wheels
RUN python -c 'import room_rank_features; assert room_rank_features.__version__ == "4.3.1"'
```

Build for the architecture of the nodes that will run the pod:

```bash
# x86-64 EKS nodes
docker build --platform linux/amd64 -t room-feature-library:4.3.1 .

# ARM64 / Graviton EKS nodes
docker build --platform linux/arm64 -t room-feature-library:4.3.1-arm64 .
```

Cross-architecture builds require Docker emulation or a builder of that
architecture. The application image must match the node architecture even though
the library's own wheel works on either platform. Use your existing image registry
and EKS deployment process for the resulting application image. If your service
also uses LightGBM, install it and its Linux runtime dependencies in that service;
this feature library does not install or import it.

## Feature configurations and serving

| Configuration | Model inputs | Count |
| --- | --- | --- |
| `R` | Commercial inputs | 8 |
| `P14` | R + four structured bed columns | 12 |
| `P14_L` | P14 + description length | 13 |
| `P14_SWL` | P14 + 41 parsed non-bed attributes + length + word indicators | 54 + V |
| `R_H2` | R + the two bed-preference features | 10 |
| `P14_H2` | P14 + the two bed-preference features | 14 |
| `P14_L_H2` | P14_L + the two bed-preference features | 15 |
| `P14_SWL_H2` | P14_SWL + the two bed-preference features | 56 + V |

`V` is the vocabulary fitted on training data, capped at 1,200 by default.
The P14 key is a historical name, not its current feature count. All configurations
exclude margin and rate_type. The supplied structured bed columns are authoritative;
text does not replace them. Empty descriptions are supported.

Each `_H2` variant adds only `history_bed_config_match_rate` and
`history_total_beds_abs_diff`, using outcomes available strictly before the current
search. The history store must be prepared separately and passed as `history=`.
For booking timestamps that are already UTC, construct it with
`booking_time_is_utc=True`; the default handles the original dataset's Pacific
clock interpretation. Set actual availability timestamps or delay to match your
serving data.

Fit preprocessing on training data and save its state with `features.save(...)`.
For serving, load the state associated with your trained model:

```python
from room_rank_features import RoomFeatureTransformer

# Example for a saved P14_L transformer. candidate_df has contiguous search rows.
features = RoomFeatureTransformer.load("room_features.json")
X, groups = features.transform(candidate_df)
```

For an `_H2` state, use
`RoomFeatureTransformer.load("room_features.json", history=history)` with a
separately rebuilt history store using the saved settings. JSON saves the feature
order, vocabulary and history configuration, not traveler booking records. The
library does not sort or filter candidates; keep labels and rows aligned in the
calling code. Size application memory for the dense feature matrix, and preserve
whole searches when batching.

Build references: [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [wheel compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).
