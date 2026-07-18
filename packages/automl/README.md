# benchcraft-automl

Benchcraft's clean-room tabular AutoML module (internal codename "AutoML",
architecture doc Part 3 "Module 1: AutoML"). This is a **scaffold-depth
pass**, not a full implementation of the module's eventual scope.

## What this package is (and isn't) right now

The full AutoML module is designed around three signature capabilities:
streaming/incremental optimization via `partial_fit` with a fading-factor
running-metric evaluator, zero-config PSI-based drift detection, and a
`.compile()` path that fuses a fitted pipeline into a single ONNX graph.

**This package currently implements exactly one of those three: `.compile()`.**
The streaming `partial_fit` evaluator and PSI drift detection are
explicitly out of scope for this pass -- they are future work, not
partially stubbed out here.

## The signature capability: `.compile()`

`benchcraft_automl.compile()` takes a **fitted** `sklearn.pipeline.Pipeline`
and a representative sample input, and returns a single, self-contained
`onnx.ModelProto` via `skl2onnx.convert_sklearn` -- every step of the
pipeline (scaler, encoder, estimator, ...) fused into one ONNX graph.

Why this matters (per the architecture doc's motivating diagnosis in
Appendix A): pickle-based serialization of a trained pipeline is fragile
across environment/version drift between training and serving -- a
pipeline pickled against one scikit-learn/numpy version can silently break
(or silently misbehave) when unpickled against a different one at serving
time, because pickle encodes Python object internals, not a portable model
representation. A compiled ONNX graph has no such dependency: it can be
loaded by `onnxruntime.InferenceSession` in a completely different
environment/language, with no scikit-learn install required at all.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchcraft_automl import compile, CompileOptions

pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
pipeline.fit(X_train, y_train)

onnx_model = compile(pipeline, X_train, options=CompileOptions(zipmap=False))

import onnxruntime
session = onnxruntime.InferenceSession(onnx_model.SerializeToString())
labels, proba = session.run(None, {session.get_inputs()[0].name: X_test.astype("float32")})
```

See `examples/compile_iris_example.py` for a complete runnable
fit-compile-infer-verify demo, and `tests/test_compile.py` for the
correctness test suite (ONNX predictions checked against the original
sklearn pipeline's own `predict`/`predict_proba`).

### Scope of this pass

- Targets pipelines over a 2-D table of **numeric** features (the common
  `StandardScaler` / linear-model / tree-model case). Pipelines needing
  heterogeneous per-column ONNX type mapping (e.g. a `ColumnTransformer`
  over mixed string+numeric raw columns) are not handled by this
  scaffold-depth pass.
- `compile()` is the **one canonical** export path in this package -- there
  is no second/parallel ONNX export function anywhere else in the codebase.

## Clean-room provenance

Per the architecture doc's licensing policy (§2.2, "source-available
non-compete license" mitigation) and CLAUDE.md's licensing rules: this
package is positioned as a successor to LazyPredict/PyCaret-style tabular
AutoML tools, several of which (notably PyCaret 4.0's core) are licensed
under FSL-1.1-MIT/BUSL-1.1 with a non-compete clause and a delayed 2-year
MIT rollover. **No code from PyCaret, LazyPredict, AutoGluon, FLAML,
MLJAR, or any other AutoML project was read, copied, or adapted to write
this package.** `compile.py` was written directly against the public
`skl2onnx`/ONNX API surface (`skl2onnx.convert_sklearn`,
`skl2onnx.common.data_types.FloatTensorType`, `onnx.checker.check_model`)
and the architecture doc's description of the desired capability. No API
surface (function names, class hierarchy, option names) was copied from
any of those projects either -- `compile()`'s signature and
`CompileOptions` are this package's own design.

TPOT (GPL-3.0) and its DEAP dependency (LGPL) are not in this package's
dependency tree, and never will be, per the architecture doc.

## Dependency surface

Per the architecture doc's AutoML dependency-surface constraint, the core
install is deliberately minimal:

- **Core (always installed):** `numpy`, `pandas`, `scikit-learn`.
- **Optional `onnx` extra:** `skl2onnx`, `onnx`, `onnxruntime` -- lazily
  imported only inside `compile()`, so `import benchcraft_automl` succeeds
  even without this extra installed. Calling `compile()` without it raises
  a clear `ONNXExtraNotInstalledError` telling you what to install.
- **Optional `dev` extra:** `pytest`.

This package also uses `lazycore.data`'s Tier-1 Arrow-tabular helpers
(`is_arrow_backed_pandas`, `pandas_arrow_dtypes`) to validate/report on a
caller's pandas DataFrame input, per the architecture doc's shared
data-tier convention (§2.1) -- it does not reimplement Arrow/pandas
interop helpers of its own. This is used purely for reporting; `compile()`
still coerces to a plain numeric numpy array for `skl2onnx`, since that is
what the ONNX conversion path actually needs, not an Arrow buffer.

`lazycore` is a local sibling package (`packages/lazycore`), not a package
published to PyPI. It **is** declared in `pyproject.toml`'s `dependencies`
(as a bare, unpinned `"benchcraft-core"`, its PyPI distribution name) so
that a resolver run without it already installed fails fast with a clear
"could not find benchcraft-core" error instead of succeeding and then
failing at import time inside
`benchcraft_automl.compile`. That declaration does **not** make a plain
`pip install packages/automl` work in isolation, though -- hatchling/pip
don't have a portable, idiomatic way to express a relative-path dependency
in `pyproject.toml` metadata the way e.g. Poetry's `path = "../lazycore"`
does. You must still install it first (see below), which satisfies the
declared dependency before it's ever resolved against PyPI.

## Installation (local dev)

```bash
# from the repo root
pip install -e packages/lazycore
pip install -e "packages/automl[onnx,dev]"
```

## Running tests

```bash
pytest packages/automl/tests
```

ONNX-dependent tests are skipped (via `pytest.importorskip`), not failed,
if the `onnx` extra isn't installed.

## Running the example

```bash
python packages/automl/examples/compile_iris_example.py
```

Fits a `StandardScaler` + `LogisticRegression` pipeline on
`sklearn.datasets.load_iris`, compiles it with `benchcraft_automl.compile`,
runs it through `onnxruntime.InferenceSession`, and asserts the ONNX
output matches the sklearn pipeline's own `predict`/`predict_proba` within
tolerance.
