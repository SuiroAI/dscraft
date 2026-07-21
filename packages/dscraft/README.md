# dscraft

A unified, MIT-licensed, local-first ML tooling platform: tabular AutoML,
data cleaning, time-series forecasting, graph ML, computer vision, LLM
fine-tuning, LLM/agent red-teaming, and agent/RAG benchmark eval, all
installed and imported as one real Python package — the way numpy or
PyTorch ships one distribution with an internal module tree, not nine
separately-installed packages.

```bash
pip install dscraft                  # base install: just dscraft.core
pip install "dscraft[automl]"        # + dscraft.automl's runtime deps
pip install "dscraft[all]"           # every subpackage's runtime deps
```

```python
import dscraft.core        # always available
import dscraft.automl      # available once installed with the `automl` extra
```

Each subpackage below is a **scaffold-depth pass**: a real, working slice
of its eventual scope (per `DSCraft_Unified_Architecture.md`), not a full
implementation. See that document for the full architecture, locked
design decisions, and per-module roadmap; this README only orients you to
what's here today and how to install/run it.

## Subpackages

| Subpackage | Extra | What it does |
|---|---|---|
| [`dscraft.core`](#dscraftcore) | *(base install)* | Shared substrate: three-tier data conventions, OTel GenAI telemetry helpers, license-isolation policy, shared sandbox executor. |
| [`dscraft.automl`](#dscraftautoml) | `automl` (+ `automl-onnx`) | Clean-room tabular AutoML — `.compile()` fuses a fitted `sklearn.pipeline.Pipeline` into one portable ONNX graph via `skl2onnx`. |
| [`dscraft.clean`](#dscraftclean) | `clean` | Data-quality firewall — ONNX Runtime (PyTorch-free) text embeddings feeding cosine-similarity near-duplicate detection. |
| [`dscraft.forecast`](#dscraftforecast) | `forecast` | Classical statistical forecasting (the full Nixtla `statsforecast` classical catalog: AutoARIMA/AutoETS/AutoCES/AutoTheta autofits, simple baselines, Croston) over a Tier-1 Arrow-backed pipeline, plus a basic backtest report. |
| [`dscraft.graph`](#dscraftgraph) | `graph` | Sparse graph ML — a concrete Tier-2 COO↔CSR/CSC tensor adapter (PyG↔SciPy) plus a minimal GCN forward pass. |
| [`dscraft.vision`](#dscraftvision) | `vision` | Computer vision — a concrete Tier-3 dense image pipeline (decode→augment→tensor) plus a small CNN exported via `torch.export()`→ONNX. |
| [`dscraft.tune`](#dscrafttune) | `tune` | Local LLM fine-tuning — an Adapter-Factory `BaseTrainingAdapter` interface with a `ProgrammaticAdapter` doing real (tiny) LoRA fine-tuning via `peft`/`transformers`. |
| [`dscraft.security`](#dscraftsecurity) | `security` | LLM red-teaming — a `BaseSecurityAdapter` running a real prompt-injection probe/detector loop against a local target inside the shared sandbox, OWASP-mapped findings. |
| [`dscraft.agent`](#dscraftagent) | `agent` | Agent/benchmark eval — a bring-your-own-agent `AgentAdapter` executing file-manipulation tool-use tasks inside the shared sandbox, scored for pass rate and latency. |
| [`dscraft.eda`](#dscrafteda) | `eda` | Exploratory data analysis — a lazy Polars profiling engine, HLL/KLL sketches, a mixed-type association matrix, and a self-contained HTML/Canvas report, composed behind one `LazyEDA` entry point. |

## Installation

```bash
# Base install — just dscraft.core (opentelemetry-api only)
pip install dscraft

# One subpackage's runtime deps
pip install "dscraft[forecast]"

# AutoML's optional ONNX export path (on top of the `automl` extra)
pip install "dscraft[automl,automl-onnx]"

# Everything (all nine subpackages' runtime deps)
pip install "dscraft[all]"
```

### Local development

```bash
cd /path/to/this/repo
pip install -e "packages/dscraft[dev,all]"
pytest packages/dscraft/tests
```

`dev` adds `pytest` plus test-only dependencies (`pandas`, `polars`,
`pyarrow`, `statsmodels`) that some subpackages' test suites need but
their runtime code does not. `dev` is deliberately kept free of any
"heavy" dependency (torch, onnx, transformers, scikit-learn) so that
`pip install -e "packages/dscraft[dev]"` alone stays minimal — those only
come in via a subpackage's own extra (e.g. `automl`, `vision`) or via
`all`. Installing `all` alongside `dev` is what lets the *entire* combined
test suite (all nine subpackages) run in one environment.

To run a single subpackage's tests in isolation, install just its extra:

```bash
pip install -e "packages/dscraft[forecast,dev]"
pytest packages/dscraft/tests/forecast
```

`dscraft.vision`'s test suite is the one exception: its real-dataset
validation test uses `sklearn.datasets.load_digits()` (test-only, not a
`vision` runtime dependency), so running it in isolation needs
`scikit-learn` from somewhere — either add the `automl` extra
(`dscraft[vision,automl,dev]`, since `automl` already depends on
`scikit-learn`) or install `scikit-learn` directly alongside
`dscraft[vision,dev]`. The full combined install (`dscraft[all,dev]`)
always has it, via `automl`.

## `dscraft.core`

The thin, shared substrate underneath every other subpackage: three-tier
data/tensor conventions (Tier 1 Arrow-backed pandas/Polars, Tier 2 sparse
graph tensor adapters, Tier 3 dense media pipelines), OpenTelemetry
GenAI-schema telemetry helpers, the license-isolation policy table and
model-tier allowlist mechanism, and the shared sandbox executor
(`SandboxPolicy` + `BaseSandboxExecutor`, with a real `SeatbeltSandboxExecutor`
on macOS) used by both `dscraft.security` and `dscraft.agent`. Its only
runtime dependency is `opentelemetry-api`; it never depends on pandas,
polars, torch, or any ML framework. Always installed — no extra needed.

## `dscraft.automl`

`dscraft.automl.compile()` takes a **fitted** `sklearn.pipeline.Pipeline`
and returns a single, portable `onnx.ModelProto` via `skl2onnx`, fusing
every pipeline step into one graph loadable by `onnxruntime` with no
scikit-learn install required at serving time. Base runtime deps
(`numpy`, `pandas`, `scikit-learn`) install via the `automl` extra;
`skl2onnx`/`onnx`/`onnxruntime` (needed only for `.compile()` itself, and
lazily imported) install via the separate `automl-onnx` extra.

`dscraft.automl.build_model(name, task, **kwargs)` builds an unfitted,
sklearn-compatible gradient-boosted-tree estimator from a caller-selected
backend -- XGBoost, LightGBM, or CatBoost, all three equally supported per
this project's multi-backend design principle (none is a "default").
`SUPPORTED_CLASSIFIERS`/`SUPPORTED_REGRESSORS` are the name -> class
allowlists `build_model` dispatches through, mirroring
`dscraft.forecast`'s `SUPPORTED_MODELS` pattern. All three libraries are
base runtime deps of the `automl` extra (not a separate extra).

`dscraft.automl.build_clusterer(name, **kwargs)` builds an unfitted,
sklearn-compatible clustering estimator -- currently HDBSCAN, via
scikit-learn's own built-in `sklearn.cluster.HDBSCAN` (no new dependency
needed), exposed through `SUPPORTED_CLUSTERERS`. This is an independent,
unsupervised capability -- it does not require the supervised model
allowlist above.

`dscraft.automl.build_resampler(name, **kwargs)` builds an unfitted
`imbalanced-learn` resampler -- `RandomOverSampler`, `SMOTE`, or
`RandomUnderSampler`, exposed through `SUPPORTED_RESAMPLERS` -- for
rebalancing a skewed classification training set ahead of the classifier
path above. `imbalanced-learn` is a base runtime dep of the `automl`
extra.

## `dscraft.clean`

A data-quality firewall for a training DataFrame. The primary entrypoint is
`Sanitizer`, which composes three independently-implemented capabilities
into one audit-then-clean workflow:

- **DeCoLe** (`label_errors.py`) — group-conditioned Confident Learning:
  per-(group, class) confidence thresholds instead of one global threshold,
  avoiding the standard confident-learning failure mode of over-pruning a
  lower-confidence-but-correctly-labeled group's examples.
- **Train/test contamination auditing** (`contamination.py`) — a two-stage
  pipeline: cheap LSHBloom MinHash/Bloom-filter candidate screening (via
  `datasketch`) over every row, then optional Min-K%++ log-probability
  validation for stage-1 candidates (only run when the caller supplies
  precomputed per-token log-probabilities from a language model this
  module never runs itself).
- **Dataset Integrity Score** (`integrity.py`) — a single weighted scalar
  combining the label-error rate, the contamination rate, and
  train/test demographic-group drift (Jensen-Shannon divergence).

```python
from dscraft.clean import Sanitizer

sanitizer = Sanitizer(train_df, target_col="text", label_col="label", group_col="group")
report = sanitizer.audit(test_df, out_of_sample_probs)  # from your own k-fold CV
print(report.integrity_report.score)  # 0.0 (worst) .. 1.0 (best)

cleaned_df = report.purge(strategy="demographic-preserving", output_path="cleaned.parquet")
```

`report.purge()`'s `"demographic-preserving"` strategy removes flagged
label-error rows and training rows matched to validated-contaminated test
items, while capping how much any single demographic group can be pruned
relative to the dataset-wide removal rate — see `SanitizerReport.purge`'s
docstring for the exact, documented algorithm.

`detect_near_duplicate_text` remains available as the lower-level building
block it always was (ONNX Runtime, deliberately PyTorch-free, text
embeddings feeding cosine-similarity near-duplicate detection — a scaffold
of the LazyClean module's D4 semantic-dedup idea; the IVF-HNSW/spherical
k-means scale-out path is still out of scope, see `dedup.py`). Zero-vector
("no extractable features") rows are honestly reported as "not
comparable," distinct from both confirmed-duplicate and confirmed-distinct
pairs. Install via the `clean` extra.

## `dscraft.forecast`

Classical statistical forecasting via Nixtla's `statsforecast` over a
Tier-1 Arrow-backed input pipeline, with a basic train/test backtest
reporting MAE/RMSE. `SUPPORTED_MODELS` covers the full classical catalog
`statsforecast` ships out of the box: autoregressive/exponential-smoothing
autofits (`AutoARIMA`, `AutoETS`, `AutoCES`, `AutoTheta`), simple baselines
(`Naive`, `SeasonalNaive`, `RandomWalkWithDrift`, `HistoricAverage`,
`WindowAverage`, `SeasonalWindowAverage`), and the Croston family for
intermittent-demand series (`CrostonClassic`, `CrostonOptimized`,
`CrostonSBA`) — zero new dependencies, since every one of these classes
already ships in `statsforecast`. Backtest MAE/RMSE are computed via
Nixtla's `utilsforecast` (the shared metrics/plotting/preprocessing-helper
layer the rest of the nixtlaverse already assumes) rather than
hand-rolled, so future backends added to this subpackage share one
canonical metric implementation. `decompose()` exposes STL/MSTL time
series decomposition (plus an optional Box-Cox transform) as a
standalone, inspectable diagnostic -- trend/seasonal/residual components
returned as a tidy long-format DataFrame, not folded silently into
`forecast()` -- via `statsmodels`/`scipy`, both already transitively
available in this extra. The tree-based ML branch, TSFM zero-shot branch,
self-healing preprocessing, and conformal-prediction leaderboard from the
full LazyForecast design are deferred. Install via the `forecast` extra.

In addition to the hermetic synthetic-panel tests and the `statsmodels`-
bundled co2/nile real-dataset validation, this subpackage's test suite
also validates `forecast()`/`backtest()` against a real published
forecasting-competition benchmark (the M3 competition's "Other" group) via
Nixtla's `datasetsforecast` -- a test-only dependency (like `statsmodels`)
that downloads its data over the network on first use and is skipped
(not failed) when offline; see
`tests/forecast/test_datasetsforecast_validation.py`.

## `dscraft.graph`

`PyGSparseAdapter` is the first concrete implementation of
`dscraft.core.data.SparseGraphTensorAdapter`: a real COO↔CSR/CSC bridge
between PyTorch Geometric edge-index tensors and `scipy.sparse`, since
DLPack cannot represent sparsity. `GCN` is a minimal two-layer graph
convolutional network built on `torch_geometric.nn.GCNConv` that consumes
the adapter directly. Install via the `graph` extra.

## `dscraft.vision`

`SimpleImagePipeline` is the first concrete implementation of
`dscraft.core.data.DenseMediaPipeline` (decode via Pillow → augment
resize/flip → dense `torch.Tensor`, DLPack-ready). `TinyCNN` is a small
LeNet-style classifier captured via `torch.export()` and exported to ONNX,
proving the export path end-to-end. `run_ocr(image, backend=...)` adds OCR
as a second, independent capability with a selectable backend — `"easyocr"`
(PyTorch-based, MPS/GPU-capable, no external binary) or `"tesseract"` (via
`pytesseract`, CPU-only, requires the separate system `tesseract` binary —
e.g. `brew install tesseract` on macOS) — per the multi-backend design
principle, neither is hard-coded as the only option. Both return a
comparably-shaped `OCRResult` (text plus per-detection bounding
boxes/confidence). Install via the `vision` extra (both OCR backends'
pip packages ship with it; the `tesseract` backend additionally needs the
system binary installed separately).

## `dscraft.tune`

`BaseTrainingAdapter` is a minimal Adapter-Factory interface
(`prepare`/`train_step`/`save_adapter`); `ProgrammaticAdapter` is the one
concrete implementation, running a real (tiny) LoRA fine-tuning step on a
small local causal LM via `peft` + `transformers` — genuine forward +
backward + optimizer-step training, not a mock. Subprocess-isolated
adapters (torchtune/Axolotl), multi-fidelity BOHB tuning, and real
GGUF/MLX export are deferred. Install via the `tune` extra.

## `dscraft.security`

A minimal, real, end-to-end slice of the LazyRed module: `BaseSecurityAdapter`
runs one probe (`PromptInjectionAdapter`) against a deliberately vulnerable
local target function, executed through the shared `dscraft.core.sandbox`
executor, with findings mapped to the OWASP LLM Top 10 and reported via
`dscraft.core.telemetry`. No extra heavy runtime deps beyond `dscraft.core`
itself — install via the `security` extra.

## `dscraft.agent`

A minimal, real bring-your-own-agent benchmark loop: `SandboxedAgentAdapter`
always executes an agent's chosen action through a caller-supplied
`dscraft.core.sandbox.BaseSandboxExecutor`; a small file-manipulation task
family (pass-designed and sandbox-escape-attempt variants) proves the
sandbox genuinely drives the scored outcome; a tiny benchmark runner
reports aggregate pass rate and latency via `dscraft.core.telemetry`. No
extra heavy runtime deps beyond `dscraft.core` itself — install via the
`agent` extra.

## `dscraft.eda`

`LazyEDA` is the single entry point over four independently-built pieces:
a lazy-Polars profiling `engine` (schema + per-column null counts + row
count, computed without materializing a source's full data), `sketches`
(HyperLogLog cardinality and KLL quantile estimation, via the Apache
Software Foundation's `datasketches` library), `associations` (a mixed
continuous/categorical/mixed-type pairwise correlation/association-matrix
suite built on SciPy), and a `report` renderer that turns already-
aggregated summaries into one self-contained, offline-friendly HTML/Canvas
document — no external CDN references, no charting library, just inlined
CSS/JS.

```python
from dscraft.eda import LazyEDA

profile = LazyEDA().profile("orders.parquet")
print(profile.null_report.columns_with_nulls())
print(profile.association_matrix.columns)

profile.export("orders_eda_report.html")
```

`LazyEDA.profile` routes each column by the coarse category
`engine.profile_schema` already assigns it: `"numeric"` columns get a KLL
quantile pass (min/p25/p50/p75/max by default) plus an equal-width
histogram; `"string"` columns get an HLL cardinality estimate plus a
top-K value-frequency histogram; `"boolean"`/`"temporal"`/`"other"`
columns get neither sketch (see the module docstring in
`dscraft/eda/__init__.py` for the full rationale). The returned
`EDAProfile` exposes every intermediate result — the schema/null reports,
the per-column `KLLResult`/`HLLResult` sketches, and the
`AssociationMatrixResult` — as inspectable attributes, not just an
`.export()` side effect, matching this platform's convention (see
`dscraft.clean`'s `SanitizerReport`) of returning programmatically usable
result objects. Outlier/anomaly detection and time-series-specific EDA
(despite `dscraft.forecast` being an obvious future consumer) are out of
scope for this pass — see `DSCraft_Unified_Architecture.md`'s LazyEDA
module entry for what's deferred. Install via the `eda` extra.

For publication-quality, static, layered exploratory plots (as an
alternative to — not a replacement for — the interactive single-file HTML
report above), `dscraft.eda.plots` exposes `association_heatmap` and
`column_distribution`, which turn an `AssociationMatrixResult` or a
`ColumnSummary` you already have (e.g. from `profile.association_matrix`
or `profile.report_data.column_summaries`) into a `plotnine.ggplot`
object you can further customize and `.save(path)` yourself:

```python
from dscraft.eda import LazyEDA, association_heatmap, column_distribution

profile = LazyEDA().profile("orders.parquet")
association_heatmap(profile.association_matrix).save("associations.png")

summaries = {s.name: s for s in profile.report_data.column_summaries}
column_distribution(summaries["amount"]).save("amount_distribution.png")
```

This is an optional, separately-gated capability: install the `eda-plotnine`
extra (`pip install "dscraft[eda,eda-plotnine]"`) to use it — the base `eda`
extra alone does not pull in `plotnine`/matplotlib.

## Further reading

See `DSCraft_Unified_Architecture.md` at the repo root for the full
locked v1 architecture — module scope, algorithms, licensing policy, and
what's deferred to later phases. Each subpackage's own docstrings and
`packages/dscraft/tests/<subpackage>/` cover implementation-level detail
this README intentionally omits.
