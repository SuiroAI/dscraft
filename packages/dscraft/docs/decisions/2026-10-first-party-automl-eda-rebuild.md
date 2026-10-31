# 2026-10 first-party rebuild of `dscraft.automl` and `dscraft.eda`

- **Status:** Accepted (plan of record; implemented incrementally across the multi-PR sequence below)
- **Scope:** The modeling, boosting, resampling, clustering, probabilistic-sketch, and
  statistical-association *algorithms* in `dscraft.automl` and `dscraft.eda` are to be
  reimplemented first-party (in-house), replacing the third-party *algorithm* libraries they
  currently delegate to. Normal foundational/substrate libraries are retained.
- **Supersedes (in part):** the multi-backend "wrap XGBoost/LightGBM/CatBoost / imbalanced-learn /
  sklearn HDBSCAN / Apache DataSketches" posture documented in `models.py`, `clustering.py`,
  `resampling.py`, and `sketches.py`. See "Relationship to the multi-backend design principle"
  below for how this interacts with that locked principle.

## Why this rebuild exists

DSCraft's stated purpose is to be a unified, one-stop data-science library that brings the whole
workflow together in one place. For the *auto-modeling* (`dscraft.automl`) and *auto-EDA*
(`dscraft.eda`) capability areas specifically, the directive is that the **modeling, boosting, and
techniques themselves must be our own, end to end** — not thin dispatch wrappers around external
algorithm libraries. The orchestration/composition layer in both subpackages is already
first-party and clean-room (there is no PyCaret / auto-sklearn / FLAML / ydata-profiling / sweetviz
/ etc. anywhere in the tree). What remains third-party is specifically the *algorithm backends*,
and those are what this rebuild makes first-party.

This is explicitly **not** a license-forced clean-room reimplementation in the sense of the
cleanlab (AGPL) or PyCaret (FSL-1.1-MIT non-compete) mitigations in CLAUDE.md's LazyIsolate policy
— every library being replaced here is permissively licensed (XGBoost Apache-2.0, LightGBM MIT,
CatBoost Apache-2.0, imbalanced-learn MIT, scikit-learn BSD, Apache DataSketches Apache-2.0) and
could legally be depended on indefinitely. The driver is a *product* decision — these capability
areas should be "something we built" — not a licensing obligation.

## The acceptable-substrate vs. must-be-first-party line

The dividing line applied throughout this rebuild:

**Acceptable to keep (normal-library substrate / plumbing):**

- `numpy`, `pandas`, `polars` — array/dataframe/lazy-execution substrate.
- `scipy` *primitives* (linear algebra, special functions, FFT) — numerical building blocks, not
  "the technique." Note: `scipy.stats`' *statistical tests themselves* (e.g. `pearsonr`,
  `chi2_contingency`) are treated as techniques to internalize (see PR 14), not primitives.
- `scikit-learn` *plumbing only*: `BaseEstimator`/mixins for the estimator protocol,
  `check_is_fitted`/validation utilities, `Pipeline`. **No scikit-learn estimator *algorithm*** is
  used in our code (no `sklearn.ensemble`, `sklearn.cluster`, `sklearn.linear_model` models). This
  distinction matters because `dscraft.automl.compile` legitimately needs
  `sklearn.pipeline.Pipeline` as the thing it converts to ONNX — that stays.
- The ONNX export stack (`skl2onnx`/`onnx`/`onnxruntime`) in `compile.py` — a serialization/export
  format, not a modeling algorithm. Stays (and remains the optional `automl-onnx` extra).
- `plotnine` in `eda/plots.py` — a rendering backend, not an EDA technique. Stays (optional
  `eda-plotnine` extra).
- The dependency-free HTML/Canvas report renderer (`eda/report.py`) and the Polars lazy engine
  (`eda/engine.py`) — already fully first-party.

**Must become first-party (the "modeling, boosting, and techniques"):**

| Current third-party backend | Where | Becomes |
|---|---|---|
| XGBoost / LightGBM / CatBoost (gradient boosting) | `automl/models.py` | in-house GBDT + linear + forest estimators |
| imbalanced-learn (SMOTE / over- / under-sampling) | `automl/resampling.py` | in-house resamplers |
| `sklearn.cluster.HDBSCAN` (+ no other clusterer today) | `automl/clustering.py` | in-house k-means / DBSCAN / agglomerative / HDBSCAN-equivalent |
| Apache DataSketches (HLL / KLL) | `eda/sketches.py` | in-house HyperLogLog + streaming-quantile sketch |
| `scipy.stats` correlation/chi-squared *tests* | `eda/associations.py` | in-house Pearson/Spearman/Kendall/chi² (Cramér's V, Theil's U, correlation ratio are already ours) |

## The sklearn-compatibility contract (how we avoid breaking `compile()` and existing pipelines)

Every first-party estimator this rebuild introduces is **scikit-learn API-compatible** — it
subclasses the appropriate `sklearn.base` mixin and implements the standard
`fit`/`predict`/`predict_proba`/`get_params`/`set_params` protocol, and sets the standard fitted
attributes (`n_features_in_`, `classes_`, etc.). This is deliberate and is *not* a violation of the
"no sklearn algorithms" rule: we use sklearn's *estimator protocol/base classes* (plumbing) while
the fit/predict *math* is entirely ours. The payoff is that:

- `dscraft.automl.compile()` keeps working unchanged — a `Pipeline` containing our estimators still
  satisfies `check_is_fitted` and still has a defined `n_features_in_`.
- Resamplers keep the `fit_resample(X, y) -> (X_res, y_res)` contract the existing tests and the
  `README` document, so `dscraft.automl`'s public surface does not change shape.
- `LazyEDA` is untouched: the first-party `HLLResult`/`KLLResult`/`estimate_cardinality`/
  `estimate_quantiles` and the association functions preserve their exact public signatures and
  documented error-bound semantics.

## Relationship to the multi-backend design principle

CLAUDE.md's "expose ALL libraries that serve the same purpose as selectable options, never pick a
single winner" principle was written about *which third-party backends to surface*. This rebuild
changes the thing being surfaced: the `SUPPORTED_*` allowlist dispatch pattern is **kept** (it is a
good internal design and the tests assert against it), but its entries become our own
implementations rather than external libraries. Where we offer several styles (e.g. multiple model
families, multiple clusterers), the allowlist still exposes all of them as equally-selectable, none
a hard-coded default — so the *spirit* of the principle (user-selectable breadth, no silent winner)
is preserved while the *letter* (wrap external libs) is superseded for these specific algorithm
areas. The third-party algorithm libraries are **removed** from the `automl`/`eda`/`all` extras as
each in-house replacement lands, because "ours end to end" is the whole point — leaving them as
optional backends would reintroduce exactly the dependency this rebuild exists to remove.

## PR sequence (plan of record)

Each PR is a self-contained, fully-tested, green, installable unit. Third-party algorithm deps are
removed from `pyproject.toml` only in the PR that lands their complete first-party replacement.

**Phase 0 — plan**
- **PR 0** — this decision document. *(no code)*

**Phase 1 — shared first-party ML primitives (`automl`)**
- **PR 1** — core numeric/optimization primitives: loss & link functions, a GD/Newton/IRLS solver
  core, split-finding helpers, metrics (accuracy/logloss/RMSE/R²).
- **PR 2** — first-party CART decision tree (classification + regression), the substrate for the
  forest and boosting PRs.

**Phase 2 — first-party modeling backends (`automl/models.py`)**
- **PR 3** — first-party linear models (ridge/OLS regressor, logistic/softmax classifier,
  elastic-net regularization).
- **PR 4** — first-party gradient boosting (`DSBoost*`), the XGBoost/LightGBM/CatBoost replacement.
- **PR 5** — first-party random forest / extra-trees ensembles.
- **PR 6** — retire `xgboost`/`lightgbm`/`catboost` from the `automl`/`all` extras; flip the
  `SUPPORTED_*` allowlists to the first-party backends; update tests/README/docstrings.

**Phase 3 — first-party resampling (`automl/resampling.py`)**
- **PR 7** — first-party `RandomOverSampler`/`RandomUnderSampler`/`SMOTE` (+ Borderline-SMOTE).
- **PR 8** — retire `imbalanced-learn` from the `automl`/`all` extras; update tests/README.

**Phase 4 — first-party clustering (`automl/clustering.py`)**
- **PR 9** — first-party `DSKMeans` + `DSDBSCAN`.
- **PR 10** — first-party agglomerative + density-based hierarchical (HDBSCAN-equivalent),
  replacing `sklearn.cluster.HDBSCAN`; update tests/README.

**Phase 5 — first-party EDA sketches (`eda/sketches.py`)**
- **PR 11** — first-party HyperLogLog (preserving `HLLResult`/`estimate_cardinality` + RSE).
- **PR 12** — first-party streaming quantile sketch (preserving `KLLResult`/`estimate_quantiles` +
  rank-error).
- **PR 13** — retire `datasketches` from the `eda`/`all` extras; update tests/README.

**Phase 6 — first-party EDA breadth ("all styles of EDA")**
- **PR 14** — internalize the remaining `scipy.stats` correlation/chi² calls in `associations.py`.
- **PR 15** — univariate distribution profiling (moments, robust stats, shape/entropy).
- **PR 16** — outlier/anomaly detection (robust-z/MAD, IQR fences, multivariate screen) — the
  deferred LazyEDA item #1.
- **PR 17** — temporal/time-series EDA (range coverage, gap/cadence, seasonality/autocorrelation) —
  the deferred LazyEDA item #2; built standalone, no `forecast` import (per §2.9).
- **PR 18** — bivariate + dimensionality-reduction EDA (first-party PCA).

**Phase 7 — integration & reconciliation**
- **PR 19** — update `examples/automl` + `examples/eda` (and quickstarts) to the first-party
  backends; add an integration test proving a first-party pipeline still `.compile()`s to ONNX;
  confirm no third-party algorithm lib remains in the two subpackages' extras.
- **PR 20** — reconcile `DSCraft_Unified_Architecture.md` (AutoML / LazyEDA module entries) and
  `CLAUDE.md`'s subpackage table to record that modeling/boosting/sketches are now first-party.

## Non-goals / out of scope for this rebuild

- Replacing acceptable substrate (numpy/pandas/polars/scipy-primitives/sklearn-plumbing/ONNX/
  plotnine) — explicitly retained.
- The streaming `partial_fit` fading-factor evaluator and PSI drift detection named in the
  architecture doc's AutoML entry — still out of scope, unchanged by this rebuild.
- Any cross-module (`eda` ↔ `forecast`) data contract — still deferred per §2.9; PR 17's temporal
  EDA is built standalone.
- Matching any specific external library's numeric output bit-for-bit — the goal is correct,
  well-tested, first-party algorithms with documented behavior, validated for *reasonable* accuracy
  parity on standard datasets, not byte-identical reproduction of a particular implementation.
