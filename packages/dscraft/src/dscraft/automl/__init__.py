"""dscraft.automl: DSCraft's clean-room tabular AutoML module.

This scaffold-depth pass implements the architecture doc's (Part 3,
"Module 1: AutoML") `.compile()` path that fuses a fitted
`sklearn.pipeline.Pipeline` into a single ONNX graph via `skl2onnx`,
replacing brittle pickle-based serialization (see `compile.py` and the
package README for details and clean-room provenance), plus:

- a multi-backend model-selection surface: first-party, caller-selectable
  model backends (`models.py`: `DSLinear`/`DSLogistic` linear models,
  `DSBoost` gradient-boosted trees, `DSRandomForest`/`DSExtraTrees` bagging
  ensembles -- all built from scratch on numpy, per the first-party
  automl/eda rebuild, with the third-party XGBoost/LightGBM/CatBoost
  backends removed; see
  docs/decisions/2026-10-first-party-automl-eda-rebuild.md);
- an unsupervised clustering allowlist (`clustering.py`: first-party
  `DSKMeans`/`DSDBSCAN`/`DSAgglomerative`/`DSHDBSCAN`, built from scratch on
  numpy in `_clustering.py` -- the `sklearn.cluster.HDBSCAN` backend it
  originally wrapped was replaced by the first-party `DSHDBSCAN`, with
  `"HDBSCAN"` kept as a backward-compatible alias), independent of the
  supervised model-selection surface; and
- an imbalanced-class resampling allowlist (`resampling.py`: first-party
  `RandomOverSampler`/`SMOTE`/`BorderlineSMOTE`/`RandomUnderSampler`, built
  from scratch on numpy in `_resampling.py` -- the third-party
  `imbalanced-learn` backends it originally wrapped were replaced by the
  first-party automl/eda rebuild), an optional preprocessing step ahead of
  the classifier path.

The streaming/incremental `partial_fit` fading-factor evaluator and the
PSI drift-detection feature from the same architecture doc section are
explicitly out of scope for this pass -- future work, not implemented
here.

Public API surface (this is the one canonical entrypoint per capability --
no parallel export path exists elsewhere in this package):
    >>> from dscraft.automl import compile, CompileOptions
    >>> from dscraft.automl import build_model, SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS
    >>> from dscraft.automl import build_clusterer, SUPPORTED_CLUSTERERS
    >>> from dscraft.automl import build_resampler, SUPPORTED_RESAMPLERS
"""

from dscraft.automl.clustering import SUPPORTED_CLUSTERERS, build_clusterer
from dscraft.automl.compile import (
    CompileOptions,
    ONNXExtraNotInstalledError,
    compile,  # noqa: A004 -- intentional public API name, see architecture doc
)
from dscraft.automl.models import SUPPORTED_CLASSIFIERS, SUPPORTED_REGRESSORS, build_model
from dscraft.automl.resampling import SUPPORTED_RESAMPLERS, build_resampler

__all__ = [
    "compile",
    "CompileOptions",
    "ONNXExtraNotInstalledError",
    "SUPPORTED_CLASSIFIERS",
    "SUPPORTED_REGRESSORS",
    "build_model",
    "SUPPORTED_CLUSTERERS",
    "build_clusterer",
    "SUPPORTED_RESAMPLERS",
    "build_resampler",
]

__version__ = "0.2.1"
