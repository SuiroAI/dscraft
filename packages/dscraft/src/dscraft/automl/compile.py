"""The `.compile()` path: fuse a fitted sklearn Pipeline into one ONNX graph.

Per the architecture doc (Part 3, "Module 1: AutoML", and Appendix A's
AutoML diagnosis), pickle-based serialization of a trained pipeline is
fragile across environment/version drift between training and serving --
the pickle format encodes Python object internals, not a portable model
representation, so a pipeline pickled with one scikit-learn/numpy version
can silently fail (or worse, silently misbehave) when unpickled against a
different one at serving time. `.compile()` replaces that with a single,
self-contained ONNX graph via `skl2onnx.convert_sklearn`, which has no
dependency on the Python/scikit-learn version that produced it -- only on
the ONNX opset it targets.

This module is a **clean-room implementation**. It was written from the
public `skl2onnx`/ONNX API surface and the architecture doc's description
of the capability, not by reading or adapting PyCaret (or any other
source-available/non-compete-licensed project's) source code. See the
package README fragment for the full clean-room provenance note required
by CLAUDE.md's licensing policy (architecture doc §2.2, "source-available
non-compete license" mitigation).

Scope for this pass (deliberately narrow, per the AutoML module's
scaffold-depth build task):

- Only the `.compile()` -> ONNX capability is implemented here. The
  streaming/incremental `partial_fit` fading-factor evaluator and the PSI
  drift-detection feature described elsewhere in the architecture doc are
  explicitly out of scope for this module file and this package version.
- `compile()` targets pipelines whose input is a 2-D table of numeric
  features (the common case for `StandardScaler` / tree / linear-model
  pipelines). Pipelines with heterogeneous/categorical raw-column ONNX
  type mapping (e.g. `ColumnTransformer` over mixed string+numeric input)
  are not handled by this scaffold-depth pass; `_as_numeric_array` will
  raise a clear `TypeError` if the input can't be coerced to a numeric
  array.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from dscraft.core.data import is_arrow_backed_pandas, pandas_arrow_dtypes

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    import onnx
    import pandas as pd

__all__ = [
    "compile",
    "CompileOptions",
    "ONNXExtraNotInstalledError",
]


class ONNXExtraNotInstalledError(ImportError):
    """Raised by :func:`compile` when the optional ``onnx`` extra is missing.

    ONNX export is a lazy-loaded optional extra of this package (per the
    architecture doc's AutoML dependency-surface constraint), not a hard
    dependency -- ``import dscraft.automl`` must succeed without
    ``skl2onnx``/``onnx``/``onnxruntime`` installed. This error is only
    raised the moment :func:`compile` is actually called.
    """


@dataclass(frozen=True)
class CompileOptions:
    """Optional knobs for :func:`compile`, kept intentionally small.

    Attributes:
        target_opset: ONNX opset to target. ``None`` lets `skl2onnx` choose
            its default (recommended unless the serving runtime pins a
            specific opset).
        doc_string: Free-text description embedded in the resulting
            `onnx.ModelProto`.
        input_name: Name of the single graph input tensor.
        zipmap: Whether to keep skl2onnx's default `ZipMap` output for
            classifiers (maps class labels to probabilities as a
            dictionary-shaped output). Set to ``False`` to get a plain
            tensor output instead, which is usually what you want for a
            single, uniform-dtype numeric feature space fed straight into
            `onnxruntime.InferenceSession`.
    """

    target_opset: int | None = None
    doc_string: str = ""
    input_name: str = "input"
    zipmap: bool = False


def _as_numeric_array(sample_input: Any) -> np.ndarray:
    """Coerce ``sample_input`` to a 2-D float32 numpy array.

    Where the caller passes a pandas DataFrame, this uses `dscraft.core.data`
    (the Tier-1 Arrow-tabular helpers, architecture doc §2.1) purely to
    *report* on the frame's dtype backing -- it does not force an Arrow
    conversion, because `skl2onnx`/ONNX Runtime need a plain numeric numpy
    array, not an Arrow buffer. Reusing `dscraft.core.data` here is about
    reusing the platform's shared validation/reporting convention, not
    about routing through Arrow because Arrow happens to be Tier 1.
    """
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core dependency
        pd = None  # type: ignore[assignment]

    if pd is not None and isinstance(sample_input, pd.DataFrame):
        if not is_arrow_backed_pandas(sample_input):
            warnings.warn(
                "compile() received a pandas DataFrame that is not fully "
                "ArrowDtype-backed (see dscraft.core.data.is_arrow_backed_pandas). "
                "This does not block compilation -- the frame will be "
                "coerced to a plain numpy array for skl2onnx -- but it does "
                "mean this frame is not on the Tier-1 zero-copy Arrow path "
                "described in the architecture doc §2.1.",
                stacklevel=3,
            )
        else:
            # Purely informational: confirms which columns are already on
            # the Tier-1 Arrow path. Not used to change compilation
            # behavior.
            pandas_arrow_dtypes(sample_input)
        try:
            array = sample_input.to_numpy(dtype=np.float32, copy=True)
        except (ValueError, TypeError) as exc:
            bad_columns = [
                str(col)
                for col in sample_input.columns
                if not pd.api.types.is_numeric_dtype(sample_input[col])
            ]
            detail = (
                f" Non-numeric column(s): {', '.join(bad_columns)}."
                if bad_columns
                else ""
            )
            raise TypeError(
                "compile() requires sample_input to be coercible to a 2-D "
                "float32 numeric array, but the provided pandas DataFrame "
                f"contains values that cannot be converted to float32.{detail} "
                f"Original error: {exc}"
            ) from exc
    else:
        try:
            array = np.asarray(sample_input, dtype=np.float32)
        except (ValueError, TypeError) as exc:
            raise TypeError(
                "compile() requires sample_input to be coercible to a 2-D "
                "float32 numeric array, but the provided value contains "
                f"data that cannot be converted to float32 (dtype "
                f"{getattr(sample_input, 'dtype', type(sample_input).__name__)!r}). "
                f"Original error: {exc}"
            ) from exc

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise TypeError(
            "compile() requires sample_input to be coercible to a 2-D "
            f"numeric array (rows x features); got array with shape "
            f"{array.shape!r}."
        )
    return array


def _require_onnx_stack() -> tuple[Any, Any]:
    """Import the optional ONNX stack, or raise a clear, actionable error.

    Kept as its own function (rather than inline try/except in `compile`)
    so the import boundary -- and therefore the "this is a lazy-loaded
    optional extra" contract -- is in exactly one place.
    """
    try:
        import onnx
        import skl2onnx
    except ImportError as exc:
        raise ONNXExtraNotInstalledError(
            "compile() requires the optional 'automl-onnx' extra (skl2onnx, "
            "onnx, onnxruntime) on top of the 'automl' extra (numpy, "
            "pandas, scikit-learn) that compile() itself depends on. "
            "Install both together with:\n"
            '    pip install "dscraft[automl,automl-onnx]"\n'
            "This keeps ONNX export opt-in and separate from AutoML's "
            "core numpy/pandas/scikit-learn dependency surface, per the "
            "architecture doc's AutoML dependency-surface constraint."
        ) from exc
    return onnx, skl2onnx


def compile(  # noqa: A001 -- intentional public API name, see architecture doc
    pipeline: Pipeline,
    sample_input: "pd.DataFrame | np.ndarray",
    *,
    options: CompileOptions | None = None,
) -> "onnx.ModelProto":
    """Fuse a fitted `sklearn.pipeline.Pipeline` into a single ONNX graph.

    This is the one canonical `.compile()` implementation for the AutoML
    module (no parallel export path exists elsewhere in this package).

    Args:
        pipeline: A **fitted** `sklearn.pipeline.Pipeline`. Unfitted
            pipelines raise `sklearn.exceptions.NotFittedError` (via
            `sklearn.utils.validation.check_is_fitted`) before any ONNX
            work is attempted.
        sample_input: A representative input sample used only to determine
            the ONNX graph's input shape/dtype -- a pandas DataFrame or
            anything coercible to a 2-D numeric numpy array (rows x
            features). Values are not otherwise inspected; only ``shape``
            and dtype matter.
        options: Optional :class:`CompileOptions`. Defaults to
            `CompileOptions()`.

    Returns:
        An `onnx.ModelProto` -- a single, self-contained ONNX graph
        representing the entire pipeline (every step fused into one
        graph), suitable for `onnxruntime.InferenceSession` without any
        further dependency on scikit-learn or this package at inference
        time.

    Raises:
        TypeError: ``pipeline`` is not an `sklearn.pipeline.Pipeline`, or
            ``sample_input`` cannot be coerced to a 2-D numeric array.
        sklearn.exceptions.NotFittedError: ``pipeline`` has not been
            fitted yet.
        ONNXExtraNotInstalledError: the optional `automl-onnx` extra (on
            top of the `automl` extra) is not installed. Install both with
            `pip install "dscraft[automl,automl-onnx]"`.
    """
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            "compile() requires an sklearn.pipeline.Pipeline instance; got "
            f"{type(pipeline).__name__}."
        )

    # Raises sklearn.exceptions.NotFittedError if any step isn't fitted.
    check_is_fitted(pipeline)

    opts = options or CompileOptions()
    array = _as_numeric_array(sample_input)

    # Guard against a sample_input whose feature count silently disagrees
    # with what the fitted pipeline was trained on -- if left unchecked,
    # this would produce an ONNX graph with the wrong input signature (or
    # fail later, more confusingly, inside skl2onnx conversion or at ONNX
    # Runtime inference time). Not every estimator/pipeline sets
    # `n_features_in_` (e.g. a first step that doesn't implement sklearn's
    # standard fitted-attribute convention), so this check is best-effort:
    # skip it rather than raising on an unrelated AttributeError.
    expected_features = getattr(pipeline, "n_features_in_", None)
    if expected_features is not None and array.shape[1] != expected_features:
        raise ValueError(
            "compile() received a sample_input with "
            f"{array.shape[1]} feature column(s), but the fitted pipeline "
            f"was trained on {expected_features} feature column(s) "
            "(pipeline.n_features_in_). sample_input must have the same "
            "number of columns as the data the pipeline was fit on."
        )

    onnx, skl2onnx = _require_onnx_stack()
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [(opts.input_name, FloatTensorType([None, array.shape[1]]))]

    convert_kwargs: dict[str, Any] = {
        "initial_types": initial_types,
        "doc_string": opts.doc_string,
    }
    if opts.target_opset is not None:
        convert_kwargs["target_opset"] = opts.target_opset
    if not opts.zipmap:
        convert_kwargs["options"] = {id(pipeline): {"zipmap": False}}

    onnx_model = skl2onnx.convert_sklearn(pipeline, **convert_kwargs)
    onnx.checker.check_model(onnx_model)
    return onnx_model
