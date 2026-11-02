"""First-party quantile demand model + complete H-day trajectory contract.

This module is DSCraft's public, reusable modeling surface for uncertainty-aware
demand forecasting. A *consumer* (e.g. SuiroAI's ``retail.replenishment`` profile)
fits a quantile model, then asks for either a per-day set of quantiles or a complete
H-day trajectory, and serializes/loads the result through a canonical JSON contract
with no pickled estimator and no executable deserialization.

Ownership boundary: DSCraft owns the numerical modeling (training, prediction,
serialization, loading, the dependence representation). The business semantics --
which service level, which dependence evidence is acceptable, how a trajectory is
turned into an order -- belong to the consumer's problem profile, not here.

Two serialized kinds:

* ``ds_quantile_v1`` -- a fitted multi-quantile model (one first-party
  :class:`~dscraft.automl._boosting.DSQuantileBoostRegressor` per requested
  quantile level, sharing training data), serialized as plain JSON.
* ``ds_trajectory_v1`` -- a complete H-day demand trajectory: the H per-day
  demand distributions (each a proper CDF over a declared integer/float support)
  PLUS the cross-day dependence structure the horizon calculation needs. A finite
  set of quantiles does not determine a distribution or the dependence across
  days, so both are represented explicitly here.

Dependence models (closed allowlist): ``independent`` (consumer convolves the
per-day marginals; permitted only with a recorded ``justification_ref``) and
``empirical_copula`` (a fitted joint carried in the trajectory). **Comonotonic is
deliberately NOT an allowed model** -- it is not a general upper bound for the
horizon percentile. When no admissible dependence model is available the trajectory
is emitted with ``dependence=None`` so the consumer fails closed (withholds) rather
than silently assuming independence.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from dscraft.automl.models import build_model

#: Serialized-format discriminators.
QUANTILE_KIND = "ds_quantile_v1"
TRAJECTORY_KIND = "ds_trajectory_v1"

#: The only dependence models the trajectory contract recognizes. Comonotonic is
#: intentionally excluded (see module docstring).
_DEPENDENCE_MODELS = frozenset({"independent", "empirical_copula"})


class QuantileModelError(ValueError):
    """Invalid quantile-model configuration, payload, or load input."""


def _canonical(payload: dict[str, Any]) -> bytes:
    """Deterministic JSON bytes: sorted keys, no NaN/Inf, compact separators."""
    return json.dumps(
        payload, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _check_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if not qs:
        raise QuantileModelError("at least one quantile level is required")
    if any(not (0.0 < q < 1.0) for q in qs):
        raise QuantileModelError("every quantile level must lie in (0, 1)")
    if list(qs) != sorted(qs) or len(set(qs)) != len(qs):
        raise QuantileModelError("quantile levels must be strictly increasing")
    return qs


@dataclass(frozen=True)
class PerDayDemandDistribution:
    """A proper demand CDF for one horizon day over a finite support.

    ``support`` is strictly increasing; ``cumulative_prob[i]`` is P(demand <= support[i]),
    non-decreasing and ending at exactly 1.0. No mass exists above the top support point
    (``tail_model='empirical_clipped'``); the method never invents tail mass.
    """

    date: str
    support: tuple[float, ...]
    cumulative_prob: tuple[float, ...]
    tail_model: str = "empirical_clipped"

    def __post_init__(self) -> None:
        if len(self.support) != len(self.cumulative_prob) or not self.support:
            raise QuantileModelError("support and cumulative_prob must be equal, non-empty length")
        if list(self.support) != sorted(set(self.support)):
            raise QuantileModelError("support must be strictly increasing")
        probs = self.cumulative_prob
        if any(not (0.0 <= p <= 1.0) for p in probs):
            raise QuantileModelError("cumulative probabilities must lie in [0, 1]")
        if list(probs) != sorted(probs):
            raise QuantileModelError("cumulative probabilities must be non-decreasing")
        if not math.isclose(probs[-1], 1.0, abs_tol=1e-9):
            raise QuantileModelError("cumulative probability must reach 1.0 (incomplete_demand_distribution)")

    def quantile(self, level: float) -> float:
        """Smallest support value whose cumulative probability >= ``level``."""
        for value, prob in zip(self.support, self.cumulative_prob):
            if prob >= level:
                return value
        return self.support[-1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "support": list(self.support),
            "cumulative_prob": list(self.cumulative_prob),
            "tail_model": self.tail_model,
        }


@dataclass(frozen=True)
class HDayTrajectory:
    """Complete H-day demand trajectory: per-day CDFs + cross-day dependence."""

    horizon_days: int
    per_day: tuple[PerDayDemandDistribution, ...]
    dependence: dict[str, Any] | None
    quantiles: tuple[float, ...]
    calibration: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon_days != len(self.per_day) or self.horizon_days < 1:
            raise QuantileModelError("horizon_days must equal the number of per-day distributions and be >= 1")
        if self.dependence is not None:
            model = self.dependence.get("model")
            if model not in _DEPENDENCE_MODELS:
                raise QuantileModelError(
                    f"dependence.model must be one of {sorted(_DEPENDENCE_MODELS)} "
                    f"(comonotonic is not an allowed model); got {model!r}"
                )
            if not self.dependence.get("justification_ref"):
                raise QuantileModelError("a dependence model requires a recorded justification_ref")

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": TRAJECTORY_KIND,
            "horizon_days": self.horizon_days,
            "per_day": [d.as_dict() for d in self.per_day],
            "dependence": self.dependence,
            "quantiles": list(self.quantiles),
            "calibration": self.calibration,
        }

    def to_bytes(self) -> bytes:
        return _canonical(self.as_dict())


@dataclass(frozen=True)
class DSQuantileModel:
    """A fitted multi-quantile demand model, serializable to canonical JSON.

    Holds one first-party quantile booster per requested level (median included for the
    point forecast). Prediction is monotone-enforced across levels at call time: a raw
    crossing (a lower quantile predicting above a higher one for some row) is repaired by
    a non-decreasing projection and recorded, never emitted.
    """

    quantiles: tuple[float, ...]
    _models: dict[float, Any]
    seed: int
    params: dict[str, Any]

    def predict_quantiles(self, X) -> list[dict[float, float]]:
        """Return, per input row, a ``{quantile_level: value}`` mapping (non-crossing)."""
        raw = {q: [float(v) for v in self._models[q].predict(X)] for q in self.quantiles}
        n = len(next(iter(raw.values())))
        out: list[dict[float, float]] = []
        for i in range(n):
            running = -math.inf
            row: dict[float, float] = {}
            for q in self.quantiles:  # ascending
                v = max(raw[q][i], running)  # non-decreasing projection
                row[q] = v
                running = v
            out.append(row)
        return out

    def serialize(self) -> bytes:
        """Serialize to canonical JSON carrying the JSON-native tree ensembles.

        For each quantile level the fitted first-party booster is distilled into
        plain nested dicts (``init``, ``learning_rate``, and a list of trees, each
        a nested split/leaf structure of JSON numbers). This makes inference a
        pure-arithmetic tree walk with no heavy dependency and no refit on load --
        the serialized form IS the model, not a pointer to a refit recipe.
        """
        payload = {
            "kind": QUANTILE_KIND,
            "quantiles": list(self.quantiles),
            "seed": self.seed,
            "params": self.params,
            "sub_model_kind": "DSQuantileBoost",
            "ensembles": {
                _q_key(q): _distill_booster(self._models[q]) for q in self.quantiles
            },
        }
        return _canonical(payload)


def _q_key(q: float) -> str:
    """Stable string key for a quantile level in the serialized payload."""
    return repr(round(float(q), 12))


def _distill_node(node: Any) -> dict[str, Any]:
    """Reduce one first-party ``TreeNode`` to a minimal JSON split/leaf dict."""
    if node.is_leaf:
        import numpy as _np

        arr = _np.asarray(node.value, dtype=float).reshape(-1)
        if arr.size != 1:
            raise QuantileModelError("expected a scalar regression leaf value")
        return {"leaf": float(arr[0])}
    return {
        "feature": int(node.feature_index),
        "threshold": float(node.threshold),
        "left": _distill_node(node.left),
        "right": _distill_node(node.right),
    }


def _distill_booster(model: Any) -> dict[str, Any]:
    """Distill a fitted ``DSQuantileBoostRegressor`` into JSON-native inference state.

    Uses only the model's public fitted attributes (``init_``, ``learning_rate``,
    ``estimators_`` and each tree's ``root_``/``TreeNode`` fields) -- the model's own
    structure, not a third-party library's internals.
    """
    return {
        "init": float(model.init_),
        "learning_rate": float(model.learning_rate),
        "n_features_in": int(model.n_features_in_),
        "trees": [_distill_node(tree.root_) for tree in model.estimators_],
    }


def _walk_tree(node: dict[str, Any], features: Sequence[float]) -> float:
    while "leaf" not in node:
        node = node["left"] if features[node["feature"]] <= node["threshold"] else node["right"]
    return node["leaf"]


def predict_from_serialized(payload: bytes | str | dict[str, Any], features: Sequence[float]) -> dict[float, float]:
    """Pure-arithmetic quantile prediction from a ``ds_quantile_v1`` payload.

    No heavy dependency, no refit: walks the distilled JSON tree ensembles. Returns a
    non-crossing ``{quantile_level: value}`` mapping for one feature row. This is the
    inference path a dependency-light consumer (e.g. SuiroAI) uses after loading a frozen
    model, so model inference never imports the training backend.
    """
    header = load_quantile_model_header(payload)
    ensembles = header.get("ensembles")
    if not ensembles:
        raise QuantileModelError("serialized quantile model carries no tree ensembles")
    quantiles = tuple(header["quantiles"])
    running = -math.inf
    out: dict[float, float] = {}
    for q in quantiles:  # ascending
        ens = ensembles[_q_key(q)]
        if len(features) != ens["n_features_in"]:
            raise QuantileModelError("inference feature alignment changed")
        total = ens["init"] + ens["learning_rate"] * sum(_walk_tree(t, features) for t in ens["trees"])
        running = max(total, running)  # non-decreasing projection across levels
        out[q] = running
    return out


def fit_quantile_model(
    X,
    y,
    quantiles: Sequence[float],
    *,
    seed: int = 0,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    subsample: float = 1.0,
) -> DSQuantileModel:
    """Fit one first-party quantile booster per requested level (shared training data).

    Deterministic for a fixed ``(X, y, quantiles, seed, params)``. Raises
    :class:`QuantileModelError` on an invalid quantile set; propagates the booster's
    own ``ValueError`` on a degenerate fit (no silent last-iterate).
    """
    qs = _check_quantiles(quantiles)
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
    }
    models: dict[float, Any] = {}
    for q in qs:
        m = build_model("DSQuantileBoost", "regression", alpha=q, random_state=seed, **params)
        m.fit(X, y)
        models[q] = m
    return DSQuantileModel(quantiles=qs, _models=models, seed=seed, params=params)


def _cdf_from_quantiles(row: dict[float, float], quantiles: tuple[float, ...]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build a step CDF (support, cumulative_prob) from one row's quantile values.

    Each quantile level ``q`` maps its predicted value to cumulative probability ``q``;
    identical values collapse (keeping the highest cumulative probability). The top point
    is forced to 1.0 (empirical_clipped tail: all remaining mass sits at the top value).
    """
    pairs: dict[float, float] = {}
    for q in quantiles:
        v = round(float(row[q]), 9)
        pairs[v] = max(pairs.get(v, 0.0), q)
    support = tuple(sorted(pairs))
    cumulative = tuple(pairs[v] for v in support)
    cumulative = cumulative[:-1] + (1.0,)  # clip tail mass onto the top support value
    return support, cumulative


def predict_trajectory(
    model: DSQuantileModel,
    X_per_day,
    *,
    dates: Sequence[str],
    dependence: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> HDayTrajectory:
    """Produce a complete H-day trajectory from a fitted quantile model.

    ``X_per_day`` is a length-H sequence of single-row feature inputs (one per horizon
    day). Each day's predicted quantiles become a step CDF (empirical_clipped tail).
    ``dependence`` is passed through after allowlist validation; ``None`` means no
    admissible dependence model, and the consumer must fail closed. Comonotonic is
    rejected by :class:`HDayTrajectory` construction.
    """
    H = len(dates)
    if len(X_per_day) != H or H < 1:
        raise QuantileModelError("dates and X_per_day must be the same, non-empty length")
    per_day: list[PerDayDemandDistribution] = []
    for day_X, date in zip(X_per_day, dates):
        row = model.predict_quantiles(day_X)[0]
        support, cumulative = _cdf_from_quantiles(row, model.quantiles)
        per_day.append(PerDayDemandDistribution(date=date, support=support, cumulative_prob=cumulative))
    return HDayTrajectory(
        horizon_days=H,
        per_day=tuple(per_day),
        dependence=dependence,
        quantiles=model.quantiles,
        calibration=calibration or {},
    )


#: Markers that identify a third-party / legacy serialized model that this
#: first-party loader must REJECT rather than silently coerce. Migration from
#: such an artifact is a re-fit from recorded training lineage, never a byte
#: re-interpretation (the formats are not compatible).
_LEGACY_MARKERS = ("lightgbm", "lgbm", "xgboost", "catboost", "tree_info", "objective")


class UnsupportedLegacyModelError(QuantileModelError):
    """A third-party/legacy model artifact was handed to the first-party loader."""


def load_quantile_model_header(data: bytes | str | dict[str, Any]) -> dict[str, Any]:
    """Validate and return the header of a ``ds_quantile_v1`` model payload.

    Rejects a legacy/third-party serialization (e.g. a LightGBM ``dump_model``
    blob, which carries ``tree_info`` / ``objective`` keys) with
    :class:`UnsupportedLegacyModelError`, naming the required remediation: re-fit
    a ``ds_quantile_v1`` model from the recorded training lineage. There is no
    in-place byte migration. If the lineage is unavailable the caller must leave
    the path blocked naming that exact missing input.
    """
    if isinstance(data, (bytes, str)):
        payload = json.loads(data)
    elif isinstance(data, dict):
        payload = data
    else:
        raise QuantileModelError("quantile-model load input must be bytes, str, or dict")
    if not isinstance(payload, dict):
        raise QuantileModelError("quantile-model payload must be a JSON object")
    kind = payload.get("kind")
    if kind != QUANTILE_KIND:
        # Distinguish a legacy third-party artifact (actionable: re-fit) from an
        # merely-unknown one.
        keys = {k.lower() for k in payload}
        if any(m in keys for m in _LEGACY_MARKERS) or (isinstance(kind, str) and any(m in kind.lower() for m in _LEGACY_MARKERS)):
            raise UnsupportedLegacyModelError(
                "unsupported_legacy_model_format: this looks like a third-party/legacy "
                "(e.g. LightGBM) model serialization. There is no byte migration; re-fit a "
                f"{QUANTILE_KIND} model from the recorded training lineage. If the lineage is "
                "unavailable, leave the path blocked naming 'legacy quantile artifact present "
                "but training lineage unavailable for re-fit'."
            )
        raise QuantileModelError(f"unknown saved model format (expected {QUANTILE_KIND}, got {kind!r})")
    return payload


def load_trajectory(data: bytes | str | dict[str, Any]) -> HDayTrajectory:
    """Load a ``ds_trajectory_v1`` artifact; reject any other kind. No code execution."""
    if isinstance(data, (bytes, str)):
        payload = json.loads(data)
    elif isinstance(data, dict):
        payload = data
    else:
        raise QuantileModelError("trajectory load input must be bytes, str, or dict")
    if not isinstance(payload, dict) or payload.get("kind") != TRAJECTORY_KIND:
        raise QuantileModelError(f"unknown saved trajectory format (expected {TRAJECTORY_KIND})")
    per_day = tuple(
        PerDayDemandDistribution(
            date=d["date"],
            support=tuple(d["support"]),
            cumulative_prob=tuple(d["cumulative_prob"]),
            tail_model=d.get("tail_model", "empirical_clipped"),
        )
        for d in payload["per_day"]
    )
    return HDayTrajectory(
        horizon_days=payload["horizon_days"],
        per_day=per_day,
        dependence=payload.get("dependence"),
        quantiles=tuple(payload.get("quantiles", ())),
        calibration=payload.get("calibration", {}),
    )
