"""Tests for the first-party quantile model + H-day trajectory contract."""

from __future__ import annotations

import numpy as np
import pytest

from dscraft.automl import (
    TRAJECTORY_KIND,
    HDayTrajectory,
    PerDayDemandDistribution,
    QuantileModelError,
    fit_quantile_model,
    load_trajectory,
    predict_trajectory,
)


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 800
    X = rng.normal(size=(n, 2))
    y = np.abs(X[:, 0] * 3.0 + rng.normal(scale=1.0, size=n)) * 5.0
    return X, y


def _fit(seed: int = 0):
    X, y = _data()
    return fit_quantile_model(
        X, y, quantiles=(0.5, 0.8, 0.95), seed=seed,
        n_estimators=40, learning_rate=0.1, max_depth=3, subsample=0.8,
    )


def test_quantile_levels_validated() -> None:
    X, y = _data()
    for bad in [(), (0.0, 0.5), (0.5, 0.5), (0.9, 0.1), (1.0,)]:
        with pytest.raises(QuantileModelError):
            fit_quantile_model(X, y, quantiles=bad)


def test_predict_quantiles_non_crossing() -> None:
    model = _fit()
    X, _ = _data()
    rows = model.predict_quantiles(X[:50])
    for row in rows:
        vals = [row[q] for q in (0.5, 0.8, 0.95)]
        assert vals == sorted(vals)  # non-decreasing in quantile level


def test_quantile_model_seed_reproducible() -> None:
    X, _ = _data()
    a = _fit(7).predict_quantiles(X[:30])
    b = _fit(7).predict_quantiles(X[:30])
    assert a == b


def test_trajectory_per_day_is_proper_cdf() -> None:
    model = _fit()
    X, _ = _data()
    traj = predict_trajectory(
        model, [X[i:i + 1] for i in range(3)],
        dates=["2025-12-01", "2025-12-02", "2025-12-03"],
        dependence={"model": "independent", "justification_ref": "ev:acf-lag1<0.1"},
    )
    assert traj.horizon_days == 3
    for day in traj.per_day:
        assert list(day.support) == sorted(set(day.support))
        assert day.cumulative_prob[-1] == 1.0
        assert list(day.cumulative_prob) == sorted(day.cumulative_prob)
        # quantile() returns the smallest support value with cum prob >= level
        assert day.quantile(0.95) >= day.quantile(0.5)


def test_trajectory_round_trip() -> None:
    model = _fit()
    X, _ = _data()
    traj = predict_trajectory(
        model, [X[i:i + 1] for i in range(2)],
        dates=["2025-12-01", "2025-12-02"],
        dependence={"model": "independent", "justification_ref": "ev:acf"},
    )
    blob = traj.to_bytes()
    back = load_trajectory(blob)
    assert back.to_bytes() == blob
    assert back.horizon_days == traj.horizon_days
    assert back.dependence == traj.dependence


def test_trajectory_rejects_comonotonic_and_unknown_dependence() -> None:
    day = PerDayDemandDistribution(date="2025-12-01", support=(0.0, 100.0), cumulative_prob=(0.96, 1.0))
    for bad_model in ("comonotonic", "perfect", "bogus"):
        with pytest.raises(QuantileModelError):
            HDayTrajectory(
                horizon_days=1, per_day=(day,),
                dependence={"model": bad_model, "justification_ref": "x"},
                quantiles=(0.95,),
            )


def test_trajectory_dependence_requires_justification() -> None:
    day = PerDayDemandDistribution(date="2025-12-01", support=(0.0, 100.0), cumulative_prob=(0.96, 1.0))
    with pytest.raises(QuantileModelError):
        HDayTrajectory(
            horizon_days=1, per_day=(day,),
            dependence={"model": "independent"},  # no justification_ref
            quantiles=(0.95,),
        )


def test_trajectory_none_dependence_allowed_for_fail_closed() -> None:
    # dependence=None is the explicit "no admissible model" signal the consumer withholds on.
    day = PerDayDemandDistribution(date="2025-12-01", support=(0.0, 100.0), cumulative_prob=(0.96, 1.0))
    traj = HDayTrajectory(horizon_days=1, per_day=(day,), dependence=None, quantiles=(0.95,))
    assert traj.dependence is None


def test_incomplete_cdf_rejected() -> None:
    with pytest.raises(QuantileModelError):
        PerDayDemandDistribution(date="2025-12-01", support=(0.0, 100.0), cumulative_prob=(0.5, 0.9))


def test_load_rejects_wrong_kind() -> None:
    with pytest.raises(QuantileModelError):
        load_trajectory({"kind": "something_else", "per_day": []})
    assert TRAJECTORY_KIND == "ds_trajectory_v1"


# --- Task 8: serialization header, legacy policy, test-only oracle agreement ---

from dscraft.automl import (  # noqa: E402
    QUANTILE_KIND,
    UnsupportedLegacyModelError,
    load_quantile_model_header,
)


def test_quantile_model_serialize_header_round_trip() -> None:
    model = _fit(3)
    blob = model.serialize()
    header = load_quantile_model_header(blob)
    assert header["kind"] == QUANTILE_KIND
    assert header["quantiles"] == [0.5, 0.8, 0.95]
    assert header["seed"] == 3


def test_legacy_lightgbm_artifact_rejected() -> None:
    # A LightGBM dump_model-style blob carries tree_info / objective keys.
    legacy = {"name": "tree", "objective": "quantile", "tree_info": [{"tree_structure": {}}]}
    with pytest.raises(UnsupportedLegacyModelError) as exc:
        load_quantile_model_header(legacy)
    assert "unsupported_legacy_model_format" in str(exc.value)
    assert "re-fit" in str(exc.value).lower()


def test_unknown_non_legacy_format_rejected_plainly() -> None:
    with pytest.raises(QuantileModelError):
        load_quantile_model_header({"kind": "mystery_v9", "foo": 1})


def test_quantile_agrees_with_statsmodels_oracle_within_tolerance() -> None:
    """Prediction agreement vs a TEST-ONLY oracle (statsmodels QuantReg); never a prod dep."""
    sm = pytest.importorskip("statsmodels.regression.quantile_regression")
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.normal(size=n)
    # linear conditional quantiles so QuantReg is a fair oracle
    y = 2.0 * x + rng.normal(scale=1.0, size=n)
    X = x.reshape(-1, 1)

    for alpha in (0.25, 0.5, 0.75):
        model = fit_quantile_model(
            X, y, quantiles=(alpha,), seed=0,
            n_estimators=150, learning_rate=0.1, max_depth=2, subsample=1.0,
        )
        import numpy as _np
        Xsm = _np.column_stack([_np.ones(n), x])
        oracle = sm.QuantReg(y, Xsm).fit(q=alpha)
        grid = _np.linspace(-1.5, 1.5, 7)
        ours = [model.predict_quantiles(_np.array([[g]]))[0][alpha] for g in grid]
        theirs = [float(oracle.params[0] + oracle.params[1] * g) for g in grid]
        # boosted trees vs linear oracle: agreement within a loose band on in-range x
        diffs = [abs(a - b) for a, b in zip(ours, theirs)]
        assert max(diffs) < 1.5, (alpha, list(zip(ours, theirs)))


# --- Serialized pure-arithmetic inference matches the live model exactly -------

from dscraft.automl import predict_from_serialized  # noqa: E402


def test_serialized_inference_matches_live_model() -> None:
    """A distilled ds_quantile_v1 payload reproduces the live model's quantiles exactly."""
    model = _fit(5)
    X, _ = _data()
    blob = model.serialize()
    live = model.predict_quantiles(X[:40])
    for i in range(40):
        row = predict_from_serialized(blob, [float(v) for v in X[i]])
        for q in (0.5, 0.8, 0.95):
            assert row[q] == pytest.approx(live[i][q], abs=1e-9)


def test_serialized_inference_non_crossing() -> None:
    model = _fit(5)
    X, _ = _data()
    blob = model.serialize()
    row = predict_from_serialized(blob, [float(v) for v in X[0]])
    vals = [row[q] for q in (0.5, 0.8, 0.95)]
    assert vals == sorted(vals)
