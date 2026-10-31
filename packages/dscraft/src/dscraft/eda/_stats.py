"""First-party statistical primitives for ``dscraft.eda.associations``.

This internal module implements the handful of classical correlation/association *statistics* that
``dscraft.eda.associations`` previously called out to ``scipy.stats`` for -- Pearson, Spearman, and
Kendall correlation, and the chi-squared contingency statistic -- from scratch on ``numpy``. Per
the first-party automl/eda rebuild (see
``docs/decisions/2026-10-first-party-automl-eda-rebuild.md``), these are *techniques* (not mere
numeric primitives), so they become first-party; Cramér's V, Theil's U, and the correlation ratio
were already implemented from scratch in ``associations.py`` and are unchanged.

Each function mirrors the shape of the SciPy result it replaces closely enough that
``associations.py`` needs only a trivial call-site change: the three correlation functions return a
``(statistic, p_value)`` named tuple, and :func:`chi2_contingency` returns the chi-squared
statistic and degrees of freedom. The p-values use the standard large-sample normal/t
approximations, which is what these exploratory association measures need (they are used to
populate an association matrix, not to run a precise hypothesis test), and matches SciPy's
asymptotic behaviour closely for the sample sizes EDA operates on.

Only ``numpy`` is used -- no ``scipy``. (``scipy`` remains a declared ``eda`` dependency for now
only as substrate; this module removes the dependence on its *statistical tests* specifically.)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

__all__ = [
    "CorrelationStat",
    "pearsonr",
    "spearmanr",
    "kendalltau",
    "chi2_contingency",
]


class CorrelationStat(NamedTuple):
    """A ``(statistic, p_value)`` pair, mirroring SciPy's correlation result objects."""

    statistic: float
    p_value: float


def _normal_sf(z: float) -> float:
    """Survival function (upper tail) of the standard normal, via the error function."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _t_two_sided_p(t: float, df: int) -> float:
    """Two-sided p-value for a t-statistic with ``df`` degrees of freedom.

    Uses the regularized incomplete beta function (computed here via a continued fraction) for the
    Student-t tail -- the same relationship SciPy uses -- so no scipy dependency is needed.
    """
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    # P(|T| > |t|) = I_x(df/2, 1/2) where I is the regularized incomplete beta.
    return _betai(df / 2.0, 0.5, x)


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via the Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b + lbeta) / a
    # Use the continued fraction where it converges fastest; otherwise apply the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a).
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - _betai(b, a, 1.0 - x)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Numerical Recipes, Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def pearsonr(x: np.ndarray, y: np.ndarray) -> CorrelationStat:
    """Pearson linear correlation coefficient and its two-sided p-value.

    ``r = cov(x, y) / (std(x) std(y))``; the p-value is from the t-statistic
    ``t = r sqrt((n - 2) / (1 - r^2))`` on ``n - 2`` degrees of freedom. Returns ``r = 0`` with a
    p-value of ``1`` for a constant input (zero variance), rather than ``nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float(np.sum(xm**2)) * float(np.sum(ym**2)))
    if denom == 0.0:
        return CorrelationStat(0.0, 1.0)
    r = float(np.sum(xm * ym) / denom)
    r = max(-1.0, min(1.0, r))
    if n <= 2 or abs(r) >= 1.0:
        return CorrelationStat(r, 0.0 if abs(r) >= 1.0 else float("nan"))
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    return CorrelationStat(r, _t_two_sided_p(t, n - 2))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Assign ranks to data, averaging ranks within ties (the standard 'average' method)."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.shape[0], dtype=np.float64)
    sorted_a = a[order]
    i = 0
    n = a.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based; average over the tie block
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> CorrelationStat:
    """Spearman rank correlation: Pearson correlation of the rank-transformed data."""
    return pearsonr(_rankdata(x), _rankdata(y))


def kendalltau(x: np.ndarray, y: np.ndarray) -> CorrelationStat:
    """Kendall's tau-b rank correlation and its two-sided (normal-approximation) p-value.

    tau-b handles ties: ``tau = (C - D) / sqrt((C + D + Tx)(C + D + Ty))`` where ``C``/``D`` are
    concordant/discordant pairs and ``Tx``/``Ty`` count pairs tied only in ``x``/``y``. The p-value
    uses the standard normal approximation for the (no-ties) variance of tau.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        prod = sx * sy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
        tie_x += int(np.sum((sx == 0) & (sy != 0)))
        tie_y += int(np.sum((sy == 0) & (sx != 0)))
    n0 = concordant + discordant + tie_x
    n1 = concordant + discordant + tie_y
    if n0 == 0 or n1 == 0:
        return CorrelationStat(0.0, 1.0)
    tau = (concordant - discordant) / math.sqrt(n0 * n1)
    tau = max(-1.0, min(1.0, tau))
    # Normal approximation to the null distribution of tau (no-ties variance).
    if n > 2:
        var = (2.0 * (2.0 * n + 5.0)) / (9.0 * n * (n - 1.0))
        z = tau / math.sqrt(var) if var > 0 else 0.0
        p = 2.0 * _normal_sf(abs(z))
    else:
        p = float("nan")
    return CorrelationStat(tau, min(1.0, p))


def chi2_contingency(table: np.ndarray) -> tuple[float, int]:
    """Pearson chi-squared statistic and degrees of freedom for a contingency ``table``.

    Computes ``chi2 = sum((O - E)^2 / E)`` with expected counts
    ``E_ij = row_total_i * col_total_j / grand_total`` (no continuity correction -- the caller,
    Cramér's V, explicitly wants the uncorrected magnitude). Returns ``(chi2, dof)`` with
    ``dof = (r - 1)(c - 1)``.
    """
    table = np.asarray(table, dtype=np.float64)
    grand = table.sum()
    if grand == 0:
        return 0.0, 0
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    expected = row_totals @ col_totals / grand
    # Guard against zero expected cells (empty row/col) -- those contribute nothing.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    chi2 = float(np.sum(terms))
    r, c = table.shape
    dof = (r - 1) * (c - 1)
    return chi2, dof
