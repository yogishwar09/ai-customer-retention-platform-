"""Frequentist A/B statistics for retention experiments (stdlib + numpy only)."""

from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()


def required_sample_size(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size to detect an absolute lift ``mde`` over ``baseline_rate``
    (two-sided two-proportion z-test)."""
    p1 = min(max(baseline_rate, 1e-4), 1 - 1e-4)
    mde = abs(mde)
    p2 = p1 + mde if p1 + mde < 1 else p1 - mde  # test the detectable direction near the bounds
    p2 = min(max(p2, 1e-4), 1 - 1e-4)
    if abs(p2 - p1) < 1e-9:
        return 10**9
    z_a = _N.inv_cdf(1 - alpha / 2)
    z_b = _N.inv_cdf(power)
    p_bar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p2 - p1) ** 2))


def unequal_allocation_total(n_per_arm_equal: int, control_share: float) -> int:
    """Total sample needed when arms are unbalanced (variance inflation 1/(4 s (1-s)))."""
    s = min(max(control_share, 0.01), 0.99)
    return int(math.ceil(2 * n_per_arm_equal / (4 * s * (1 - s))))


def two_proportion_test(k_t: int, n_t: int, k_c: int, n_c: int, alpha: float = 0.05) -> dict:
    if n_t == 0 or n_c == 0:
        return {"status": "insufficient"}
    p_t, p_c = k_t / n_t, k_c / n_c
    diff = p_t - p_c
    pooled = (k_t + k_c) / (n_t + n_c)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - _N.cdf(abs(z)))
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    z_a = _N.inv_cdf(1 - alpha / 2)
    return {"status": "ok", "treatment_rate": p_t, "control_rate": p_c, "difference": diff,
            "relative_lift": diff / p_c if p_c else None, "ci_low": diff - z_a * se, "ci_high": diff + z_a * se,
            "z": z, "p_value": p_value, "significant": p_value < alpha}


def achieved_power(p_c: float, mde: float, n_t: int, n_c: int, alpha: float = 0.05) -> float:
    if n_t == 0 or n_c == 0:
        return 0.0
    p_t = min(max(p_c + mde, 1e-4), 1 - 1e-4)
    se = math.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c)
    if se == 0:
        return 0.0
    z_a = _N.inv_cdf(1 - alpha / 2)
    return float(_N.cdf(abs(mde) / se - z_a))
