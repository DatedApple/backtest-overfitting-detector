"""Bailey & López de Prado (2014) Deflated Sharpe Ratio.

All Sharpe ratios inside these formulas are *per period* (same frequency as T),
not annualized. Annualize only when you print numbers for humans.

Paper: https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329  # γ in the paper
TRADING_DAYS = 252


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(periods_per_year))


def per_period_sharpe(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    return float(np.mean(r) / np.std(r, ddof=1))


def moments(returns: np.ndarray) -> tuple[float, float, int]:
    """Skewness and *raw* kurtosis (3 for a Normal), plus sample length T."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    skew = float(stats.skew(r, bias=False)) if r.size > 2 else 0.0
    kurt = float(stats.kurtosis(r, fisher=False, bias=False)) if r.size > 3 else 3.0
    return skew, kurt, int(r.size)


def expected_max_z(n_trials: int) -> float:
    """E[max] of N i.i.d. standard Normal draws (paper Eq. 1, mean 0, var 1).

    For N=10 this is about 1.57 — a sanity check from related López de Prado notes.
    """
    if n_trials < 2:
        return 0.0
    n = float(n_trials)
    return (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n) + EULER_GAMMA * stats.norm.ppf(
        1.0 - math.exp(-1.0) / n
    )


def expected_max_sharpe(n_trials: int, mean_sr: float = 0.0, std_sr: float = 1.0) -> float:
    """How good the *best* Sharpe looks after N independent trials of pure noise."""
    return mean_sr + std_sr * expected_max_z(n_trials)


def effective_trials(n_trials: int, mean_abs_corr: float) -> float:
    """Independent-trial count when rules are correlated.

    Perfectly correlated rules → 1 trial. Uncorrelated → N trials.
    """
    rho = min(max(float(mean_abs_corr), 0.0), 1.0)
    return n_trials / (1.0 + (n_trials - 1) * rho)


def probabilistic_sharpe(
    sr: float,
    t: int,
    skew: float,
    kurtosis: float,
    sr_threshold: float = 0.0,
) -> float:
    """P(true SR > threshold). Corrects for short samples and non-Normal returns."""
    if t < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr))
    z = (sr - sr_threshold) * math.sqrt(t - 1) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe(
    sr: float,
    t: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    std_sr: float,
) -> tuple[float, float]:
    """DSR = PSR with threshold = expected max Sharpe under the null.

    Returns (dsr, sr0) in *per-period* units.
    DSR is a probability: chance the result is still real after multiple testing.
    Convention: DSR ≥ 0.95 is the usual 'survives' bar.
    """
    sr0 = expected_max_sharpe(n_trials, mean_sr=0.0, std_sr=std_sr)
    return probabilistic_sharpe(sr, t, skew, kurtosis, sr_threshold=sr0), sr0


def min_track_record_length(
    sr: float,
    skew: float,
    kurtosis: float,
    sr_threshold: float = 0.0,
    prob: float = 0.95,
) -> float:
    """Observations needed for PSR(sr, threshold) to reach `prob`."""
    if sr <= sr_threshold:
        return float("inf")
    z = stats.norm.ppf(prob)
    scale = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    return 1.0 + scale * (z / (sr - sr_threshold)) ** 2


def paper_example() -> dict[str, float]:
    """Reproduce the treasury-seasonality example in the paper.

    Reported annualized SR = 2.5, N = 100 trials, Var(SR) = 0.5 (annualized),
    T = 1250 days, skew = -3, kurtosis = 10.
    Investor should get DSR ≈ 0.90 — not a 95% discovery.
    """
    periods = 250  # paper uses 250, not 252
    sr = 2.5 / math.sqrt(periods)
    std_sr = math.sqrt(0.5 / periods)
    dsr, sr0 = deflated_sharpe(sr, t=1250, skew=-3.0, kurtosis=10.0, n_trials=100, std_sr=std_sr)
    return {
        "sr_annual": 2.5,
        "sr_period": sr,
        "sr0_period": sr0,
        "sr0_annual": sr0 * math.sqrt(periods),
        "dsr": dsr,
        "psr_vs_zero": probabilistic_sharpe(sr, 1250, -3.0, 10.0, 0.0),
    }
