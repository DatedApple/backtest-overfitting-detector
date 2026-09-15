"""Random trading rules + one 'I believe in this' benchmark.

Every rule is a function of past prices only. Position is shifted by 1 day
so we never trade on the same bar we used to form the signal (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trial:
    name: str
    kind: str
    params: dict
    returns: np.ndarray
    sharpe_ann: float


def _shifted_returns(prices: pd.Series, position: pd.Series) -> np.ndarray:
    daily = prices.pct_change()
    # Trade tomorrow on today's signal.
    pnl = position.shift(1) * daily
    return pnl.dropna().to_numpy()


def sma_crossover(prices: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    pos = pd.Series(np.where(fast_ma > slow_ma, 1.0, -1.0), index=prices.index)
    pos[slow_ma.isna()] = np.nan
    return pos


def momentum(prices: pd.Series, lookback: int) -> pd.Series:
    past = prices.pct_change(lookback)
    pos = pd.Series(np.where(past > 0, 1.0, -1.0), index=prices.index)
    pos[past.isna()] = np.nan
    return pos


def breakout(prices: pd.Series, window: int) -> pd.Series:
    high = prices.rolling(window).max()
    low = prices.rolling(window).min()
    pos = pd.Series(
        np.where(prices >= high, 1.0, np.where(prices <= low, -1.0, 0.0)),
        index=prices.index,
    )
    pos[high.isna()] = np.nan
    return pos


def candidate_strategy(prices: pd.Series) -> tuple[str, pd.Series]:
    """Classic 12-month time-series momentum: long if up over 252d, else short.

    Published rule (Moskowitz, Ooi, Pedersen 2012), not searched on this ticker.
    """
    pos = momentum(prices, lookback=252)
    return "12-month time-series momentum (Moskowitz, Ooi, Pedersen 2012)", pos


def random_trials(prices: pd.Series, n: int, seed: int, sharpe_fn) -> list[Trial]:
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []
    for i in range(n):
        kind = rng.choice(["sma", "momentum", "breakout"])
        if kind == "sma":
            fast = int(rng.integers(5, 60))
            slow = int(rng.integers(fast + 10, 250))
            params = {"fast": fast, "slow": slow}
            pos = sma_crossover(prices, fast, slow)
            name = f"SMA {fast}/{slow}"
        elif kind == "momentum":
            lookback = int(rng.integers(5, 252))
            params = {"lookback": lookback}
            pos = momentum(prices, lookback)
            name = f"MOM {lookback}d"
        else:
            window = int(rng.integers(10, 120))
            params = {"window": window}
            pos = breakout(prices, window)
            name = f"BO {window}d"

        rets = _shifted_returns(prices, pos)
        trials.append(
            Trial(
                name=name,
                kind=kind,
                params=params,
                returns=rets,
                sharpe_ann=sharpe_fn(rets),
            )
        )
    return trials
