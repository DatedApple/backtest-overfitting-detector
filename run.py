"""Backtest Overfitting Detector — run this file.

python run.py                 # 1000 random rules on SPY + DSR on momentum
python run.py --paper-example # Bailey & López de Prado numerical example
python run.py --ticker QQQ --n-trials 1000
"""

from __future__ import annotations

# Windows consoles often cannot print the ó in López
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_prices
from rules import Trial, _shifted_returns, candidate_strategy, random_trials
from stats import (
    TRADING_DAYS,
    annualized_sharpe,
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    paper_example,
    per_period_sharpe,
    probabilistic_sharpe,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"


def _mean_abs_corr(trials: list[Trial]) -> float:
    usable = [t.returns for t in trials if t.returns.size >= 50]
    if len(usable) < 2:
        return 0.0
    min_len = min(r.size for r in usable)
    mat = np.column_stack([r[-min_len:] for r in usable])
    # Sample a subset if N is large — full NxN corr on 1000 series is fine.
    c = np.corrcoef(mat, rowvar=False)
    iu = np.triu_indices_from(c, k=1)
    vals = c[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(np.abs(vals))) if vals.size else 0.0


def _report_strategy(title: str, returns: np.ndarray, n_trials: int, std_sr_period: float) -> None:
    sr_ann = annualized_sharpe(returns)
    sr = per_period_sharpe(returns)
    skew, kurt, t = moments(returns)
    psr = probabilistic_sharpe(sr, t, skew, kurt, 0.0)
    dsr, sr0 = deflated_sharpe(sr, t, skew, kurt, n_trials=max(n_trials, 2), std_sr=std_sr_period)
    mtrl = min_track_record_length(sr, skew, kurt, sr_threshold=sr0, prob=0.95)
    years = t / TRADING_DAYS
    mtrl_years = mtrl / TRADING_DAYS if np.isfinite(mtrl) else float("inf")
    survives = dsr >= 0.95
    print(f"\n=== {title} ===")
    print(f"  Annualized Sharpe:          {sr_ann:8.3f}")
    print(f"  T (days / years):           {t:8d}  /  {years:.2f}")
    print(f"  Skew, kurtosis (raw):       {skew:8.3f}  /  {kurt:.3f}")
    print(f"  PSR (true SR > 0):          {psr:8.3f}   [ignores how many rules you tried]")
    print(f"  Noise ceiling SR0 (ann.):   {sr0 * np.sqrt(TRADING_DAYS):8.3f}   [best Sharpe luck can hand you]")
    print(f"  Deflated Sharpe (prob real):{dsr:8.3f}   [{'SURVIVES 95% bar' if survives else 'DOES NOT SURVIVE 95% bar'}]")
    print(f"  Min track record @ 95%:     {mtrl_years:8.2f} years of this same SR to beat the noise ceiling")


def run_detector(ticker: str, n_trials: int, seed: int, start: str) -> None:
    print(f"Downloading {ticker} from {start} ...")
    prices = load_prices(ticker, start=start)
    print(f"  {len(prices)} daily closes, {prices.index[0].date()} → {prices.index[-1].date()}")

    print(f"\nBacktesting {n_trials} random rules (seed={seed}) ...")
    trials = random_trials(prices, n=n_trials, seed=seed, sharpe_fn=annualized_sharpe)
    sharpes = np.array([t.sharpe_ann for t in trials])
    best: Trial = max(trials, key=lambda t: t.sharpe_ann)

    std_ann = float(np.std(sharpes, ddof=1))
    std_period = std_ann / np.sqrt(TRADING_DAYS)
    mean_corr = _mean_abs_corr(trials)
    n_eff = effective_trials(n_trials, mean_corr)
    emax_indep = expected_max_sharpe(n_trials, sharpes.mean(), std_ann)
    emax_null = expected_max_sharpe(n_trials, 0.0, std_ann)

    print("\n=== Random-rule lottery (this is the trap) ===")
    print(f"  Mean Sharpe of all rules:   {sharpes.mean():8.3f}")
    print(f"  Std Sharpe across rules:    {std_ann:8.3f}")
    print(f"  Best rule:                  {best.name:8s}   Sharpe {best.sharpe_ann:.3f}")
    print(f"  Formula E[max SR] = mean + std * z(N={n_trials}): {emax_indep:.3f}")
    print(f"  Noise ceiling if true skill is 0 (DSR's SR0):     {emax_null:.3f}")
    print(f"  Mean |corr| of rule PnL:    {mean_corr:.3f}  →  N_eff ≈ {n_eff:.1f} (correlated rules count as fewer independent tests)")
    print("  If the winner sits near E[max], you mostly found luck, not a law of nature.")

    _report_strategy(f"WINNER of {n_trials} random rules — {best.name}", best.returns, n_trials, std_period)

    cname, cpos = candidate_strategy(prices)
    crets = _shifted_returns(prices, cpos)
    _report_strategy(f"STRATEGY YOU MIGHT BELIEVE IN — {cname}", crets, n_trials, std_period)

    bh = prices.pct_change().dropna().to_numpy()
    _report_strategy(f"BUY AND HOLD {ticker} (charged the same N={n_trials} — unfair but shows the bar)", bh, n_trials, std_period)

    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(sharpes, bins=40, color="#4c78a8", edgecolor="white")
    axes[0].axvline(best.sharpe_ann, color="#e45756", lw=2, label=f"best random  {best.sharpe_ann:.2f}")
    axes[0].axvline(emax_indep, color="#f58518", lw=2, ls="--", label=f"E[max] luck {emax_indep:.2f}")
    axes[0].axvline(annualized_sharpe(crets), color="#54a24b", lw=2, label=f"momentum     {annualized_sharpe(crets):.2f}")
    axes[0].set_title(f"{n_trials} random rules on {ticker}")
    axes[0].set_xlabel("Annualized Sharpe")
    axes[0].legend(fontsize=8)

    def equity(r):
        return pd.Series(np.cumprod(1 + r))

    axes[1].plot(equity(best.returns).values, color="#e45756", label=best.name)
    axes[1].plot(equity(crets).values, color="#54a24b", label="12m momentum")
    axes[1].plot(equity(bh).values, color="#4c78a8", label=f"Buy {ticker}")
    axes[1].set_title("Equity curves (growth of $1)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "overfitting_demo.png"
    fig.savefig(out, dpi=140)
    print(f"\nSaved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest overfitting detector (Deflated Sharpe Ratio)")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--n-trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--paper-example", action="store_true")
    args = p.parse_args()

    if args.paper_example:
        out = paper_example()
        print("Bailey & López de Prado (2014) numerical example")
        print("  Reported SR 2.5, N=100, Var(SR)=0.5, T=1250, skew=-3, kurtosis=10")
        print(f"  SR0 annualized (noise ceiling): {out['sr0_annual']:.3f}")
        print(f"  PSR vs 0 (no multiple testing): {out['psr_vs_zero']:.3f}")
        print(f"  DSR (prob result is real):      {out['dsr']:.3f}   [paper: ~0.90, fails 95%]")
        return

    run_detector(args.ticker, args.n_trials, args.seed, args.start)


if __name__ == "__main__":
    main()
