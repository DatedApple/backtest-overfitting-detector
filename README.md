# Backtest Overfitting Detector

A small, readable implementation of the **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) — the statistic that tells you whether a backtested trading rule is a real edge or just the best result out of however many you tried.

Search a thousand random SMA, momentum, and breakout variants on real SPY prices, and the best one will look great. That's not evidence of skill — it's what you'd expect from pure noise once you let yourself pick the winner. The Deflated Sharpe Ratio (DSR) asks a sharper question: *given the number of trials, the length of the track record, and how non-normal the returns are, what's the probability this Sharpe ratio reflects a real effect?*

Most "best" rules don't survive that question.

Paper (free PDF): https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf

---

## Table of contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Command-line options](#command-line-options)
- [Reading the output](#reading-the-output)
- [The math](#the-math)
- [Project layout](#project-layout)
- [Dependencies](#dependencies)
- [Limitations](#limitations)
- [Reference](#reference)

---

## What it does

Running `run.py` walks through four things, in order:

1. **The random-rule lottery.** 1,000 SMA crossover, momentum, and breakout variants, backtested on SPY with no look-ahead (positions are lagged one day). The best Sharpe among them tends to land close to the analytic **E[max Sharpe]** for that trial count — that formula *is* what luck looks like at that sample size.
2. **DSR on the winner.** Treat that lucky rule as if someone had proposed it as a discovery, and run the deflated Sharpe test on it. It's almost always below the 0.95 bar.
3. **DSR on a rule you might actually believe.** 12-month time-series momentum (Moskowitz, Ooi, Pedersen, 2012) — a published strategy, not something searched for on this ticker. Same test, same trial count charged against it. Even well-known rules can fail once you charge them for the search you actually did.
4. **The paper's own numerical example.** SR 2.5, 100 trials, Var(SR) = 0.5, five years of daily data, fat tails → DSR ≈ 0.90. Below the bar, despite a Sharpe of 2.5.

The rule of thumb from the paper: **DSR ≥ 0.95** to call a result real.

## Quick start

Needs Python 3.10+.

**macOS / Linux**

```bash
git clone https://github.com/DatedApple/backtest-overfitting-detector.git
cd backtest-overfitting-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python run.py --paper-example
python run.py
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/DatedApple/backtest-overfitting-detector.git
cd backtest-overfitting-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python run.py --paper-example
python run.py
```

Output: a printed DSR report for each of the three strategies, plus `figures/overfitting_demo.png` (a Sharpe histogram and equity curves). Price data is cached under `data/` after the first run.

## Command-line options

| Flag | Default | Description |
|---|---|---|
| `--ticker` | `SPY` | Symbol to pull from Yahoo's chart API |
| `--n-trials` | `1000` | Number of random rules to generate |
| `--seed` | `42` | RNG seed for the random rules |
| `--start` | `2010-01-01` | Start date for price history |
| `--paper-example` | off | Skip the backtest and reproduce the paper's own worked example |

Example: `python run.py --ticker QQQ --n-trials 1000`

## Reading the output

For each strategy, `run.py` prints:

- **Annualized Sharpe** — the headline number everyone quotes.
- **T (days / years)** — sample length. DSR gets more forgiving as this grows.
- **Skew, kurtosis** — non-normality in the returns. Negative skew and fat tails both push DSR down for a given Sharpe.
- **PSR (true SR > 0)** — probability the strategy beats zero, *ignoring* how many variants were tried. On its own this number overstates how real a result is.
- **Noise ceiling (SR0)** — the best Sharpe pure luck would be expected to produce, given the trial count. This becomes the threshold DSR tests against instead of zero.
- **Deflated Sharpe** — the actual answer: probability the result is still real after accounting for multiple testing and non-normal returns. `≥ 0.95` survives.
- **Min track record length** — how many more years of this same Sharpe ratio it would take to clear the 95% bar.

The script also reports the mean absolute correlation between the 1,000 random rules and an effective trial count (`N_eff`). Correlated variants of the same idea don't count as independent tries — the script accounts for that rather than charging the full N.

## The math

All formulas below use **per-period** Sharpe ratios (`mean / std`), not annualized ones. Annualizing only happens when printing numbers for a human — multiply by `√252`.

**Expected best Sharpe after N independent noise trials** (paper, Eq. 1, null mean 0):

```
SR₀ = σ_SR × [(1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e))]
```

`γ ≈ 0.5772` (Euler–Mascheroni constant). `σ_SR` is the standard deviation of Sharpe ratios across the N trials.

**Probabilistic Sharpe Ratio (PSR)** — P(true SR > threshold), corrected for skew and kurtosis:

```
PSR = Φ[ (SR − SR*) · √(T − 1) / √(1 − γ₃·SR + ((γ₄ − 1)/4)·SR²) ]
```

`T` = number of observations, `γ₃` = skewness, `γ₄` = raw kurtosis (3 for a normal distribution).

**Deflated Sharpe Ratio (DSR)** = PSR with the threshold set to `SR0` instead of zero. It's a probability, not a rescaled Sharpe number.

**Effective trial count**, since random rules are correlated with each other:

```
N_eff = N / (1 + (N − 1)·ρ̄)
```

`ρ̄` is the mean absolute pairwise correlation of daily PnL across trials. Using the raw trial count N is the stricter, more conservative choice — the script reports both.

## Project layout

Suggested reading order:

| File | Contents |
|---|---|
| `stats.py` | The formulas — PSR, DSR, expected-max-Sharpe, minimum track record length. Start with `expected_max_z`, then `deflated_sharpe`. |
| `rules.py` | Random rule generation (SMA, momentum, breakout) plus the momentum benchmark. Note the `.shift(1)` — no look-ahead. |
| `data.py` | Price download from Yahoo's chart API, with local CSV caching. |
| `run.py` | Orchestration — runs the lottery, prints the DSR reports, saves the plot. |

## Dependencies

| Package | Used for |
|---|---|
| `numpy` | Arrays, random rule parameters, correlation of trial PnL |
| `pandas` | Price series, rolling averages, percent returns |
| `scipy` | `norm.cdf` / `norm.ppf`, skewness, kurtosis |
| `matplotlib` | The Sharpe histogram and equity curve figure |
| `curl_cffi` | Fetches SPY prices from Yahoo's chart API |

No dependency on the `pypbo` package. The DSR, PSR, and minimum-track-record-length formulas are reimplemented directly in `stats.py`, so every line is readable and there's nothing to trust blindly.

## Limitations

- This is a demonstration of the DSR methodology, not a production risk tool or investment advice. Don't use a single script's output to size real capital.
- Yahoo's chart API is unofficial and can change or rate-limit without notice. If `data.py` starts failing, that's the likely cause.
- The random rule set (SMA, momentum, breakout) is a stand-in for "the kind of thing a person searching for an edge would try." A real research process usually explores a much larger and stranger space, which means the true trial count — and the resulting noise ceiling — is probably higher than what's charged here.
- `N_eff` is one reasonable way to account for correlated trials. The paper itself doesn't prescribe an exact method for real-world trial correlation, so treat it as a lower bound on the true multiple-testing penalty, not an exact figure.

## Reference

Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* [Free PDF](https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
