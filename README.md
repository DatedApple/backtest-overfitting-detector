# Backtest Overfitting Detector

You search a thousand random trading rules on **real prices**, keep the best Sharpe, and it looks brilliant. Then the **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) asks: given how many tries you had, the length of the track record, and how ugly the return distribution is, what is the probability that number is real?

Most “best” rules do not survive.

Paper (free PDF): https://davidhbailey.com/dhbpapers/deflated-sharpe.pdf

## Run (5 minutes)

```powershell
cd $env:USERPROFILE\backtest-overfitting-detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py --paper-example
python run.py
```

Optional: `python run.py --ticker QQQ --n-trials 1000`

Output: printed DSR report + `figures/overfitting_demo.png`.

## What you will see

1. **Random-rule lottery** — 1000 SMA / momentum / breakout variants on SPY. The winner’s Sharpe is usually close to the analytic **E[max Sharpe]** for that many trials. That formula is *what luck looks like*.
2. **Winner DSR** — treat the lucky rule as if it were a discovery. DSR is almost always below 0.95.
3. **A strategy you might believe in** — 12-month time-series momentum (Moskowitz, Ooi, Pedersen 2012), not searched on this ticker. Same DSR test. Many “classic” rules still fail once you charge them for the search you *actually* did (here: 1000 nearby technical rules).
4. **Paper example** — SR 2.5, 100 trials, Var(SR)=0.5, 5 years of daily data, fat tails → DSR ≈ 0.90. The investor says no.

Rule of thumb in the paper: **DSR ≥ 0.95** means it survives.

## Tools (the only stack)

| Tool | Why it is here |
|---|---|
| **Python** | One language, whole demo. |
| **curl_cffi** | Talks to Yahoo's chart API (real SPY prices). Cached under `data/`. |
| **pandas** | Price series, rolling SMAs, percent returns. |
| **numpy** | Arrays, random rule parameters, correlation of PnL. |
| **scipy.stats** | `norm.cdf` / `norm.ppf` (Φ and Φ⁻¹ in the paper), skew, kurtosis. |
| **matplotlib** | Histogram of 1000 Sharpes + equity curves. |

You do **not** need `esvhd/pypbo` installed. That repo is the same math (PSR, DSR, minTRL, PBO). We reimplemented the DSR pieces in `stats.py` so you can read every line.

## Math, in one screen

All formulas use **per-day** Sharpe (`mean/std`), not the annualized number. Annualize only to print: `× √252`.

**Expected best Sharpe after N independent noise trials** (paper Eq. 1, null mean 0):

`SR₀ = σ_SR × [(1−γ) Φ⁻¹(1−1/N) + γ Φ⁻¹(1−1/(N e))]`

γ ≈ 0.577 (Euler–Mascheroni). `σ_SR` is the **standard deviation of Sharpes across trials**.

**Probabilistic Sharpe (PSR)** — P(true SR > threshold), with a non-Normal correction:

`PSR = Φ[ (SR − SR*) √(T−1) / √(1 − γ₃ SR + ((γ₄−1)/4) SR²) ]`

T = days, γ₃ = skew, γ₄ = **raw** kurtosis (3 if Normal).

**Deflated Sharpe (DSR)** = PSR with threshold `SR* = SR₀`.  
It is a probability, not a deflated “Sharpe number”.

**N_eff** because random SMA rules are correlated:

`N_eff = N / (1 + (N−1) ρ̄)` where ρ̄ is mean absolute pairwise correlation of daily PnL.

Using raw N=1000 is harsher. The demo reports both.

## Files to read, in order

1. `stats.py` — formulas only. Start with `expected_max_z`, then `deflated_sharpe`.
2. `rules.py` — random rules + the momentum candidate. Note `shift(1)` (no look-ahead).
3. `data.py` — Yahoo chart API download.
4. `run.py` — glue: lottery, DSR reports, plots.

## If someone asks “did we overfit?”

Say: we generated a thousand technically plausible rules on real SPY, kept the max Sharpe, and compared it to the closed-form noise ceiling. Then we ran DSR on that winner **and** on 12-month momentum, charging both for the trial count. DSR is the probability the result is still above that ceiling after selection bias and non-Normal returns.
