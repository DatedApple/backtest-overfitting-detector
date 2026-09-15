"""Download real daily adjusted closes from Yahoo's chart API.

Use period1/period2 unix timestamps. `range=max` can downsample to monthly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from curl_cffi import requests as cffi_requests

DATA_DIR = Path(__file__).resolve().parent / "data"


def _from_yahoo(ticker: str, start: str, end: str | None) -> pd.Series:
    session = cffi_requests.Session(impersonate="chrome")
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp()) if end else int(pd.Timestamp.utcnow().timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = session.get(
        url,
        params={
            "interval": "1d",
            "period1": start_ts,
            "period2": end_ts,
            "includeAdjustedClose": "true",
        },
        timeout=120,
    )
    r.raise_for_status()
    payload = r.json()
    result = (payload.get("chart") or {}).get("result")
    if not result:
        raise RuntimeError(payload.get("chart", {}).get("error") or "yahoo empty")
    block = result[0]
    ts = block["timestamp"]
    adj = (block.get("indicators") or {}).get("adjclose") or []
    if adj and adj[0].get("adjclose"):
        closes = adj[0]["adjclose"]
    else:
        closes = block["indicators"]["quote"][0]["close"]
    s = pd.Series(closes, index=pd.to_datetime(ts, unit="s"), name="Close", dtype=float)
    s = s.dropna().sort_index()
    s.index.name = "Date"
    if s.empty:
        raise RuntimeError("yahoo empty after date filter")
    return s


def load_prices(ticker: str = "SPY", start: str = "2010-01-01", end: str | None = None) -> pd.Series:
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"{ticker}_{start}_{end or 'latest'}.csv"
    if cache.exists():
        s = pd.read_csv(cache, parse_dates=["Date"], index_col="Date")["Close"]
        return s.astype(float).sort_index()

    close = _from_yahoo(ticker, start, end)
    close.to_frame().to_csv(cache)
    print(f"  price source: yahoo chart API ({len(close)} daily bars)")
    return close
