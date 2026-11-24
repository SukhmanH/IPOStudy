"""
fetch_prices.py -- daily OHLCV + split/dividend actions from Yahoo (yfinance)
for every ticker in data/universe.csv plus benchmarks.

* Prices come back SPLIT-ADJUSTED (Open/High/Low/Close) with a separate
  'Stock Splits' column.  build_events.py reconstructs the *traded* price
  level from the split history so that price filters ($5 floor) are not
  contaminated by later reverse splits (a look-ahead that would otherwise
  let historical penny stocks into the "$5+" universe).
* Raw batches are cached to data/raw_yahoo/ (git-ignored, ~GBs) so reruns are
  free; data/price_fetch_log.csv (committed) records every ticker with a
  status code: ok | empty | error:<msg>.
* Pull window is pinned: 2008-06-01 .. 2026-11-23 (end exclusive 2026-11-24).

Run:  python short_gainers/src/fetch_prices.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CACHE = DATA / "raw_yahoo"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = DATA / "price_fetch_log.csv"
START, END = "2008-06-01", "2026-11-24"
BENCH = ["SPY", "QQQ", "IWM", "^VIX"]
BATCH = 100

yf.set_tz_cache_location(str(DATA / "raw" / "yf_tz_cache"))


def to_long(df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, dict]:
    out, status = [], {}
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
        except KeyError:
            status[t] = "empty"
            continue
        sub = sub.dropna(subset=["Open", "High", "Low", "Close"], how="all")
        if sub.empty:
            status[t] = "empty"
            continue
        sub = sub.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                  "Adj Close": "adj_close", "Volume": "volume",
                                  "Dividends": "dividends", "Stock Splits": "splits"})
        cols = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
        sub = sub[[c for c in cols if c in sub.columns]].copy()
        sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
        sub.index.name = "date"
        sub.insert(0, "ticker", t)
        out.append(sub.reset_index())
        status[t] = "ok"
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), status


def fetch(tickers: list[str], threads: bool = True) -> tuple[pd.DataFrame, dict]:
    df = yf.download(tickers, start=START, end=END, auto_adjust=False, actions=True,
                     group_by="ticker", threads=threads, progress=False, repair=False)
    try:
        from yfinance import shared as _shared
        errs = dict(getattr(_shared, "_ERRORS", {}) or {})
    except Exception:
        errs = {}
    long, status = to_long(df, tickers)
    for t, e in errs.items():
        if t in status and status[t] != "ok":
            status[t] = "error:" + str(e)[:80].replace(",", ";").replace("\n", " ")
    return long, status


def main() -> None:
    uni = pd.read_csv(DATA / "universe.csv")
    tickers = BENCH + sorted(set(uni["yahoo"].dropna()) - set(BENCH))
    batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]
    log: dict[str, str] = {}
    for bi, b in enumerate(batches):
        f = CACHE / f"batch_{bi:03d}.parquet"
        st = CACHE / f"batch_{bi:03d}.status.csv"
        if f.exists() and st.exists():
            log.update(pd.read_csv(st).set_index("ticker")["status"].to_dict())
            continue
        for attempt in range(4):
            try:
                long, status = fetch(b)
                break
            except Exception as e:  # network hiccup / rate limit
                wait = 15 * (attempt + 1)
                print(f"batch {bi} attempt {attempt} failed: {e!r}; sleeping {wait}s", flush=True)
                time.sleep(wait)
        else:
            long, status = pd.DataFrame(), {t: "error:batch_failed" for t in b}
        # retry anything that did not come back, one at a time, unthreaded
        retry = [t for t in b if status.get(t) != "ok"]
        if retry:
            time.sleep(2)
            parts = [long] if not long.empty else []
            for t in retry:
                try:
                    l2, s2 = fetch([t], threads=False)
                    if s2.get(t) == "ok":
                        parts.append(l2)
                    status[t] = s2.get(t, status.get(t, "empty"))
                except Exception as e:
                    status[t] = "error:" + str(e)[:80].replace(",", ";")
            long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not long.empty:
            long.to_parquet(f, index=False)
        else:
            pd.DataFrame(columns=["ticker", "date"]).to_parquet(f, index=False)
        pd.DataFrame({"ticker": list(status), "status": list(status.values())}).to_csv(st, index=False)
        log.update(status)
        n_ok = sum(v == "ok" for v in status.values())
        print(f"batch {bi + 1}/{len(batches)}: ok {n_ok}/{len(b)}  rows {len(long)}", flush=True)
        time.sleep(1.0)
    pd.DataFrame({"ticker": list(log), "status": list(log.values())}).to_csv(LOG, index=False)
    s = pd.Series(log)
    print("DONE", s.map(lambda v: v.split(":")[0]).value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
