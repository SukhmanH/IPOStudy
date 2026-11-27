"""
fetch_intraday.py -- hourly bars (Yahoo, regular session, ~730 trading days:
2023-12-29 .. 2026-11-25) plus matching daily bars with split history, for
every ticker that could have been an intraday top gainer in that window
(data/intraday_tickers.csv: some day with high >= +10% over the prior close,
a traded high >= $5 and >= $2M traded) and SPY.

Daily bars are re-pulled at the same moment as the hourly ones so both carry
the same split adjustment.  Caches (git-ignored): data/raw_intraday/*.parquet.
Log (committed): data/intraday_fetch_log.csv.

Run:  python short_gainers/src/fetch_intraday.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CACHE = DATA / "raw_intraday"
CACHE.mkdir(parents=True, exist_ok=True)
BATCH = 80
yf.set_tz_cache_location(str(DATA / "raw" / "yf_tz_cache"))


def long_frame(df: pd.DataFrame, tickers: list[str], intraday: bool) -> tuple[pd.DataFrame, dict]:
    out, st = [], {}
    for t in tickers:
        try:
            s = df[t] if isinstance(df.columns, pd.MultiIndex) else df
        except KeyError:
            st[t] = "empty"
            continue
        s = s.dropna(subset=["Open", "High", "Low", "Close"], how="all")
        if s.empty:
            st[t] = "empty"
            continue
        s = s.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                              "Volume": "volume", "Stock Splits": "splits"})
        keep = [c for c in ["open", "high", "low", "close", "volume", "splits"] if c in s.columns]
        s = s[keep].copy()
        idx = pd.DatetimeIndex(s.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        s.index = idx if intraday else idx.normalize()
        s.index.name = "ts" if intraday else "date"
        s.insert(0, "ticker", t)
        out.append(s.reset_index())
        st[t] = "ok"
    return (pd.concat(out, ignore_index=True) if out else pd.DataFrame()), st


def main() -> None:
    tick = ["SPY"] + [t for t in pd.read_csv(DATA / "intraday_tickers.csv")["ticker"] if t != "SPY"]
    batches = [tick[i:i + BATCH] for i in range(0, len(tick), BATCH)]
    log = {}
    for bi, b in enumerate(batches):
        fh, fd = CACHE / f"h_{bi:03d}.parquet", CACHE / f"d_{bi:03d}.parquet"
        if fh.exists() and fd.exists():
            continue
        for attempt in range(4):
            try:
                h = yf.download(b, period="730d", interval="1h", auto_adjust=False, prepost=False,
                                group_by="ticker", threads=True, progress=False)
                d = yf.download(b, start="2023-10-01", end="2026-11-27", auto_adjust=False, actions=True,
                                group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"batch {bi} attempt {attempt}: {e!r}", flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            continue
        hl, hs = long_frame(h, b, True)
        dl, ds = long_frame(d, b, False)
        hl.to_parquet(fh, index=False)
        dl.to_parquet(fd, index=False)
        for t in b:
            log[t] = f"hourly:{hs.get(t, 'empty')} daily:{ds.get(t, 'empty')}"
        print(f"batch {bi + 1}/{len(batches)}: hourly ok {sum(v == 'ok' for v in hs.values())}/{len(b)} "
              f"rows {len(hl)}", flush=True)
        time.sleep(1.0)
    pd.Series(log, name="status").rename_axis("ticker").reset_index().to_csv(DATA / "intraday_fetch_log.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
