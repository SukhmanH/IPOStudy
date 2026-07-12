"""
fetch_prices_t1t4.py — price-history fetch + retention accounting for the
T1-T4 universe (headline names + too-recent 'considered' names, which enter
short-horizon exploratory cells only).

- yfinance primary via the shared spcx-report cache (utils.fetch_history,
  raw=True: unadjusted OHLC + AdjClose; retried, negative-cached, resumable).
- Stooq FALLBACK for names yfinance drops (brief requirement): daily CSV from
  stooq.com (<sym>.us), cached to data/raw/<TKR>__stooq.csv. Stooq data is
  split-adjusted but NOT dividend-adjusted — tagged so the panel builder can
  treat it accordingly (documented limitation).
- Recycled-ticker guard: earliest bar >270 days from the listing date =>
  status recycled_mismatch (cache kept, name flagged; wrong-entity prices
  must not enter the panel).
- Retention table by tier x era with a reason-coded status for EVERY name
  (nothing silent) -> results/retention_by_tier_era.csv +
  data/price_fetch_log_t1t4.csv.

Run:  python fetch_prices_t1t4.py     (hours on first run; cached after)
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(r"c:\Users\brylan\Desktop\spaceX")
GEN = REPO / "general"
sys.path.insert(0, str(REPO / "spcx-report" / "src"))
import utils as u  # noqa: E402

UNIVERSE = GEN / "data" / "ipo_universe_t1t4.csv"
FETCH_LOG = GEN / "data" / "price_fetch_log_t1t4.csv"
RETENTION = GEN / "results" / "retention_by_tier_era.csv"
RECYCLE_GAP_DAYS = 270
STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"
STOOQ_PAUSE = 1.0          # be polite; stooq enforces a daily hit limit
_stooq_limit_hit = False


def stooq_fetch(ticker: str) -> pd.DataFrame | None:
    """Daily history from Stooq (cached). Returns None if unavailable."""
    global _stooq_limit_hit
    cache = u.RAW_DIR / f"{ticker.replace('/', '-').replace('^', '_')}__stooq.csv"
    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            return df if len(df) else None
        except Exception:
            pass
    if _stooq_limit_hit:
        return None
    sym = ticker.lower().replace("-", ".") + ".us"
    try:
        r = requests.get(STOOQ_URL.format(sym=sym), timeout=20,
                         headers={"User-Agent": "Mozilla/5.0 (research)"})
        time.sleep(STOOQ_PAUSE)
        txt = r.text
        if "Exceeded the daily hits limit" in txt:
            _stooq_limit_hit = True
            print("  [WARN] Stooq daily hit limit reached — remaining fallbacks skipped this run")
            return None
        if not txt.startswith("Date,"):
            return None
        df = pd.read_csv(io.StringIO(txt), index_col=0, parse_dates=True)
        if len(df) == 0:
            return None
        df.index.name = "Date"
        df.to_csv(cache)
        return df
    except Exception:
        return None


def main():
    print("FETCH PRICES T1-T4 — yfinance primary, Stooq fallback, retention log")
    print("=" * 68)
    uni = pd.read_csv(UNIVERSE, keep_default_na=False)
    names = uni[(uni["bucket"] == "headline")
                | ((uni["bucket"] == "considered")
                   & uni["reason"].str.contains("too recent", na=False))].copy()
    names["listing_dt"] = pd.to_datetime(names["first_trade"], errors="coerce")
    print(f"names to fetch: {len(names)}  (headline {sum(names.bucket=='headline')}, "
          f"too-recent {sum(names.bucket=='considered')})")

    rows = []
    n = len(names)
    for i, r in enumerate(names.itertuples(), 1):
        tkr = r.ticker
        status, src, first_bar, nbars = "", "", "", 0
        hist = u.fetch_history(tkr, raw=True)
        if hist is not None and len(hist):
            src = "yfinance"
        else:
            hist = stooq_fetch(tkr)
            if hist is not None and len(hist):
                src = "stooq"
        if hist is None or len(hist) == 0:
            status = "missing_both_sources"
        else:
            nbars = len(hist)
            fb = hist.index.min()
            first_bar = fb.date().isoformat()
            if pd.notna(r.listing_dt) and abs((fb - r.listing_dt).days) > RECYCLE_GAP_DAYS:
                # earliest bar far from listing: pre-listing artifact OR recycled
                # symbol. If the history simply STARTS long before the IPO it is
                # recycled/wrong-entity; resolve_first_trade can't save that.
                status = "recycled_mismatch" if fb < r.listing_dt else "late_start_gap"
            else:
                status = "ok"
        rows.append({
            "ticker": tkr, "tier": r.tier, "bucket": r.bucket,
            "listing": r.first_trade, "year": str(r.first_trade)[:4],
            "status": status, "source": src, "first_bar": first_bar, "n_bars": nbars,
        })
        if i % 100 == 0:
            print(f"  ...{i}/{n}")
            pd.DataFrame(rows).to_csv(FETCH_LOG, index=False)  # checkpoint

    log = pd.DataFrame(rows)
    log.to_csv(FETCH_LOG, index=False)

    # ---- retention by tier x era ----
    log["era"] = pd.cut(pd.to_numeric(log["year"], errors="coerce"),
                        [2009, 2014, 2019, 2026],
                        labels=["2010-2014", "2015-2019", "2020+"])
    ret = (log.assign(ok=(log["status"] == "ok").astype(int),
                      stooq=((log["status"] == "ok") & (log["source"] == "stooq")).astype(int))
              .groupby(["tier", "era"], observed=True)
              .agg(universe=("ticker", "count"), kept=("ok", "sum"),
                   via_stooq=("stooq", "sum"))
              .reset_index())
    ret["dropped"] = ret["universe"] - ret["kept"]
    ret["pct_kept"] = (100 * ret["kept"] / ret["universe"]).round(1)
    RETENTION.parent.mkdir(parents=True, exist_ok=True)
    ret.to_csv(RETENTION, index=False)

    print("\n" + "=" * 68)
    print("RETENTION by tier x era:")
    print(ret.to_string(index=False))
    print("\nStatus counts:")
    print(log["status"].value_counts().to_string())
    print("\nSource mix (ok names):")
    print(log[log["status"] == "ok"]["source"].value_counts().to_string())
    print(f"\nWrote:\n  {FETCH_LOG}\n  {RETENTION}")


if __name__ == "__main__":
    main()
