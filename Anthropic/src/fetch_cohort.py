"""
fetch_cohort.py - comparable cohort + price fetch for the ANT application.

Cohort (fixed BEFORE looking at any ANT-specific result):
  * HISTORICAL: the parent study's T3+T4 headline universe (traditional
    underwritten IPOs >= $5B at listing, 2010+, SPACs / direct listings /
    spinoffs / uplistings excluded) - exactly the pre-registered primary-spec
    universe. T4 (> $10B) is ANT's own tier.
  * LIVE (out-of-sample): T3/T4 names the parent logged as "too recent"
    (listed Oct-2025 .. Jul-2026) plus the two T3/T4 IPOs its seed calendar
    missed (INIO Jun-2026, JMKE Jul-2026). These have no +180 outcome yet;
    they test whether the early-fade pattern still holds in the 2025-26 regime.
  * SUBJECT: ANT itself (listed 2026-11-19).
Every candidate gets a reason code in data/cohort_universe.csv; every fetch
gets a status in data/price_fetch_log.csv. Nothing is dropped silently.

Outputs:
  data/cohort_universe.csv
  data/price_fetch_log.csv
  data/cohort_prices_window.csv.gz   (day 0..259 bars per event; committed)
  data/benchmarks.csv.gz             (QQQ/SPY/IWM AdjClose, VIX close)
  data/ant_prices.csv                (every ANT bar Yahoo has, placeholder rows flagged)
Raw full-history caches live in data/raw/ (git-ignored; rebuilt on demand).

Run:  python Anthropic/src/fetch_cohort.py [--refresh-ant] [--refresh-all]
"""

from __future__ import annotations

import sys
import datetime as dt

import numpy as np
import pandas as pd

import common as c

# Secondary-source corrections for the LIVE cohort (StockTitan "Biggest IPOs of
# 2026", value at IPO price; fetched 2026-11-19). Parent tiers for these rows were
# low-confidence single-source (yfinance share count) estimates.
LIVE_CORRECTIONS = {
    "BSP": dict(tier="T3", mcap_b=9.42, why="parent T4 $18.4B (low-conf yf shares) -> StockTitan $9.42B"),
    "DSC": dict(tier="", mcap_b=np.nan, why="parent T4 $16.0B (low-conf yf shares) but absent from StockTitan's "
                                         "list of every 2026 IPO >= $1B -> size unverified; excluded from live T3/T4 check"),
}
LIVE_ADDITIONS = [
    dict(ticker="INIO", name="Innio N.V.", listing="2026-06-04", offer=27.00, tier="T4", mcap_b=20.25,
         why="in parent seed but mcap unresolved; StockTitan $20.25B -> T4"),
    dict(ticker="JMKE", name="Jersey Mike's Subs", listing="2026-07-30", offer=23.00, tier="T3", mcap_b=7.31,
         why="priced 2026-07-29, after the parent seed calendar ended; StockTitan $7.31B -> T3"),
]
EXCLUDED_EXTRA = [
    dict(ticker="SKHY", name="SK hynix Inc. (ADS)", listing="2026-07-10", offer=149.00,
         why="cross-listing of an already-public company (KRX:000660): no IPO price discovery, "
             "treated like the spec's uplisting exclusion"),
]


def build_cohort() -> pd.DataFrame:
    uni = pd.read_csv(c.PARENT_UNIVERSE, keep_default_na=False)
    aud = pd.read_csv(c.PARENT_AUDIT, keep_default_na=False, low_memory=False)
    adr = (aud.drop_duplicates("ticker").set_index("ticker")["adr"]
              .map(lambda v: str(v).strip().lower() in ("1", "1.0", "true")))
    rows = []
    for r in uni[uni["tier"].isin(["T3", "T4"])].itertuples():
        mcap = pd.to_numeric(r.mcap_listing_b, errors="coerce")
        base = dict(ticker=r.ticker, name=r.name, listing_expected=r.first_trade,
                    offer=pd.to_numeric(r.offer_price, errors="coerce"),
                    tier=r.tier, mcap_b=mcap, mcap_source=f"parent:{r.mcap_src}/{r.mcap_conf}",
                    adr=bool(adr.get(r.ticker, False)))
        if r.bucket == "headline":
            rows.append({**base, "role": "historical", "reason": "parent T3/T4 headline universe"})
        elif "too recent" in r.reason:
            fix = LIVE_CORRECTIONS.get(r.ticker)
            if fix:
                base.update(tier=fix["tier"], mcap_b=fix["mcap_b"], mcap_source="StockTitan 2026 IPO list")
                role = "live" if fix["tier"] in ("T3", "T4") else "excluded"
                rows.append({**base, "role": role, "reason": "tier corrected: " + fix["why"]})
            else:
                rows.append({**base, "role": "live", "reason": "parent 'too recent' T3/T4 (no +180 outcome at parent freeze)"})
        else:
            rows.append({**base, "role": "excluded", "reason": "parent: " + r.reason})
    for a in LIVE_ADDITIONS:
        rows.append(dict(ticker=a["ticker"], name=a["name"], listing_expected=a["listing"], offer=a["offer"],
                         tier=a["tier"], mcap_b=a["mcap_b"], mcap_source="StockTitan 2026 IPO list",
                         adr=False, role="live", reason="added: " + a["why"]))
    for e in EXCLUDED_EXTRA:
        rows.append(dict(ticker=e["ticker"], name=e["name"], listing_expected=e["listing"], offer=e["offer"],
                         tier="", mcap_b=np.nan, mcap_source="", adr=True, role="excluded", reason=e["why"]))
    rows.append(dict(ticker=c.ANT_TICKER, name="Anthropic, PBC", listing_expected=c.ANT_LISTING.date().isoformat(),
                     offer=c.ANT_OFFER, tier="T4", mcap_b=c.ANT_MCAP_BASIC_B,
                     mcap_source="derived: 13.24B basic shares x $83 (FD ~$1.5T)", adr=False,
                     role="subject", reason="study subject"))
    df = pd.DataFrame(rows)
    df.to_csv(c.DATA / "cohort_universe.csv", index=False)
    return df


def fetch_benchmarks(today: pd.Timestamp) -> pd.DataFrame:
    out = {}
    for t, col in [("QQQ", "AdjClose"), ("SPY", "AdjClose"), ("IWM", "AdjClose"), ("^VIX", "Close")]:
        h = c.cached_history(t, "2009-11-01", today + pd.Timedelta(days=1), refresh=True)
        if h is None:
            raise SystemExit(f"benchmark {t} unavailable - cannot continue")
        h = h[h.index < c.FETCH_END]          # complete sessions only
        out[t.replace("^", "")] = h[col].astype(float)
    b = pd.DataFrame(out).sort_index()
    b.to_csv(c.DATA / "benchmarks.csv.gz", float_format="%.6f")
    # calendar self-check against the realized QQQ trading dates (2011..last bar)
    lo, hi = pd.Timestamp("2011-01-01"), b["QQQ"].dropna().index.max()
    cal = c.trading_days(lo, hi)
    q = b["QQQ"].dropna().loc[lo:hi].index
    extra, missing = cal.difference(q), q.difference(cal)
    print(f"[calendar] NYSE rule calendar vs QQQ bars {lo.date()}..{hi.date()}: "
          f"{len(extra)} calendar-only days, {len(missing)} bar-only days")
    if len(extra) or len(missing):
        print("   calendar-only:", [d.date().isoformat() for d in extra[:10]],
              " bar-only:", [d.date().isoformat() for d in missing[:10]])
    return b


def fetch_events(cohort: pd.DataFrame, today: pd.Timestamp, refresh_all: bool) -> None:
    todo = cohort[cohort["role"].isin(["historical", "live"])].copy()
    # The parent's regression-tested day-0 anchors are authoritative where they exist
    parent_anchor = (pd.read_csv(c.PARENT_PANEL, keep_default_na=False, usecols=["ticker", "listing"])
                       .drop_duplicates("ticker").set_index("ticker")["listing"])
    log, frames = [], []
    n = len(todo)
    for i, r in enumerate(todo.itertuples(), 1):
        exp = pd.Timestamp(r.listing_expected)
        h = c.cached_history(r.ticker, exp - pd.Timedelta(days=40), today + pd.Timedelta(days=1),
                             refresh=refresh_all)
        if h is not None:
            h = h[h.index < c.FETCH_END]      # complete sessions only
        rec = dict(ticker=r.ticker, role=r.role, tier=r.tier, listing_expected=r.listing_expected,
                   status="", first_bar="", n_bars_from_day0=0, split_factor_after_day0=np.nan,
                   raw_close0=np.nan, offer=r.offer, pop_close=np.nan,
                   anchor_source="parent_panel" if r.ticker in parent_anchor.index else "resolver")
        if h is None or len(h) == 0:
            rec["status"] = "missing_yahoo"
            log.append(rec)
            continue
        auth = r.ticker in parent_anchor.index
        anchor, status = c.resolve_first_trade(h, pd.Timestamp(parent_anchor[r.ticker]) if auth else exp,
                                               r.offer, authoritative=auth)
        if anchor is None:
            rec["status"] = status
            log.append(rec)
            continue
        win = h.loc[anchor:].iloc[: c.WINDOW_BARS].copy()
        sf = c.split_factor_after(h, anchor)
        raw_close0 = float(win["Close"].iloc[0]) * sf
        pop = raw_close0 / r.offer - 1 if pd.notna(r.offer) and r.offer > 0 else np.nan
        if status.startswith("ok") and pd.notna(pop) and not (0.25 <= 1 + pop <= 6.0):
            status = "offer_price_mismatch"      # split/units problem or wrong entity
        rec.update(status=status, first_bar=anchor.date().isoformat(), n_bars_from_day0=len(win),
                   split_factor_after_day0=sf, raw_close0=raw_close0, pop_close=pop)
        log.append(rec)
        if not status.startswith("ok"):
            continue                              # logged above; never enters the panel
        win.insert(0, "day", np.arange(len(win)))
        win.insert(0, "ticker", r.ticker)
        win["split_factor_after_day0"] = sf
        frames.append(win.reset_index())
        if i % 25 == 0 or i == n:
            print(f"  ...{i}/{n}")
    logdf = pd.DataFrame(log)
    logdf.to_csv(c.DATA / "price_fetch_log.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(c.DATA / "cohort_prices_window.csv.gz", index=False,
                                                float_format="%.6f")
    print("\nfetch status by role x tier:")
    print(logdf.groupby(["role", "tier", "status"]).size().to_string())


def fetch_ant(today: pd.Timestamp) -> None:
    h = c.yf_history(c.ANT_TICKER, c.ANT_LISTING - pd.Timedelta(days=10), today + pd.Timedelta(days=1))
    now_et = pd.Timestamp.now(tz="America/New_York")
    if h is None or len(h) == 0:
        print("[ANT] no Yahoo rows yet")
        pd.DataFrame(columns=["Open", "High", "Low", "Close", "AdjClose", "Volume", "placeholder"]) \
          .to_csv(c.DATA / "ant_prices.csv")
        return
    h["placeholder"] = (h["Open"].fillna(0) <= 0) | (h["Volume"].fillna(0) <= 0)
    h["fetched_at_et"] = now_et.strftime("%Y-%m-%d %H:%M")
    h.to_csv(c.DATA / "ant_prices.csv")
    print("[ANT] Yahoo rows:\n" + h.to_string())


def main():
    refresh_all = "--refresh-all" in sys.argv
    today = c.DATA_PULL_DATE
    print(f"FETCH COHORT - data pull {today.date()} (UTC {dt.datetime.now(dt.timezone.utc):%H:%M}); "
          f"complete sessions before {c.FETCH_END.date()} only")
    print("=" * 72)
    cohort = build_cohort()
    print(cohort.groupby(["role", "tier"]).size().to_string())
    fetch_benchmarks(today)
    fetch_events(cohort, today, refresh_all)
    fetch_ant(today)


if __name__ == "__main__":
    main()
