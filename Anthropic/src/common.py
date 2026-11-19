"""
common.py - shared paths, constants, NYSE trading calendar, and a cached
yfinance fetcher for the Anthropic (ANT) application of the IPO-fade study.

Conventions (identical to the parent study's build_panel.py):
  * day 0  = the listing day (first real trading bar). "+k" = k trading days later.
  * Executable prices only: every return is close-to-close on AdjClose.
    The day-0 intraday HIGH is a measurement reference (legacy Study-1 metric),
    never an entry price.
  * hedged / fade metric x = qqq_ret - stock_ret over the same dates
    (dollar-neutral short stock / long QQQ; positive = the fade was captured).
  * For PRICE TARGETS the QQQ-neutral ratio form is used instead:
        rel = (1 + stock_ret) / (1 + qqq_ret) - 1
    (always yields a positive price; target = anchor x (1 + rel) x (1 + QQQ path)).

Everything is relative to this file, so the study runs on any machine:
    python Anthropic/src/run_all.py
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- paths
ANT_DIR = Path(__file__).resolve().parents[1]
REPO = ANT_DIR.parent
DATA = ANT_DIR / "data"
RAW = DATA / "raw"
RESULTS = ANT_DIR / "results"
FIGS = ANT_DIR / "figs"
for _p in (RAW, RESULTS, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

PARENT_UNIVERSE = REPO / "data" / "ipo_universe_t1t4.csv"
PARENT_AUDIT = REPO / "data" / "universe_audit_t1t4.csv"
PARENT_FETCH_LOG = REPO / "data" / "price_fetch_log_t1t4.csv"
PARENT_PANEL = REPO / "results" / "event_panel.csv"
PARENT_PRIMARY = REPO / "results" / "primary_spec.csv"

DEAL_TERMS = DATA / "ant_deal_terms.csv"
LOCKUP = DATA / "ant_lockup_schedule.csv"

# ----------------------------------------------------------------------------- constants
HORIZONS = [5, 10, 20, 40, 60, 90, 120, 180, 250]
TARGET_HORIZONS = [20, 60, 120, 180, 250]
WINDOW_BARS = 260                       # bars kept per event (day 0 .. day 259)
DISCOVERY_END = pd.Timestamp("2018-12-31")   # parent study's fixed OOS split
PRIMARY_ENTRY, PRIMARY_EXIT = 20, 180   # parent study's pre-registered trade
BORROWS = [0.00, 0.03, 0.10, 0.30]
N_BOOT = 10_000
BLOCK_LEN = 4                           # quarters per moving block (parent: 4)

ANT_TICKER = "ANT"
ANT_LISTING = pd.Timestamp("2026-11-19")
ANT_OFFER = 83.00
ANT_MCAP_BASIC_B = 1099.0               # 13.24B basic shares x $83 (see deal terms)
ANT_MCAP_FD_B = 1500.0                  # CNBC "around $1.5 trillion" fully diluted
DATA_PULL_DATE = pd.Timestamp.today().normalize()
# Only COMPLETE sessions enter the cohort/benchmark data: during market hours
# today's partial bar is dropped (ANT's own status is read separately).
_NOW_ET = pd.Timestamp.now(tz="America/New_York")
_TODAY_ET = _NOW_ET.normalize().tz_localize(None)
MARKET_CLOSED_TODAY = (_NOW_ET.hour, _NOW_ET.minute) >= (16, 15) or _TODAY_ET.weekday() >= 5
FETCH_END = _TODAY_ET + pd.Timedelta(days=1) if MARKET_CLOSED_TODAY else _TODAY_ET   # exclusive

# ----------------------------------------------------------------------------- calendar
_SPECIAL_CLOSURES = {            # unscheduled full-day NYSE closures since 2010
    dt.date(2012, 10, 29), dt.date(2012, 10, 30),   # Hurricane Sandy
    dt.date(2018, 12, 5),                           # President G.H.W. Bush
    dt.date(2025, 1, 9),                            # President Carter
}


def _easter(y: int) -> dt.date:
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = ((h + l_ - 7 * m + 114) % 31) + 1
    return dt.date(y, month, day)


def _nth_weekday(y, month, weekday, n):
    d = dt.date(y, month, 1)
    return d + dt.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(y, month, weekday):
    d = (dt.date(y, month + 1, 1) if month < 12 else dt.date(y + 1, 1, 1)) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def nyse_holidays(y: int) -> set[dt.date]:
    h = set()
    nyd = dt.date(y, 1, 1)
    if nyd.weekday() == 6:
        h.add(nyd + dt.timedelta(days=1))
    elif nyd.weekday() != 5:          # Saturday NYD is not observed on Friday
        h.add(nyd)
    h.add(_nth_weekday(y, 1, 0, 3))   # MLK
    h.add(_nth_weekday(y, 2, 0, 3))   # Washington's Birthday
    h.add(_easter(y) - dt.timedelta(days=2))   # Good Friday
    h.add(_last_weekday(y, 5, 0))     # Memorial Day
    if y >= 2022:
        h.add(_observed(dt.date(y, 6, 19)))    # Juneteenth
    h.add(_observed(dt.date(y, 7, 4)))
    h.add(_nth_weekday(y, 9, 0, 1))   # Labor Day
    h.add(_nth_weekday(y, 11, 3, 4))  # Thanksgiving
    h.add(_observed(dt.date(y, 12, 25)))
    return h


def trading_days(start, end) -> pd.DatetimeIndex:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    hol = set(_SPECIAL_CLOSURES)
    for y in range(start.year, end.year + 1):
        hol |= nyse_holidays(y)
    return pd.DatetimeIndex([d for d in pd.bdate_range(start, end) if d.date() not in hol])


def ant_calendar_day(k: int) -> pd.Timestamp:
    """Calendar date of ANT trading day +k (day 0 = 2026-11-19)."""
    days = trading_days(ANT_LISTING, ANT_LISTING + pd.Timedelta(days=int(k * 1.6) + 30))
    return days[k]


def ant_trading_day_of(date) -> int:
    """ANT trading-day index of the first trading day on/after `date`."""
    date = pd.Timestamp(date)
    days = trading_days(ANT_LISTING, date + pd.Timedelta(days=10))
    return int(np.searchsorted(days.values, np.datetime64(date), side="left"))


# ----------------------------------------------------------------------------- fetch
def _safe(t: str) -> str:
    return t.replace("^", "_").replace("/", "-").replace(".", "-")


def _normalize(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    if getattr(h.index, "tz", None) is not None:
        h.index = h.index.tz_localize(None)
    h.index = pd.DatetimeIndex(h.index).normalize()
    h.index.name = "Date"
    ren = {"Adj Close": "AdjClose", "Stock Splits": "Splits"}
    h = h.rename(columns=ren)
    keep = [c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume", "Splits"] if c in h.columns]
    h = h[keep]
    if "Splits" not in h.columns:
        h["Splits"] = 0.0
    return h[~h.index.duplicated(keep="last")].sort_index()


def yf_history(ticker: str, start, end) -> pd.DataFrame | None:
    """Raw yfinance call with retries. Returns normalized frame or None."""
    import yfinance as yf
    last_err = None
    for attempt in range(4):
        try:
            h = yf.Ticker(ticker).history(start=pd.Timestamp(start).date().isoformat(),
                                          end=pd.Timestamp(end).date().isoformat(),
                                          auto_adjust=False, actions=True)
            if h is not None and len(h):
                return _normalize(h)
            return None
        except Exception as e:  # network / rate limit
            last_err = e
            time.sleep(2 ** attempt)
    print(f"  [WARN] {ticker}: fetch failed after retries ({str(last_err)[:80]})")
    return None


def cached_history(ticker: str, start, end, refresh: bool = False) -> pd.DataFrame | None:
    """Full history in [start, end) cached at data/raw/<TICKER>.csv (+ .missing
    marker for negative results, so reruns do not hammer Yahoo)."""
    path = RAW / f"{_safe(ticker)}.csv"
    miss = RAW / f"{_safe(ticker)}.missing"
    if not refresh:
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        if miss.exists():
            return None
    h = yf_history(ticker, start, end)
    time.sleep(0.4)
    if h is None or len(h) == 0:
        miss.write_text(json.dumps({"ticker": ticker, "checked": str(dt.datetime.now(dt.timezone.utc))}))
        return None
    if miss.exists():
        miss.unlink()
    h.to_csv(path)
    return h


def placeholder_mask(h: pd.DataFrame, offer: float | None = None) -> pd.Series:
    """Yahoo placeholder rows: Open<=0 (e.g. ANT pre-open: O=0 H=L=C=$83 V=0), or a
    flat zero-volume bar sitting exactly on the offer price. NOTE: zero volume
    alone is NOT a placeholder - Yahoo reports V=0 on Venture Global's real first
    four sessions (2025-01-24..29), so a volume filter would mis-anchor it."""
    o = h["Open"].fillna(0)
    v = h["Volume"].fillna(0)
    flat = (h["High"] == h["Low"]) & (h["Low"] == h["Close"])
    at_offer = (h["Close"] - offer).abs() < 0.011 if offer is not None and pd.notna(offer) else False
    return (o <= 0) | ((v <= 0) & flat & at_offer)


def resolve_first_trade(h: pd.DataFrame, expected: pd.Timestamp, offer: float | None = None,
                        authoritative: bool = False):
    """Day-0 anchor. If `authoritative` (the parent study's regression-tested
    anchor date is known), take the first non-placeholder bar on/after it.
    Otherwise: first real (volume>0) bar on/after expected-5d, with guards for
    pre-listing OTC prints (VICI, PECO junk bars) and recycled/truncated symbols.
    Returns (anchor_date or None, status)."""
    expected = pd.Timestamp(expected)
    usable = ~placeholder_mask(h, offer) & (h["Open"].fillna(0) > 0)
    if authoritative:
        cand = h[(h.index >= expected) & usable]
        if len(cand) == 0:
            return None, "no_real_bar_yet"
        anchor = cand.index[0]
        if (anchor - expected).days > 4:
            return anchor, "late_start_gap"      # Yahoo history truncated (KVUE, OLPX)
        return anchor, "ok"
    lo = expected - pd.Timedelta(days=5)
    real = usable & (h["Volume"].fillna(0) > 0)
    pre = h[(h.index < lo) & real]
    if len(pre) > 5:
        # Symbol traded before the IPO (OTC pre-listing, e.g. VICI post-bankruptcy;
        # or an earlier entity). Anchor strictly on the audited listing date.
        cand = h[(h.index >= expected) & real]
        if len(cand) == 0:
            return None, "no_real_bar_yet"
        anchor = cand.index[0]
        if (anchor - expected).days > 4:
            return anchor, "prelisting_bars_unresolved"
        return anchor, "ok_prelisting_trimmed"
    cand = h[(h.index >= lo) & real]
    if len(cand) == 0:
        return None, "no_real_bar_yet"
    anchor = cand.index[0]
    if (anchor - expected).days > 10:
        # first real bar far after the listing: Yahoo history truncated or the
        # symbol now belongs to a different (later) entity -> unusable
        return anchor, "late_start_gap"
    return anchor, "ok"


def split_factor_after(h: pd.DataFrame, date) -> float:
    """Product of split ratios strictly after `date` (Yahoo OHLC are adjusted for
    all later splits, so raw price at `date` = adjusted price x this factor)."""
    s = h.loc[h.index > pd.Timestamp(date), "Splits"].replace(0, np.nan).dropna()
    return float(np.prod(s.values)) if len(s) else 1.0


def quarter_of(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{(ts.month - 1) // 3 + 1}"


def era_of(year: int) -> str:
    return "2010-2014" if year <= 2014 else "2015-2019" if year <= 2019 else "2020+"


def block_bootstrap_ci(values: np.ndarray, quarters: np.ndarray, stat_fn,
                       n_boot: int = N_BOOT, block: int = BLOCK_LEN, alpha: float = 0.05,
                       seed: int = 42) -> tuple[float, float]:
    """Moving-block bootstrap clustered by listing quarter - same construction as
    the parent's primary_spec.py: quarters ordered chronologically, blocks of
    `block` consecutive quarters drawn with replacement until the quarter count
    is covered, every event in a drawn quarter kept (with multiplicity)."""
    rng = np.random.default_rng(seed)
    qs = sorted(pd.unique(quarters))
    nq = len(qs)
    if nq == 0:
        return np.nan, np.nan
    by_q = [values[quarters == q] for q in qs]
    starts_all = np.arange(1) if nq <= block else np.arange(nq - block + 1)
    n_blocks = int(np.ceil(nq / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        vals = np.concatenate([by_q[s + j] for s in starts for j in range(block) if s + j < nq])
        out[b] = stat_fn(vals) if len(vals) else np.nan
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fmt_pct(x, nd=1):
    return "" if pd.isna(x) else f"{x:+.{nd}%}"
