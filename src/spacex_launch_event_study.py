"""
SpaceX launch event study
Pre-IPO Hyperliquid perps (Ventuals SPACEX, TradeXYZ SPCX) + post-IPO SPCX stock.

Run in Claude Code or locally (this needs open network access to
api.hyperliquid.xyz, ll.thespacedevs.com, and Yahoo Finance).

    pip install requests pandas numpy matplotlib yfinance

Data landscape as of 2026-07-15:
  - Ventuals SPACEX perp (HIP-3, USDH margin): first pre-IPO market on HL,
    live since roughly Nov 2025. Actual start = first candle returned.
  - TradeXYZ SPCX perp (HIP-3): live 2026-05-18 at a $150 reference.
  - SPCX Nasdaq IPO: 2026-06-12. Priced $135, opened $150, closed $160.95.

Known landmines handled below:
  1. 2026-05-28 Ventuals oracle flash crash: bad Notice.co feed mishandled
     the 5-for-1 split, printed a fake -45% move ($2,277 -> $1,254) and
     liquidated ~1,400 positions. Not a real price. Excluded.
  2. The 5:1 split itself creates a persistent scale break in whichever
     series carried it. Handled by NaN-ing single-bar rescale returns.
  3. HL candleSnapshot returns only the ~5,000 most recent candles per
     coin+interval. At 1h that reaches back ~208 days (about 2025-12-20).
     Nov to mid-Dec history needs 1d candles or the HL S3 archive
     (s3://hyperliquid-archive, requester-pays). If HL delisted or settled
     the perp at IPO and returns nothing, use the Binance fallback below.
  4. These perps trade on crypto rails and BTC fell ~50% from its Oct 2025
     high inside the sample. All event returns are reported raw AND
     BTC-beta-adjusted.
"""

import json
import time
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

HL_INFO = "https://api.hyperliquid.xyz/info"
LL2 = "https://ll.thespacedevs.com/2.2.0/launch/previous/"
OUT = "./out"
IPO_DATE = pd.Timestamp("2026-06-12", tz="UTC")
SAMPLE_START = pd.Timestamp("2025-11-01", tz="UTC")

# Fake prints to exclude (UTC). Ventuals oracle incident window, May 28 2026.
EXCLUDE_WINDOWS = [
    (pd.Timestamp("2026-05-28 13:00", tz="UTC"),
     pd.Timestamp("2026-05-28 22:00", tz="UTC")),
]

# Event windows in hours relative to launch T0
POST_WINDOWS = [1, 4, 24, 72]


# ---------------------------------------------------------------- HL helpers
def hl_post(payload, retries=3):
    for i in range(retries):
        try:
            r = requests.post(HL_INFO, json=payload, timeout=20)
            if r.status_code == 200:
                return r.json()
            time.sleep(1 + i)
        except requests.RequestException:
            time.sleep(1 + i)
    return None


def discover_spacex_markets():
    """Enumerate all perp dexs (core + HIP-3 builders), grep for SpaceX."""
    found = []
    dexs = hl_post({"type": "perpDexs"}) or []
    names = [""] + [d.get("name", "") for d in dexs if isinstance(d, dict)]
    for dex in dict.fromkeys(names):  # dedupe, keep order
        meta = hl_post({"type": "meta", "dex": dex} if dex else {"type": "meta"})
        if not meta or "universe" not in meta:
            continue
        for a in meta["universe"]:
            nm = a.get("name", "")
            if any(k in nm.upper() for k in ("SPACEX", "SPCX")):
                found.append({"dex": dex, "coin": nm,
                              "delisted": a.get("isDelisted", False)})
        time.sleep(0.3)
    print("Discovered SpaceX markets:", json.dumps(found, indent=2))
    return found


def get_candles(coin, interval, start_ms, end_ms, dex=""):
    """candleSnapshot. Tries the raw universe name, then dex-prefixed."""
    for c in dict.fromkeys([coin, f"{dex}:{coin}" if dex else coin]):
        res = hl_post({"type": "candleSnapshot",
                       "req": {"coin": c, "interval": interval,
                               "startTime": int(start_ms),
                               "endTime": int(end_ms)}})
        if res:
            df = pd.DataFrame(res)
            if len(df):
                df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                for col in ("o", "h", "l", "c", "v"):
                    df[col] = df[col].astype(float)
                df = df.set_index("t").sort_index()
                print(f"  {c} {interval}: {len(df)} bars, "
                      f"{df.index[0]} -> {df.index[-1]}")
                return df[["o", "h", "l", "c", "v"]]
    print(f"  {coin} {interval}: no candles returned")
    return pd.DataFrame()


def binance_fallback():
    """If HL history is gone post-settlement, Binance listed pre-IPO SPCX
    perps too and its kline history has no recency cap."""
    try:
        info = requests.get(
            "https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
        syms = [s["symbol"] for s in info.get("symbols", [])
                if "SPCX" in s["symbol"] or "SPACEX" in s["symbol"]]
        print("Binance candidate symbols:", syms)
        out = {}
        for sym in syms:
            rows, start = [], int(SAMPLE_START.timestamp() * 1000)
            while True:
                kl = requests.get(
                    "https://fapi.binance.com/fapi/v1/klines",
                    params={"symbol": sym, "interval": "1h",
                            "startTime": start, "limit": 1500},
                    timeout=20).json()
                if not isinstance(kl, list) or not kl:
                    break
                rows += kl
                start = kl[-1][6] + 1
                if len(kl) < 1500:
                    break
                time.sleep(0.25)
            if rows:
                df = pd.DataFrame(rows).iloc[:, :6]
                df.columns = ["t", "o", "h", "l", "c", "v"]
                df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                df = df.set_index("t").astype(float).sort_index()
                out[sym] = df
        return out
    except Exception as e:
        print("Binance fallback failed:", e)
        return {}


# ------------------------------------------------------------------ cleaning
def clean_returns(px):
    """Log returns with artifact handling:
    - NaN out bars inside EXCLUDE_WINDOWS (oracle flash crash)
    - NaN out single-bar rescales (splits / oracle re-denomination):
      |log ret| > 0.7 counts as a rescale, real news does not move a
      trillion-dollar name 100 percent in one bar."""
    r = np.log(px["c"]).diff()
    for lo, hi in EXCLUDE_WINDOWS:
        r.loc[lo:hi] = np.nan
    n_scale = int((r.abs() > 0.7).sum())
    if n_scale:
        print(f"  flagged {n_scale} rescale/artifact bars as NaN")
    r[r.abs() > 0.7] = np.nan
    return r


def rolling_beta(r_asset, r_btc, window_bars):
    df = pd.concat([r_asset, r_btc], axis=1, keys=["a", "b"]).dropna()
    cov = df["a"].rolling(window_bars, min_periods=window_bars // 3).cov(df["b"])
    var = df["b"].rolling(window_bars, min_periods=window_bars // 3).var()
    beta = (cov / var).reindex(r_asset.index).ffill()
    return beta.fillna(beta.median() if beta.notna().any() else 1.0)


# ------------------------------------------------------------------ launches
def fetch_launches(net_gte="2025-11-01T00:00:00Z"):
    """SpaceX launches from Launch Library 2 (free tier, ~15 req/hr)."""
    rows, url = [], (f"{LL2}?limit=100&ordering=net"
                     f"&lsp__name=SpaceX&net__gte={net_gte}")
    while url:
        r = requests.get(url, timeout=30).json()
        for L in r.get("results", []):
            cfg = (L.get("rocket") or {}).get("configuration") or {}
            rows.append({
                "name": L.get("name"),
                "t0": pd.Timestamp(L.get("net")),
                "status": (L.get("status") or {}).get("abbrev"),
                "rocket": cfg.get("name", ""),
            })
        url = r.get("next")
        time.sleep(2)
    ev = pd.DataFrame(rows)
    ev["t0"] = pd.to_datetime(ev["t0"], utc=True)
    ev["is_starship"] = ev["rocket"].str.contains("Starship", case=False)
    ev["is_success"] = ev["status"].eq("Success")
    print(f"Launches fetched: {len(ev)} "
          f"({int(ev['is_starship'].sum())} Starship)")
    return ev.sort_values("t0").reset_index(drop=True)


# --------------------------------------------------------------- event study
def window_ret(r, ab, t0, hours, direction=1):
    """Cumulative (raw, abnormal) log return over a window vs T0."""
    if direction > 0:
        seg_r = r.loc[t0: t0 + timedelta(hours=hours)]
        seg_a = ab.loc[t0: t0 + timedelta(hours=hours)]
    else:
        seg_r = r.loc[t0 - timedelta(hours=hours): t0]
        seg_a = ab.loc[t0 - timedelta(hours=hours): t0]
    if seg_r.notna().sum() == 0:
        return np.nan, np.nan
    return seg_r.sum(skipna=True), seg_a.sum(skipna=True)


def event_study(px, events, r_btc, label, bars_per_day=24):
    r = clean_returns(px)
    beta = rolling_beta(r, r_btc, 30 * bars_per_day)
    ab = r - beta * r_btc.reindex(r.index)
    recs = []
    for _, e in events.iterrows():
        t0 = e["t0"]
        if t0 < r.index[0] or t0 > r.index[-1]:
            continue
        row = {"name": e["name"], "t0": t0, "rocket": e["rocket"],
               "status": e["status"], "is_starship": e["is_starship"],
               "market": label}
        row["pre24_raw"], row["pre24_abn"] = window_ret(r, ab, t0, 24, -1)
        for h in POST_WINDOWS:
            if t0 + timedelta(hours=h) > r.index[-1] + timedelta(hours=1):
                row[f"post{h}_raw"] = row[f"post{h}_abn"] = np.nan
            else:
                row[f"post{h}_raw"], row[f"post{h}_abn"] = window_ret(
                    r, ab, t0, h, +1)
        recs.append(row)
    return pd.DataFrame(recs)


def summarize(res):
    if res.empty:
        print("No in-window events.")
        return
    print("\n================ EVENT STUDY SUMMARY ================")
    for (mkt, star), g in res.groupby(["market", "is_starship"]):
        tag = "STARSHIP" if star else "Falcon/other (pooled)"
        print(f"\n[{mkt}] {tag}  n={len(g)}")
        for h in POST_WINDOWS:
            col = g[f"post{h}_abn"].dropna() * 100
            if col.empty:
                continue
            line = (f"  +{h:>3}h abn:  mean {col.mean():+6.2f}%  "
                    f"median {col.median():+6.2f}%  "
                    f"IQR [{col.quantile(.25):+.2f}, {col.quantile(.75):+.2f}]")
            if len(col) >= 20 and not star:
                t = col.mean() / (col.std(ddof=1) / np.sqrt(len(col)))
                line += (f"  t={t:+.2f} (overlapping windows, treat the"
                         f" t-stat as optimistic)")
            print(line)
        if star:
            print("  Starship rows (case studies, not statistics):")
            cols = ["name", "t0", "status"] + \
                   [f"post{h}_abn" for h in POST_WINDOWS]
            print(g[cols].to_string(index=False))
    print("\nCaveats: pre-IPO Starship n is ~1 (Flight 12, a partial"
          "\nfailure), so no success-premium estimate exists pre-IPO."
          "\nFalcon pooled events overlap heavily at ~3 launches/week;"
          "\nindependence is violated and effects near zero are expected.")


# -------------------------------------------------------- post-IPO stock leg
def stock_leg(events):
    """SPCX vs QQQ after 2026-06-12. Launches at ~22:30 UTC land after the
    NYSE close, so the tradeable stock reaction is the next-day open gap.
    The perp, if still listed, gives the true overnight reaction."""
    try:
        import yfinance as yf
    except ImportError:
        print("pip install yfinance for the stock leg")
        return pd.DataFrame()
    px = yf.download("SPCX QQQ", start="2026-06-12", interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    opens = yf.download("SPCX QQQ", start="2026-06-12", interval="1d",
                        auto_adjust=True, progress=False)["Open"]
    rows = []
    for _, e in events[events["t0"] >= IPO_DATE].iterrows():
        d = e["t0"].tz_convert("America/New_York").normalize().tz_localize(None)
        idx = px.index[px.index > d]
        prev = px.index[px.index <= d]
        if len(idx) == 0 or len(prev) == 0:
            continue
        nxt, prv = idx[0], prev[-1]
        gap = {t: np.log(opens.loc[nxt, t] / px.loc[prv, t])
               for t in ("SPCX", "QQQ")}
        day = {t: np.log(px.loc[nxt, t] / px.loc[prv, t])
               for t in ("SPCX", "QQQ")}
        rows.append({"name": e["name"], "t0": e["t0"],
                     "status": e["status"], "is_starship": e["is_starship"],
                     "next_open_gap_abn": gap["SPCX"] - gap["QQQ"],
                     "next_close_abn": day["SPCX"] - day["QQQ"]})
    out = pd.DataFrame(rows)
    if len(out):
        print("\n============ POST-IPO SPCX (QQQ-adjusted) ============")
        print(out.to_string(index=False))
    return out


# ----------------------------------------------------------------- main
def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    now_ms = int(time.time() * 1000)
    start_ms = int(SAMPLE_START.timestamp() * 1000)

    events = fetch_launches()
    events.to_csv(f"{OUT}/launches.csv", index=False)

    markets = discover_spacex_markets()
    btc_1h = get_candles("BTC", "1h", now_ms - 5000 * 3600 * 1000, now_ms)
    btc_1d = get_candles("BTC", "1d", start_ms, now_ms)
    r_btc_1h = np.log(btc_1h["c"]).diff() if len(btc_1h) else pd.Series(dtype=float)
    r_btc_1d = np.log(btc_1d["c"]).diff() if len(btc_1d) else pd.Series(dtype=float)

    all_res = []
    got_any = False
    for m in markets:
        label = f"{m['dex'] or 'core'}:{m['coin']}"
        print(f"\n--- {label} ---")
        px1h = get_candles(m["coin"], "1h",
                           now_ms - 5000 * 3600 * 1000, now_ms, m["dex"])
        px1d = get_candles(m["coin"], "1d", start_ms, now_ms, m["dex"])
        for tag, px, rb, bpd in (("1h", px1h, r_btc_1h, 24),
                                 ("1d", px1d, r_btc_1d, 1)):
            if len(px):
                got_any = True
                px.to_csv(f"{OUT}/{label.replace(':', '_')}_{tag}.csv")
                all_res.append(event_study(px, events, rb,
                                           f"{label} {tag}", bpd))

    if not got_any:
        print("\nHL returned nothing. The perps may have been settled and"
              "\npurged at IPO. Trying Binance pre-IPO perp fallback...")
        for sym, px in binance_fallback().items():
            px.to_csv(f"{OUT}/binance_{sym}_1h.csv")
            all_res.append(event_study(px, events, r_btc_1h,
                                       f"binance:{sym} 1h", 24))

    res = pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame()
    if len(res):
        res.to_csv(f"{OUT}/event_returns.csv", index=False)
        summarize(res)

    stock = stock_leg(events)
    if len(stock):
        stock.to_csv(f"{OUT}/spcx_stock_events.csv", index=False)

    print(f"\nOutputs in {OUT}/. Flight 13 window opens 2026-07-16 evening;"
          "\nrerun this after T0 (or ~24h later) and it lands in the tables"
          "\nautomatically via Launch Library.")


if __name__ == "__main__":
    main()
