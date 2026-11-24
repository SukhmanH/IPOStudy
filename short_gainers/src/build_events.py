"""
build_events.py -- event panel for the short-the-top-gainers study.

Implements PREREGISTRATION.md sections 3-4 and the feature table in 10.

Steps
  1. Load cached Yahoo batches; keep rows with positive O/H/L/C and V >= 0.
  2. Reconstruct the TRADED price level from the split history
        raw_t = adj_t * prod(split ratios dated strictly after t)
     (Yahoo back-adjusts all rows before a split; the split-day row is actual).
  3. Per-ticker features known at the close of t (and the t+1 open gap).
  4. Candidate pool: 2010-01-01 <= t <= last date, r_t >= +10%, traded close
     >= $1, day-t dollar volume >= $2M.  Exclusions are reason-coded.
  5. Universes: U5 (>= $5, DV >= $10M) primary; U1 ($1-$5, DV >= $2M) and
     U5lo (>= $5, $2M <= DV < $10M) exploratory.  Rank by r_t within each
     universe/day (ties: dollar volume).
  6. Forward returns (stock and SPY/IWM over identical windows).  Multi-day
     windows use dividend-adjusted closes so the short pays dividends.

Out (committed):  data/events.parquet, data/event_exclusions.csv,
                  data/coverage_by_year.csv, data/market_daily.csv
Run:  python short_gainers/src/build_events.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CACHE = DATA / "raw_yahoo"
START = pd.Timestamp("2010-01-01")
FWD = [2, 5, 10, 20]


def load_prices() -> pd.DataFrame:
    parts = []
    for f in sorted(CACHE.glob("batch_*.parquet")):
        d = pd.read_parquet(f)
        if len(d):
            parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    px = px.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return px


def market_frame(px: pd.DataFrame) -> pd.DataFrame:
    def one(t):
        s = px[px.ticker == t].set_index("date")
        return s
    spy, iwm, vix = one("SPY"), one("IWM"), one("^VIX")
    m = pd.DataFrame(index=spy.index)
    m["spy_o"], m["spy_c"], m["spy_ac"] = spy["open"], spy["close"], spy["adj_close"]
    m["iwm_o"], m["iwm_c"], m["iwm_ac"] = iwm["open"], iwm["close"], iwm["adj_close"]
    m["vix"] = vix["close"].reindex(m.index).ffill()
    m["spy_sma200"] = m["spy_c"].rolling(200, min_periods=200).mean()
    m["spy_ret"] = m["spy_c"].pct_change()
    m["spy_above200"] = m["spy_c"] > m["spy_sma200"]
    m.index.name = "date"
    return m


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn > 0, 100.0).where(up.notna())


def build() -> None:
    px = load_prices()
    m = market_frame(px)
    cal = m.index
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)

    uni = pd.read_csv(DATA / "universe.csv")
    tags = uni.set_index("yahoo")[["symbol", "is_adr", "is_spac", "is_healthcare"]]

    px = px[~px.ticker.isin(["SPY", "QQQ", "IWM", "^VIX"])]
    ok = (px[["open", "high", "low", "close"]] > 0).all(axis=1) & (px["volume"] >= 0)
    n_bad_rows = int((~ok).sum())
    px = px[ok].drop(columns=["dividends"], errors="ignore").reset_index(drop=True)
    px["ticker"] = px["ticker"].astype("category")
    N = len(px)
    tc = px["ticker"].cat.codes.values
    starts = np.r_[0, np.flatnonzero(np.diff(tc)) + 1]
    ends = np.r_[starts[1:] - 1, N - 1]
    sizes = ends - starts + 1
    g_start, g_end = np.repeat(starts, sizes), np.repeat(ends, sizes)
    row_i = np.arange(N) - g_start

    def shift(arr: np.ndarray, k: int, idx: np.ndarray | None = None) -> np.ndarray:
        """Value k rows ahead (k>0) or behind (k<0) within the same ticker; NaN outside."""
        idx = np.arange(N) if idx is None else idx
        j = idx + k
        ok_ = (j <= g_end[idx]) & (j >= g_start[idx])
        if np.issubdtype(arr.dtype, np.datetime64):
            out = np.full(len(idx), np.datetime64("NaT", "ns"), dtype=arr.dtype)
        else:
            out = np.full(len(idx), np.nan)
        out[ok_] = arr[j[ok_]]
        return out

    o, h, lo_, c = (px[k].values.astype(float) for k in ["open", "high", "low", "close"])
    ac, v = px["adj_close"].values.astype(float), px["volume"].values.astype(float)
    dates = px["date"].values
    splits = np.nan_to_num(px["splits"].values.astype(float))
    tick = px["ticker"]

    # --- split reconstruction: raw_t = adj_t * prod(ratios strictly after t) --------------
    ratio = np.where(splits > 0, splits, 1.0)
    log_r = pd.Series(np.log(ratio))
    rev_cum = log_r[::-1].groupby(tc[::-1]).cumsum()[::-1].values      # sum over rows >= t
    split_factor = np.exp(rev_cum - np.log(ratio))
    raw_close = c * split_factor
    cs = pd.Series((splits > 0).astype(float)).groupby(tc).cumsum().values
    div_f = np.where(ac > 0, ac / c, np.nan)

    # --- calendar alignment ----------------------------------------------------------------
    cal_i = pd.Series(dates).map(cal_pos).values.astype(float)
    prev_cal_i, next_cal_i = shift(cal_i, -1), shift(cal_i, 1)

    # --- day-t features -------------------------------------------------------------------
    prev_close = shift(c, -1)
    ret = c / prev_close - 1
    dv = c * v
    rng = h - lo_
    clv = np.where(rng > 0, (c - lo_) / np.where(rng > 0, rng, 1), 0.5)
    gap_t = o / prev_close - 1
    vs = pd.Series(v)
    vol_prev20 = vs.groupby(tc).transform(lambda s: s.shift(1).rolling(20, min_periods=20).mean()).values
    rvol = np.where(vol_prev20 > 0, v / np.where(vol_prev20 > 0, vol_prev20, 1), np.nan)
    ret20_prior = prev_close / shift(c, -21) - 1
    cser = pd.Series(c)
    prior_max = cser.groupby(tc).transform(lambda s: s.shift(1).rolling(252, min_periods=60).max()).values
    new_high = np.where(np.isnan(prior_max), np.nan, (c >= prior_max).astype(float))
    rsi14 = cser.groupby(tc).transform(rsi_wilder).values
    up = pd.Series((ret > 0).astype(int))
    blk = (1 - up).groupby(tc).cumsum()
    streak = up.groupby([tc, blk.values]).cumsum().values
    del vs, cser, up, blk, vol_prev20, prior_max

    # --- candidate pool (signal-time screens only) -----------------------------------------
    # pool also keeps rows that only a NAIVE (split-adjusted) $5 screen would admit, so the
    # look-ahead bias of that screen can be measured and the shibui cross-check matched.
    in_pool = ((dates >= np.datetime64(START)) & (ret >= 0.10) & (dv >= 2e6) &
               ((raw_close >= 1.0) | (c >= 5.0)))
    idx = np.flatnonzero(in_pool)
    n_pool = len(idx)
    cand = pd.DataFrame({
        "ticker": tick.values[idx].astype(str), "date": dates[idx], "close": c[idx],
        "raw_close": raw_close[idx], "ret": ret[idx], "dv": dv[idx], "clv": clv[idx],
        "gap_t": gap_t[idx], "rvol": rvol[idx], "ret20_prior": ret20_prior[idx],
        "new_high": new_high[idx], "rsi14": rsi14[idx], "streak": streak[idx],
        "row_i": row_i[idx], "cal_i": cal_i[idx], "prev_cal_i": prev_cal_i[idx],
        "next_cal_i": next_cal_i[idx], "div_f": div_f[idx]})
    cs_i = cs[idx]
    cs_p2 = np.where(idx + 2 <= g_end[idx], cs[np.minimum(idx + 2, N - 1)], cs[g_end[idx]])
    cs_m3 = np.where(idx - 3 >= g_start[idx], cs[np.maximum(idx - 3, 0)], 0.0)
    cand["split_near"] = (cs_p2 - cs_m3) > 0
    for k in [1] + FWD:
        cand[f"split_fwd{k}"] = (shift(cs, k, idx) - cs_i) > 0
    cand["open_n1"], cand["high_n1"] = shift(o, 1, idx), shift(h, 1, idx)
    cand["low_n1"], cand["close_n1"] = shift(lo_, 1, idx), shift(c, 1, idx)
    cand["volume_n1"], cand["div_f_n1"] = shift(v, 1, idx), shift(div_f, 1, idx)
    cand["date_n1"] = shift(dates, 1, idx)
    for k in FWD:
        cand[f"close_n{k}"] = shift(c, k, idx)
        cand[f"divf_n{k}"] = shift(div_f, k, idx)
        cand[f"date_n{k}"] = shift(dates, k, idx)
    cov = (pd.DataFrame({"year": pd.DatetimeIndex(dates).year, "t": tc})
           .query("year >= 2010").groupby("year")["t"].nunique().rename("tickers_with_data"))
    del o, h, lo_, c, ac, v, splits, raw_close, cs, div_f, ret, dv, clv, gap_t, rvol
    del ret20_prior, new_high, rsi14, streak, px

    reason = pd.Series("eligible", index=cand.index)
    reason[cand["row_i"] < 60] = "short_history"
    reason[(reason == "eligible") & (cand["prev_cal_i"] != cand["cal_i"] - 1)] = "gap_day_return"
    reason[(reason == "eligible") & cand["split_near"]] = "split_within_2d"
    reason[(reason == "eligible") & (cand["ret"] >= 1.0) & ~(cand["rvol"] >= 1.5)] = "probable_data_error"
    cand["excl_reason"] = reason
    cand["universe"] = np.select(
        [(cand["raw_close"] >= 5) & (cand["dv"] >= 10e6),
         (cand["raw_close"] >= 5) & (cand["dv"] < 10e6),
         (cand["raw_close"] >= 1) & (cand["raw_close"] < 5)],
        ["U5", "U5lo", "U1"], default="none")

    elig = cand["excl_reason"] == "eligible"
    cand["rank"] = np.nan
    e = cand[elig & (cand["universe"] != "none")].sort_values(
        ["date", "universe", "ret", "dv"], ascending=[True, True, False, False])
    cand.loc[e.index, "rank"] = e.groupby(["date", "universe"]).cumcount() + 1
    cand["n_elig_day"] = cand.groupby(["date", "universe"])["rank"].transform("max")
    # naive screen on split-ADJUSTED close (what a vendor table without split history gives)
    cand["u5_naive"] = (cand["close"] >= 5) & (cand["dv"] >= 10e6)
    cand["rank_naive"] = np.nan
    e = cand[elig & cand["u5_naive"]].sort_values(["date", "ret", "dv"], ascending=[True, False, False])
    cand.loc[e.index, "rank_naive"] = e.groupby("date").cumcount() + 1

    # tradeability at t+1 (known only at t+1; events are NOT replaced)
    cand["tradeable_t1"] = ((cand["next_cal_i"] == cand["cal_i"] + 1) & (cand["open_n1"] > 0) &
                            (cand["volume_n1"] > 0) & cand["close_n1"].notna())

    # --- market fields --------------------------------------------------------
    mt = m.reindex(cand["date"])
    for c in ["vix", "spy_above200", "spy_ret"]:
        cand[c] = mt[c].values
    mn = m.reindex(cand["date_n1"])
    cand["spy_oc1"] = (mn["spy_c"] / mn["spy_o"] - 1).values
    cand["iwm_oc1"] = (mn["iwm_c"] / mn["iwm_o"] - 1).values
    cand["spy_on1"] = (mn["spy_o"].values / mt["spy_c"].values) - 1
    cand["spy_cc1"] = (mn["spy_ac"].values / mt["spy_ac"].values) - 1
    spy_o_tr = mn["spy_o"].values * (mn["spy_ac"].values / mn["spy_c"].values)
    for k in FWD:
        mk = m.reindex(cand[f"date_n{k}"])
        cand[f"spy_oC{k}"] = mk["spy_ac"].values / spy_o_tr - 1
        cand[f"spy_cC{k}"] = mk["spy_ac"].values / mt["spy_ac"].values - 1

    # --- stock forward returns ------------------------------------------------
    cand["oc1"] = cand["close_n1"] / cand["open_n1"] - 1
    cand["on1"] = cand["open_n1"] / cand["close"] - 1          # = entry-day gap (F14)
    cand["cc1"] = (cand["close_n1"] * cand["div_f_n1"]) / (cand["close"] * cand["div_f"]) - 1
    cand["hi1"] = cand["high_n1"] / cand["open_n1"] - 1
    cand["lo1"] = cand["low_n1"] / cand["open_n1"] - 1
    o_tr = cand["open_n1"] * cand["div_f_n1"]
    for k in FWD:
        tr_k = cand[f"close_n{k}"] * cand[f"divf_n{k}"]
        cand[f"oC{k}"] = (tr_k / o_tr - 1).where(~cand[f"split_fwd{k}"])
        cand[f"cC{k}"] = (tr_k / (cand["close"] * cand["div_f"]) - 1).where(~cand[f"split_fwd{k}"])
    for c in ["oc1", "on1", "cc1", "hi1", "lo1"]:
        cand[c] = cand[c].where(cand["tradeable_t1"] & ~cand["split_fwd1"])

    # --- earnings tag (report dated t-1 or t, Nasdaq symbol) -------------------
    ef = DATA / "earnings_dates.csv.gz"
    cand["symbol"] = cand["ticker"].map(tags["symbol"])
    if ef.exists():
        ed = pd.read_csv(ef, usecols=["date", "symbol"])
        ed["date"] = pd.to_datetime(ed["date"])
        keys = set(zip(ed["symbol"], ed["date"]))
        prev_date = pd.Series(cal, index=np.arange(len(cal)))
        pdates = cand["cal_i"].sub(1).map(prev_date)
        cand["earnings"] = [((s, d) in keys) or ((s, p) in keys)
                            for s, d, p in zip(cand["symbol"], cand["date"], pdates)]
        cand["earnings_cal_covered"] = cand["date"].isin(set(ed["date"]))
    else:
        cand["earnings"] = np.nan
        cand["earnings_cal_covered"] = False
    for c in ["is_adr", "is_spac", "is_healthcare"]:
        cand[c] = cand["ticker"].map(tags[c]).astype(bool)

    keep = ["ticker", "symbol", "date", "date_n1", "universe", "excl_reason", "rank", "n_elig_day",
            "u5_naive", "rank_naive",
            "tradeable_t1", "ret", "raw_close", "close", "dv", "clv", "gap_t", "rvol", "ret20_prior",
            "new_high", "rsi14", "streak", "vix", "spy_above200", "spy_ret", "earnings",
            "earnings_cal_covered", "is_adr", "is_spac", "is_healthcare",
            "on1", "oc1", "cc1", "hi1", "lo1", "spy_oc1", "iwm_oc1", "spy_on1", "spy_cc1"] + \
           [f"oC{k}" for k in FWD] + [f"cC{k}" for k in FWD] + \
           [f"spy_oC{k}" for k in FWD] + [f"spy_cC{k}" for k in FWD] + [f"date_n{k}" for k in FWD]
    ev = cand[keep].reset_index(drop=True)
    for col in ev.columns:  # float32 keeps 7 significant digits: ample for returns/prices
        if ev[col].dtype == "float64":
            ev[col] = ev[col].astype("float32")
    ev.to_parquet(DATA / "events.parquet", index=False, compression="zstd")

    # --- audit outputs ---------------------------------------------------------
    ex = (cand.assign(year=cand["date"].dt.year)
          .groupby(["universe", "excl_reason"]).size().rename("n").reset_index())
    ex.loc[len(ex)] = ["all", "rows_dropped_nonpositive_ohlc", n_bad_rows]
    ex.loc[len(ex)] = ["all", "candidate_pool_rows", n_pool]
    ex.to_csv(DATA / "event_exclusions.csv", index=False)
    cov.reset_index().to_csv(DATA / "coverage_by_year.csv", index=False)
    m.to_csv(DATA / "market_daily.csv")
    top = ev[(ev["universe"] == "U5") & (ev["rank"] <= 10)]
    print(f"price rows {N:,}  tickers {len(starts):,}  bad rows dropped {n_bad_rows}")
    print(f"candidate pool {n_pool:,}; eligible by universe:\n"
          f"{cand[elig].groupby('universe').size().to_string()}")
    print(f"U5 top-10 events {len(top):,} over {top['date'].nunique():,} days; "
          f"tradeable at t+1 {int(top['tradeable_t1'].sum()):,}")


if __name__ == "__main__":
    build()
