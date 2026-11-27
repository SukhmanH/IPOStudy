"""
intraday_analysis.py -- Tests I, II and III of PREREGISTRATION_INTRADAY.md.

  I    short the running gainer at checkpoint T on the gain day (ranked by its gain AT T),
       cover at that day's official close: no overnight position.
  II   short the main study's top-10 gainers on day 2 at T (or the official open), cover at
       day 2's official close.
  III  Test I's 15:30 selection held overnight to day 2's open / close (descriptive).

Out: results/intraday_*.csv, results/intraday_summary.json, figs/fig7_intraday_timing.png
Run: python short_gainers/src/intraday_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parents[1]
DATA, RES, FIGS = HERE / "data", HERE / "results", HERE / "figs"
CACHE = DATA / "raw_intraday"
BARS = ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
CHECKS = ["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
BAR_ENDING = {c: BARS[i] for i, c in enumerate(CHECKS)}   # price at 12:30 = close of the 11:30 bar
COSTS = [0.0, 0.0010, 0.0025, 0.0050, 0.0100, 0.0300]
BASE, SLOT = 0.0025, 0.10
N_BOOT, BLOCK, SEED = 10_000, 21, 42
WIN_START, WIN_END = pd.Timestamp("2023-12-29"), pd.Timestamp("2026-11-25")
HALF_END = pd.Timestamp("2025-05-30")
BONF = 1 - 0.05 / 6
PRIMARY = [("I", "all", "12:30"), ("I", "all", "15:30"), ("I", "NE", "12:30"), ("I", "NE", "15:30"),
           ("II", "all", "12:30"), ("II", "NE", "12:30")]


# ----------------------------------------------------------------------------- stats
def boot(x: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    T = len(x)
    nb = int(np.ceil(T / BLOCK))
    out = np.empty(N_BOOT)
    for i in range(0, N_BOOT, 500):
        m = min(500, N_BOOT - i)
        st = rng.integers(0, T - BLOCK + 1, size=(m, nb))
        idx = (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(m, -1)[:, :T]
        out[i:i + m] = x[idx].mean(axis=1)
    a = (1 - BONF) / 2 * 100
    lo, hi, blo, bhi = np.percentile(out, [2.5, 97.5, a, 100 - a])
    return {"ci_lo": lo, "ci_hi": hi, "ci_lo_bonf": blo, "ci_hi_bonf": bhi}


def nw_t(x: np.ndarray) -> float:
    if x.std() == 0:
        return np.nan
    r = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return float(r.tvalues[0])


def portfolio(df: pd.DataFrame, col: str, cost: float, days: pd.DatetimeIndex, date_col: str) -> pd.Series:
    s = df[df[col].notna()]
    return (SLOT * (s[col] - cost)).groupby(s[date_col]).sum().reindex(days, fill_value=0.0)


def cell_stats(df: pd.DataFrame, col: str, days: pd.DatetimeIndex, date_col: str, ci: bool = True) -> dict:
    v = df[col].dropna()
    out = {"n": len(v), "gross_mean": v.mean(), "median": v.median(), "hit": (v > 0).mean(),
           "p01": v.quantile(0.01), "p99": v.quantile(0.99), "worst": v.min(), "best": v.max()}
    for c in COSTS:
        r = portfolio(df, col, c, days, date_col)
        x = r.values
        tag = f"{c * 100:.2f}".rstrip("0").rstrip(".")
        out[f"mean_daily_{tag}"] = x.mean()
        out[f"sharpe_{tag}"] = x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std() > 0 else np.nan
        if c == BASE:
            eq = np.cumprod(1 + x)
            out.update({"nw_t_0.25": nw_t(x), "max_dd_0.25": (eq / np.maximum.accumulate(eq) - 1).min(),
                        "worst_day_0.25": x.min(),
                        "half1_mean_daily_0.25": r[r.index <= HALF_END].mean(),
                        "half2_mean_daily_0.25": r[r.index > HALF_END].mean()})
            if ci:
                out.update({k + "_0.25": v_ for k, v_ in boot(x).items()})
    return out


# ----------------------------------------------------------------------------- data
def load_daily() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(f) for f in sorted(CACHE.glob("d_*.parquet"))], ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return d.drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)


def first_dates() -> pd.Series:
    parts = [pd.read_parquet(f, columns=["ticker", "date"]) for f in sorted((DATA / "raw_yahoo").glob("batch_*.parquet"))]
    x = pd.concat(parts)
    return pd.to_datetime(x.groupby("ticker")["date"].min())


def hourly_for(pairs: set[tuple[str, pd.Timestamp]], tickers: set[str]) -> pd.DataFrame:
    """Hourly bars for the needed (ticker, date) pairs, wide by bar start time."""
    out = []
    files = sorted(CACHE.glob("h_*.parquet")) + sorted(CACHE.glob("r_*.parquet"))  # disjoint tickers
    want = pd.MultiIndex.from_tuples(sorted(pairs), names=["ticker", "date"])
    for f in files:
        h = pd.read_parquet(f)
        h = h[h["ticker"].isin(tickers)]
        if h.empty:
            continue
        h["date"] = h["ts"].dt.normalize()
        h["bar"] = h["ts"].dt.strftime("%H:%M")
        h = h[pd.MultiIndex.from_arrays([h["ticker"], h["date"]]).isin(want)]
        out.append(h)
    h = pd.concat(out, ignore_index=True).drop_duplicates(["ticker", "date", "bar"], keep="last")
    h = h[h["bar"].isin(BARS)]
    h["dvol"] = h["close"] * h["volume"]
    w = h.pivot_table(index=["ticker", "date"], columns="bar", values=["open", "close", "high", "low", "dvol"], aggfunc="last")
    return w


def main() -> None:
    d = load_daily()
    spy_d = d[d["ticker"] == "SPY"].set_index("date")
    cal = spy_d.index[(spy_d.index >= WIN_START - pd.Timedelta(days=10)) & (spy_d.index <= WIN_END + pd.Timedelta(days=3))]
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    days = cal[(cal >= WIN_START) & (cal <= WIN_END)]

    g = d.groupby("ticker", sort=False)
    d["prev_close"], d["prev_date"] = g["close"].shift(1), g["date"].shift(1)
    d["next_open"], d["next_close"], d["next_date"] = g["open"].shift(-1), g["close"].shift(-1), g["date"].shift(-1)
    ratio = d["splits"].where(d["splits"] > 0, 1.0).fillna(1.0)
    rev = np.log(ratio).groupby(d["ticker"]).transform(lambda s: s[::-1].cumsum()[::-1])
    d["sf"] = np.exp(rev - np.log(ratio))                       # product of split ratios after t
    cs = (ratio != 1.0).astype(int).groupby(d["ticker"]).cumsum()
    d["split_near"] = (cs.groupby(d["ticker"]).shift(-2).fillna(cs) - cs.groupby(d["ticker"]).shift(3).fillna(0)) > 0
    d["cal_i"] = d["date"].map(cal_pos)
    d["prev_ok"] = d["prev_date"].map(cal_pos) == d["cal_i"] - 1
    d["next_ok"] = d["next_date"].map(cal_pos) == d["cal_i"] + 1
    fd = first_dates()
    first = d["ticker"].map(fd).fillna(g["date"].transform("min"))
    d["age"] = d["cal_i"] - first.map(lambda x: cal_pos.get(x, -10_000) if x >= cal[0] else -10_000)
    d.loc[first < cal[0], "age"] = 10_000                       # listed before the window: old enough
    d["hi_gain"] = d["high"] / d["prev_close"] - 1

    # earnings tag (report dated t-1 or t), Nasdaq symbols
    uni = pd.read_csv(DATA / "universe.csv").set_index("yahoo")["symbol"]
    ed = pd.read_csv(DATA / "earnings_dates.csv.gz", usecols=["date", "symbol"])
    ekeys = set(zip(ed["symbol"], pd.to_datetime(ed["date"])))
    sym = d["ticker"].map(uni)
    d["earnings"] = [((s, t) in ekeys) or ((s, p) in ekeys) for s, t, p in zip(sym, d["date"], d["prev_date"])]

    # ------------------------------------------------------------------ Test I candidates + hourly bars
    win = (d["date"] >= WIN_START) & (d["date"] <= WIN_END)
    cand = d[win & (d["hi_gain"] >= 0.10) & (d["ticker"] != "SPY")].copy()
    ev = pd.read_parquet(DATA / "events.parquet")
    e2 = ev[(ev["excl_reason"] == "eligible") & (ev["universe"] == "U5") & (ev["rank"] <= 10) &
            (ev["date"] >= WIN_START) & (ev["date"] < WIN_END)][["ticker", "date", "earnings"]].copy()
    e2 = e2.merge(d[["ticker", "date", "next_date", "next_ok"]], on=["ticker", "date"], how="left")
    pairs = set(zip(cand["ticker"], cand["date"])) | set(zip(cand["ticker"], cand["next_date"].dropna())) | \
        set(zip(e2["ticker"], e2["next_date"])) | \
        {("SPY", x) for x in days} | {("SPY", x) for x in e2["next_date"].dropna()}
    tick = set(cand["ticker"]) | set(e2["ticker"]) | {"SPY"}
    W = hourly_for(pairs, tick)
    # Price at T = close of the hourly bar ending at T; no forward fill: an hour without prints
    # (halt, or a Yahoo gap) means no entry at T.
    closeW = W["close"].reindex(columns=BARS)
    highW = W["high"].reindex(columns=BARS)
    # Yahoo hourly prices are split-adjusted but hourly volume is NOT (checked on NVDA's and
    # SMCI's 10:1 splits), so traded dollars = adjusted price x split factor x raw volume.
    sf_at = d.set_index(["ticker", "date"])["sf"].reindex(W.index).values
    dvW = W["dvol"].reindex(columns=BARS).fillna(0).cumsum(axis=1).mul(sf_at, axis=0)
    spyW = closeW.xs("SPY", level="ticker")
    dd = d.set_index(["ticker", "date"])
    # DEVIATION (disclosed in INTRADAY.md): for 0.58% of ticker-days (556 tickers, e.g. KALA x30,
    # TGL x24) Yahoo's hourly prices sit on a different split basis than its daily prices.  A
    # ticker-day is used only if every hourly bar lies inside the official daily range (+-2%);
    # an hourly bar cannot legitimately trade outside it, so this cannot select on outcomes.
    hh = W["high"].max(axis=1)
    hl = W["low"].min(axis=1)
    cons = (hh <= dd["high"].reindex(W.index) * 1.02) & (hl >= dd["low"].reindex(W.index) * 0.98)
    # Exit inside the same series as the entry: last hourly print (15:30-16:00 bar); the official
    # close only when that bar is missing on a consistent day.
    exitH = closeW["15:30"].where(closeW["15:30"].notna(), dd["close"].reindex(W.index))
    openH = W["open"].reindex(columns=BARS)["09:30"]
    last_bar_ok = closeW["15:30"].notna()
    # Coverage audit: daily-data proxies for "could have qualified at some checkpoint"
    # (high >= +10%, traded close >= $5, daily $ volume >= $5M) that have no hourly bars that day.
    has_h = pd.MultiIndex.from_frame(cand[["ticker", "date"]]).isin(closeW.index)
    proxy = (cand["close"] * cand["sf"] >= 5) & (cand["close"] * cand["volume"] >= 5e6)
    audit = {"testI_candidate_ticker_days": int(len(cand)),
             "testI_candidate_days_without_hourly": int((~has_h).sum())}
    for lab, msk in [("half1", cand["date"] <= HALF_END), ("half2", cand["date"] > HALF_END)]:
        sel = proxy & msk
        audit[f"coverage_{lab}_proxy_eligible"] = int(sel.sum())
        audit[f"coverage_{lab}_missing_hourly_share"] = float((~has_h[sel.values]).mean()) if sel.any() else None

    def at(df_keys: pd.DataFrame, bar: str, frame: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(df_keys)
        return frame[bar].reindex(idx).values

    def max_after(df_keys: pd.DataFrame, start_bar: str) -> np.ndarray:
        cols = BARS[BARS.index(start_bar):]
        return highW[cols].max(axis=1).reindex(pd.MultiIndex.from_frame(df_keys)).values

    spy_close = spy_d["close"]
    # ------------------------------------------------------------------ Test I
    rowsI = []
    selI = {}
    for T in CHECKS:
        b = BAR_ENDING[T]
        c = cand.copy()
        keys = c[["ticker", "date"]]
        c["P"] = at(keys, b, closeW)
        c["dv_T"] = at(keys, b, dvW)
        c["gain_T"] = c["P"] / c["prev_close"] - 1
        c["raw_P"] = c["P"] * c["sf"]
        c["cons"] = cons.reindex(pd.MultiIndex.from_frame(keys)).fillna(False).astype(bool).values
        c["X"] = exitH.reindex(pd.MultiIndex.from_frame(keys)).values
        ok = ((c["gain_T"] >= 0.10) & (c["raw_P"] >= 5) & (c["dv_T"] >= 5e6) & (c["age"] >= 60) &
              c["prev_ok"] & ~c["split_near"] & c["P"].notna() & c["cons"] & c["X"].notna())
        c = c[ok].sort_values(["date", "gain_T", "dv_T"], ascending=[True, False, False])
        c = c[c.groupby("date").cumcount() < 10].copy()
        c["max_up"] = max_after(c[["ticker", "date"]], BARS[BARS.index(b) + 1]) / c["P"] - 1  # high after entry
        spy_P = spyW[b].reindex(c["date"]).values
        spy_X = spyW["15:30"].reindex(c["date"]).values
        c["s_h"] = (1 - c["X"] / c["P"]) + (spy_X / spy_P - 1)
        selI[T] = c
        for var, sub in [("all", c), ("NE", c[~c["earnings"]])]:
            st = cell_stats(sub, "s_h", days, "date")
            lb = last_bar_ok.reindex(pd.MultiIndex.from_frame(sub[["ticker", "date"]])).fillna(False).astype(bool).values
            rowsI.append({"test": "I", "variant": var, "entry": T, **st,
                          "pushed_up_5pct": (sub["max_up"] >= 0.05).mean(), "pushed_up_10pct": (sub["max_up"] >= 0.10).mean(),
                          "median_gain_at_entry": sub["gain_T"].median(),
                          "share_exit_from_official_close": 1 - lb.mean() if len(lb) else np.nan})
    I = pd.DataFrame(rowsI)
    I.to_csv(RES / "intraday_I_gain_day.csv", index=False)

    # ------------------------------------------------------------------ Test II
    e2 = e2[e2["next_ok"] == True].copy()  # noqa: E712
    k2 = e2[["ticker", "next_date"]].rename(columns={"next_date": "date"})
    dn = d.set_index(["ticker", "date"])
    idx2 = pd.MultiIndex.from_frame(k2)
    e2["O2"] = dn["open"].reindex(idx2).values
    e2["H2"] = dn["high"].reindex(idx2).values
    e2["C2"] = dn["close"].reindex(idx2).values
    spy_o2 = spy_d["open"].reindex(e2["next_date"]).values
    spy_c2 = spy_close.reindex(e2["next_date"]).values
    rowsII = []
    for T in ["open"] + CHECKS:
        x = e2.copy()
        if T == "open":
            x["P"] = x["O2"]
            spy_P = spy_o2
            x["max_up"] = x["H2"] / x["P"] - 1
        else:
            b = BAR_ENDING[T]
            x["P"] = at(k2, b, closeW)
            spy_P = spyW[b].reindex(x["next_date"]).values
            x["max_up"] = max_after(k2, BARS[BARS.index(b) + 1]) / x["P"] - 1
            i2 = pd.MultiIndex.from_frame(k2)
            x["C2"] = exitH.reindex(i2).values
            x.loc[~cons.reindex(i2).fillna(False).astype(bool).values, "P"] = np.nan
            spy_c2_h = spyW["15:30"].reindex(x["next_date"]).values
        x["s_h"] = (1 - x["C2"] / x["P"]) + ((spy_c2 if T == "open" else spy_c2_h) / spy_P - 1)
        x = x[x["P"].notna() & x["C2"].notna()]
        for var, sub in [("all", x), ("NE", x[x["earnings"] != 1.0])]:
            st = cell_stats(sub, "s_h", days, "date")
            rowsII.append({"test": "II", "variant": var, "entry": T, **st,
                           "pushed_up_5pct": (sub["max_up"] >= 0.05).mean(), "pushed_up_10pct": (sub["max_up"] >= 0.10).mean()})
    II = pd.DataFrame(rowsII)
    II.to_csv(RES / "intraday_II_day_two.csv", index=False)

    # ------------------------------------------------------------------ Test III (overnight, descriptive)
    c = selI["15:30"].copy()
    c = c[c["next_ok"]]
    inext = pd.MultiIndex.from_frame(c[["ticker", "next_date"]].rename(columns={"next_date": "date"}))
    ok3 = cons.reindex(inext).fillna(False).astype(bool).values
    c["O1"] = openH.reindex(inext).values
    c["C1"] = exitH.reindex(inext).values
    c = c[ok3 & c["O1"].notna() & c["C1"].notna()]
    so = W["open"]["09:30"].xs("SPY", level="ticker").reindex(c["next_date"]).values
    sc = spyW["15:30"].reindex(c["next_date"]).values
    sp = spyW["14:30"].reindex(c["date"]).values
    c["s_open"] = (1 - c["O1"] / c["P"]) + (so / sp - 1)
    c["s_close"] = (1 - c["C1"] / c["P"]) + (sc / sp - 1)
    rowsIII = []
    for var, sub in [("all", c), ("NE", c[~c["earnings"]])]:
        for col, lab in [("s_open", "cover next open"), ("s_close", "cover next close")]:
            st = cell_stats(sub, col, days, "date", ci=False)
            rowsIII.append({"test": "III", "variant": var, "exit": lab, **st,
                            **{f"net_borrow{int(b * 100)}": st["gross_mean"] - BASE - b / 252 for b in [0.10, 0.30, 1.00]}})
    III = pd.DataFrame(rowsIII)
    III.to_csv(RES / "intraday_III_overnight.csv", index=False)

    # ------------------------------------------------------------------ POST-HOC (suggested by Test II):
    # the day-2 fade path. Short at the first print of day 2 (open of the 09:30 bar, same hourly
    # series as the exits), cover at each checkpoint or at the last print. Descriptive only.
    f = e2.copy()
    fi = pd.MultiIndex.from_frame(f[["ticker", "next_date"]].rename(columns={"next_date": "date"}))
    f["O"] = openH.reindex(fi).values
    f = f[cons.reindex(fi).fillna(False).astype(bool).values & f["O"].notna()]
    fi = pd.MultiIndex.from_frame(f[["ticker", "next_date"]].rename(columns={"next_date": "date"}))
    spy_o = W["open"]["09:30"].xs("SPY", level="ticker").reindex(f["next_date"]).values
    rowsF = []
    for T in CHECKS + ["close"]:
        x = f.copy()
        if T == "close":
            x["X"] = exitH.reindex(fi).values
            spy_x = spyW["15:30"].reindex(x["next_date"]).values
        else:
            x["X"] = closeW[BAR_ENDING[T]].reindex(fi).values
            spy_x = spyW[BAR_ENDING[T]].reindex(x["next_date"]).values
        x["s_h"] = (1 - x["X"] / x["O"]) + (spy_x / spy_o - 1)
        x = x[x["X"].notna()]
        for var, sub in [("all", x), ("NE", x[x["earnings"] != 1.0])]:
            st = cell_stats(sub, "s_h", days, "date", ci=(T in ("10:30", "close")))
            rowsF.append({"test": "POSTHOC-fade-path", "variant": var, "cover_at": T, **st})
    FP = pd.DataFrame(rowsF)
    FP.to_csv(RES / "intraday_posthoc_day2_fade_path.csv", index=False)

    # ------------------------------------------------------------------ verdicts on the six primary cells
    allc = pd.concat([I, II])
    rows = []
    for test, var, T in PRIMARY:
        r = allc[(allc["test"] == test) & (allc["variant"] == var) & (allc["entry"] == T)].iloc[0]
        works = bool(r["mean_daily_0.25"] > 0 and r["ci_lo_bonf_0.25"] > 0 and
                     r["half1_mean_daily_0.25"] > 0 and r["half2_mean_daily_0.25"] > 0)
        rows.append({"cell": f"{test}{'-NE' if var == 'NE' else ''}-{T.replace(':', '')}", "n": int(r["n"]),
                     "gross_mean": r["gross_mean"], "hit": r["hit"], "mean_daily_0.25": r["mean_daily_0.25"],
                     "ci_lo_bonf": r["ci_lo_bonf_0.25"], "ci_hi_bonf": r["ci_hi_bonf_0.25"],
                     "half1": r["half1_mean_daily_0.25"], "half2": r["half2_mean_daily_0.25"],
                     "sharpe_0.25": r["sharpe_0.25"], "mean_daily_3_fx": r["mean_daily_3"],
                     "verdict": "WORKS" if works else "DOES NOT WORK"})
    V = pd.DataFrame(rows)
    V.to_csv(RES / "intraday_primary_verdicts.csv", index=False)
    ic = pd.MultiIndex.from_frame(cand[["ticker", "date"]])
    audit["testI_candidate_days_hourly_inconsistent_excluded"] = int((~cons.reindex(ic).fillna(True).astype(bool)).sum())
    audit.update({"days": int(len(days)), "window": [str(days[0].date()), str(days[-1].date())],
                  "testII_events": int(len(e2))})
    (RES / "intraday_summary.json").write_text(json.dumps({"audit": audit, "verdicts": rows}, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(json.dumps(audit, indent=1))
    print(V.round(4).to_string(index=False))
    show = ["test", "variant", "entry", "n", "gross_mean", "median", "hit", "mean_daily_0.25", "sharpe_0.25",
            "pushed_up_10pct", "worst"]
    print(pd.concat([I, II])[show].round(4).to_string(index=False))
    print(III[["variant", "exit", "n", "gross_mean", "median", "hit", "net_borrow10", "net_borrow30", "net_borrow100"]].round(4).to_string(index=False))


def make_fig() -> None:
    """figs/fig7_intraday_timing.png -- three small panels, dataviz reference palette."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED, GRID, BASE_ = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    C1, C2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": BASE_, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "legend.frameon": False,
                         "axes.titlelocation": "left", "axes.titlesize": 10.5, "font.size": 9.5})
    I = pd.read_csv(RES / "intraday_I_gain_day.csv")
    II = pd.read_csv(RES / "intraday_II_day_two.csv")
    FP = pd.read_csv(RES / "intraday_posthoc_day2_fade_path.csv").rename(columns={"cover_at": "entry"})
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.2f}%")
    fmt0 = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    panels = [(axes[0, 0], I, "gross_mean", "A. Gain day: short while it runs (entry time), cover at the close", fmt),
              (axes[0, 1], II, "gross_mean", "B. Day 2: short at (entry time), cover at the close", fmt),
              (axes[1, 0], FP, "gross_mean", "C. Day 2: short at the open, cover at (time)  [post-hoc]", fmt),
              (axes[1, 1], I, "pushed_up_10pct", "D. Gain day: stock rose 10%+ more after you shorted", fmt0)]
    for ax, df, col, title, f in panels:
        for var, color, lab in [("all", C1, "All top-10 gainers"), ("NE", C2, "Skip earnings-day gainers")]:
            sub = df[df["variant"] == var]
            ax.plot(range(len(sub)), sub[col].values, color=color, lw=2, marker="o", ms=5, mec=SURF, mew=1.5, label=lab)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([{"open": "9:30 open", "close": "16:00 close"}.get(e, e) for e in sub["entry"]], fontsize=8.5)
        if col == "gross_mean":
            ax.axhline(0, color=BASE_, lw=1)
            ax.axhline(0.0025, color=MUTED, lw=1)
            ax.text(len(sub) - 1, 0.0025, "0.25% cost", color=MUTED, fontsize=8, va="bottom", ha="right")
            ax.set_ylabel("Mean hedged return per short trade")
        else:
            ax.set_ylim(0, None)
            ax.set_ylabel("Share of trades")
        ax.yaxis.set_major_formatter(f)
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_intraday_timing.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
    make_fig()
