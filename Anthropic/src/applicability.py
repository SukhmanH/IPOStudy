"""
applicability.py - does the IPO-fade study apply to Anthropic (ANT)?

Order of operations (fixed before computing anything ANT-specific):
  A. Reproduce the parent's pre-registered primary spec (T3+T4, short +20 close
     -> cover +180 close, long-QQQ hedge) from the parent panel AND from this
     rebuild; report the post-verdict matured events separately.
  B. Base rates for ANT's tier (T4) and the pre-registered pool (T3+T4):
     horizon stats from the day-0 close, the day-0 high, and the offer price.
  C. Does fade depth depend on size? (ANT is ~6x the largest IPO with a +180
     outcome.) Cross-sectional OLS with listing-quarter-clustered SEs + the
     >$50B mega-IPO roster, event by event.
  D. Conditioning: day-1 pop terciles and pre-listing VIX terciles.
  E. Live out-of-sample check: the 2025-26 T3/T4 cohort (incl. SPCX) vs the
     historical distribution at each horizon they have reached.
  F. SPCX case study: the closest structural analog (mega-cap, ~4% float,
     staggered lockups, Nasdaq-100 fast entry on day +15).

All cells are descriptive except A (the parent's one confirmatory test).
Every table carries n; n<25 is flagged.

Outputs: results/primary_spec_reproduction.csv, results/horizon_stats.csv,
         results/size_regression.csv, results/mega_ipo_roster.csv,
         results/conditioning.csv, results/live_check_2025_26.csv,
         results/spcx_case_study.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import common as c

HOLD_YEARS = (c.PRIMARY_EXIT - c.PRIMARY_ENTRY) / 252.0


def summarize(v: pd.Series, quarters: pd.Series | None = None, boot: bool = False) -> dict:
    v = pd.to_numeric(v, errors="coerce")
    mask = v.notna()
    v = v[mask]
    n = len(v)
    out = dict(n=n, flag_small_n=n < 25)
    if n == 0:
        return out
    out.update(median=v.median(), mean=v.mean(), p10=v.quantile(.10), p25=v.quantile(.25),
               p75=v.quantile(.75), p90=v.quantile(.90), hit_rate=(v > 0).mean())
    if n > 2:
        sd = v.std(ddof=1)
        out["naive_t"] = v.mean() / (sd / np.sqrt(n)) if sd > 0 else np.nan
    if boot and quarters is not None and n > 3:
        qv = quarters[mask].to_numpy()
        out["mean_ci_lo"], out["mean_ci_hi"] = c.block_bootstrap_ci(v.to_numpy(), qv, np.mean)
        out["median_ci_lo"], out["median_ci_hi"] = c.block_bootstrap_ci(v.to_numpy(), qv, np.median)
    return out


# --------------------------------------------------------------------------- A
def primary_reproduction(ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    par = pd.read_csv(c.PARENT_PANEL, keep_default_na=False)
    par["y"] = pd.to_numeric(par["hedged_e20_x180"], errors="coerce")
    par = par[par["tier"].isin(["T3", "T4"]) & par["y"].notna()]
    hist = ev[(ev["role"] == "historical") & ev["hedged_e20_x180"].notna()]
    matured = ev[(ev["role"] == "live") & ev["hedged_e20_x180"].notna()]
    sources = [("parent_panel (exact replay)", par, "split"),
               ("this_rebuild (Yahoo pull " + str(c.DATA_PULL_DATE.date()) + ")", hist, "split")]
    for label, df, scol in sources:
        for split in ["discovery", "validation"]:
            sub = df[df[scol] == split]
            for b in [0.0, 0.10]:
                y = sub["y" if "y" in sub else "hedged_e20_x180"] - b * HOLD_YEARS
                s = summarize(y, sub["quarter"], boot=True)
                rows.append(dict(source=label, split=split, borrow_annual=b, **s))
    for b in [0.0, 0.10]:
        y = matured["hedged_e20_x180"] - b * HOLD_YEARS
        rows.append(dict(source="post-verdict matured (listed Oct-25..Feb-26)", split="out-of-sample",
                         borrow_annual=b, **summarize(y)))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "primary_spec_reproduction.csv", index=False)
    return out


# --------------------------------------------------------------------------- B
def horizon_stats(ev: pd.DataFrame) -> pd.DataFrame:
    hist = ev[ev["role"] == "historical"]
    groups = {
        "T4 (ANT's tier)": hist[hist["tier"] == "T4"],
        "T3+T4 (pre-registered pool)": hist,
        "T4 non-ADR": hist[(hist["tier"] == "T4") & (~hist["adr"])],
        "T4 >= $50B": hist[(hist["tier"] == "T4") & (hist["mcap_b"] >= 50)],
        "T4 2019+ (validation era)": hist[(hist["tier"] == "T4") & (hist["split"] == "validation")],
    }
    metrics = [("x", "hedged fade from day-0 close (q - s)"),
               ("rel", "QQQ-neutral relative return from day-0 close"),
               ("x_hi", "hedged give-back from day-0 HIGH (legacy metric)"),
               ("po_rel", "price / offer, QQQ-neutral")]
    rows = []
    for gname, g in groups.items():
        for key, desc in metrics:
            for h in c.HORIZONS:
                s = summarize(g[f"{key}_{h}"])
                if key == "po_rel":   # 'hit' = broke issue (below offer)
                    v = g[f"{key}_{h}"].dropna()
                    s["hit_rate"] = (v < 1).mean() if len(v) else np.nan
                rows.append(dict(group=gname, metric=key, metric_desc=desc, horizon=h, **s))
        rows.append(dict(group=gname, metric="primary_e20_x180", metric_desc="parent primary spec (gross)",
                         horizon=180, **summarize(g["hedged_e20_x180"], g["quarter"], boot=True)))
        rows.append(dict(group=gname, metric="pop_close", metric_desc="day-1 pop (close / offer - 1)",
                         horizon=0, **summarize(g["pop_close"])))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "horizon_stats.csv", index=False)
    return out


# --------------------------------------------------------------------------- C
def size_regression(ev: pd.DataFrame) -> pd.DataFrame:
    hist = ev[ev["role"] == "historical"].copy()
    hist["log_mcap"] = np.log(hist["mcap_b"])
    rows = []
    specs = [("hedged_e20_x180", ["log_mcap"]),
             ("hedged_e20_x180", ["log_mcap", "pop_close", "vix_pre21", "rv_d1_20"]),
             ("x_180", ["log_mcap"]),
             ("x_180", ["log_mcap", "pop_close", "vix_pre21", "rv_d1_20"]),
             ("x_hi_180", ["log_mcap", "pop_close", "vix_pre21"])]
    for sample, df in [("T3+T4", hist), ("T4", hist[hist["tier"] == "T4"])]:
        for y, xs in specs:
            d = df[[y, "quarter"] + xs].dropna()
            if len(d) < 15:
                continue
            X = sm.add_constant(d[xs])
            fit = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["quarter"])[0]})
            for k in ["const"] + xs:
                rows.append(dict(sample=sample, y=y, spec="+".join(xs), term=k, coef=fit.params[k],
                                 se_clustered=fit.bse[k], t=fit.tvalues[k], p=fit.pvalues[k],
                                 n=len(d), n_clusters=d["quarter"].nunique(), r2=fit.rsquared))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "size_regression.csv", index=False)
    # mega roster
    cols = ["ticker", "name", "listing", "mcap_b", "offer", "pop_close", "pop_high", "x_20", "x_60", "x_120",
            "x_180", "x_hi_180", "hedged_e20_x180", "po_rel_180", "peak_day_0_20", "maxdd_rel_0_180"]
    mega = ev[(ev["mcap_b"] >= 50) | (ev["ticker"] == "SPCX")].sort_values("mcap_b", ascending=False)[cols]
    mega.to_csv(c.RESULTS / "mega_ipo_roster.csv", index=False)
    return out


# --------------------------------------------------------------------------- D
def conditioning(ev: pd.DataFrame) -> pd.DataFrame:
    hist = ev[ev["role"] == "historical"].copy()
    rows = []
    for var, label in [("pop_close", "day-1 pop tercile"), ("vix_pre21", "pre-listing VIX tercile")]:
        cuts = hist[var].quantile([1 / 3, 2 / 3]).to_numpy()
        hist[f"{var}_terc"] = pd.cut(hist[var], [-np.inf, cuts[0], cuts[1], np.inf],
                                     labels=["low", "mid", "high"])
        for terc, g in hist.groupby(f"{var}_terc", observed=True):
            for y in ["x_60", "x_180", "x_hi_180", "hedged_e20_x180"]:
                rows.append(dict(conditioner=label, tercile=str(terc),
                                 bounds=f"{g[var].min():.3f}..{g[var].max():.3f}", y=y,
                                 **summarize(g[y])))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "conditioning.csv", index=False)
    return out, {v: hist[v].quantile([1 / 3, 2 / 3]).tolist() for v in ["pop_close", "vix_pre21"]}


# --------------------------------------------------------------------------- E
def live_check(ev: pd.DataFrame) -> pd.DataFrame:
    hist = ev[ev["role"] == "historical"]
    live = ev[ev["role"] == "live"].sort_values("listing")
    rows = []
    for r in live.itertuples():
        n = r.n_bars - 1                       # latest day index observed
        rec = dict(ticker=r.ticker, name=r.name, tier=r.tier, listing=r.listing, mcap_b=r.mcap_b,
                   pop_close=r.pop_close, days_observed=n)
        for h in [20, 60, 90, 120, 180]:
            v = getattr(r, f"x_{h}")
            ref = hist[f"x_{h}"].dropna()
            rec[f"x_{h}"] = v
            rec[f"x_{h}_pctile_vs_hist"] = (stats.percentileofscore(ref, v) / 100) if pd.notna(v) else np.nan
        rec["primary_e20_x180"] = r.hedged_e20_x180
        rows.append(rec)
    out = pd.DataFrame(rows)
    # aggregate: realized vs historical, by horizon
    agg = []
    for h in [20, 60, 90, 120, 180]:
        v = out[f"x_{h}"].dropna()
        ref = hist[f"x_{h}"].dropna()
        if len(v) == 0:
            continue
        k = int((v > 0).sum())
        p_hist = (ref > 0).mean()
        agg.append(dict(ticker=f"ALL live (h={h})", days_observed=h, **{
            f"x_{h}": v.median(), f"x_{h}_pctile_vs_hist": stats.percentileofscore(ref, v.median()) / 100},
            name=f"n={len(v)}; faded {k}/{len(v)}; hist median {ref.median():+.3f}, hist hit {p_hist:.0%}; "
                 f"binomial p(fades LESS than history)={stats.binomtest(k, len(v), p_hist, alternative='less').pvalue:.3f}, "
                 f"p(fades MORE)={stats.binomtest(k, len(v), p_hist, alternative='greater').pvalue:.3f} "
                 f"(events overlap in one 2026 regime - not independent)"))
    out = pd.concat([out, pd.DataFrame(agg)], ignore_index=True)
    out.to_csv(c.RESULTS / "live_check_2025_26.csv", index=False)
    return out


# --------------------------------------------------------------------------- F
def spcx_case(ev: pd.DataFrame) -> pd.DataFrame:
    win = pd.read_csv(c.DATA / "cohort_prices_window.csv.gz", parse_dates=["Date"])
    bench = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    g = win[win["ticker"] == "SPCX"].set_index("Date").sort_index()
    q = bench["QQQ"].reindex(g.index).ffill()
    adj = g["AdjClose"]
    rel = (adj / adj.iloc[0]) / (q / q.iloc[0])
    pk = adj.iloc[:21].idxmax()
    tr = adj.loc[pk:].idxmin()
    ann, eff = pd.Timestamp("2026-06-26"), pd.Timestamp("2026-07-07")
    pre_eff = g.index[g.index < eff][-1]          # index funds trade at this close
    items = [
        ("offer", 135.0, ""),
        ("day0 open / high / close", f"{g['Open'].iloc[0]:.2f} / {g['High'].iloc[0]:.2f} / {g['Close'].iloc[0]:.2f}",
         g.index[0].date()),
        ("peak close (days 0-20)", round(float(adj.loc[pk]), 2), f"{pk.date()} (day {g.index.get_loc(pk)})"),
        ("trough close after peak", round(float(adj.loc[tr]), 2), f"{tr.date()} (day {g.index.get_loc(tr)})"),
        ("peak-to-trough", f"{adj.loc[tr] / adj.loc[pk] - 1:+.1%}", ""),
        ("latest close", round(float(adj.iloc[-1]), 2), f"{g.index[-1].date()} (day {len(g) - 1})"),
        ("day0 close -> latest, stock", f"{adj.iloc[-1] / adj.iloc[0] - 1:+.1%}", ""),
        ("day0 close -> latest, QQQ-neutral rel", f"{rel.iloc[-1] - 1:+.1%}", ""),
        ("min QQQ-neutral rel level", f"{rel.min() - 1:+.1%}", f"{rel.idxmin().date()}"),
        ("N100 announcement -> effective (close to close)",
         f"{adj.loc[pre_eff] / adj.loc[:ann].iloc[-1] - 1:+.1%} (QQQ {q.loc[pre_eff] / q.loc[:ann].iloc[-1] - 1:+.1%})",
         f"{ann.date()} -> {pre_eff.date()} close"),
        ("N100 effective -> +20 sessions after",
         f"{adj.iloc[g.index.get_loc(pre_eff) + 20] / adj.loc[pre_eff] - 1:+.1%}", ""),
        ("parent primary spec so far: short day-20 close -> latest (hedged vs QQQ)",
         f"{(q.iloc[-1] / q.iloc[20] - 1) - (adj.iloc[-1] / adj.iloc[20] - 1):+.1%} "
         f"(stock {adj.iloc[-1] / adj.iloc[20] - 1:+.1%}, QQQ {q.iloc[-1] / q.iloc[20] - 1:+.1%})",
         f"entry {g.index[20].date()} at {adj.iloc[20]:.2f}; day {len(g) - 1} of 180"),
        ("x_20 / x_60 / x_90 (hedged from day0 close)",
         " / ".join(f"{ev.set_index('ticker').loc['SPCX', f'x_{h}']:+.3f}" for h in [20, 60, 90]), ""),
    ]
    out = pd.DataFrame(items, columns=["item", "value", "date_or_note"])
    out.to_csv(c.RESULTS / "spcx_case_study.csv", index=False)
    return out


def main():
    ev = pd.read_csv(c.RESULTS / "cohort_event_table.csv", keep_default_na=False)
    for col in ev.columns:
        if col not in ("ticker", "name", "role", "tier", "listing", "quarter", "era", "split", "adr"):
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    ev["adr"] = ev["adr"].astype(str) == "True"

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("A. PRIMARY SPEC REPRODUCTION")
    a = primary_reproduction(ev)
    print(a[["source", "split", "borrow_annual", "n", "mean", "median", "hit_rate", "mean_ci_lo", "mean_ci_hi",
             "median_ci_lo", "median_ci_hi"]].round(4).to_string(index=False))
    print("\nB. HORIZON STATS (selected)")
    b = horizon_stats(ev)
    sel = b[(b.metric.isin(["x", "x_hi", "po_rel"])) & (b.horizon.isin([20, 60, 120, 180, 250]))
            & b.group.isin(["T4 (ANT's tier)", "T3+T4 (pre-registered pool)"])]
    print(sel[["group", "metric", "horizon", "n", "median", "mean", "p25", "p75", "hit_rate"]].round(3).to_string(index=False))
    print(b[b.metric.isin(["primary_e20_x180", "pop_close"])][["group", "metric", "n", "median", "mean", "hit_rate",
          "mean_ci_lo", "mean_ci_hi"]].round(3).to_string(index=False))
    print("\nC. SIZE REGRESSION (log mcap terms)")
    cr = size_regression(ev)
    print(cr[cr.term == "log_mcap"][["sample", "y", "spec", "coef", "se_clustered", "t", "p", "n", "n_clusters"]].round(4).to_string(index=False))
    print(pd.read_csv(c.RESULTS / "mega_ipo_roster.csv").round(3).to_string(index=False))
    print("\nD. CONDITIONING")
    d, cuts = conditioning(ev)
    print("tercile cut points:", {k: [round(x, 3) for x in v] for k, v in cuts.items()})
    print(d[d.y.isin(["x_180", "hedged_e20_x180"])][["conditioner", "tercile", "bounds", "y", "n", "median", "mean", "hit_rate"]].round(3).to_string(index=False))
    print("\nE. LIVE CHECK 2025-26")
    e = live_check(ev)
    print(e.round(3).to_string(index=False))
    print("\nF. SPCX CASE STUDY")
    print(spcx_case(ev).to_string(index=False))


if __name__ == "__main__":
    main()
