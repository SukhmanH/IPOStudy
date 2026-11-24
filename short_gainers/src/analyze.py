"""
analyze.py -- H1 confirmatory test, H2 rule search (discovery) + single
validation test, and the exploratory battery A-F of PREREGISTRATION.md.

Every number is computed from data/events.parquet (built by build_events.py).
Outputs one CSV per table in results/, plus results/summary.json and the
multiple-testing ledger results/ledger.csv.

Run:  python short_gainers/src/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parents[1]
DATA, RES = HERE / "data", HERE / "results"
RES.mkdir(exist_ok=True)

COSTS = [0.0, 0.0010, 0.0025, 0.0050, 0.0100]
BASE_COST, DECISION_COST = 0.0025, 0.0050
BORROW_BASE, BORROW_SENS = 0.10, [0.30, 1.00]
SLOT = 0.10
N_BOOT, BLOCK, SEED = 10_000, 21, 42
MIN_DISC_EVENTS, STEP2_MIN_GAIN = 300, 0.20
SPLITS = {"discovery": ("2010-01-04", "2018-12-31"), "validation": ("2019-01-02", "2099-12-31")}
LEDGER: list[dict] = []

FEATURES = {
    "F1_gain": ("ret", [-np.inf, 0.15, 0.25, 0.50, 1.00, np.inf]),
    "F2_clv": ("clv", [-np.inf, 0.50, 0.85, np.inf]),
    "F3_gap_t": ("gap_t", [-np.inf, 0.02, 0.10, np.inf]),
    "F4_rvol": ("rvol", [-np.inf, 3, 10, np.inf]),
    "F5_ret20_prior": ("ret20_prior", [-np.inf, -0.20, 0.20, np.inf]),
    "F6_new_high": ("new_high", "bool"),
    "F7_rsi14": ("rsi14", [-np.inf, 70, 85, np.inf]),
    "F8_streak": ("streak", [-np.inf, 1.5, 2.5, np.inf]),
    "F9_price": ("raw_close", [-np.inf, 10, 30, np.inf]),
    "F10_dollar_vol": ("dv", [-np.inf, 30e6, 100e6, np.inf]),
    "F11_spy_above200": ("spy_above200", "bool"),
    "F12_vix": ("vix", [-np.inf, 15, 25, np.inf]),
    "F13_spy_day": ("spy_ret", [-np.inf, 0.0, np.inf]),
    "F14_entry_gap": ("on1", [-np.inf, -0.05, 0.0, 0.05, np.inf]),
    "F15_earnings": ("earnings", "bool"),
    "F16_healthcare": ("is_healthcare", "bool"),
    "F17_spac": ("is_spac", "bool"),
}


def ledger(section: str, what: str, n: int) -> None:
    LEDGER.append({"section": section, "what": what, "cells": int(n)})


# ----------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    ev = pd.read_parquet(DATA / "events.parquet")
    mkt = pd.read_csv(DATA / "market_daily.csv", parse_dates=["date"]).set_index("date")
    cal = mkt.index[:-1]                       # signal days need a following market day
    ev["split"] = np.where(ev["date"] <= pd.Timestamp(SPLITS["discovery"][1]), "discovery", "validation")
    ev["s_raw"] = -ev["oc1"]
    ev["s_h"] = -ev["oc1"] + ev["spy_oc1"]
    ev["s_iwm"] = -ev["oc1"] + ev["iwm_oc1"]
    ev["raw_prev_close"] = ev["raw_close"] / (1 + ev["ret"])
    for c in ["new_high", "earnings", "spy_above200", "is_healthcare", "is_spac"]:
        ev[c] = ev[c].map({True: 1.0, False: 0.0, 1.0: 1.0, 0.0: 0.0})
    ev["year"] = ev["date"].dt.year
    return ev, cal


def days_of(cal: pd.DatetimeIndex, split: str) -> pd.DatetimeIndex:
    lo, hi = SPLITS[split]
    return cal[(cal >= pd.Timestamp(lo)) & (cal <= pd.Timestamp(hi))]


def topn(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """Re-rank eligible rows within each signal day by gain (ties: dollar volume)."""
    d = df.sort_values(["date", "ret", "dv"], ascending=[True, False, False])
    if n is None:
        return d
    return d[d.groupby("date").cumcount() < n]


# ----------------------------------------------------------------------------- stats
def portfolio(sub: pd.DataFrame, col: str, cost: float, days: pd.DatetimeIndex,
              slot: float | None = SLOT) -> pd.Series:
    s = sub[sub[col].notna()]
    if slot is None:  # equal weight across the day's events (100% gross when active)
        r = (s[col] - cost).groupby(s["date"]).mean()
    else:
        r = (slot * (s[col] - cost)).groupby(s["date"]).sum()
    return r.reindex(days, fill_value=0.0)


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    T = len(x)
    nb = int(np.ceil(T / BLOCK))
    out = np.empty(N_BOOT)
    for i in range(0, N_BOOT, 500):
        m = min(500, N_BOOT - i)
        st = rng.integers(0, T - BLOCK + 1, size=(m, nb))
        idx = (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(m, -1)[:, :T]
        out[i:i + m] = x[idx].mean(axis=1)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def nw_t(x: np.ndarray, lags: int = 5) -> float:
    x = np.asarray(x, float)
    if np.allclose(x.std(), 0):
        return np.nan
    res = sm.OLS(x, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def port_stats(r: pd.Series, ci: bool = True) -> dict:
    x = r.values
    sd = x.std(ddof=1)
    eq = np.cumprod(1 + x)
    dd = eq / np.maximum.accumulate(eq) - 1
    out = {"days": len(x), "active_days": int((x != 0).sum()), "mean_daily": x.mean(),
           "ann_return": x.mean() * 252, "ann_vol": sd * np.sqrt(252),
           "sharpe": x.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
           "nw_t": nw_t(x), "max_dd": dd.min(), "worst_day": x.min(), "best_day": x.max(),
           "cum_return": eq[-1] - 1}
    if ci:
        out["ci_lo"], out["ci_hi"] = boot_ci(x)
    return out


def event_stats(v: pd.Series) -> dict:
    v = v.dropna()
    if len(v) == 0:
        return {"n": 0}
    pos, neg = v[v > 0].sum(), -v[v < 0].sum()
    return {"n": len(v), "mean": v.mean(), "median": v.median(), "hit": (v > 0).mean(),
            "pf": pos / neg if neg > 0 else np.nan, "p01": v.quantile(0.01),
            "p99": v.quantile(0.99), "worst": v.min(), "best": v.max(), "sd": v.std()}


def cohort_t(df: pd.DataFrame, col: str, lags: int) -> tuple[float, int]:
    s = df[df[col].notna()].groupby("date")[col].mean()
    if len(s) < 30:
        return np.nan, len(s)
    return nw_t(s.values, max(1, lags)), len(s)


# ----------------------------------------------------------------------------- filters
def candidate_filters() -> list[dict]:
    out = []
    for f, (col, spec) in FEATURES.items():
        if spec == "bool":
            out += [{"feature": f, "col": col, "kind": "eq", "val": 1.0, "name": f"{f}=yes"},
                    {"feature": f, "col": col, "kind": "eq", "val": 0.0, "name": f"{f}=no"}]
            continue
        k = len(spec) - 1
        for i in range(k):
            for j in range(i, k):
                if i == 0 and j == k - 1:
                    continue
                lo, hi = spec[i], spec[j + 1]
                out.append({"feature": f, "col": col, "kind": "range", "val": (lo, hi),
                            "name": f"{f} in [{lo:g},{hi:g})"})
    return out


def fmask(df: pd.DataFrame, flt: dict) -> pd.Series:
    x = df[flt["col"]]
    if flt["kind"] == "eq":
        return x == flt["val"]
    lo, hi = flt["val"]
    return (x >= lo) & (x < hi)


def bucket_labels(df: pd.DataFrame, f: str) -> pd.Series:
    col, spec = FEATURES[f]
    if spec == "bool":
        return df[col].map({1.0: "yes", 0.0: "no"})
    cats = pd.cut(df[col], bins=spec, right=False)
    return cats.astype(str).where(df[col].notna())


# ----------------------------------------------------------------------------- main
def main() -> None:
    ev, cal = load()
    D = {s: days_of(cal, s) for s in SPLITS}
    elig = ev[(ev["excl_reason"] == "eligible")]
    u5 = elig[elig["universe"] == "U5"]
    prim = u5[u5["rank"] <= 10]
    prim_t = prim[prim["s_h"].notna()]
    summary: dict = {"n_events_top10": int(len(prim)), "n_tradeable": int(len(prim_t)),
                     "n_untradeable_t1": int(len(prim) - len(prim_t))}

    # ---------------------------------------------------------------- H1
    rows = []
    for sp in SPLITS:
        sub = prim_t[prim_t["split"] == sp]
        es = event_stats(sub["s_h"])
        for c in COSTS:
            r = portfolio(sub, "s_h", c, D[sp])
            rows.append({"split": sp, "cost": c, **port_stats(r), "n_events": es["n"],
                         "events_per_active_day": es["n"] / max(1, (r != 0).sum()),
                         "gross_mean_event": es["mean"], "median_event": es["median"],
                         "hit": es["hit"], "pf": es["pf"], "p01": es["p01"], "p99": es["p99"],
                         "worst_event": es["worst"], "best_event": es["best"]})
    h1 = pd.DataFrame(rows)
    h1.to_csv(RES / "h1_primary.csv", index=False)
    ledger("H1", "primary spec: 2 splits x 5 costs (verdict uses 1 cell)", 10)
    v = h1[(h1.split == "validation") & np.isclose(h1.cost, DECISION_COST)].iloc[0]
    h1_pass = bool(v["mean_daily"] > 0 and v["ci_lo"] > 0)
    summary["H1"] = {"validation_mean_daily_at_0.50%": v["mean_daily"],
                     "ci": [v["ci_lo"], v["ci_hi"]], "sharpe": v["sharpe"],
                     "verdict": "PASS" if h1_pass else "FAIL"}
    print("H1 verdict:", summary["H1"]["verdict"], {k: round(x, 5) if isinstance(x, float) else x
                                                    for k, x in summary["H1"].items()})

    # ---------------------------------------------------------------- H2 rule search
    disc = prim_t[prim_t["split"] == "discovery"]
    val = prim_t[prim_t["split"] == "validation"]
    filters = candidate_filters()
    assert len(filters) == 80, len(filters)

    def disc_eval(mask: pd.Series) -> dict:
        n = int(mask.sum())
        if n < MIN_DISC_EVENTS:
            return {"n_disc": n, "eligible": False}
        r = portfolio(disc[mask], "s_h", BASE_COST, D["discovery"])
        st = port_stats(r, ci=False)
        return {"n_disc": n, "eligible": True, "sharpe_disc": st["sharpe"],
                "mean_daily_disc": st["mean_daily"], "gross_mean_event_disc": disc.loc[mask, "s_h"].mean()}

    base_disc = port_stats(portfolio(disc, "s_h", BASE_COST, D["discovery"]), ci=False)
    step1 = pd.DataFrame([{**{"step": 1, "filter": f["name"], "feature": f["feature"]},
                           **disc_eval(fmask(disc, f))} for f in filters])
    ledger("H2", "step-1 single filters evaluated on discovery", len(step1))
    e1 = step1[step1["eligible"]].sort_values("sharpe_disc", ascending=False)
    best1 = e1.iloc[0]
    f1 = next(f for f in filters if f["name"] == best1["filter"])
    m1 = fmask(disc, f1)
    step2 = pd.DataFrame([{**{"step": 2, "filter": f["name"], "feature": f["feature"]},
                           **disc_eval(m1 & fmask(disc, f))}
                          for f in filters if f["feature"] != f1["feature"]])
    ledger("H2", "step-2 added filters evaluated on discovery", len(step2))
    e2 = step2[step2["eligible"]].sort_values("sharpe_disc", ascending=False)
    rule = [f1]
    if len(e2) and e2.iloc[0]["sharpe_disc"] >= best1["sharpe_disc"] + STEP2_MIN_GAIN:
        rule.append(next(f for f in filters if f["name"] == e2.iloc[0]["filter"]))
    pd.concat([step1, step2]).to_csv(RES / "h2_rule_search_discovery.csv", index=False)
    rule_name = " AND ".join(f["name"] for f in rule)

    def rule_mask(df):
        m = pd.Series(True, index=df.index)
        for f in rule:
            m &= fmask(df, f)
        return m

    rows = []
    for sp, sub in [("discovery", disc), ("validation", val)]:
        s2 = sub[rule_mask(sub)]
        es = event_stats(s2["s_h"])
        for c in COSTS:
            r = portfolio(s2, "s_h", c, D[sp])
            rows.append({"rule": rule_name, "split": sp, "cost": c, **port_stats(r),
                         "n_events": es["n"], "gross_mean_event": es["mean"], "median_event": es["median"],
                         "hit": es["hit"], "pf": es["pf"], "p01": es["p01"], "worst_event": es["worst"]})
    h2 = pd.DataFrame(rows)
    h2.to_csv(RES / "h2_selected_rule.csv", index=False)
    ledger("H2", "selected rule validated once (verdict cell)", 1)
    v2 = h2[(h2.split == "validation") & np.isclose(h2.cost, DECISION_COST)].iloc[0]
    h2_pass = bool(v2["mean_daily"] > 0 and v2["ci_lo"] > 0)
    summary["H2"] = {"rule": rule_name, "step1_best": best1["filter"],
                     "step1_sharpe_disc": best1["sharpe_disc"],
                     "unconditional_sharpe_disc_base": base_disc["sharpe"],
                     "step2_best": (e2.iloc[0]["filter"] if len(e2) else None),
                     "step2_sharpe_disc": (e2.iloc[0]["sharpe_disc"] if len(e2) else None),
                     "validation_mean_daily_at_0.50%": v2["mean_daily"], "ci": [v2["ci_lo"], v2["ci_hi"]],
                     "validation_sharpe_at_0.50%": v2["sharpe"], "verdict": "PASS" if h2_pass else "FAIL"}
    print("H2 rule:", rule_name, "| verdict:", summary["H2"]["verdict"])

    # H2 diagnostic (exploratory, NOT a verdict): every step-1 filter's out-of-sample result.
    # Does discovery Sharpe predict validation Sharpe across the whole menu?
    rows = []
    for f in filters:
        md, mv = fmask(disc, f), fmask(val, f)
        if md.sum() < MIN_DISC_EVENTS:
            continue
        sd = port_stats(portfolio(disc[md], "s_h", BASE_COST, D["discovery"]), ci=False)
        sv = port_stats(portfolio(val[mv], "s_h", BASE_COST, D["validation"]), ci=False)
        rows.append({"filter": f["name"], "n_disc": int(md.sum()), "n_val": int(mv.sum()),
                     "gross_mean_disc": disc.loc[md, "s_h"].mean(), "gross_mean_val": val.loc[mv, "s_h"].mean(),
                     "sharpe_disc": sd["sharpe"], "sharpe_val": sv["sharpe"]})
    diag = pd.DataFrame(rows).sort_values("sharpe_disc", ascending=False)
    diag.to_csv(RES / "h2_diagnostic_all_filters_validation.csv", index=False)
    ledger("H2-diag", "all eligible step-1 filters re-scored on validation (exploratory)", len(diag))
    # same diagnostic for step 2 (second condition given the step-1 filter)
    rows2 = []
    mv1 = fmask(val, f1)
    for f in filters:
        if f["feature"] == f1["feature"]:
            continue
        md = m1 & fmask(disc, f)
        if md.sum() < MIN_DISC_EVENTS:
            continue
        mv = mv1 & fmask(val, f)
        sd = port_stats(portfolio(disc[md], "s_h", BASE_COST, D["discovery"]), ci=False)
        sv = port_stats(portfolio(val[mv], "s_h", BASE_COST, D["validation"]), ci=False)
        rows2.append({"filter": f1["name"] + " AND " + f["name"], "n_disc": int(md.sum()), "n_val": int(mv.sum()),
                      "gross_mean_disc": disc.loc[md, "s_h"].mean(), "gross_mean_val": val.loc[mv, "s_h"].mean(),
                      "sharpe_disc": sd["sharpe"], "sharpe_val": sv["sharpe"]})
    diag2 = pd.DataFrame(rows2).sort_values("sharpe_disc", ascending=False)
    diag2.to_csv(RES / "h2_diagnostic_step2_validation.csv", index=False)
    ledger("H2-diag", "eligible step-2 combinations re-scored on validation (exploratory)", len(diag2))
    summary["H2_diagnostic"] = {
        "step2_spearman_sharpe_disc_vs_val": float(diag2["sharpe_disc"].corr(diag2["sharpe_val"], method="spearman")),
        "step1_rule_val_sharpe_base": float(port_stats(portfolio(val[mv1], "s_h", BASE_COST, D["validation"]), ci=False)["sharpe"]),
        "filters": int(len(diag)),
        "spearman_sharpe_disc_vs_val": float(diag["sharpe_disc"].corr(diag["sharpe_val"], method="spearman")),
        "spearman_mean_disc_vs_val": float(diag["gross_mean_disc"].corr(diag["gross_mean_val"], method="spearman")),
        "top5_by_disc_sharpe_median_val_sharpe": float(diag.head(5)["sharpe_val"].median()),
        "all_filters_median_val_sharpe": float(diag["sharpe_val"].median())}

    # ---------------------------------------------------------------- A. entry x exit grid
    rows = []
    legs = [("close_t", "open_t1", "on1", "spy_on1", 1), ("close_t", "close_t1", "cc1", "spy_cc1", 1)]
    legs += [("close_t", f"close_t{k}", f"cC{k}", f"spy_cC{k}", k) for k in [2, 5, 10, 20]]
    legs += [("open_t1", "close_t1", "oc1", "spy_oc1", 0)]
    legs += [("open_t1", f"close_t{k}", f"oC{k}", f"spy_oC{k}", k - 1) for k in [2, 5, 10, 20]]
    for entry, exit_, col, hcol, nights in legs:
        for sp in SPLITS:
            sub = prim[(prim["split"] == sp)].copy()
            sub["x"] = -sub[col] + sub[hcol]
            sub = sub[sub["x"].notna()]
            es = event_stats(sub["x"])
            t, nd = cohort_t(sub, "x", lags=max(1, nights))
            row = {"entry": entry, "exit": exit_, "nights": nights, "split": sp, **es,
                   "cohort_nw_t": t, "cohort_days": nd,
                   "net_mean_base": es.get("mean", np.nan) - BASE_COST - nights * BORROW_BASE / 252}
            for b in BORROW_SENS:
                row[f"net_mean_borrow{int(b * 100)}"] = es.get("mean", np.nan) - BASE_COST - nights * b / 252
            rows.append(row)
    pd.DataFrame(rows).to_csv(RES / "A_entry_exit_grid.csv", index=False)
    ledger("A", "entry x exit x split cells", len(rows))

    # ---------------------------------------------------------------- B. feature profiles
    rows = []
    for f in FEATURES:
        lab = bucket_labels(prim_t, f)
        for (sp, b), sub in prim_t.assign(b=lab).dropna(subset=["b"]).groupby(["split", "b"]):
            es = event_stats(sub["s_h"])
            rows.append({"feature": f, "bucket": b, "split": sp, **es, "flag_small_n": es["n"] < 100})
    B = pd.DataFrame(rows)
    B.to_csv(RES / "B_feature_profiles.csv", index=False)
    ledger("B", "feature bucket x split cells", len(B))
    # B2: do the bucket effects replicate?  Compare each bucket's discovery mean with its
    # validation mean, relative to the split's unconditional mean (buckets with n>=100 in both).
    pv = B.pivot_table(index=["feature", "bucket"], columns="split", values=["mean", "n"])
    pv = pv[(pv[("n", "discovery")] >= 100) & (pv[("n", "validation")] >= 100)]
    dm = pv[("mean", "discovery")] - prim_t.loc[prim_t.split == "discovery", "s_h"].mean()
    vm = pv[("mean", "validation")] - prim_t.loc[prim_t.split == "validation", "s_h"].mean()
    summary["B2_feature_replication"] = {
        "buckets": int(len(pv)), "pearson": float(np.corrcoef(dm, vm)[0, 1]),
        "spearman": float(pd.Series(dm.values).corr(pd.Series(vm.values), method="spearman")),
        "sign_agreement": float((np.sign(dm) == np.sign(vm)).mean())}
    pd.DataFrame({"disc_excess": dm, "val_excess": vm}).to_csv(RES / "B2_feature_replication.csv")

    # ---------------------------------------------------------------- C. signal variants
    rows = []
    for n in [5, 10, 20, None]:
        for floor in [0.10, 0.20, 0.30]:
            base = topn(u5[u5["ret"] >= floor], n)
            base = base[base["s_h"].notna()]
            for sp in SPLITS:
                sub = base[base["split"] == sp]
                es = event_stats(sub["s_h"])
                st = port_stats(portfolio(sub, "s_h", BASE_COST, D[sp], slot=None), ci=False)
                rows.append({"top_n": n or "all", "gain_floor": floor, "split": sp, "n": es["n"],
                             "gross_mean_event": es["mean"], "median_event": es["median"], "hit": es["hit"],
                             "ew_sharpe_base": st["sharpe"], "ew_mean_daily_base": st["mean_daily"],
                             "ew_max_dd": st["max_dd"], "ew_worst_day": st["worst_day"]})
    pd.DataFrame(rows).to_csv(RES / "C_signal_variants.csv", index=False)
    ledger("C", "top-N x gain-floor x split cells", len(rows))

    # ---------------------------------------------------------------- D. universe variants
    naive = elig[(elig["u5_naive"] == True) & (elig["rank_naive"] <= 10)]  # noqa: E712
    variants = {
        "U5 primary (pre-registered)": (topn(u5, 10), "s_h"),
        "U5 raw (unhedged)": (topn(u5, 10), "s_raw"),
        "U5 IWM-hedged": (topn(u5, 10), "s_iwm"),
        "U5 ex-SPAC": (topn(u5[u5["is_spac"] != 1.0], 10), "s_h"),
        "U5 ex-Health Care": (topn(u5[u5["is_healthcare"] != 1.0], 10), "s_h"),
        "U5lo (>=$5, $2-10M DV)": (topn(elig[elig["universe"] == "U5lo"], 10), "s_h"),
        "U1 ($1-5, borrow-flagged)": (topn(elig[elig["universe"] == "U1"], 10), "s_h"),
        "POST-HOC U5 & prior close >= $5": (topn(u5[u5["raw_prev_close"] >= 5], 10), "s_h"),
        "POST-HOC U5 & prior close >= $1": (topn(u5[u5["raw_prev_close"] >= 1], 10), "s_h"),
        "POST-HOC U5 & prior close >= $5 & gain < 50%": (topn(u5[(u5["raw_prev_close"] >= 5) & (u5["ret"] < 0.5)], 10), "s_h"),
        "NAIVE split-adjusted $5 screen (look-ahead)": (naive, "s_h"),
    }
    rows = []
    for name, (df, col) in variants.items():
        for sp in SPLITS:
            sub = df[(df["split"] == sp) & df[col].notna()]
            es = event_stats(sub[col])
            for c in [BASE_COST, DECISION_COST]:
                st = port_stats(portfolio(sub, col, c, D[sp]), ci=(c == DECISION_COST))
                rows.append({"variant": name, "split": sp, "cost": c, "n": es["n"],
                             "gross_mean_event": es["mean"], "median_event": es["median"], "hit": es["hit"],
                             **{k: st[k] for k in ["mean_daily", "sharpe", "nw_t", "max_dd", "worst_day"]},
                             "ci_lo": st.get("ci_lo"), "ci_hi": st.get("ci_hi")})
    pd.DataFrame(rows).to_csv(RES / "D_universe_variants.csv", index=False)
    ledger("D", "universe variant x split x cost cells", len(rows))

    # ---------------------------------------------------------------- E. squeeze stops
    rows = []
    for stop in [None, 0.10, 0.20, 0.30]:
        df = prim_t.copy()
        if stop is not None:
            hit = df["hi1"] >= stop
            df["x"] = np.where(hit, -(stop + 0.01), -df["oc1"]) + df["spy_oc1"]
        else:
            hit = pd.Series(False, index=df.index)
            df["x"] = df["s_h"]
        for sp in SPLITS:
            sub = df[df["split"] == sp]
            es = event_stats(sub["x"])
            st = port_stats(portfolio(sub, "x", BASE_COST, D[sp]), ci=False)
            rows.append({"stop": stop or "none", "split": sp, "n": es["n"],
                         "stopped_frac": float(hit[sub.index].mean()), "gross_mean_event": es["mean"],
                         "hit": es["hit"], "p01_event": es["p01"], "worst_event": es["worst"],
                         "sharpe_base": st["sharpe"], "mean_daily_base": st["mean_daily"],
                         "max_dd": st["max_dd"], "worst_day": st["worst_day"]})
    pd.DataFrame(rows).to_csv(RES / "E_stop_overlay.csv", index=False)
    ledger("E", "stop level x split cells", len(rows))

    # ---------------------------------------------------------------- F. per-year
    rows = []
    rm_all = rule_mask(prim_t)
    for y, sub in prim_t.groupby("year"):
        days = cal[(cal.year == y)]
        r = portfolio(sub, "s_h", BASE_COST, days)
        es = event_stats(sub["s_h"])
        st = port_stats(r, ci=False)
        sr = sub[rm_all[sub.index]]
        rr = portfolio(sr, "s_h", BASE_COST, days)
        rows.append({"year": y, "n": es["n"], "gross_mean_event": es["mean"], "median_event": es["median"],
                     "hit": es["hit"], "worst_event": es["worst"], "year_return_base": r.sum(),
                     "sharpe_base": st["sharpe"], "max_dd": st["max_dd"], "worst_day": st["worst_day"],
                     "rule_n": len(sr), "rule_gross_mean_event": sr["s_h"].mean(),
                     "rule_year_return_base": rr.sum(), "rule_sharpe_base": port_stats(rr, ci=False)["sharpe"]})
    pd.DataFrame(rows).to_csv(RES / "F_per_year.csv", index=False)
    ledger("F", "years x (primary, H2 rule)", 2 * len(rows))

    # ---------------------------------------------------------------- cost at which the validation CI stops clearing zero
    rows = []
    for name, sub in [("H1 primary", val), ("H2 rule", val[rule_mask(val)])]:
        for c in np.round(np.arange(0.0, 0.01001, 0.0005), 4):
            st = port_stats(portfolio(sub, "s_h", c, D["validation"]), ci=True)
            rows.append({"strategy": name, "cost": c, "mean_daily": st["mean_daily"], "ci_lo": st["ci_lo"],
                         "ci_hi": st["ci_hi"], "sharpe": st["sharpe"]})
    BE = pd.DataFrame(rows)
    BE.to_csv(RES / "break_even_costs_validation.csv", index=False)
    ledger("BE", "validation cost grid (descriptive; verdicts unchanged)", len(BE))
    summary["validation_cost_where_ci_still_clears_zero"] = {
        k: float(g.loc[g["ci_lo"] > 0, "cost"].max()) if (g["ci_lo"] > 0).any() else 0.0
        for k, g in BE.groupby("strategy")}
    summary["validation_gross_break_even"] = {
        "H1 primary": float(val["s_h"].mean()), "H2 rule": float(val.loc[rule_mask(val), "s_h"].mean())}

    # ---------------------------------------------------------------- P. POST-HOC candidates
    # Examined AFTER the validation results above were seen.  Not a verdict; candidates for a
    # fresh forward (paper-trading) test only.  Counted in the ledger.
    est = prim_t["raw_prev_close"] >= 5
    noe = prim_t["earnings"] == 0.0
    cands = {
        "no earnings & prior close>=$5 (H2 rule, established names)": noe & est,
        "no earnings & entry gap>=+5% & prior close>=$5": noe & (prim_t["on1"] >= 0.05) & est,
        "no earnings & RSI14>=85 & prior close>=$5": noe & (prim_t["rsi14"] >= 85) & est,
        "gain>=25% (preliminary-run pick, before earnings data loaded)": prim_t["ret"] >= 0.25,
        "gain>=25% & prior close>=$5": (prim_t["ret"] >= 0.25) & est,
        "RSI14>=85": prim_t["rsi14"] >= 85,
        "RSI14>=85 & prior close>=$5": (prim_t["rsi14"] >= 85) & est,
        "entry gap>=+5%": prim_t["on1"] >= 0.05,
        "entry gap>=+5% & prior close>=$5": (prim_t["on1"] >= 0.05) & est,
        "SPY>200d & gain>=25%": (prim_t["spy_above200"] == 1.0) & (prim_t["ret"] >= 0.25),
    }
    rows = []
    for name, msk in cands.items():
        for sp in SPLITS:
            sub = prim_t[msk & (prim_t["split"] == sp)]
            es = event_stats(sub["s_h"])
            sb = port_stats(portfolio(sub, "s_h", BASE_COST, D[sp]), ci=False)
            sd = port_stats(portfolio(sub, "s_h", DECISION_COST, D[sp]), ci=True)
            rows.append({"candidate": name, "split": sp, "n": es["n"], "gross_mean_event": es["mean"],
                         "median_event": es["median"], "hit": es["hit"], "worst_event": es["worst"],
                         "sharpe_0.25": sb["sharpe"], "sharpe_0.50": sd["sharpe"],
                         "mean_daily_0.50": sd["mean_daily"], "ci_lo_0.50": sd["ci_lo"], "ci_hi_0.50": sd["ci_hi"],
                         "max_dd_0.25": sb["max_dd"], "worst_day_0.25": sb["worst_day"]})
    pd.DataFrame(rows).to_csv(RES / "P_posthoc_candidates.csv", index=False)
    ledger("P", "post-hoc candidate x split cells (seen after validation)", len(rows))

    # ---------------------------------------------------------------- daily series for figures
    out = pd.DataFrame(index=cal[cal >= pd.Timestamp("2010-01-04")])
    out["primary_base"] = portfolio(prim_t, "s_h", BASE_COST, out.index)
    out["primary_gross"] = portfolio(prim_t, "s_h", 0.0, out.index)
    rm = prim_t[rule_mask(prim_t)]
    out["rule_base"] = portfolio(rm, "s_h", BASE_COST, out.index)
    out["rule_gross"] = portfolio(rm, "s_h", 0.0, out.index)
    out["rule_n"] = rm.groupby("date").size().reindex(out.index, fill_value=0)
    out["primary_n"] = prim_t.groupby("date").size().reindex(out.index, fill_value=0)
    est = topn(u5[u5["raw_prev_close"] >= 5], 10)
    out["posthoc_established_base"] = portfolio(est[est["s_h"].notna()], "s_h", BASE_COST, out.index)
    out["naive_base"] = portfolio(naive[naive["s_h"].notna()], "s_h", BASE_COST, out.index)
    out.index.name = "signal_date"
    nv = naive[naive["s_h"].notna()]
    (nv.assign(month=nv["date"].dt.strftime("%Y-%m"), pnl=SLOT * (nv["s_h"] - BASE_COST))
       .groupby("month").agg(n_events=("s_h", "size"), month_pnl_base=("pnl", "sum"),
                             mean_event_gross=("s_h", "mean"))
       .to_csv(RES / "G_yahoo_naive_monthly.csv"))
    out.to_csv(RES / "daily_portfolio_returns.csv")

    L = pd.DataFrame(LEDGER)
    L.to_csv(RES / "ledger.csv", index=False)
    summary["cells_examined_total"] = int(L["cells"].sum())
    summary["rule_events_validation"] = int(rule_mask(val).sum())
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
