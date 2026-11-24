"""
make_report_page.py -- builds the interactive HTML report (published as an
Artifact) from results/*.csv, so every number on the page comes from the
committed result files.  Charts are inline SVG drawn by vanilla JS; colors
come from CSS tokens (light + dark), every chart has hover tooltips and a
table view.

Run:  python short_gainers/src/make_report_page.py <out.html>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA, RES = HERE / "data", HERE / "results"


def r(x, n=6):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def payload() -> dict:
    S = json.loads((RES / "summary.json").read_text())
    h1 = pd.read_csv(RES / "h1_primary.csv")
    h2 = pd.read_csv(RES / "h2_selected_rule.csv")
    A = pd.read_csv(RES / "A_entry_exit_grid.csv")
    B = pd.read_csv(RES / "B_feature_profiles.csv")
    D = pd.read_csv(RES / "D_universe_variants.csv")
    E = pd.read_csv(RES / "E_stop_overlay.csv")
    F = pd.read_csv(RES / "F_per_year.csv")
    P = pd.read_csv(RES / "P_posthoc_candidates.csv")
    G = pd.read_csv(RES / "G_shibui_replication.csv")
    L = pd.read_csv(RES / "ledger.csv")
    daily = pd.read_csv(RES / "daily_portfolio_returns.csv", parse_dates=["signal_date"]).set_index("signal_date")
    ev = pd.read_parquet(DATA / "events.parquet")
    prim = ev[(ev["excl_reason"] == "eligible") & (ev["universe"] == "U5") & (ev["rank"] <= 10) & ev["oc1"].notna()]
    s_h = (-prim["oc1"] + prim["spy_oc1"]).astype(float)
    cols = {}
    for tag, g, n in [("a", daily["primary_gross"], daily["primary_n"]), ("b", daily["rule_gross"], daily["rule_n"])]:
        cols[f"g{tag}"] = g.cumsum()
        cols[f"c25{tag}"] = (g - 0.1 * n * 0.0025).cumsum()
        cols[f"c50{tag}"] = (g - 0.1 * n * 0.0050).cumsum()
    cum = pd.DataFrame(cols)[["ga", "c25a", "c50a", "gb", "c25b", "c50b"]]
    wk = cum.resample("W-FRI").last().dropna()
    cum_pts = [[d.strftime("%Y-%m-%d")] + [r(v, 4) for v in row] for d, row in zip(wk.index, wk.values)]

    feats = [("F15_earnings", "Earnings report on the day or the day before"), ("F1_gain", "Gain on the signal day"), ("F14_entry_gap", "Next-day opening gap"),
             ("F7_rsi14", "RSI(14) on the signal day"), ("F2_clv", "Where it closed in the day's range"),
             ("F5_ret20_prior", "Return over the prior 20 days"), ("F11_spy_above200", "S&P 500 above its 200-day average")]
    feat_out = []
    for f, title in feats:
        sub = B[B["feature"] == f]
        order = (["yes", "no"] if f == "F15_earnings" else ["no", "yes"] if f == "F11_spy_above200"
                 else sorted(sub["bucket"].unique(), key=bkey))
        rows = []
        for b in order:
            d = sub[(sub["bucket"] == b) & (sub["split"] == "discovery")]
            v = sub[(sub["bucket"] == b) & (sub["split"] == "validation")]
            rows.append({"label": pretty(b), "d": r(d["mean"].iloc[0]) if len(d) else None,
                         "v": r(v["mean"].iloc[0]) if len(v) else None,
                         "nd": int(d["n"].iloc[0]) if len(d) else 0, "nv": int(v["n"].iloc[0]) if len(v) else 0,
                         "hd": r(d["hit"].iloc[0], 4) if len(d) else None, "hv": r(v["hit"].iloc[0], 4) if len(v) else None})
        feat_out.append({"title": title, "rows": rows})

    a = A[A["entry"] == "open_t1"].copy()
    a["k"] = a["exit"].str.extract(r"close_t(\d+)").astype(int)
    horizon = [{"k": int(k), "d": r(g[g.split == "discovery"]["mean"].iloc[0]),
                "v": r(g[g.split == "validation"]["mean"].iloc[0]),
                "nd": int(g[g.split == "discovery"]["n"].iloc[0]), "nv": int(g[g.split == "validation"]["n"].iloc[0])}
               for k, g in a.groupby("k")]

    def dv(name, split, cost, col):
        x = D[(D["variant"] == name) & (D["split"] == split) & np.isclose(D["cost"], cost)]
        return r(x[col].iloc[0]) if len(x) else None
    screens = [("NAIVE split-adjusted $5 screen (look-ahead)", "Naive screen: $5+ on split-adjusted price"),
               ("U5 primary (pre-registered)", "Pre-registered: $5+ actual traded price"),
               ("POST-HOC U5 & prior close >= $5", "Established: $5+ on the day before too"),
               ("U1 ($1-5, borrow-flagged)", "Under $5 (borrow usually unavailable)")]
    screen_out = [{"label": lab, "d": dv(n, "discovery", 0.0025, "gross_mean_event"),
                   "v": dv(n, "validation", 0.0025, "gross_mean_event"),
                   "nd": dv(n, "discovery", 0.0025, "n"), "nv": dv(n, "validation", 0.0025, "n"),
                   "sd": dv(n, "discovery", 0.0025, "sharpe"), "sv": dv(n, "validation", 0.0025, "sharpe"),
                   "wd": dv(n, "validation", 0.0025, "worst_day")} for n, lab in screens]

    bins = np.linspace(-1.0, 0.6, 65)
    cnt, _ = np.histogram(s_h.clip(-1.0, 0.6 - 1e-9), bins=bins)
    hist = {"edges": [r(b, 4) for b in bins], "counts": [int(c) for c in cnt],
            "q01": r(s_h.quantile(0.01), 4), "q50": r(s_h.quantile(0.5), 4), "q99": r(s_h.quantile(0.99), 4),
            "worst": r(s_h.min(), 4), "best": r(s_h.max(), 4), "n": int(len(s_h)),
            "share_pos": r((s_h > 0).mean(), 4)}

    def rows_of(df, cols):
        return [{c: (r(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
                 for c, v in zip(cols, rec)} for rec in df[cols].itertuples(index=False)]

    cov = pd.read_csv(DATA / "coverage_by_year.csv")
    return {
        "summary": S,
        "h1": rows_of(h1, ["split", "cost", "n_events", "gross_mean_event", "median_event", "hit", "mean_daily",
                           "ci_lo", "ci_hi", "sharpe", "max_dd", "worst_day", "worst_event"]),
        "h2": rows_of(h2, ["split", "cost", "n_events", "gross_mean_event", "median_event", "hit", "mean_daily",
                           "ci_lo", "ci_hi", "sharpe", "max_dd", "worst_event"]),
        "cum": cum_pts, "features": feat_out, "horizon": horizon, "screens": screen_out,
        "stops": rows_of(E.astype({"stop": str}), ["stop", "split", "n", "stopped_frac", "gross_mean_event",
                                                    "sharpe_base", "worst_event", "worst_day", "max_dd"]),
        "years": rows_of(F, ["year", "n", "gross_mean_event", "year_return_base", "sharpe_base", "worst_day",
                             "rule_n", "rule_gross_mean_event", "rule_year_return_base", "rule_sharpe_base"]),
        "posthoc": rows_of(P, ["candidate", "split", "n", "gross_mean_event", "median_event", "hit",
                               "sharpe_0.25", "sharpe_0.50", "ci_lo_0.50", "ci_hi_0.50", "worst_event"]),
        "repl": rows_of(G, ["split", "months", "events_shibui", "events_yahoo", "mean_event_gross_shibui",
                            "mean_event_gross_yahoo", "corr_month_pnl"]),
        "hist": hist, "ledger_total": int(L["cells"].sum()),
        "rule_est": rows_of(P[P["candidate"].str.startswith("no earnings & prior close")],
                            ["split", "n", "gross_mean_event", "median_event", "hit", "sharpe_0.25", "sharpe_0.50"]),
        "coverage": {"y2010": int(cov.iloc[0]["tickers_with_data"]), "y2026": int(cov.iloc[-1]["tickers_with_data"])},
        "n_tickers": int(prim["ticker"].nunique()),
    }


def bkey(b: str):
    try:
        return float(b.strip("[(").split(",")[0].replace("-inf", "-1e18"))
    except ValueError:
        return 0.0


def pretty(b: str) -> str:
    """Compact bucket label for narrow small multiples: <15%, 15–25%, ≥100%, <−5%."""
    if b in ("yes", "no"):
        return b
    lo, hi = [x.strip() for x in b.strip("[)").split(",")]

    def f(v, unit=True):
        v = float(v)
        s = f"{v * 100:.0f}" if abs(v) < 5 else f"{v:.0f}"
        s = s.replace("-", "−")
        return s + ("%" if unit and abs(v) < 5 else "")
    if lo == "-inf":
        return f"<{f(hi)}"
    if hi == "inf":
        return f"≥{f(lo)}"
    return f"{f(lo, False)}–{f(hi)}"


def main(out: str) -> None:
    tpl = (HERE / "src" / "report_page_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload(), separators=(",", ":")))
    Path(out).write_text(html)
    print("wrote", out, f"{len(html) / 1024:.0f} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(HERE / "results" / "report_page.html"))
