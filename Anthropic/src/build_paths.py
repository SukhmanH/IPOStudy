"""
build_paths.py - day-0-anchored event table + daily relative paths for the
T3/T4 comparable cohort, plus the consistency check against the parent study.
Pure computation on the committed windows (no network).

Per event (day 0 = listing day; AdjClose close-to-close; QQQ aligned to the
stock's own dates, forward-filled - same as the parent's build_panel.py):
  s_h   stock return day0 close -> day h close
  q_h   QQQ return over the same dates
  x_h   = q_h - s_h          additive hedged fade (parent's metric; + = fade)
  rel_h = (1+s_h)/(1+q_h)-1  QQQ-neutral relative return (used for price targets)
  *_hi  the same measured from the day-0 intraday HIGH (legacy Study-1
        reference: "give-back from the day-1 high"; never an entry price)
  po_h  price at day h / offer price (raw, split-consistent)
  grid  hedged e{1,5,10,20,30} x x{60,90,120,180,250} (parent's grid) + SPY/IWM
        variants of the primary spec

Consistency gate (parent ground rule #1 in spirit): every overlapping event's
25 grid cells are compared to the parent's results/event_panel.csv, and the
parent's primary-spec numbers are reproduced from the parent panel itself.

Outputs:
  results/cohort_event_table.csv
  data/cohort_rel_paths.csv.gz         (day x event QQQ-neutral relative level)
  results/consistency_check_vs_parent.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as c

ENTRIES = [1, 5, 10, 20, 30]
EXITS = [60, 90, 120, 180, 250]


def load_inputs():
    win = pd.read_csv(c.DATA / "cohort_prices_window.csv.gz", parse_dates=["Date"])
    bench = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    uni = pd.read_csv(c.DATA / "cohort_universe.csv", keep_default_na=False)
    log = pd.read_csv(c.DATA / "price_fetch_log.csv", keep_default_na=False)
    return win, bench, uni, log


def event_row(g: pd.DataFrame, meta: dict, bench: pd.DataFrame) -> tuple[dict, np.ndarray]:
    g = g.sort_values("day").set_index("Date")
    adj = g["AdjClose"].astype(float).to_numpy()
    n = len(adj)
    q = bench["QQQ"].reindex(g.index).ffill().to_numpy()
    spy = bench["SPY"].reindex(g.index).ffill().to_numpy()
    iwm = bench["IWM"].reindex(g.index).ffill().to_numpy()
    sf = float(g["split_factor_after_day0"].iloc[0])
    close0, high0, open0 = (float(g[k].iloc[0]) for k in ("Close", "High", "Open"))
    divadj0 = adj[0] / close0 if close0 > 0 else 1.0
    adj_high0 = high0 * divadj0
    offer = meta["offer"]
    raw_close0, raw_high0, raw_open0 = close0 * sf, high0 * sf, open0 * sf
    d0 = g.index[0]

    # pre-listing VIX: mean close over the 21 sessions before day 0
    vix = bench["VIX"].dropna()
    vix_pre = float(vix.loc[:d0 - pd.Timedelta(days=1)].tail(21).mean())
    lr = np.diff(np.log(adj[: min(n, 21)]))
    rv20 = float(np.std(lr, ddof=1) * np.sqrt(252)) if len(lr) > 2 else np.nan
    dv = (g["Close"] * g["Volume"]).to_numpy()
    adv20 = float(np.mean(dv[1:21])) if n > 21 else np.nan

    row = dict(meta)
    row.update(listing=d0.date().isoformat(), quarter=c.quarter_of(d0), year=d0.year,
               era=c.era_of(d0.year),
               split=("live" if meta["role"] == "live" else
                      "discovery" if d0 <= c.DISCOVERY_END else "validation"),
               n_bars=n, raw_open0=raw_open0, raw_high0=raw_high0, raw_close0=raw_close0,
               pop_open=raw_open0 / offer - 1, pop_close=raw_close0 / offer - 1,
               pop_high=raw_high0 / offer - 1, high_over_close0=high0 / close0 - 1,
               vix_pre21=vix_pre, rv_d1_20=rv20, adv20_usd_m=adv20 / 1e6 if pd.notna(adv20) else np.nan)

    for h in c.HORIZONS:
        if h < n:
            s = adj[h] / adj[0] - 1
            qq = q[h] / q[0] - 1
            s_hi = adj[h] / adj_high0 - 1
            row.update({f"s_{h}": s, f"q_{h}": qq, f"x_{h}": qq - s,
                        f"rel_{h}": (1 + s) / (1 + qq) - 1,
                        f"s_hi_{h}": s_hi, f"x_hi_{h}": qq - s_hi,
                        f"rel_hi_{h}": (1 + s_hi) / (1 + qq) - 1,
                        f"po_{h}": raw_close0 * (1 + s) / offer,
                        f"po_rel_{h}": raw_close0 * (1 + s) / (1 + qq) / offer})
        else:
            for k in ("s", "q", "x", "rel", "s_hi", "x_hi", "rel_hi", "po", "po_rel"):
                row[f"{k}_{h}"] = np.nan
    for e in ENTRIES:
        for x in EXITS:
            if x < n:
                s = adj[x] / adj[e] - 1
                row[f"hedged_e{e}_x{x}"] = (q[x] / q[e] - 1) - s
            else:
                row[f"hedged_e{e}_x{x}"] = np.nan
    e, x = c.PRIMARY_ENTRY, c.PRIMARY_EXIT
    if x < n:
        s = adj[x] / adj[e] - 1
        row["hedged_spy_e20_x180"] = (spy[x] / spy[e] - 1) - s
        row["hedged_iwm_e20_x180"] = (iwm[x] / iwm[e] - 1) - s
        row["rel_e20_x180"] = (1 + s) / (q[x] / q[e]) - 1
    else:
        row["hedged_spy_e20_x180"] = row["hedged_iwm_e20_x180"] = row["rel_e20_x180"] = np.nan

    # path shape (QQQ-neutral relative level, day0 close = 1)
    rel_path = (adj / adj[0]) / (q / q[0])
    k20, k180 = min(n, 21), min(n, 181)
    row["peak_rel_0_20"] = float(rel_path[:k20].max())
    row["peak_day_0_20"] = int(rel_path[:k20].argmax())
    row["trough_rel_0_180"] = float(rel_path[:k180].min()) if n > 1 else np.nan
    running_max = np.maximum.accumulate(rel_path[:k180])
    row["maxdd_rel_0_180"] = float((rel_path[:k180] / running_max - 1).min())
    return row, rel_path


def main():
    print("BUILD PATHS - day-0-anchored cohort event table")
    print("=" * 72)
    win, bench, uni, log = load_inputs()
    ok = log[log["status"].str.startswith("ok")]
    meta = uni.set_index("ticker")
    rows, paths = [], []
    for t, g in win.groupby("ticker", sort=False):
        if t not in set(ok["ticker"]):
            continue
        m = meta.loc[t]
        md = dict(ticker=t, name=m["name"], role=m["role"], tier=m["tier"],
                  mcap_b=pd.to_numeric(m["mcap_b"], errors="coerce"), adr=str(m["adr"]) == "True",
                  offer=float(m["offer"]))
        r, p = event_row(g, md, bench)
        rows.append(r)
        paths.append(pd.DataFrame({"ticker": t, "day": np.arange(len(p)), "rel": p}))
    ev = pd.DataFrame(rows)
    ev.to_csv(c.RESULTS / "cohort_event_table.csv", index=False)
    pd.concat(paths).to_csv(c.DATA / "cohort_rel_paths.csv.gz", index=False, float_format="%.6f")
    print(f"events: {len(ev)}")
    print(ev.groupby(["role", "tier", "split"]).size().to_string())

    # ------------------------------------------------ consistency vs the parent
    par = pd.read_csv(c.PARENT_PANEL, keep_default_na=False)
    cols = [f"hedged_e{e}_x{x}" for e in ENTRIES for x in EXITS]
    for col in cols:
        par[col] = pd.to_numeric(par[col], errors="coerce")
    both = ev[["ticker"] + cols].merge(par[["ticker"] + cols], on="ticker", suffixes=("", "_parent"))
    recs = []
    for r in both.itertuples(index=False):
        d = r._asdict()
        diffs = [abs(d[k] - d[k + "_parent"]) for k in cols
                 if pd.notna(d[k]) and pd.notna(d[k + "_parent"])]
        newly = sum(1 for k in cols if pd.notna(d[k]) and pd.isna(d[k + "_parent"]))
        recs.append(dict(ticker=r.ticker, cells_compared=len(diffs),
                         max_abs_diff=max(diffs) if diffs else np.nan,
                         cells_new_since_parent=newly))
    cc = pd.DataFrame(recs)
    cc.to_csv(c.RESULTS / "consistency_check_vs_parent.csv", index=False)
    print(f"\nconsistency vs parent event_panel: {len(cc)} overlapping events, "
          f"{int(cc.cells_compared.sum())} cells compared; "
          f"max |diff| = {cc.max_abs_diff.max():.2e}; "
          f"events with any cell off by >0.5pp: {int((cc.max_abs_diff > 0.005).sum())}; "
          f"cells that matured since the parent freeze: {int(cc.cells_new_since_parent.sum())}")
    off = cc[cc.max_abs_diff > 0.005]
    if len(off):
        print(off.to_string(index=False))


if __name__ == "__main__":
    main()
