"""
replicate_shibui.py -- exploratory item G: independent replication of the
top-gainer short on a second vendor (shibui DB, server-side DuckDB SQL).

shibui has no split history, so its $5 floor can only be applied to
split-ADJUSTED closes.  The comparison is therefore against this study's
"NAIVE split-adjusted $5 screen" variant (same screen, Yahoo data, pandas
pipeline).  The monthly series came back from the SQL queries recorded in
REPORT.md (Appendix G) and are stored verbatim in data/shibui/*.json.

Run:  python short_gainers/src/replicate_shibui.py
Out:  results/G_shibui_replication.csv, results/G_shibui_monthly_compare.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
DATA, RES = HERE / "data", HERE / "results"


def main() -> None:
    sh = pd.concat([pd.DataFrame(json.loads((DATA / "shibui" / f).read_text()))
                    for f in ["naive_monthly_discovery.json", "naive_monthly_validation.json"]])
    sh = sh.rename(columns={"ym": "month"}).set_index("month")
    yh = pd.read_csv(RES / "G_yahoo_naive_monthly.csv").set_index("month")
    m = sh.join(yh, lsuffix="_shibui", rsuffix="_yahoo", how="outer")
    m.to_csv(RES / "G_shibui_monthly_compare.csv")
    m["split"] = np.where(m.index <= "2018-12", "discovery", "validation")
    rows = []
    for sp, g in m.groupby("split"):
        g = g.dropna()
        rows.append({
            "split": sp, "months": len(g),
            "events_shibui": int(g["n_events_shibui"].sum()), "events_yahoo": int(g["n_events_yahoo"].sum()),
            "mean_event_gross_shibui": float(np.average(g["mean_event_gross_shibui"], weights=g["n_events_shibui"])),
            "mean_event_gross_yahoo": float(np.average(g["mean_event_gross_yahoo"], weights=g["n_events_yahoo"])),
            "month_pnl_mean_shibui": g["month_pnl_base_shibui"].mean(),
            "month_pnl_mean_yahoo": g["month_pnl_base_yahoo"].mean(),
            "corr_month_pnl": g["month_pnl_base_shibui"].corr(g["month_pnl_base_yahoo"]),
            "corr_n_events": g["n_events_shibui"].corr(g["n_events_yahoo"]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(RES / "G_shibui_replication.csv", index=False)
    print(out.round(4).to_string(index=False))
    d = m.dropna().assign(diff=lambda x: x["month_pnl_base_shibui"] - x["month_pnl_base_yahoo"])
    print("largest monthly disagreements:")
    print(d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(6)[
        ["n_events_shibui", "n_events_yahoo", "month_pnl_base_shibui", "month_pnl_base_yahoo"]].round(4))


if __name__ == "__main__":
    main()
