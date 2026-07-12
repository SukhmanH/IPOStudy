"""
build_panel.py — per-event hedged-return panel for the fade-generalization
study. Pure computation on cached prices (no network).

For every universe name with clean prices, computes close-to-close returns
from entry-day close to exit-day close over the full pre-registered +
exploratory grid:
    entry td   e in {1, 5, 10, 20, 30}
    exit  td   x in {60, 90, 120, 180, 250}
using split/dividend-adjusted closes (AdjClose), windows anchored on the
audited listing date (utils.resolve_first_trade — kills pre-IPO placeholder
rows). QQQ is aligned to each stock's own trading dates (same timestamps).

Hedged return per event, per $1 of short notional (dollar-neutral spread):
    hedged(e,x) = qqq_ret(e->x) - stock_ret(e->x)      (positive = fade captured)
Borrow cost is NOT applied here — the panel stores gross legs; cost scenarios
are applied by the analysis scripts so sensitivities stay explicit.

Executable prices only: all entries/exits at daily closes. The day-1 high is
never used as an entry. Events missing the exit bar (delisted/acquired
mid-window) are recorded with NaN for that pair and counted — that attrition
is survivorship-relevant and reported, not hidden.

Output: results/event_panel.csv — one row per event with columns
    stock_e{e}_x{x}, qqq_e{e}_x{x}, hedged_e{e}_x{x}
plus ticker/tier/listing/quarter/era/split metadata.

Run:  python build_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"c:\Users\brylan\Desktop\spaceX")
GEN = REPO / "general"
sys.path.insert(0, str(REPO / "spcx-report" / "src"))
import utils as u  # noqa: E402

UNIVERSE = GEN / "data" / "ipo_universe_t1t4.csv"
FETCH_LOG = GEN / "data" / "price_fetch_log_t1t4.csv"
PANEL_OUT = GEN / "results" / "event_panel.csv"

ENTRIES = [1, 5, 10, 20, 30]
EXITS = [60, 90, 120, 180, 250]

# Out-of-sample split — FIXED by the brief before any results:
DISCOVERY_END = pd.Timestamp("2018-12-31")   # discovery: 2010-01-01..2018-12-31
# validation: 2019-01-01 .. latest listing with complete +180 history


def main():
    print("BUILD PANEL — per-event hedged returns, entries x exits grid")
    print("=" * 68)
    uni = pd.read_csv(UNIVERSE, keep_default_na=False)
    log = pd.read_csv(FETCH_LOG, keep_default_na=False)
    ok = set(log[log["status"] == "ok"]["ticker"])

    names = uni[(uni["bucket"] == "headline")
                | ((uni["bucket"] == "considered")
                   & uni["reason"].str.contains("too recent", na=False))].copy()
    names = names[names["ticker"].isin(ok)]
    names["listing_dt"] = pd.to_datetime(names["first_trade"], errors="coerce")
    names = names[names["listing_dt"].notna()]
    print(f"events with clean prices: {len(names)}")

    qqq = u.fetch_history("QQQ", raw=True)
    if qqq is None:
        raise SystemExit("QQQ cache missing")
    qqq_adj = qqq["AdjClose"]

    max_x = max(EXITS)
    rows, miss_pair = [], {(e, x): 0 for e in ENTRIES for x in EXITS}
    for i, r in enumerate(names.itertuples(), 1):
        hist = u.fetch_history(r.ticker, raw=True)   # cache hit, no network
        if hist is None or "AdjClose" not in hist.columns:
            continue
        anchor = u.resolve_first_trade(hist, expected=r.listing_dt)
        win = hist.loc[anchor:].iloc[: max_x + 1]
        adj = win["AdjClose"].astype(float)
        q = qqq_adj.reindex(win.index).ffill().astype(float)

        row = {
            "ticker": r.ticker, "tier": r.tier, "bucket": r.bucket,
            "listing": anchor.date().isoformat(),
            "quarter": f"{anchor.year}Q{(anchor.month - 1)//3 + 1}",
            "year": anchor.year,
            "era": ("2010-2014" if anchor.year <= 2014 else
                    "2015-2019" if anchor.year <= 2019 else "2020+"),
            "split": "discovery" if anchor <= DISCOVERY_END else "validation",
            "n_bars": len(win),
        }
        for e in ENTRIES:
            for x in EXITS:
                if x < len(adj) and e < len(adj) and adj.iloc[e] > 0 and q.iloc[e] > 0:
                    s_ret = adj.iloc[x] / adj.iloc[e] - 1.0
                    q_ret = q.iloc[x] / q.iloc[e] - 1.0
                    row[f"stock_e{e}_x{x}"] = s_ret
                    row[f"qqq_e{e}_x{x}"] = q_ret
                    row[f"hedged_e{e}_x{x}"] = q_ret - s_ret
                else:
                    row[f"stock_e{e}_x{x}"] = np.nan
                    row[f"qqq_e{e}_x{x}"] = np.nan
                    row[f"hedged_e{e}_x{x}"] = np.nan
                    miss_pair[(e, x)] += 1
        rows.append(row)
        if i % 200 == 0:
            print(f"  ...{i}/{len(names)}")

    panel = pd.DataFrame(rows)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_OUT, index=False)

    print(f"\npanel events: {len(panel)}")
    print("by tier x split (events with a complete e20->x180 pair):")
    comp = panel[panel["hedged_e20_x180"].notna()]
    print(comp.groupby(["tier", "split"], observed=True).size().to_string())
    print("\nincomplete-pair counts (delisted/acquired mid-window or too recent):")
    for (e, x), c in sorted(miss_pair.items()):
        if c and x in (180, 250):
            print(f"  e{e}->x{x}: {c}")
    print(f"\nWrote {PANEL_OUT}")


if __name__ == "__main__":
    main()
