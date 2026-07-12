"""
primary_spec.py — the ONE pre-registered confirmatory test, plus the
exploratory entry x exit grid (reported in full, never promoted).

PRIMARY SPEC (verbatim from the brief, committed before any exploration):
  Universe: T3+T4 (>=$5B at listing), 2010-present, standard exclusions.
  Trade: short at close of +20, cover at close of +180; hedge long QQQ,
         equal dollar, same timestamps (dollar-neutral spread).
  Metric: hedged return per event; median, mean, hit rate.
  Inference: moving-block bootstrap clustered by listing quarter
         (10,000 draws) for the CI; naive t reported alongside, labeled
         as assuming independence.
  Costs: 3% annualized borrow base case over the ~160-td hold;
         sensitivities at 0%, 10%, 30%.
  DECISION RULE (stated verbatim): the edge "exists and is plausibly
  capturable in T3/T4" iff the VALIDATION-period hedged mean is profitable
  to the short with the 95% bootstrap CI excluding zero at the 10% borrow
  sensitivity. Otherwise the honest conclusion is: no systematic edge after
  costs; the fade is real but only tradeable when listing structure is
  special (SPCX-style float/lockup calendars).

Outputs: results/primary_spec.csv, results/entry_exit_grid.csv
Run:  python primary_spec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(r"c:\Users\brylan\Desktop\spaceX")
GEN = REPO / "general"
PANEL = GEN / "results" / "event_panel.csv"
OUT_PRIMARY = GEN / "results" / "primary_spec.csv"
OUT_GRID = GEN / "results" / "entry_exit_grid.csv"

ENTRY, EXIT = 20, 180
HOLD_YEARS = (EXIT - ENTRY) / 252.0          # ~160 td
BORROWS = [0.00, 0.03, 0.10, 0.30]           # annualized; 3% = base case
N_BOOT = 10_000
BLOCK_LEN = 4                                 # quarters per moving block (~1yr)
RNG = np.random.default_rng(42)

ENTRIES = [1, 5, 10, 20, 30]
EXITS = [60, 90, 120, 180, 250]


def block_bootstrap_ci(df: pd.DataFrame, col: str, stat_fn, n_boot=N_BOOT,
                       block=BLOCK_LEN, alpha=0.05) -> tuple[float, float]:
    """Moving-block bootstrap clustered by listing quarter: quarters are the
    cluster unit, ordered chronologically; blocks of `block` consecutive
    quarters are drawn with replacement until the quarter count is covered;
    each draw keeps every event in the chosen quarters (with multiplicity)."""
    quarters = sorted(df["quarter"].unique())
    nq = len(quarters)
    by_q = {q: df.loc[df["quarter"] == q, col].to_numpy() for q in quarters}
    if nq <= block:
        starts_all = np.arange(1)
    else:
        starts_all = np.arange(nq - block + 1)
    n_blocks = int(np.ceil(nq / block))
    stats_out = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.choice(starts_all, size=n_blocks, replace=True)
        vals = np.concatenate([
            by_q[quarters[s + j]]
            for s in starts for j in range(block) if s + j < nq
        ]) if nq > 0 else np.array([])
        stats_out[b] = stat_fn(vals) if len(vals) else np.nan
    lo, hi = np.nanpercentile(stats_out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def spec_stats(df: pd.DataFrame, borrow: float) -> dict:
    h = df["hedged_net"] = df[f"hedged_e{ENTRY}_x{EXIT}"] - borrow * HOLD_YEARS
    n = len(h)
    mean, med = float(h.mean()), float(h.median())
    hit = float((h > 0).mean())
    sd = float(h.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if n > 1 else np.nan
    mlo, mhi = block_bootstrap_ci(df.assign(hedged_net=h), "hedged_net", np.mean)
    dlo, dhi = block_bootstrap_ci(df.assign(hedged_net=h), "hedged_net", np.median)
    return dict(n=n, mean=mean, median=med, hit_rate=hit,
                naive_t=t, naive_p=p,
                mean_ci_lo=mlo, mean_ci_hi=mhi,
                median_ci_lo=dlo, median_ci_hi=dhi)


def main():
    print("PRIMARY CONFIRMATORY SPEC — T3+T4, short close +20 -> cover close +180")
    print("=" * 72)
    panel = pd.read_csv(PANEL, keep_default_na=False)
    col = f"hedged_e{ENTRY}_x{EXIT}"
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

    t34 = panel[panel["tier"].isin(["T3", "T4"]) & panel[col].notna()].copy()
    print(f"T3+T4 events with complete +{ENTRY}->+{EXIT}: {len(t34)} "
          f"(discovery {sum(t34.split=='discovery')}, validation {sum(t34.split=='validation')})")

    rows = []
    for split in ["discovery", "validation", "full"]:
        sub = t34 if split == "full" else t34[t34["split"] == split]
        for borrow in BORROWS:
            s = spec_stats(sub.copy(), borrow)
            rows.append({"split": split, "borrow_annual": borrow, **s})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_PRIMARY, index=False)

    for split in ["discovery", "validation"]:
        print(f"\n--- {split.upper()} ---")
        for r in res[res.split == split].itertuples():
            print(f"  borrow {r.borrow_annual:>4.0%}: n={r.n}  mean={r.mean:+.2%} "
                  f"[{r.mean_ci_lo:+.2%},{r.mean_ci_hi:+.2%}]  median={r.median:+.2%} "
                  f"[{r.median_ci_lo:+.2%},{r.median_ci_hi:+.2%}]  hit={r.hit_rate:.0%}  "
                  f"naive_t={r.naive_t:+.2f} (independence-assuming)")

    # ---- the decision, by the pre-registered rule, on validation @10% borrow ----
    v10 = res[(res.split == "validation") & (res.borrow_annual == 0.10)].iloc[0]
    passes = (v10["mean"] > 0) and (v10["mean_ci_lo"] > 0)
    print("\n" + "=" * 72)
    print("DECISION RULE (verbatim): the edge \"exists and is plausibly capturable")
    print("in T3/T4\" iff the validation-period hedged mean is profitable to the")
    print("short with the 95% bootstrap CI excluding zero at the 10% borrow")
    print("sensitivity.")
    print(f"\nVALIDATION @ 10% borrow: mean {v10['mean']:+.2%}, "
          f"95% CI [{v10['mean_ci_lo']:+.2%}, {v10['mean_ci_hi']:+.2%}], n={int(v10['n'])}")
    print(f"\nVERDICT: {'PASS — edge exists and is plausibly capturable in T3/T4'
          if passes else
          'FAIL — no systematic edge after costs; the fade is real but only'
          ' tradeable when listing structure is special (SPCX-style float/lockup'
          ' calendars)'}")

    # ---- exploratory entry x exit x tier grid (report ALL cells) ----
    grid = []
    for tier in ["T1", "T2", "T3", "T4"]:
        for e in ENTRIES:
            for x in EXITS:
                c = f"hedged_e{e}_x{x}"
                for split in ["discovery", "validation"]:
                    v = pd.to_numeric(
                        panel.loc[(panel.tier == tier) & (panel.split == split), c],
                        errors="coerce").dropna()
                    grid.append({"tier": tier, "entry": e, "exit": x, "split": split,
                                 "n": len(v),
                                 "median_hedged": v.median() if len(v) else np.nan,
                                 "mean_hedged": v.mean() if len(v) else np.nan,
                                 "hit_rate": (v > 0).mean() if len(v) else np.nan,
                                 "flag_small_n": len(v) < 25})
    gdf = pd.DataFrame(grid)
    gdf.to_csv(OUT_GRID, index=False)
    n_cells = len(gdf)
    print(f"\nExploratory grid written: {n_cells} cells examined "
          f"(tier x entry x exit x split) — multiple-testing count for the report; "
          f"{int(gdf.flag_small_n.sum())} cells flagged n<25.")
    print(f"\nWrote:\n  {OUT_PRIMARY}\n  {OUT_GRID}")


if __name__ == "__main__":
    main()
