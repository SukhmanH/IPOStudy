"""
make_figs.py - the three figures for the ANT report (static PNG; each has a
table twin in results/).

  figs/fig1_t4_median_path_with_ant_calendar.png
  figs/fig2_ant_price_target_fan.png
  figs/fig3_live_check_2025_26.png

Palette: validated reference slots (blue #2a78d6 = T4/ANT, orange #eb6834 =
the one highlighted comparison), muted gray for context, hairline solid grid.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as c  # noqa: E402
from price_targets import ant_state, load_events  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONTEXT = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9, "legend.labelcolor": INK2,
    "font.family": "sans-serif", "lines.solid_capstyle": "round",
})


def _paths() -> pd.DataFrame:
    p = pd.read_csv(c.DATA / "cohort_rel_paths.csv.gz")
    return p.pivot(index="day", columns="ticker", values="rel")


def fig1(ev: pd.DataFrame, P: pd.DataFrame) -> None:
    hist = ev[ev["role"] == "historical"]
    t4 = P[[t for t in hist[hist["tier"] == "T4"]["ticker"] if t in P]].iloc[:251] * 100
    t3 = P[[t for t in hist[hist["tier"] == "T3"]["ticker"] if t in P]].iloc[:251] * 100
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    # ANT unlock window shading (tranches 2..12: day 17 -> day 121)
    ax.axvspan(17, 121, color=MUTED, alpha=0.07, lw=0)
    ax.text(69, 57, "ANT staggered-unlock window (days 17-121): up to 8.9x the IPO\nshare count becomes sellable; dates after day 49 assume FY2026\nresults ~Mar 4, 2027",
            ha="center", va="bottom", fontsize=8, color=INK2)
    ax.fill_between(t4.index, t4.quantile(.25, axis=1), t4.quantile(.75, axis=1), color=BLUE, alpha=0.10, lw=0,
                    label="T4 interquartile range")
    ax.plot(t4.index, t4.median(axis=1), color=BLUE, lw=2, label=f"T4 median (> $10B; n={t4.shape[1]})")
    ax.plot(t3.index, t3.median(axis=1), color=ORANGE, lw=2, label=f"T3 median ($5-10B; n={t3.shape[1]})")
    ax.axhline(100, color=AXIS, lw=0.8)
    for d, lab in [(20, "Day 20 = Dec 18, 2026: projected Nasdaq-100 index buying\nat the close = the parent study's +20 short entry"),
                   (180, "Day 180 = Aug 11, 2027:\nprimary-spec exit")]:
        ax.axvline(d, color=INK2, lw=0.8, alpha=0.6)
        ax.text(d + 2, 134, lab, fontsize=8, color=INK2, va="top")
    for d in [17, 30, 39, 49, 72, 121]:
        ax.plot([d], [68], marker="v", ms=5, color=INK2)
    ax.text(124, 68, "lockup tranches 2-12", fontsize=8, color=INK2, va="center")
    m4 = t4.median(axis=1)
    for d in [60, 120, 180, 250]:
        ax.plot([d], [m4[d]], marker="o", ms=6, color=BLUE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{m4[d] - 100:+.0f}%", (d, m4[d]), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 252)
    ax.set_ylim(55, 136)
    ax.set_xlabel("Trading days after listing (day 0 = first close)")
    ax.set_ylabel("Price relative to QQQ (day-0 close = 100)")
    ax.set_title("Mega-IPOs drift down vs QQQ after listing; ANT's unlocks land in the steepest stretch")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86))
    fig.text(0.01, 0.005, "Historical T3/T4 traditional IPOs 2010-2025 (parent universe, SPACs/direct listings excluded). "
             "QQQ-neutral: (1+stock)/(1+QQQ). Data: Yahoo Finance, pulled " + str(c.DATA_PULL_DATE.date()) + ".",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(c.FIGS / "fig1_t4_median_path_with_ant_calendar.png")
    plt.close(fig)


def fig2(ev: pd.DataFrame, P: pd.DataFrame) -> None:
    st = ant_state()
    hist = ev[(ev["role"] == "historical") & (ev["tier"] == "T4")].set_index("ticker")
    T = P[[t for t in hist.index if t in P]].iloc[:251]
    if st["state"] == "day0_final":
        anchor, lab = st["close0"], f"anchored on ANT's day-0 close \\${st['close0']:.2f}"
        lvl = T * anchor
    elif st["state"] == "day0_intraday_provisional":
        anchor, lab = st["close0"], f"PROVISIONAL: anchored on ANT intraday \\${st['close0']:.2f} ({st['as_of_et']})"
        lvl = T * anchor
    else:
        anchor, lab = c.ANT_OFFER, "anchored on the \\$83 offer (ANT not yet trading): each comparable's own day-1 pop + path"
        lvl = T.mul(1 + hist.loc[T.columns, "pop_close"], axis=1) * c.ANT_OFFER
    days = [c.ant_calendar_day(k) for k in range(len(lvl))]
    q = {k: lvl.quantile(v, axis=1).to_numpy() for k, v in
         [("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90)]}
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    ax.fill_between(days, q["p10"], q["p90"], color=BLUE, alpha=0.07, lw=0, label="10th-90th percentile")
    ax.fill_between(days, q["p25"], q["p75"], color=BLUE, alpha=0.16, lw=0, label="25th-75th percentile")
    ax.plot(days, q["p50"], color=BLUE, lw=2, label="Median (base case)")
    ax.axhline(c.ANT_OFFER, color=INK2, lw=0.9)
    ax.text(days[-1], c.ANT_OFFER + 1.5, "IPO price $83", ha="right", va="bottom", fontsize=8.5, color=INK2)
    for k in c.TARGET_HORIZONS:
        x, y = days[k], q["p50"][k]
        ax.plot([x], [y], marker="o", ms=6, color=BLUE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"${y:.0f}\n+{k}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
    lock = pd.read_csv(c.LOCKUP)
    ymin = max(0, np.nanmin(q["p10"]) * 0.85)
    for r in lock.itertuples():
        d = pd.Timestamp(r.date)
        if d <= days[-1] and r.tranche > 1:
            ax.plot([d], [ymin + 2], marker="v", ms=4 + 5 * (r.max_shares_m / 3000), color=INK2,
                    alpha=0.9 if str(r.date_is_assumed) != "True" else 0.45)
    ax.text(days[3], ymin + 5, "lockup tranches (marker size = shares; faded = date assumed)", fontsize=7.5,
            color=INK2, va="bottom")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(ymin, np.nanmax(q["p90"]) * 1.08)
    ax.set_xlim(days[0], days[-1] + pd.Timedelta(days=5))
    ax.set_ylabel("ANT price, USD (assumes QQQ flat)")
    ax.set_title("ANT study-implied price range: base case slides below the $83 IPO price by mid-2027")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.005, f"Base rates from {T.shape[1]} historical T4 IPOs (> \\$10B at listing, 2010-2025), "
             + lab + ".\nNot a fundamental valuation. A QQQ move of +/-10% scales every level by the same %. "
             "Exact values: results/ant_price_targets.csv, ant_price_fan_daily.csv.",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(c.FIGS / "fig2_ant_price_target_fan.png")
    plt.close(fig)
    pd.DataFrame({"date": [d.date() for d in days], "trading_day": range(len(days)), **q}) \
      .to_csv(c.RESULTS / "ant_price_fan_daily.csv", index=False, float_format="%.2f")


def fig3(ev: pd.DataFrame, P: pd.DataFrame) -> None:
    hist = ev[ev["role"] == "historical"]
    live = ev[ev["role"] == "live"]
    t4 = P[[t for t in hist[hist["tier"] == "T4"]["ticker"] if t in P]].iloc[:251] * 100
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    ax.fill_between(t4.index, t4.quantile(.25, axis=1), t4.quantile(.75, axis=1), color=BLUE, alpha=0.10, lw=0,
                    label="Historical T4 interquartile range")
    ax.plot(t4.index, t4.median(axis=1), color=BLUE, lw=2, label="Historical T4 median")
    first = True
    for t in live["ticker"]:
        if t == "SPCX" or t not in P:
            continue
        s = P[t].dropna() * 100
        ax.plot(s.index, s.values, color=CONTEXT, lw=1, alpha=0.9,
                label=f"Other 2025-26 T3/T4 IPOs (n={len(live) - 1})" if first else None)
        first = False
    s = P["SPCX"].dropna() * 100
    ax.plot(s.index, s.values, color=ORANGE, lw=2, label="SpaceX (SPCX), listed Jun 12, 2026", zorder=6)
    ax.plot([s.index[-1]], [s.values[-1]], marker="o", ms=6, color=ORANGE, mec=SURFACE, mew=2, zorder=7)
    ax.annotate(f"SPCX at day {s.index[-1]}: {s.values[-1] - 100:+.0f}% vs QQQ", (s.index[-1], s.values[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=8.5, color=INK)
    imin = int(np.argmin(s.values))
    ax.annotate(f"{s.values[imin] - 100:+.0f}% (Aug 5)", (s.index[imin], s.values[imin]), xytext=(6, -10),
                textcoords="offset points", fontsize=8.5, color=INK)
    ax.axvline(15, color=INK2, lw=0.8, alpha=0.6)
    ax.text(16, 200, "SPCX joins\nNasdaq-100\n(day 15)", fontsize=8, color=INK2, va="top")
    ax.axhline(100, color=AXIS, lw=0.8)
    ax.set_xlim(0, 262)
    ax.set_ylim(25, 205)
    ax.set_xlabel("Trading days after listing")
    ax.set_ylabel("Price relative to QQQ (day-0 close = 100)")
    ax.set_title("Out-of-sample: 2025-26 mega-IPOs faded at least as much as history")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.005, "Live cohort = T3/T4 IPOs listed Oct-2025..Jul-2026 (parent 'too recent' names + INIO, JMKE). "
             "16 of 19 were below QQQ-neutral par by day 60.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(c.FIGS / "fig3_live_check_2025_26.png")
    plt.close(fig)


def main():
    ev = load_events()
    P = _paths()
    fig1(ev, P)
    fig2(ev, P)
    fig3(ev, P)
    print("figures written to", c.FIGS)


if __name__ == "__main__":
    main()
