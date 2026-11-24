"""
make_figs.py -- static figures for REPORT.md (reads results/*.csv and the
event panel).  Palette and mark specs follow the dataviz reference palette:
categorical slots 1-3 (blue, orange, aqua; validated all-pairs, light mode),
blue<->red diverging poles for gains/losses, hairline recessive grids, 2px
lines, direct labels on lines (aqua is <3:1 contrast, so labels are required).

Run:  python short_gainers/src/make_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA, RES, FIGS = HERE / "data", HERE / "results", HERE / "figs"
FIGS.mkdir(exist_ok=True)

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
POS, NEG = "#2a78d6", "#e34948"
SPLIT_DATE = pd.Timestamp("2019-01-01")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "font.family": "DejaVu Sans",
})


def pct(ax, axis="y", dec=1):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:.{dec}f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def split_marker(ax, y):
    ax.axvline(SPLIT_DATE, color=BASE, lw=1)
    ax.text(SPLIT_DATE, y, "  validation →", color=MUTED, fontsize=9, va="top")
    ax.text(SPLIT_DATE, y, "← discovery  ", color=MUTED, fontsize=9, va="top", ha="right")


def fig_cumulative() -> None:
    d = pd.read_csv(RES / "daily_portfolio_returns.csv", parse_dates=["signal_date"]).set_index("signal_date")
    panels = [("All top-10 gainers (H1, pre-registered)", d["primary_gross"], d["primary_n"]),
              ("Skip gainers with an earnings report (H2 rule, chosen on 2010–18)", d["rule_gross"], d["rule_n"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for ax, (title, gross, n_day) in zip(axes, panels):
        series = {"No costs": gross, "0.25% per trade": gross - 0.10 * n_day * 0.0025,
                  "0.50% per trade": gross - 0.10 * n_day * 0.0050}
        for (name, s_), col in zip(series.items(), [C1, C2, C3]):
            cs = s_.cumsum()
            ax.plot(cs.index, cs.values, color=col, label=name)
            ax.annotate(f"{cs.iloc[-1] * 100:+.0f}%", (cs.index[-1], cs.iloc[-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.axhline(0, color=BASE, lw=1)
        pct(ax, dec=0)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlim(d.index[0], d.index[-1] + pd.Timedelta(days=700))
        ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(2010, 2027, 4)])
        ax.set_xticklabels([str(y) for y in range(2010, 2027, 4)])
    for ax in axes:
        split_marker(ax, axes[0].get_ylim()[1])
    axes[0].set_ylabel("Cumulative P&L, % of capital\n(10% per trade, additive, SPY-hedged)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Short at the next open, cover at the close: before and after trading costs",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "fig1_cumulative_pnl.png", dpi=160)
    plt.close(fig)


def load_primary() -> pd.DataFrame:
    ev = pd.read_parquet(DATA / "events.parquet")
    p = ev[(ev["excl_reason"] == "eligible") & (ev["universe"] == "U5") & (ev["rank"] <= 10) & ev["oc1"].notna()]
    return p.assign(s_h=-p["oc1"] + p["spy_oc1"])


def fig_features() -> None:
    B = pd.read_csv(RES / "B_feature_profiles.csv")
    panels = [("F15_earnings", "Earnings report on t−1 or t", ["yes", "no"]),
              ("F1_gain", "Gain on signal day", None),
              ("F14_entry_gap", "Next-day opening gap", None),
              ("F7_rsi14", "RSI(14) on signal day", None),
              ("F2_clv", "Close location on signal day", None),
              ("F11_spy_above200", "SPY above its 200-day average", ["no", "yes"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey=False)
    for ax, (f, title, order) in zip(axes.flat, panels):
        sub = B[B["feature"] == f]
        labs = order or sorted(sub["bucket"].unique(), key=bucket_key)
        x = np.arange(len(labs))
        for k, (sp, col) in enumerate([("discovery", C1), ("validation", C2)]):
            s = sub[sub["split"] == sp].set_index("bucket").reindex(labs)
            ax.bar(x + (k - 0.5) * 0.26, s["mean"].values, width=0.24, color=col,
                   label=f"{sp.capitalize()} {'2010–18' if sp == 'discovery' else '2019–26'}")
        ax.axhline(0, color=BASE, lw=1)
        ax.axhline(0.005, color=MUTED, lw=1)
        ax.set_xticks(x)
        nd = sub[sub["split"] == "discovery"].set_index("bucket").reindex(labs)["n"]
        nv = sub[sub["split"] == "validation"].set_index("bucket").reindex(labs)["n"]
        ax.set_xticklabels([f"{pretty(l)}\n{kfmt(a)}/{kfmt(b)}" for l, a, b in zip(labs, nd, nv)], fontsize=8)
        ax.set_title(title, fontsize=10.5)
        pct(ax, dec=1)
        ax.grid(axis="x", visible=False)
    axes.flat[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Mean gross return per short trade by signal-day feature (hedged, before costs)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.text(0.01, 0.935, "Under each bucket: trades in discovery/validation. The grey line marks a 0.50% round-trip cost.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "fig2_feature_profiles.png", dpi=160)
    plt.close(fig)


def kfmt(v) -> str:
    v = 0 if v != v else int(v)
    return f"{v / 1000:.1f}k" if v >= 1000 else str(v)


def bucket_key(b: str):
    try:
        return float(b.strip("[(").split(",")[0].replace("-inf", "-1e18"))
    except ValueError:
        return 0.0


def pretty(b: str) -> str:
    if b in ("yes", "no"):
        return b
    lo, hi = b.strip("[)").split(",")
    lo, hi = lo.strip(), hi.strip()

    def f(v):
        v = float(v)
        return f"{v * 100:.0f}%" if abs(v) < 5 else f"{v:.0f}"
    if lo == "-inf":
        return f"<{f(hi)}"
    if hi == "inf":
        return f"≥{f(lo)}"
    return f"{f(lo)}–{f(hi)}"


def fig_horizon() -> None:
    A = pd.read_csv(RES / "A_entry_exit_grid.csv")
    a = A[A["entry"] == "open_t1"].copy()
    a["k"] = a["exit"].str.extract(r"close_t(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for sp, col in [("discovery", C1), ("validation", C2)]:
        s = a[a["split"] == sp].sort_values("k")
        ax.plot(s["k"], s["mean"], color=col, marker="o", ms=6, mec=SURF, mew=2,
                label=f"Gross, {sp} {'2010–18' if sp == 'discovery' else '2019–26'}")
        ax.annotate(f"{s['mean'].iloc[-1] * 100:+.1f}%", (s["k"].iloc[-1], s["mean"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
    ks = np.array([1, 2, 5, 10, 20])
    for b, lab in [(0.30, "cost at 30%/yr borrow"), (1.00, "cost at 100%/yr borrow")]:
        cost = 0.0025 + (ks - 1) * b / 252
        ax.plot(ks, cost, color=MUTED, lw=1)
        ax.annotate(lab, (ks[-1], cost[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=MUTED)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"t+{k}" for k in ks])
    ax.set_xlim(0.5, 26)
    pct(ax, dec=1)
    ax.set_xlabel("Exit at the close of day …  (entry at the open of t+1)")
    ax.set_ylabel("Mean hedged return per trade")
    ax.set_title("The fade keeps going after day 2 — but so does the borrow bill")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_holding_horizon.png", dpi=160)
    plt.close(fig)


def fig_per_year() -> None:
    F = pd.read_csv(RES / "F_per_year.csv")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = F["year"].values
    for k, (col_, colr, lab) in enumerate([("year_return_base", C1, "All top-10 gainers"),
                                           ("rule_year_return_base", C2, "Skip earnings-day gainers")]):
        ax.bar(x + (k - 0.5) * 0.36, F[col_], width=0.34, color=colr, label=lab)
    ax.axhline(0, color=BASE, lw=1)
    ax.axvline(2018.5, color=BASE, lw=1)
    ax.text(2018.5, ax.get_ylim()[1], "  validation →", color=MUTED, fontsize=9, va="top")
    ax.text(2018.5, ax.get_ylim()[1], "← discovery  ", color=MUTED, fontsize=9, va="top", ha="right")
    pct(ax, dec=0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in x], fontsize=8.5)
    ax.set_ylabel("Year P&L, % of capital")
    ax.set_title("P&L by year, net of 0.25% per trade")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig4_per_year.png", dpi=160)
    plt.close(fig)


def fig_screen_bias() -> None:
    D = pd.read_csv(RES / "D_universe_variants.csv")
    D = D[np.isclose(D["cost"], 0.0025)]
    rows = [("NAIVE split-adjusted $5 screen (look-ahead)", "Naive:\n$5+ on split-adjusted price"),
            ("U5 primary (pre-registered)", "Pre-registered:\n$5+ actual traded price"),
            ("POST-HOC U5 & prior close >= $5", "Established (post-hoc):\n$5+ on both t−1 and t")]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(rows))
    for k, (sp, col) in enumerate([("discovery", C1), ("validation", C2)]):
        vals = [D[(D["variant"] == v) & (D["split"] == sp)]["gross_mean_event"].iloc[0] for v, _ in rows]
        bars = ax.bar(x + (k - 0.5) * 0.26, vals, width=0.24, color=col,
                      label=f"{sp.capitalize()} {'2010–18' if sp == 'discovery' else '2019–26'}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0002, f"{v * 100:.2f}%", ha="center",
                    va="bottom", fontsize=8.5, color=INK2)
    for c, lab in [(0.0025, "0.25% base cost"), (0.005, "0.50% stress cost")]:
        ax.axhline(c, color=MUTED, lw=1)
        ax.text(2.42, c + 0.0001, " " + lab, color=MUTED, fontsize=8.5, va="bottom", ha="left")
    ax.set_xlim(-0.55, 3.25)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in rows], fontsize=9)
    pct(ax, dec=2)
    ax.set_ylabel("Mean gross return per trade")
    ax.set_title("Half of the 'edge' in a naive backtest is a look-ahead artifact")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig5_screen_bias.png", dpi=160)
    plt.close(fig)


def fig_tail() -> None:
    p = load_primary()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = p["s_h"].clip(-1.0, 0.6)
    bins = np.linspace(-1.0, 0.6, 81)
    ax.hist(x, bins=bins, color=C1, edgecolor=SURF, linewidth=0.6)
    ax.set_yscale("log")
    ax.axvline(0, color=BASE, lw=1)
    q = p["s_h"].quantile([0.01, 0.5, 0.99])
    for v, lab in [(q[0.01], "1st pct"), (q[0.99], "99th pct")]:
        ax.axvline(v, color=MUTED, lw=1)
        ax.text(v, ax.get_ylim()[1] * 0.6, f" {lab}\n {v * 100:+.0f}%", color=INK2, fontsize=8.5, va="top")
    pct(ax, axis="x", dec=0)
    ax.set_xlabel("Hedged return per short trade (clipped at −100%; worst actual −302%)")
    ax.set_ylabel("Trades (log scale)")
    ax.set_title(f"Many small wins, a few catastrophic squeezes (all {len(p):,} trades, 2010–2026)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_trade_distribution.png", dpi=160)
    plt.close(fig)


def main() -> None:
    fig_cumulative()
    fig_features()
    fig_horizon()
    fig_per_year()
    fig_screen_bias()
    fig_tail()
    print("figures:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
