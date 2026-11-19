"""
price_targets.py - study-implied ANT price targets + the ANT event calendar.

These are BASE-RATE targets: the empirical distribution of what comparable
IPOs did, mapped onto ANT's prices and trading calendar. They are not a
fundamental valuation and not a forecast of the market.

Cohort: historical T4 (> $10B at listing, n~61; ANT's tier and the legacy
Study-1 population). T3+T4 (the pre-registered pool) and T4 x high pre-listing
VIX (ANT's regime) are reported as sensitivities, never as the headline.

QQQ convention: targets are QQQ-NEUTRAL (they assume QQQ is flat from the
anchor date). Event i contributes  anchor x (1 + rel_h,i)  where
rel = (1+stock)/(1+QQQ) - 1. A different market path scales every target
1:1 - e.g. QQQ +10% => targets x 1.10 (sensitivity rows provided).

Anchors (all three reported):
  OFFER   $83 x (event's own close0/offer) x (1+rel_h)  - usable before ANT opens
  CLOSE0  ANT day-0 close x (1+rel_h)                    - once day 0 is final
  HIGH0   ANT day-0 high  x (1+rel_hi_h)                 - legacy Study-1 metric
plus a scenario grid over hypothetical day-0 closes (day-1 pop does not
predict the later fade in this cohort - see results/conditioning.csv - so
targets scale ~linearly with the day-0 close).

Outputs: results/ant_status.csv, results/ant_calendar.csv,
         results/ant_price_targets.csv, results/ant_scenarios_by_day0_close.csv,
         results/ant_primary_trade_view.csv, results/tail_risk.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as c

QS = [0.10, 0.25, 0.50, 0.75, 0.90]
QNAMES = ["p10", "p25", "p50", "p75", "p90"]
FD_SHARES_B = c.ANT_MCAP_FD_B / c.ANT_OFFER          # ~18.07B fully diluted shares
RUN_RATE_REV_B = 81.0                                  # Oct-2026 run-rate (CNBC)


def load_events() -> pd.DataFrame:
    ev = pd.read_csv(c.RESULTS / "cohort_event_table.csv", keep_default_na=False)
    for col in ev.columns:
        if col not in ("ticker", "name", "role", "tier", "listing", "quarter", "era", "split", "adr"):
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    ev["adr"] = ev["adr"].astype(str) == "True"
    return ev


def ant_state() -> dict:
    """Read data/ant_prices.csv (written by fetch_cohort.py) and classify."""
    st = dict(state="not_yet_trading", open0=np.nan, high0=np.nan, low0=np.nan, close0=np.nan,
              volume0=np.nan, as_of_et="", note="Yahoo shows only a placeholder bar (O=0, H=L=C=$83, V=0)")
    p = c.DATA / "ant_prices.csv"
    if not p.exists():
        return st
    h = pd.read_csv(p, index_col=0, parse_dates=True)
    if len(h) == 0:
        return st
    real = h[~h["placeholder"].astype(bool)] if "placeholder" in h else h
    real = real[real.index >= c.ANT_LISTING]
    if len(real) == 0:
        st["as_of_et"] = str(h.get("fetched_at_et", pd.Series([""])).iloc[-1])
        return st
    d0 = real.iloc[0]
    fetched = pd.Timestamp(str(h["fetched_at_et"].iloc[-1])) if "fetched_at_et" in h else pd.Timestamp.now()
    final = (len(real) > 1) or (fetched.normalize() > real.index[0]) or (fetched.hour >= 16 and fetched.minute >= 15) \
        or fetched.hour >= 17
    st.update(state="day0_final" if final else "day0_intraday_provisional",
              open0=float(d0["Open"]), high0=float(d0["High"]), low0=float(d0["Low"]), close0=float(d0["Close"]),
              volume0=float(d0["Volume"]), as_of_et=fetched.strftime("%Y-%m-%d %H:%M ET"),
              note=("day-0 bar final" if final else "intraday snapshot - day-0 high/close not final; rerun after 16:15 ET"))
    st["latest_close"] = float(real["Close"].iloc[-1])
    st["latest_day"] = len(real) - 1
    return st


def quantile_row(values: np.ndarray, scale: float) -> dict:
    v = values[~np.isnan(values)]
    out = {k: scale * np.quantile(v, q) for k, q in zip(QNAMES, QS)} if len(v) else {k: np.nan for k in QNAMES}
    out["mean"] = scale * v.mean() if len(v) else np.nan
    out["n"] = len(v)
    out["prob_below_offer"] = float((scale * v < c.ANT_OFFER).mean()) if len(v) else np.nan
    return out


def build_calendar(ev: pd.DataFrame) -> pd.DataFrame:
    hist = ev[(ev["role"] == "historical") & (ev["tier"] == "T4")]
    paths = pd.read_csv(c.DATA / "cohort_rel_paths.csv.gz")
    paths = paths[paths["ticker"].isin(hist["ticker"])]
    med = paths.groupby("day")["rel"].median()
    lock = pd.read_csv(c.LOCKUP)
    rows = [
        (0, "Listing day (day 0); up to 315M insider shares sellable", False),
        (6, "7th trading day: Nasdaq-100 fast-entry rank date (projected)", False),
        (15, "Nasdaq-100 inclusion announcement after the close (projected)", False),
        (16, "Quiet period ends (~25 calendar days): first sell-side price targets", False),
        (20, "Nasdaq-100 index buying at this close (projected) | quad witching | parent primary-spec ENTRY (+20 close)", False),
        (21, "Nasdaq-100 inclusion effective at the open (projected)", False),
    ]
    for r in lock.itertuples():
        if r.tranche == 1:
            continue
        k = c.ant_trading_day_of(r.date)
        rows.append((k, f"Lockup tranche {r.tranche}: up to {r.max_shares_m:,}M shares ({r.trigger})"
                         + (" [date ASSUMED - results date not announced]" if str(r.date_is_assumed) == "True" else ""),
                     str(r.date_is_assumed) == "True"))
    rows += [(h, f"Study horizon +{h}" + (" (parent primary-spec EXIT)" if h == 180 else ""), False)
             for h in [60, 120, 180, 250]]
    cal = pd.DataFrame(rows, columns=["trading_day", "event", "date_assumed"]).sort_values(["trading_day", "event"])
    cal["date"] = [c.ant_calendar_day(int(k)).date().isoformat() for k in cal["trading_day"]]
    cal["T4_median_rel_level"] = [med.get(int(k), np.nan) for k in cal["trading_day"]]
    cum = 0
    cum_list = []
    lock_days = {c.ant_trading_day_of(r.date): r.max_shares_m for r in lock.itertuples()}
    for k in cal["trading_day"]:
        cum_list.append(sum(v for d, v in lock_days.items() if d <= k))
    cal["cum_unlockable_shares_m"] = cum_list
    cal["cum_unlock_x_ipo_shares"] = cal["cum_unlockable_shares_m"] / 980.0
    cal = cal[["trading_day", "date", "event", "date_assumed", "cum_unlockable_shares_m",
               "cum_unlock_x_ipo_shares", "T4_median_rel_level"]]
    cal.to_csv(c.RESULTS / "ant_calendar.csv", index=False)
    return cal


def main():
    ev = load_events()
    hist = ev[ev["role"] == "historical"]
    vix_cut_hi = hist["vix_pre21"].quantile(2 / 3)
    cohorts = {
        "T4": hist[hist["tier"] == "T4"],
        "T3+T4": hist,
        "T4 x high-VIX": hist[(hist["tier"] == "T4") & (hist["vix_pre21"] > vix_cut_hi)],
    }
    st = ant_state()
    pd.DataFrame([st]).to_csv(c.RESULTS / "ant_status.csv", index=False)
    print("ANT state:", st)

    cal = build_calendar(ev)
    print("\nANT CALENDAR")
    print(cal.round(3).to_string(index=False))

    rows = []
    for cname, g in cohorts.items():
        for h in c.TARGET_HORIZONS:
            date = c.ant_calendar_day(h).date().isoformat()
            base = dict(cohort=cname, horizon=h, date=date)
            rows.append({**base, "anchor": "OFFER $83 (pre-open, unconditional on day 1)", "anchor_price": c.ANT_OFFER,
                         **quantile_row(g[f"po_rel_{h}"].to_numpy(dtype=float), c.ANT_OFFER)})
            if pd.notna(st["close0"]):
                lab = "CLOSE0 " + ("final" if st["state"] == "day0_final" else "PROVISIONAL intraday")
                rows.append({**base, "anchor": lab, "anchor_price": st["close0"],
                             **quantile_row(1 + g[f"rel_{h}"].to_numpy(dtype=float), st["close0"])})
                rows.append({**base, "anchor": "HIGH0 (legacy day-1-high metric) " +
                             ("final" if st["state"] == "day0_final" else "PROVISIONAL"), "anchor_price": st["high0"],
                             **quantile_row(1 + g[f"rel_hi_{h}"].to_numpy(dtype=float), st["high0"])})
    tg = pd.DataFrame(rows)
    tg["implied_fd_mcap_b_at_p50"] = tg["p50"] * FD_SHARES_B
    tg["p50_x_runrate_rev"] = tg["implied_fd_mcap_b_at_p50"] / RUN_RATE_REV_B
    tg.to_csv(c.RESULTS / "ant_price_targets.csv", index=False)
    print("\nANT PRICE TARGETS (QQQ-neutral)")
    print(tg[["cohort", "anchor", "horizon", "date", "n", "p10", "p25", "p50", "p75", "p90", "prob_below_offer",
              "implied_fd_mcap_b_at_p50"]].round(2).to_string(index=False))

    # ---- scenario grid over hypothetical day-0 closes (T4 cohort)
    g = cohorts["T4"]
    sc = []
    for close0 in [75, 83, 90, 100, 110, 125, 140, 166]:
        for h in c.TARGET_HORIZONS:
            q = quantile_row(1 + g[f"rel_{h}"].to_numpy(dtype=float), close0)
            sc.append(dict(day0_close=close0, day1_pop=close0 / c.ANT_OFFER - 1, horizon=h,
                           date=c.ant_calendar_day(h).date().isoformat(), **q))
    sc = pd.DataFrame(sc)
    sc.to_csv(c.RESULTS / "ant_scenarios_by_day0_close.csv", index=False)
    print("\nSCENARIOS (T4, QQQ-neutral): median target [P25, P75] and P(below $83)")
    piv = sc.assign(cell=sc.apply(lambda r: f"${r.p50:.0f} [{r.p25:.0f}-{r.p75:.0f}] {r.prob_below_offer:.0%}", axis=1)) \
            .pivot(index="day0_close", columns="horizon", values="cell")
    print(piv.to_string())

    # ---- the parent's primary trade, mapped onto ANT's calendar
    pv = []
    for cname, gg in cohorts.items():
        v = gg["rel_e20_x180"].dropna().to_numpy()
        x = gg["hedged_e20_x180"].dropna().to_numpy()
        pv.append(dict(cohort=cname, n=len(v), entry_date=c.ant_calendar_day(20).date().isoformat(),
                       exit_date=c.ant_calendar_day(180).date().isoformat(),
                       **{f"rel_{k}": np.quantile(v, q) for k, q in zip(QNAMES, QS)},
                       hedged_mean_gross=x.mean(), hedged_median_gross=np.median(x),
                       hedged_mean_net_10pct_borrow=x.mean() - 0.10 * (160 / 252),
                       hit_rate=(x > 0).mean(), share_loss_gt_25pct=(x < -0.25).mean(),
                       share_loss_gt_50pct=(x < -0.50).mean(), worst=x.min()))
    pv = pd.DataFrame(pv)
    pv.to_csv(c.RESULTS / "ant_primary_trade_view.csv", index=False)
    print("\nPRIMARY TRADE VIEW (short ANT at the +20 close, cover at +180; QQQ hedge)")
    print(pv.round(3).to_string(index=False))

    # ---- tail risk: who ran against the short?
    tail = hist[hist["hedged_e20_x180"] < -0.25][["ticker", "name", "tier", "listing", "mcap_b", "pop_close",
                                                    "hedged_e20_x180", "x_180"]].sort_values("hedged_e20_x180")
    tail.to_csv(c.RESULTS / "tail_risk.csv", index=False)
    print("\nRIGHT-TAIL EVENTS (primary-spec hedged loss worse than -25%):")
    print(tail.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
