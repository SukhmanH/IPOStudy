"""
entry_today.py - is there still an edge if you buy ANT Jul-16-2027 puts on
2026-11-24 (trading day 3, spot $79.33) instead of day 2?

Physical (study) distributions of S_T at expiry (day 162):
  BASE      forward view: S_T = S (1 + rel_i[3->162]) M_j   (61 T4 comparables x
            re-centred QQQ factor, 8%/yr). Supported by the data: the day-3 fade
            does not predict the forward fade (slope t = 0.05).
  CATCH-UP  ANT converges to the T4 path anchored on its $97.17 day-0 close:
            S_T = 97.17 (QQQ_now/QQQ_day0) (1 + rel_i[0->162]) M_j.
            Not supported by the data, but the conservative case after a fast drop.
  EARLY-FADERS (exploratory) T3+T4 comparables already >= 10% below QQQ by day 3
            (n = 22): S_T = S (1 + rel_i[3->162]) M_j.
Market quotes: broker snapshot 2026-11-24 (buy at ask, sell at bid).
Outputs: results/entry_today_single_puts.csv, results/entry_today_spreads.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as c
import options_quant as oq

SNAP = c.DATA / "ant_put_chain_2027-07-16_snapshot_2026-11-24.csv"
S, T, R = 79.33, 234 / 365, oq.R
CLOSE0, QQQ_DAY0, QQQ_NOW = 97.17, 645.04, 637.00
DAY_NOW, DAY_EXP = 3, 162


def dists():
    f3 = oq.cohort_forward(DAY_NOW, DAY_EXP)
    f0 = oq.cohort_forward(0, DAY_EXP)
    f03 = oq.cohort_forward(0, DAY_NOW)
    M = oq.market_factor(DAY_EXP - DAY_NOW)
    t4 = f3[f3["tier"] == "T4"]
    base = S * (1 + t4["rel"].to_numpy())[:, None] * M[None, :]
    c0 = f0[f0["tier"] == "T4"]
    catch = CLOSE0 * (QQQ_NOW / QQQ_DAY0) * (1 + c0["rel"].to_numpy())[:, None] * M[None, :]
    early = f3.merge(f03[["ticker", "rel"]].rename(columns={"rel": "rel03"}), on="ticker")
    early = early[early["rel03"] <= -0.10]
    ef = S * (1 + early["rel"].to_numpy())[:, None] * M[None, :]
    return dict(base=base, catchup=catch, early=ef), t4["quarter"].to_numpy(), len(early)


def ev(ST, pay, cost):
    p = pay(ST)
    return np.exp(-R * T) * p.mean() / cost - 1, float((p > cost * np.exp(R * T)).mean())


def main():
    ch = pd.read_csv(SNAP)
    ch["mid"] = (ch["bid"] + ch["ask"]) / 2
    ch["iv"] = [oq.implied_vol(m, S, k, T, R, True) for m, k in zip(ch["mid"], ch["strike"])]
    D, qtr, n_early = dists()
    q = ch.set_index("strike")

    def row(name, khi, klo, cost):
        pay = (lambda x: oq.payoff_put(x, khi)) if klo is None else (lambda x: oq.payoff_spread(x, khi, klo))
        out = dict(structure=name, cost=cost, breakeven=khi - cost,
                   max_gain=(khi - klo - cost) if klo is not None else np.nan)
        for k, ST in D.items():
            e, w = ev(ST, pay, cost)
            out[f"ev_{k}"], out[f"pwin_{k}"] = e, w
        Rb = (pay(D["base"]) / cost - 1).ravel()
        out["kelly_base"] = oq.kelly(Rb)[0]
        e_i = pay(D["base"]).mean(axis=1)
        boot = [(np.exp(-R * T) * e_i[idx].mean() - cost) / cost for idx in oq.block_boot_idx(qtr, 3000)]
        out["ev_base_ci90"] = f"[{np.percentile(boot, 5):+.2f}, {np.percentile(boot, 95):+.2f}]"
        out["p_ev_le_0_boot"] = float(np.mean(np.array(boot) <= 0))
        return out

    singles = [row(f"Long {k:g} put", k, None, q.loc[k, "ask"]) for k in q.index]
    spreads = []
    for hi in q.index:
        for lo in q.index:
            if lo < hi:
                cost = q.loc[hi, "ask"] - q.loc[lo, "bid"]
                if 0.25 < cost < hi - lo:
                    r_ = row(f"{hi:g}/{lo:g} spread", hi, lo, cost)
                    r_["min_leg_volume"] = int(min(q.loc[hi, "volume"], q.loc[lo, "volume"]))
                    spreads.append(r_)
    s1, s2 = pd.DataFrame(singles), pd.DataFrame(spreads)
    s1.to_csv(c.RESULTS / "entry_today_single_puts.csv", index=False, float_format="%.4f")
    s2.to_csv(c.RESULTS / "entry_today_spreads.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(f"S = {S}, T = {T:.4f}; early-fader comparables n = {n_early}")
    print("IV by strike:", dict(zip(ch["strike"], ch["iv"].round(3))))
    for k, ST in D.items():
        print(f"  {k:8s} median S_T ${np.median(ST):.2f}  P(S_T < S) {np.mean(ST < S):.2f}")
    cols = ["structure", "cost", "breakeven", "max_gain", "ev_base", "ev_base_ci90", "p_ev_le_0_boot", "pwin_base",
            "ev_catchup", "pwin_catchup", "ev_early", "kelly_base"]
    print("\nSINGLE PUTS (buy at ask)")
    print(s1[cols].round(3).to_string(index=False))
    liq = s2[s2["min_leg_volume"] >= 100]
    print("\nLIQUID SPREADS (both legs >= 100 contracts today) by base EV, cost >= $3")
    print(liq[liq.cost >= 3].sort_values("ev_base", ascending=False)[cols + ["min_leg_volume"]].head(12)
          .round(3).to_string(index=False))
    print("\nNAMED CANDIDATES")
    named = ["90/60 spread", "80/55 spread", "80/60 spread", "85/60 spread", "80/70 spread", "85/75 spread",
             "95/80 spread", "75/60 spread"]
    print(s2[s2.structure.isin(named)][cols + ["min_leg_volume"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
