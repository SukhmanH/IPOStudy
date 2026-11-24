"""
mark_to_market.py - what the ANT Jul-16-2027 put trades from 2026-11-23 are
worth on 2026-11-24, from the broker's own quotes on both days.

  entry   : 2026-11-23 snapshot (S = $87.48), buy at ask / sell at bid
            (and at mid, for a patient limit order)
  today   : 2026-11-24 snapshot (S = $79.33)
            * mid mark        = long mid - short mid
            * close-out value = long bid - short ask (what you get selling now)
  attribution (first-order, per share): delta x dS + 1/2 gamma x dS^2
            + vega x dIV + theta x 1 day, from Black-Scholes at each strike's
            implied vol on both days (r = 4.1%, no borrow - verified below).
  borrow  : put-call parity on today's simultaneous call/put snapshots
            F = K + e^{rT} (C - P),   q_implied = r - ln(F/S)/T

Run:  python Anthropic/src/mark_to_market.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import common as c
import options_quant as oq

D0, D1 = c.DATA / "ant_put_chain_2027-07-16_snapshot.csv", c.DATA / "ant_put_chain_2027-07-16_snapshot_2026-11-24.csv"
CALLS1 = c.DATA / "ant_call_chain_2027-07-16_snapshot_2026-11-24.csv"
S_0, S_1, S_CALL = 87.48, 79.33, 79.18
T_0, T_1 = 235 / 365, 234 / 365
R = oq.R
QQQ_0, QQQ_1 = 652.68, 637.00          # QQQ when each snapshot was taken (approx., intraday)
STRUCTS = [("Long 90 put", 90, None), ("90/60 put spread", 90, 60), ("90/70 put spread", 90, 70),
           ("85/60 put spread", 85, 60), ("90/80 put spread", 90, 80), ("95/85 put spread", 95, 85),
           ("105/85 put spread", 105, 85)]


def leg_iv(df, S, T):
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["iv"] = [oq.implied_vol(m, S, k, T, R, True) for m, k in zip(df["mid"], df["strike"])]
    g = [oq.greeks(S, k, T, R, s) for k, s in zip(df["strike"], df["iv"])]
    for col in ("delta", "gamma", "vega_1pt", "theta_day"):
        df[col] = [x[col] for x in g]
    return df.set_index("strike")


def main():
    p0, p1 = leg_iv(pd.read_csv(D0), S_0, T_0), leg_iv(pd.read_csv(D1), S_1, T_1)
    dS = S_1 - S_0
    rows = []
    for name, hi, lo in STRUCTS:
        legs = [(hi, +1)] + ([(lo, -1)] if lo else [])
        entry_nat = sum(w * (p0.loc[k, "ask"] if w > 0 else p0.loc[k, "bid"]) for k, w in legs)
        entry_mid = sum(w * p0.loc[k, "mid"] for k, w in legs)
        now_mid = sum(w * p1.loc[k, "mid"] for k, w in legs)
        now_exit = sum(w * (p1.loc[k, "bid"] if w > 0 else p1.loc[k, "ask"]) for k, w in legs)
        delta = sum(w * p0.loc[k, "delta"] for k, w in legs)
        gamma = sum(w * p0.loc[k, "gamma"] for k, w in legs)
        vega = sum(w * p0.loc[k, "vega_1pt"] * (p1.loc[k, "iv"] - p0.loc[k, "iv"]) * 100 for k, w in legs)
        theta = sum(w * p0.loc[k, "theta_day"] for k, w in legs)
        rows.append(dict(
            structure=name, entry_paid=entry_nat, entry_mid=entry_mid, value_mid_now=now_mid,
            value_closeout_now=now_exit, pnl_mid_per_contract=(now_mid - entry_nat) * 100,
            pnl_closeout_per_contract=(now_exit - entry_nat) * 100, pnl_mid_pct=now_mid / entry_nat - 1,
            pnl_closeout_pct=now_exit / entry_nat - 1, max_value=(hi - lo) if lo else np.nan,
            attr_delta=delta * dS * 100, attr_gamma=0.5 * gamma * dS ** 2 * 100, attr_vega=vega * 100,
            attr_theta=theta * 100))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "options_mtm_2026-11-24.csv", index=False, float_format="%.3f")

    # implied borrow from today's (near-simultaneous) call and put snapshots
    calls = pd.read_csv(CALLS1).set_index("strike")
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    S_avg = (S_1 + S_CALL) / 2
    par = []
    for k in calls.index.intersection(p1.index):
        F = k + np.exp(R * T_1) * (calls.loc[k, "mid"] - p1.loc[k, "mid"])
        par.append(dict(strike=k, implied_forward=F, implied_q=R - np.log(F / S_avg) / T_1))
    par = pd.DataFrame(par)
    par.to_csv(c.RESULTS / "options_parity_borrow_2026-11-24.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"ANT ${S_0} -> ${S_1} ({S_1 / S_0 - 1:+.1%});  QQQ {QQQ_0} -> {QQQ_1} ({QQQ_1 / QQQ_0 - 1:+.1%})")
    print("\nIMPLIED VOL by strike (American, mid):")
    print(pd.DataFrame({"iv_nov23": p0["iv"], "iv_nov24": p1["iv"]}).dropna().round(3).T.to_string())
    print("\nMARK-TO-MARKET (entry = Nov-23 natural price; $ per 1 contract = 100 shares)")
    print(out.round(2).to_string(index=False))
    print("\nPUT-CALL PARITY implied borrow (q): median %.2f%%/yr, range %.2f%% to %.2f%%"
          % (100 * par["implied_q"].median(), 100 * par["implied_q"].min(), 100 * par["implied_q"].max()))
    hedge_units = 3.22
    print(f"\nQQQ hedge for one 90/60 spread ({hedge_units} sh): {hedge_units * (QQQ_1 - QQQ_0):+.0f} $")


if __name__ == "__main__":
    main()
