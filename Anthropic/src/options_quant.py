"""
options_quant.py - price ANT Jul-16-2027 puts against the IPO-fade study.

Market side (risk-neutral, Q)
  * Black-Scholes put  P = K e^{-rT} N(-d2) - S N(-d1),
        d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt T),  d2 = d1 - sigma sqrt T
  * Implied vol per strike from the mid: American (CRR binomial, 400 steps) and
    European BS. No borrow/dividend yield is assumed, so if ANT is hard to borrow
    these IVs are upper bounds (puts carry the borrow fee).
  * Risk-neutral CDF (Breeden-Litzenberger): Q(S_T < K) = e^{rT} dP/dK. Raw mids
    are noisy (two small butterfly violations), so the derivative is taken on a
    fitted smile sigma(k) = a + b k + c k^2, k = ln(K/F):
        Q(S_T < K) = N(-d2) + e^{rT} * Vega(K) * dsigma/dK

Study side (physical, P)
  * For each historical T4 IPO i (n=61): QQQ-neutral relative return from
    trading day 2 (= ANT today) to day 162 (= the Jul-16-2027 expiry):
        rel_i = (1 + stock_i) / (1 + QQQ_i) - 1
  * Market factor M_j: every rolling 160-session QQQ gross return 2010-2026,
    re-centred to E[M] = e^{mu T} with mu = 8%/yr (neutral drift, not the
    2010s bull market). Physical terminal price:
        S_T(i,j) = S0 (1 + rel_i) M_j     (beta = 1; cohort median beta 1.05)
  * Sensitivities: QQQ flat (M = 1); raw historical paths; and the exploratory
    'weak day-0 close' subgroup (T3+T4 IPOs whose day-0 close was >5% below
    the open, like ANT's -12.9%).

Per structure (buy at the ask, sell at the bid):
    EV      = e^{-rT} E_P[payoff] - cost             (expected profit, $/share)
    EV%     = EV / cost                               (return on premium)
    P(win)  = P(payoff > cost e^{rT})
    Kelly   f* = argmax_f E_P[ ln(1 + f R) ],  R = payoff/cost - 1
  Estimation risk: moving-block bootstrap over listing quarters (the study's
  dependence-honest resampling) -> 90% CI for EV% and the 10th-pct Kelly.

Outputs: results/options_single_puts.csv, results/options_put_spreads.csv,
         results/options_summary.csv, figs/fig4_ant_options_P_vs_Q.png
Run:     python Anthropic/src/options_quant.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

import common as c

# ----------------------------------------------------------------- inputs
SNAP = c.DATA / "ant_put_chain_2027-07-16_snapshot.csv"
S0 = 87.48                         # spot shown on the chain screenshot (2026-11-23 ~10:14)
QUOTE_DATE = pd.Timestamp("2026-11-23")
EXPIRY = pd.Timestamp("2027-07-16")
R = 0.041                          # 13-week T-bill ~4.15% (^IRX); continuous approx
MU_MKT = 0.08                      # physical market drift for the base case (r + ~4% ERP)
DAY_NOW = 2                        # ANT trading day of the quote (day 0 = 2026-11-19)
T = (EXPIRY - QUOTE_DATE).days / 365.0
N_BOOT = 4000
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------- pricing
def bs_put(S, K, T, r, sig):
    sig = np.maximum(sig, 1e-9)
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def crr_american_put(S, K, T, r, sig, n=400):
    dt = T / n
    u = np.exp(sig * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    j = np.arange(n + 1)
    v = np.maximum(K - S * u ** (n - j) * d ** j, 0.0)
    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        v = np.maximum(disc * (p * v[:-1] + (1 - p) * v[1:]), K - S * u ** (i - j) * d ** j)
    return v[0]


def implied_vol(price, S, K, T, r, american=True):
    f = (lambda s: crr_american_put(S, K, T, r, s) - price) if american else \
        (lambda s: bs_put(S, K, T, r, s) - price)
    try:
        return optimize.brentq(f, 0.03, 4.0, xtol=1e-6)
    except ValueError:
        return np.nan


def greeks(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    pdf = stats.norm.pdf(d1)
    return dict(delta=-stats.norm.cdf(-d1), gamma=pdf / (S * sig * np.sqrt(T)),
                vega_1pt=S * pdf * np.sqrt(T) / 100,
                theta_day=(-S * pdf * sig / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365,
                q_itm_bs=stats.norm.cdf(-d2))


# ----------------------------------------------------------------- study distribution
def cohort_forward(day_from: int, day_to: int):
    """QQQ-neutral rel and raw stock return from day_from to day_to for each
    historical event, plus quarter (for the block bootstrap) and day-0 open->close."""
    ev = pd.read_csv(c.RESULTS / "cohort_event_table.csv", keep_default_na=False)
    w = pd.read_csv(c.DATA / "cohort_prices_window.csv.gz", parse_dates=["Date"])
    b = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    rows = []
    for r_ in ev[ev["role"] == "historical"].itertuples():
        g = w[w["ticker"] == r_.ticker].sort_values("day").set_index("Date")
        if len(g) <= day_to:
            continue
        adj = g["AdjClose"].to_numpy()
        q = b["QQQ"].reindex(g.index).ffill().to_numpy()
        s = adj[day_to] / adj[day_from] - 1
        qq = q[day_to] / q[day_from] - 1
        rows.append(dict(ticker=r_.ticker, tier=r_.tier, quarter=r_.quarter,
                         rel=(1 + s) / (1 + qq) - 1, s=s, q=qq,
                         oc0=g["Close"].iloc[0] / g["Open"].iloc[0] - 1))
    return pd.DataFrame(rows)


def market_factor(n_days: int) -> np.ndarray:
    b = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    q = b["QQQ"].dropna().to_numpy()
    gross = q[n_days:] / q[:-n_days]
    return gross * np.exp(MU_MKT * n_days / 252) / gross.mean()


def payoff_put(ST, K):
    return np.maximum(K - ST, 0.0)


def payoff_spread(ST, Khi, Klo):
    return np.clip(Khi - ST, 0.0, Khi - Klo)


def kelly(R_):
    """f* = argmax E[ln(1+fR)] on a grid (R >= -1)."""
    grid = np.linspace(0, 0.99, 199)
    g = [np.mean(np.log1p(f * R_)) for f in grid]
    k = int(np.argmax(g))
    return grid[k], g[k]


def block_boot_idx(quarters: np.ndarray, n_boot: int, block: int = c.BLOCK_LEN):
    qs = np.array(sorted(pd.unique(quarters)))
    nq = len(qs)
    members = [np.where(quarters == q)[0] for q in qs]
    starts_all = np.arange(max(1, nq - block + 1))
    n_blocks = int(np.ceil(nq / block))
    for _ in range(n_boot):
        starts = RNG.choice(starts_all, size=n_blocks, replace=True)
        yield np.concatenate([members[s + j] for s in starts for j in range(block) if s + j < nq])


def evaluate(name, cost, pay_fn, ST_by_event, ST_flat, ST_raw, quarters, extra=None):
    """ST_by_event: (n_events, n_mkt) physical terminal prices."""
    pay = pay_fn(ST_by_event)                    # (n_i, n_j)
    e_i = pay.mean(axis=1)                       # E[payoff | event i]
    ev = np.exp(-R * T) * e_i.mean() - cost
    R_all = (pay / cost - 1).ravel()
    f_star, g_star = kelly(R_all)
    boot = []
    for idx in block_boot_idx(quarters, N_BOOT):
        boot.append((np.exp(-R * T) * e_i[idx].mean() - cost) / cost)
    lo, hi = np.percentile(boot, [5, 95])
    pnl = (pay.ravel() * np.exp(-R * T) - cost) * 100
    out = dict(structure=name, cost=cost, ev_per_share=ev, ev_pct=ev / cost, ev_pct_ci90_lo=lo, ev_pct_ci90_hi=hi,
               p_win=float((pay.ravel() > cost * np.exp(R * T)).mean()),
               p_max_loss=float((pay.ravel() <= 1e-12).mean()),
               pnl_p10=np.percentile(pnl, 10), pnl_p50=np.percentile(pnl, 50), pnl_p90=np.percentile(pnl, 90),
               kelly_full=f_star, kelly_growth=g_star,
               ev_pct_qqq_flat=(np.exp(-R * T) * pay_fn(ST_flat).mean() - cost) / cost,
               ev_pct_raw_hist=(np.exp(-R * T) * pay_fn(ST_raw).mean() - cost) / cost)
    if extra:
        out.update(extra)
    return out


def main():
    chain = pd.read_csv(SNAP)
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2
    n_days = int(c.ant_trading_day_of(EXPIRY)) - DAY_NOW
    print(f"S0={S0}  T={T:.4f}y ({(EXPIRY - QUOTE_DATE).days} cal days, {n_days} trading days)  r={R}")

    # ---------- market side
    chain["iv_amer"] = [implied_vol(m, S0, k, T, R, True) for m, k in zip(chain["mid"], chain["strike"])]
    chain["iv_euro"] = [implied_vol(m, S0, k, T, R, False) for m, k in zip(chain["mid"], chain["strike"])]
    chain["iv_amer_bid"] = [implied_vol(m, S0, k, T, R, True) for m, k in zip(chain["bid"], chain["strike"])]
    chain["iv_amer_ask"] = [implied_vol(m, S0, k, T, R, True) for m, k in zip(chain["ask"], chain["strike"])]
    F = S0 * np.exp(R * T)
    k = np.log(chain["strike"] / F)
    wts = 1 / (chain["ask"] - chain["bid"]).clip(lower=0.05)
    coef = np.polyfit(k, chain["iv_amer"], 2, w=wts)          # sigma(k) = c2 k^2 + c1 k + c0
    sig_fit = np.polyval(coef, k)
    dsig_dK = np.polyval(np.polyder(coef), k) / chain["strike"]
    gk = [greeks(S0, K, T, R, s) for K, s in zip(chain["strike"], sig_fit)]
    G = pd.DataFrame(gk)
    vega_raw = G["vega_1pt"] * 100
    chain["q_itm_smile"] = G["q_itm_bs"] + np.exp(R * T) * vega_raw * dsig_dK
    for col in ["delta", "gamma", "vega_1pt", "theta_day"]:
        chain[col] = [greeks(S0, K, T, R, s)[col] for K, s in zip(chain["strike"], chain["iv_amer"])]

    # ---------- study side
    fwd = cohort_forward(DAY_NOW, DAY_NOW + n_days)
    t4 = fwd[fwd["tier"] == "T4"].reset_index(drop=True)
    weak = fwd[fwd["oc0"] < -0.05].reset_index(drop=True)         # exploratory sensitivity
    M = market_factor(n_days)
    print(f"T4 events with a complete day {DAY_NOW}->{DAY_NOW + n_days} window: {len(t4)}; "
          f"market-factor windows: {len(M)} (E[M]={M.mean():.4f}, sd={M.std():.3f})")
    ST = S0 * (1 + t4["rel"].to_numpy())[:, None] * M[None, :]
    ST_flat = S0 * (1 + t4["rel"].to_numpy())
    ST_raw = S0 * (1 + t4["s"].to_numpy())
    ST_weak = S0 * (1 + weak["rel"].to_numpy())[:, None] * M[None, :]
    qtr = t4["quarter"].to_numpy()

    chain["p_itm_study"] = [float((ST < K).mean()) for K in chain["strike"]]
    chain["p_itm_weakclose"] = [float((ST_weak < K).mean()) for K in chain["strike"]]

    # ---------- single puts
    singles = []
    for r_ in chain.itertuples():
        K = r_.strike
        singles.append(evaluate(f"Long {K:g} put", r_.ask, lambda x, K=K: payoff_put(x, K), ST, ST_flat, ST_raw, qtr,
                                extra=dict(strike=K, iv_mid=r_.iv_amer, delta=r_.delta, vega_1pt=r_.vega_1pt,
                                           theta_day=r_.theta_day, q_itm=r_.q_itm_smile, p_itm=r_.p_itm_study,
                                           breakeven=K - r_.ask,
                                           ev_pct_weakclose=(np.exp(-R * T) * payoff_put(ST_weak, K).mean() - r_.ask) / r_.ask)))
    singles = pd.DataFrame(singles)

    # ---------- vertical put spreads (buy Khi at ask, sell Klo at bid)
    spreads = []
    rows = list(chain.itertuples())
    for hi in rows:
        for lo in rows:
            if lo.strike >= hi.strike:
                continue
            cost = hi.ask - lo.bid
            width = hi.strike - lo.strike
            if cost <= 0.25 or cost >= width:
                continue
            fn = (lambda x, a=hi.strike, b_=lo.strike: payoff_spread(x, a, b_))
            spreads.append(evaluate(f"{hi.strike:g}/{lo.strike:g} put spread", cost, fn, ST, ST_flat, ST_raw, qtr,
                                    extra=dict(k_long=hi.strike, k_short=lo.strike, width=width,
                                               max_gain=width - cost, reward_risk=(width - cost) / cost,
                                               breakeven=hi.strike - cost,
                                               net_delta=hi.delta - lo.delta, net_vega_1pt=hi.vega_1pt - lo.vega_1pt,
                                               net_theta_day=hi.theta_day - lo.theta_day,
                                               ev_pct_weakclose=(np.exp(-R * T) * fn(ST_weak).mean() - cost) / cost)))
    spreads = pd.DataFrame(spreads)

    chain.to_csv(c.RESULTS / "options_chain_analytics.csv", index=False, float_format="%.5f")
    singles.to_csv(c.RESULTS / "options_single_puts.csv", index=False, float_format="%.5f")
    spreads.to_csv(c.RESULTS / "options_put_spreads.csv", index=False, float_format="%.5f")

    # ---------- distributions summary
    pct = [5, 10, 25, 50, 75, 90, 95]
    summ = pd.DataFrame({
        "percentile": pct,
        "study_P_base": np.percentile(ST.ravel(), pct),
        "study_P_qqq_flat": np.percentile(ST_flat, pct),
        "study_P_raw_hist": np.percentile(ST_raw, pct),
        "study_P_weak_close": np.percentile(ST_weak.ravel(), pct),
    })
    # risk-neutral quantiles by inverting the smile-adjusted CDF on a fine grid
    Kg = np.linspace(20, 200, 1801)
    kg = np.log(Kg / F)
    sg = np.clip(np.polyval(coef, kg), 0.05, 3.0)
    d1 = (np.log(S0 / Kg) + (R + 0.5 * sg ** 2) * T) / (sg * np.sqrt(T))
    d2 = d1 - sg * np.sqrt(T)
    vega_g = S0 * stats.norm.pdf(d1) * np.sqrt(T)
    cdf_q = np.clip(stats.norm.cdf(-d2) + np.exp(R * T) * vega_g * np.polyval(np.polyder(coef), kg) / Kg, 0, 1)
    cdf_q = np.maximum.accumulate(cdf_q)
    summ["market_Q_smile"] = [Kg[np.searchsorted(cdf_q, p / 100)] if (cdf_q >= p / 100).any() else np.nan for p in pct]
    summ.to_csv(c.RESULTS / "options_summary.csv", index=False, float_format="%.2f")
    pd.DataFrame({"price": Kg, "cdf_market_Q": cdf_q,
                  "cdf_study_P": [float((ST < x).mean()) for x in Kg]}) \
      .iloc[::10].to_csv(c.RESULTS / "options_cdf_P_vs_Q.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\nSMILE fit sigma(k) = %.3f k^2 %+.3f k %+.3f" % tuple(coef))
    print(chain[["strike", "bid", "ask", "mid", "iv_amer_bid", "iv_amer", "iv_amer_ask", "iv_euro", "delta", "vega_1pt",
                 "theta_day", "q_itm_smile", "p_itm_study", "p_itm_weakclose"]].round(3).to_string(index=False))
    print("\nTERMINAL PRICE DISTRIBUTIONS (Jul-16-2027)")
    print(summ.round(2).to_string(index=False))
    print("\nSINGLE PUTS (buy at ask) - study base case")
    print(singles[["structure", "cost", "breakeven", "iv_mid", "q_itm", "p_itm", "ev_per_share", "ev_pct",
                   "ev_pct_ci90_lo", "ev_pct_ci90_hi", "p_win", "kelly_full", "ev_pct_qqq_flat", "ev_pct_raw_hist",
                   "ev_pct_weakclose"]].round(3).to_string(index=False))
    print("\nTOP PUT SPREADS by EV% (cost >= $2)")
    sp = spreads[spreads.cost >= 2].sort_values("ev_pct", ascending=False)
    cols = ["structure", "cost", "max_gain", "reward_risk", "breakeven", "ev_per_share", "ev_pct", "ev_pct_ci90_lo",
            "ev_pct_ci90_hi", "p_win", "p_max_loss", "kelly_full", "kelly_growth", "ev_pct_qqq_flat", "ev_pct_raw_hist",
            "ev_pct_weakclose", "net_delta", "net_vega_1pt", "net_theta_day"]
    print(sp[cols].head(12).round(3).to_string(index=False))
    print("\nTOP PUT SPREADS by Kelly growth")
    print(spreads.sort_values("kelly_growth", ascending=False)[cols].head(8).round(3).to_string(index=False))


# ============================================================================ risk layer
CANDIDATES = [("Long 90 put", 90, None), ("90/60 put spread", 90, 60), ("90/70 put spread", 90, 70),
              ("85/60 put spread", 85, 60)]


def _price_struct(S, Tleft, khi, klo, sig_fn, iv_shift=0.0):
    v = bs_put(S, khi, Tleft, R, max(sig_fn(khi) + iv_shift, 0.05))
    if klo is not None:
        v -= bs_put(S, klo, Tleft, R, max(sig_fn(klo) + iv_shift, 0.05))
    return v


def risk_layer():
    chain = pd.read_csv(c.RESULTS / "options_chain_analytics.csv")
    q = chain.set_index("strike")
    F = S0 * np.exp(R * T)
    coef = np.polyfit(np.log(chain["strike"] / F), chain["iv_amer"], 2,
                      w=1 / (chain["ask"] - chain["bid"]).clip(lower=0.05))
    sig_fn = lambda K: float(np.polyval(coef, np.log(K / F)))
    n_days = int(c.ant_trading_day_of(EXPIRY)) - DAY_NOW
    fwd = cohort_forward(DAY_NOW, DAY_NOW + n_days)
    t4 = fwd[fwd["tier"] == "T4"].reset_index(drop=True)
    M = market_factor(n_days)
    Msub = RNG.choice(M, 300, replace=False)
    rel = t4["rel"].to_numpy()
    qtr = t4["quarter"].to_numpy()

    # ---- bootstrap Kelly (estimation risk): resample events by listing-quarter blocks
    rows = []
    for name, khi, klo in CANDIDATES:
        cost = q.loc[khi, "ask"] - (q.loc[klo, "bid"] if klo else 0.0)
        pay = lambda ST: payoff_put(ST, khi) - (payoff_put(ST, klo) if klo else 0.0)
        ks = []
        for b_i, idx in enumerate(block_boot_idx(qtr, 600)):
            ST = S0 * (1 + rel[idx])[:, None] * Msub[None, :]
            ks.append(kelly((pay(ST) / cost - 1).ravel())[0])
        ks = np.array(ks)
        ST_full = S0 * (1 + rel)[:, None] * M[None, :]
        f_full = kelly((pay(ST_full) / cost - 1).ravel())[0]
        rows.append(dict(structure=name, cost=cost, kelly_full=f_full, kelly_boot_p10=np.percentile(ks, 10),
                         kelly_boot_p25=np.percentile(ks, 25), kelly_boot_median=np.median(ks),
                         share_boot_kelly_zero=float((ks == 0).mean()),
                         quarter_kelly=f_full / 4))
    kb = pd.DataFrame(rows)
    kb.to_csv(c.RESULTS / "options_kelly_bootstrap.csv", index=False, float_format="%.4f")

    # ---- mark-to-market stress grid at three dates
    dates = {"Dec 18 2026 (day 20, index buying)": pd.Timestamp("2026-12-18"),
             "Mar 8 2027 (~1.7B unlock)": pd.Timestamp("2027-03-08"),
             "May 17 2027 (3.0B unlock)": pd.Timestamp("2027-05-17")}
    stress = []
    for name, khi, klo in CANDIDATES:
        cost = q.loc[khi, "ask"] - (q.loc[klo, "bid"] if klo else 0.0)
        for dlab, dte in dates.items():
            Tleft = (EXPIRY - dte).days / 365
            for S in [60, 70, 80, 87.48, 95, 100, 110, 120]:
                for dv in [-0.15, 0.0, +0.10]:
                    v = _price_struct(S, Tleft, khi, klo, sig_fn, dv)
                    stress.append(dict(structure=name, date=dlab, spot=S, iv_shift_pts=round(dv * 100),
                                       value=v, pnl_per_contract=(v - cost) * 100, pnl_pct=v / cost - 1))
    st = pd.DataFrame(stress)
    st.to_csv(c.RESULTS / "options_stress_grid.csv", index=False, float_format="%.3f")

    # ---- study-anchored levels (QQQ-neutral, day-0 close $97.17 anchor; QQQ flat)
    ev = pd.read_csv(c.RESULTS / "cohort_event_table.csv", keep_default_na=False)
    t4names = ev[(ev["role"] == "historical") & (ev["tier"] == "T4")]["ticker"]
    P = pd.read_csv(c.DATA / "cohort_rel_paths.csv.gz").pivot(index="day", columns="ticker", values="rel")
    P = P[[t for t in t4names if t in P]]
    close0 = 97.17
    lv = []
    for d in [20, 40, 60, 90, 121, 162]:
        lev = P.loc[d]
        lv.append(dict(trading_day=d, date=c.ant_calendar_day(d).date().isoformat(),
                       p10=close0 * lev.quantile(.10), p25=close0 * lev.quantile(.25), median=close0 * lev.median(),
                       p75=close0 * lev.quantile(.75), p90=close0 * lev.quantile(.90)))
    levels = pd.DataFrame(lv)
    levels.to_csv(c.RESULTS / "options_study_path_levels.csv", index=False, float_format="%.2f")

    # ---- beta hedge (isolate the QQQ-relative fade the study measures)
    beta = 1.05                                   # T4 cohort median beta to QQQ, days 2-162
    qqq_px = 652.68
    hedge = []
    for name, khi, klo in CANDIDATES:
        d_net = q.loc[khi, "delta"] - (q.loc[klo, "delta"] if klo else 0.0)
        hedge.append(dict(structure=name, net_delta=d_net, dollar_delta=d_net * 100 * S0,
                          qqq_shares_per_contract=beta * abs(d_net) * 100 * S0 / qqq_px))
    hg = pd.DataFrame(hedge)
    hg.to_csv(c.RESULTS / "options_beta_hedge.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 250)
    print("\nKELLY UNDER ESTIMATION RISK (block bootstrap over listing quarters, 600 draws)")
    print(kb.round(3).to_string(index=False))
    print("\nSTUDY PATH LEVELS for ANT ($, QQQ flat; anchored on the $97.17 day-0 close)")
    print(levels.round(2).to_string(index=False))
    print("\nSTRESS: 90/60 spread and 90 put, P&L % of cost")
    for nm in ["90/60 put spread", "Long 90 put"]:
        sub = st[(st.structure == nm)]
        piv = sub.pivot_table(index=["date", "iv_shift_pts"], columns="spot", values="pnl_pct")
        print(f"\n{nm}"); print(piv.round(2).to_string())
    print("\nBETA HEDGE"); print(hg.round(3).to_string(index=False))
    # theta drag until the pre-registered entry (Dec 18)
    days_to_entry = (pd.Timestamp("2026-12-18") - QUOTE_DATE).days
    for name, khi, klo in CANDIDATES:
        th = q.loc[khi, "theta_day"] - (q.loc[klo, "theta_day"] if klo else 0.0)
        cost = q.loc[khi, "ask"] - (q.loc[klo, "bid"] if klo else 0.0)
        print(f"theta drag to Dec 18 ({days_to_entry} cal days), {name}: {th * days_to_entry:+.2f}/sh "
              f"({th * days_to_entry / cost:+.1%} of cost)")


if __name__ == "__main__":
    main()
    risk_layer()
