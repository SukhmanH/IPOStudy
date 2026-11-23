"""
edge_test.py - the pre-registered test of the ANT put-spread edge and the
pre-registered sizing rule. Spec: Anthropic/PREREG_EDGE_TEST.md (commit
4360faf, pushed before this file existed). Nothing here deviates from it;
anything extra is labelled "reported, not gating".

Run:  python Anthropic/src/edge_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import common as c

SNAP = c.DATA / "ant_put_chain_2027-07-16_snapshot.csv"
S0, RF, T = 87.48, 0.041, 235 / 365
DAY_FROM, DAY_TO = 2, 162
N_BOOT, SEED, BLOCK = 20_000, 2026, 4
PRIMARY = "90/60 spread"
EULER = 0.5772156649
F_GRID = np.round(np.arange(0.0, 0.9901, 0.0025), 4)


# ============================================================ data
def load_forward() -> pd.DataFrame:
    ev = pd.read_csv(c.RESULTS / "cohort_event_table.csv", keep_default_na=False)
    w = pd.read_csv(c.DATA / "cohort_prices_window.csv.gz", parse_dates=["Date"])
    b = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    rows = []
    for r in ev.itertuples():
        if r.role not in ("historical", "live"):
            continue
        g = w[w["ticker"] == r.ticker].sort_values("day").set_index("Date")
        if len(g) <= DAY_TO:
            continue
        adj = g["AdjClose"].to_numpy()
        q = b["QQQ"].reindex(g.index).ffill().to_numpy()
        s = adj[DAY_TO] / adj[DAY_FROM] - 1
        qq = q[DAY_TO] / q[DAY_FROM] - 1
        rows.append(dict(ticker=r.ticker, role=r.role, tier=r.tier, quarter=r.quarter, split=r.split,
                         entry=g.index[DAY_FROM], exit=g.index[DAY_TO], rel=(1 + s) / (1 + qq) - 1, s=s, q=qq))
    return pd.DataFrame(rows)


def structures() -> pd.DataFrame:
    ch = pd.read_csv(SNAP)
    out = [dict(name=f"{h.strike:g} put", m_hi=h.strike / S0, m_lo=np.nan, cost=h.ask / S0) for h in ch.itertuples()]
    for h in ch.itertuples():
        for lo in ch.itertuples():
            if lo.strike >= h.strike:
                continue
            cost = h.ask - lo.bid
            if cost <= 0.25 or cost >= h.strike - lo.strike:
                continue
            out.append(dict(name=f"{h.strike:g}/{lo.strike:g} spread", m_hi=h.strike / S0, m_lo=lo.strike / S0,
                            cost=cost / S0))
    return pd.DataFrame(out)


def returns(st: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """R[k, i] = e^{-rT} payoff_k(x_i) / cost_k - 1, x = S_T / S_2."""
    mh = st["m_hi"].to_numpy()[:, None]
    ml = st["m_lo"].to_numpy()[:, None]
    pay = np.maximum(mh - x[None, :], 0) - np.where(np.isnan(ml), 0.0, np.maximum(np.nan_to_num(ml) - x[None, :], 0))
    return np.exp(-RF * T) * pay / st["cost"].to_numpy()[:, None] - 1


# ============================================================ inference
def boot_indices(quarters: np.ndarray, n_boot: int, rng):
    """Moving-block bootstrap over chronologically ordered listing quarters
    (blocks of BLOCK consecutive quarters; parent study's construction)."""
    qs = sorted(pd.unique(quarters))
    nq = len(qs)
    members = [np.where(quarters == q)[0] for q in qs]
    starts_all = np.arange(nq - BLOCK + 1) if nq > BLOCK else np.arange(1)
    n_blocks = int(np.ceil(nq / BLOCK))
    for _ in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        yield np.concatenate([members[s + j] for s in starts for j in range(BLOCK) if s + j < nq])


def test_mean(y: np.ndarray, quarters: np.ndarray, label: str, n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y)
    m = float(np.mean(y))
    out = dict(sample=label, n=n, G=len(pd.unique(quarters)), mean=m, median=float(np.median(y)),
               sd=float(np.std(y, ddof=1)) if n > 1 else np.nan, share_total_loss=float((y <= -0.999).mean()))
    if n < 3:
        return out
    t_iid = m / (out["sd"] / np.sqrt(n))
    out.update(t_iid=t_iid, p_iid=1 - stats.t.cdf(t_iid, n - 1))
    if out["G"] >= 3:
        fit = sm.OLS(y, np.ones((n, 1))).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(quarters)[0]})
        out.update(se_cluster=float(fit.bse[0]), t_cluster=float(fit.tvalues[0]),
                   p_cluster=float(1 - stats.t.cdf(fit.tvalues[0], out["G"] - 1)))
        bm = np.array([y[idx].mean() for idx in boot_indices(quarters, n_boot, rng)])
        out.update(p_boot=float(np.mean(bm - m >= m)), ci90_lo=float(np.percentile(bm, 5)),
                   ci90_hi=float(np.percentile(bm, 95)), lcb90=float(np.percentile(bm, 10)),
                   se_boot=float(bm.std(ddof=1)))
        out["n_eff"] = n * (out["sd"] / np.sqrt(n)) ** 2 / out["se_cluster"] ** 2
    return out


def snooping(Rm: np.ndarray, quarters: np.ndarray, names: list[str]) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    K, n = Rm.shape
    mbar = Rm.mean(1)
    boot = np.empty((N_BOOT, K))
    for b, idx in enumerate(boot_indices(quarters, N_BOOT, rng)):
        boot[b] = Rm[:, idx].mean(1)
    dev = np.sqrt(n) * (boot - mbar)
    omega = dev.std(0, ddof=1)
    tstat = np.sqrt(n) * mbar / omega
    # White (2000) Reality Check
    p_rc = float(np.mean(dev.max(1) >= np.max(np.sqrt(n) * mbar)))
    # Hansen (2005) SPA_c
    thresh = -np.sqrt(2 * np.log(np.log(n)))
    keep_neg = np.where(tstat < thresh, np.sqrt(n) * mbar, 0.0)
    T_spa = max(float(tstat.max()), 0.0)
    T_star = np.maximum(((dev + keep_neg) / omega).max(1), 0.0)
    p_spa = float(np.mean(T_star >= T_spa))
    # single-step max-t (Romano-Wolf / Westfall-Young) adjusted p-values
    maxt = (dev / omega).max(1)
    p_maxt = np.array([np.mean(maxt >= tk) for tk in tstat])
    per = pd.DataFrame(dict(structure=names, mean_R=mbar, t_boot=tstat, p_unadj=[np.mean(dev[:, k] / omega[k] >= tstat[k])
                                                                                for k in range(K)], p_maxt_adj=p_maxt))
    return dict(K=K, n=n, p_reality_check=p_rc, p_spa_c=p_spa, best=names[int(np.argmax(tstat))],
                best_t=float(tstat.max())), per


def deflated_sharpe(y: np.ndarray, sr_all: np.ndarray, n_obs: float) -> dict:
    sr = y.mean() / y.std(ddof=1)
    g3 = stats.skew(y)
    g4 = stats.kurtosis(y, fisher=False)
    N = len(sr_all)
    sr0 = np.sqrt(np.var(sr_all, ddof=1)) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N)
                                              + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))
    den = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return dict(sr_per_trade=sr, sr0_expected_max_of_N=sr0, skew=g3, kurt=g4, n_obs=n_obs,
                psr_vs_0=stats.norm.cdf(sr * np.sqrt(n_obs - 1) / den),
                dsr=stats.norm.cdf((sr - sr0) * np.sqrt(n_obs - 1) / den))


# ============================================================ sizing
def kelly(R: np.ndarray, grid=F_GRID) -> float:
    g = [np.mean(np.log1p(f * R)) for f in grid]
    return float(grid[int(np.argmax(g))])


def rescale(R: np.ndarray, target_mean: float) -> np.ndarray:
    """Scale payoffs (not returns) so mean R = target; keeps max loss at -100%."""
    lam = (1 + target_mean) / (1 + R.mean())
    return lam * (1 + R) - 1


def bayes_kelly(R: np.ndarray, se: float, tau: float | None, n_draw: int = 2000) -> dict:
    m = R.mean()
    if tau is None:                                   # flat prior
        v, mp = se ** 2, m
    else:
        v = 1 / (1 / tau ** 2 + 1 / se ** 2)
        mp = v * m / se ** 2
    mus = np.clip(np.random.default_rng(SEED).normal(mp, np.sqrt(v), n_draw), -0.95, None)
    lam = (1 + mus) / (1 + m)                         # (D,)
    best, bf = -np.inf, 0.0
    for f in F_GRID[F_GRID <= 0.6]:
        val = np.mean(np.log1p(f * (lam[:, None] * (1 + R[None, :]) - 1)))
        if val > best:
            best, bf = val, f
    return dict(tau=tau, post_mean=mp, post_sd=np.sqrt(v), p_edge_positive=1 - stats.norm.cdf(0, mp, np.sqrt(v)),
                f_bayes=bf)


def backtest_paths(ev: pd.DataFrame, R: np.ndarray, f_vec: np.ndarray, n_boot: int, rng) -> dict:
    """Calendar-time backtest: every T4 IPO entered at day 2 with fraction f of
    equity (marked at cost), exited at day 162 with return R (excess of cash).
    Bootstrap = moving blocks of 4 consecutive CALENDAR quarters (2010Q1-2025Q4,
    empty quarters included), events keep their within-quarter timing."""
    qstart = pd.period_range("2010Q1", "2025Q4", freq="Q").to_timestamp()
    nq = len(qstart)
    qi = np.array([qstart.searchsorted(d, side="right") - 1 for d in ev["entry"]])
    off = np.array([(d - qstart[k]).days for d, k in zip(ev["entry"], qi)])
    hold = np.array([(x - d).days for d, x in zip(ev["entry"], ev["exit"])])
    by_q = [np.where(qi == k)[0] for k in range(nq)]

    def run(src_q):
        entries, exits = [], []
        for slot, sq in enumerate(src_q):
            for i in by_q[sq]:
                e = qstart[slot] + pd.Timedelta(days=int(off[i]))
                entries.append((e, len(entries), R[i], e + pd.Timedelta(days=int(hold[i]))))
        steps = sorted([(e, 1, j, r) for e, j, r, _ in entries] + [(x, 0, j, r) for _, j, r, x in entries])
        F = len(f_vec)
        cash = np.ones(F)
        open_ = np.zeros((F, max(1, len(entries))))
        eq_hist, expo = [np.ones(F)], np.zeros(F)
        for d, typ, j, r in steps:
            if typ == 0:
                cash += open_[:, j] * (1 + r)
                open_[:, j] = 0
            else:
                eq = cash + open_.sum(1)
                prem = f_vec * eq
                cash -= prem
                open_[:, j] = prem
                expo = np.maximum(expo, open_.sum(1) / eq)
            eq_hist.append(cash + open_.sum(1))
        E = np.vstack(eq_hist)
        dd = (E / np.maximum.accumulate(E, axis=0) - 1).min(0)
        return E[-1], dd, expo

    hist_final, hist_dd, hist_expo = run(np.arange(nq))
    starts_all = np.arange(nq - BLOCK + 1)
    finals, dds = np.empty((n_boot, len(f_vec))), np.empty((n_boot, len(f_vec)))
    for b in range(n_boot):
        src = np.concatenate([np.arange(s, s + BLOCK) for s in rng.choice(starts_all, nq // BLOCK)])[:nq]
        finals[b], dds[b], _ = run(src)
    years = nq / 4
    return dict(hist_final=hist_final, hist_dd=hist_dd, hist_expo=hist_expo, finals=finals, dds=dds, years=years)


def _tau_label(tau):
    return "flat" if tau is None else f"{tau:.0%}"


# ============================================================ main
def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    fw = load_forward()
    st = structures()
    k_pri = int(np.where(st["name"] == PRIMARY)[0][0])
    t4 = fw[(fw["role"] == "historical") & (fw["tier"] == "T4")].reset_index(drop=True)
    print(f"structures: {len(st)} (13 puts + {len(st) - 13} spreads); T4 events: {len(t4)}; "
          f"primary {PRIMARY}: cost {st.loc[k_pri, 'cost']:.4f} x spot")

    x_mn = (1 + t4["rel"].to_numpy()) * np.exp(RF * T)      # primary: market-neutral
    x_raw = 1 + t4["s"].to_numpy()                            # secondary: unhedged, actual history
    R_mn = returns(st, x_mn)
    R_raw = returns(st, x_raw)
    y = R_mn[k_pri]
    qtr = t4["quarter"].to_numpy()

    # ---------------- gating tests
    rows = [test_mean(y, qtr, "T1/T2 primary: T4 full, market-neutral")]
    for sp in ["discovery", "validation"]:
        msk = (t4["split"] == sp).to_numpy()
        rows.append(test_mean(y[msk], qtr[msk], f"T3 {sp} (T4, market-neutral)"))
    # ---------------- reported, not gating
    rows.append(test_mean(R_raw[k_pri], qtr, "secondary: T4 full, unhedged (actual history)"))
    pool = fw[fw["role"] == "historical"].reset_index(drop=True)
    yp = returns(st.iloc[[k_pri]], (1 + pool["rel"].to_numpy()) * np.exp(RF * T))[0]
    rows.append(test_mean(yp, pool["quarter"].to_numpy(), "T3+T4 pool, market-neutral"))
    live = fw[fw["role"] == "live"].reset_index(drop=True)
    if len(live):
        yl = returns(st.iloc[[k_pri]], (1 + live["rel"].to_numpy()) * np.exp(RF * T))[0]
        rows.append(test_mean(yl, live["quarter"].to_numpy(), "out-of-sample: matured 2025-26 listings (T3+T4)"))
        print("matured live events:", ", ".join(f"{t} ({r:+.0%})" for t, r in zip(live["ticker"], yl)))
    res = pd.DataFrame(rows)
    res.to_csv(c.RESULTS / "edge_test_primary.csv", index=False, float_format="%.5f")
    print("\nMEAN-RETURN TESTS (R = PV return on premium, 90/60 spread)")
    print(res[["sample", "n", "G", "mean", "median", "share_total_loss", "t_iid", "p_iid", "t_cluster", "p_cluster",
               "p_boot", "ci90_lo", "ci90_hi", "n_eff"]].round(4).to_string(index=False))

    snoop, per = snooping(R_mn, qtr, st["name"].tolist())
    per.to_csv(c.RESULTS / "edge_test_snooping.csv", index=False, float_format="%.5f")
    sr_all = R_mn.mean(1) / R_mn.std(1, ddof=1)
    dsr_iid = deflated_sharpe(y, sr_all, len(y))
    dsr_eff = deflated_sharpe(y, sr_all, res.loc[0, "n_eff"])
    print(f"\nDATA-SNOOPING over {snoop['K']} structures: Reality Check p = {snoop['p_reality_check']:.4f}; "
          f"SPA_c p = {snoop['p_spa_c']:.4f}; best = {snoop['best']} (t = {snoop['best_t']:.2f}); "
          f"max-t adjusted p for {PRIMARY} = {per.loc[k_pri, 'p_maxt_adj']:.4f}")
    print(per.sort_values("t_boot", ascending=False).head(8).round(4).to_string(index=False))
    print(f"\nDEFLATED SHARPE (per trade): SR = {dsr_iid['sr_per_trade']:.3f}, expected max SR of {len(sr_all)} "
          f"trials SR0 = {dsr_iid['sr0_expected_max_of_N']:.3f}; PSR(0) = {dsr_iid['psr_vs_0']:.3f}, "
          f"DSR = {dsr_iid['dsr']:.3f} with n = {len(y)};  with n_eff = {dsr_eff['n_obs']:.1f}: "
          f"PSR(0) = {dsr_eff['psr_vs_0']:.3f}, DSR = {dsr_eff['dsr']:.3f}")

    # ---------------- decision rule (verbatim from the pre-registration)
    full, disc, val = res.iloc[0], res.iloc[1], res.iloc[2]
    c1 = (full["p_boot"] < 0.05) and (full["p_cluster"] < 0.05)
    c2 = (val["mean"] > 0) and (val["p_boot"] < 0.10) and (disc["mean"] > 0)
    c3 = snoop["p_spa_c"] < 0.05
    established = bool(c1 and c2 and c3)
    print("\nDECISION RULE")
    print(f"  1. full-sample T1 p = {full['p_boot']:.4f}, T2 p = {full['p_cluster']:.4f}  -> {'PASS' if c1 else 'FAIL'}")
    print(f"  2. validation mean {val['mean']:+.3f} (T1 p = {val['p_boot']:.4f}); discovery mean {disc['mean']:+.3f}"
          f"  -> {'PASS' if c2 else 'FAIL'}")
    print(f"  3. SPA_c p = {snoop['p_spa_c']:.4f}  -> {'PASS' if c3 else 'FAIL'}")
    print(f"  VERDICT: {'EDGE STATISTICALLY ESTABLISHED' if established else 'EDGE NOT STATISTICALLY ESTABLISHED'}")

    # ---------------- power
    sig_eff = full["se_cluster"] * np.sqrt(full["n"])
    n_req = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) * sig_eff / full["mean"]) ** 2
    rate = full["n"] / 16.0
    print(f"\nPOWER: effective per-event sd {sig_eff:.3f}; n for 80% power at one-sided 5% = {n_req:.0f} events "
          f"(have {full['n']}, n_eff {full['n_eff']:.1f}); at ~{rate:.1f} T4 IPOs/yr that is ~{(n_req - full['n']) / rate:.0f} "
          f"more years of data (if new events are as clustered as old ones)")

    # ---------------- sizing (pre-registered)
    f_k = kelly(y)
    lcb = full["lcb90"]
    f_lcb = 0.0 if lcb <= 0 else kelly(rescale(y, lcb))
    bayes = [bayes_kelly(y, full["se_cluster"], tau) for tau in (0.05, 0.10, 0.20, None)]
    rng = np.random.default_rng(SEED)
    f_vec = np.array([0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.065, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30])
    bt = backtest_paths(t4, y, f_vec, 5000, rng)
    bt_raw = backtest_paths(t4, R_raw[k_pri], f_vec, 5000, np.random.default_rng(SEED))
    p_dd = (bt["dds"] < -0.25).mean(0)
    ok = f_vec[p_dd <= 0.05]
    f_dd = float(ok.max()) if len(ok) else 0.0
    grid = pd.DataFrame(dict(
        f=f_vec,
        hist_cagr=bt["hist_final"] ** (1 / bt["years"]) - 1, hist_maxdd=bt["hist_dd"], hist_max_exposure=bt["hist_expo"],
        boot_cagr_p05=np.percentile(bt["finals"] ** (1 / bt["years"]) - 1, 5, axis=0),
        boot_cagr_med=np.median(bt["finals"] ** (1 / bt["years"]) - 1, axis=0),
        boot_cagr_p95=np.percentile(bt["finals"] ** (1 / bt["years"]) - 1, 95, axis=0),
        boot_maxdd_med=np.median(bt["dds"], axis=0), boot_maxdd_p95=np.percentile(bt["dds"], 5, axis=0),
        p_maxdd_gt_25=p_dd, p_maxdd_gt_50=(bt["dds"] < -0.50).mean(0), p_lose_money_16y=(bt["finals"] < 1).mean(0),
        raw_hist_cagr=bt_raw["hist_final"] ** (1 / bt_raw["years"]) - 1, raw_hist_maxdd=bt_raw["hist_dd"],
        raw_p_maxdd_gt_25=(bt_raw["dds"] < -0.25).mean(0),
        raw_boot_cagr_med=np.median(bt_raw["finals"] ** (1 / bt_raw["years"]) - 1, axis=0)))
    grid.to_csv(c.RESULTS / "edge_backtest_grid.csv", index=False, float_format="%.5f")
    f_b10 = [b for b in bayes if b["tau"] == 0.10][0]["f_bayes"]
    rec = min(f_k / 4, f_b10, f_dd) if established else min(f_b10, 0.01)
    sizing = pd.DataFrame([
        dict(rule="f_K plug-in Kelly", f=f_k),
        dict(rule="f_K / 4 (quarter-Kelly)", f=f_k / 4),
        dict(rule=f"f_LCB (Kelly at one-sided 90% LCB of mean R = {lcb:+.3f})", f=f_lcb),
        *[dict(rule=f"f_B Bayesian Kelly, prior tau = {_tau_label(b['tau'])}"
                    f" (posterior mean {b['post_mean']:+.3f}, sd {b['post_sd']:.3f}, P(edge>0) {b['p_edge_positive']:.2f})",
               f=b["f_bayes"]) for b in bayes],
        dict(rule="f_DD: largest f with P(max drawdown > 25%) <= 5% (calendar-time block bootstrap)", f=f_dd),
        dict(rule=("RECOMMENDED = min(f_K/4, f_B(10%), f_DD)  [edge established]" if established else
                   "RECOMMENDED = min(f_B(10%), 1%)  [edge NOT established -> discretionary]"), f=rec),
    ])
    sizing.to_csv(c.RESULTS / "edge_test_sizing.csv", index=False, float_format="%.4f")
    print("\nSIZING (fraction of capital put into premium)")
    print(sizing.round(4).to_string(index=False))
    print("\nCALENDAR-TIME BACKTEST (every T4 IPO 2010-2025, 90/60 analog at ANT's quotes, market-neutral; "
          "5,000 block-bootstrap timelines)")
    print(grid.round(4).to_string(index=False))
    pd.to_pickle(dict(t4=t4, y=y, f_vec=f_vec, bt=bt, established=established, rec=rec),
                 c.DATA / "raw" / "_edge_test_state.pkl")


if __name__ == "__main__":
    main()


# ============================================================ EXPLORATORY (NOT pre-registered)
# Run after the pre-registered verdict. Near-the-money spreads were the only
# structures sign-consistent across discovery, validation and the matured
# 2025-26 listings; this battery is reported as exploratory evidence and as the
# candidate hypothesis for a FUTURE pre-registered test. It does not change the
# pre-registered verdict.
EXPLORE = ["95/85 spread", "90/80 spread", "90/85 spread", "95/80 spread", "100/85 spread", "105/85 spread",
           "90/60 spread"]


def exploratory():
    fw = load_forward()
    st = structures()
    t4 = fw[(fw["role"] == "historical") & (fw["tier"] == "T4")].reset_index(drop=True)
    live = fw[fw["role"] == "live"].reset_index(drop=True)
    qtr = t4["quarter"].to_numpy()
    Rm = returns(st, (1 + t4["rel"].to_numpy()) * np.exp(RF * T))
    Rl = returns(st, (1 + live["rel"].to_numpy()) * np.exp(RF * T))
    Rraw = returns(st, 1 + t4["s"].to_numpy())
    sn = pd.read_csv(c.RESULTS / "edge_test_snooping.csv")
    ch = pd.read_csv(SNAP).set_index("strike")
    disc = (t4["split"] == "discovery").to_numpy()
    f_vec = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08, 0.10])
    rows = []
    for nm in EXPLORE:
        k = int(np.where(st["name"] == nm)[0][0])
        y = Rm[k]
        full = test_mean(y, qtr, nm)
        d_ = test_mean(y[disc], qtr[disc], nm)
        v_ = test_mean(y[~disc], qtr[~disc], nm)
        hi, lo = (float(s_) for s_ in nm.split(" ")[0].split("/"))
        cost = ch.loc[hi, "ask"] - ch.loc[lo, "bid"]
        bt = backtest_paths(t4, y, f_vec, 3000, np.random.default_rng(SEED))
        btr = backtest_paths(t4, Rraw[k], f_vec, 3000, np.random.default_rng(SEED))
        pdd = (bt["dds"] < -0.25).mean(0)
        pddr = (btr["dds"] < -0.25).mean(0)
        b10 = bayes_kelly(y, full["se_cluster"], 0.10)
        rows.append(dict(
            structure=nm, debit=cost, max_gain=(hi - lo) - cost, breakeven=hi - cost,
            long_leg_vol=int(ch.loc[hi, "volume"]), short_leg_vol=int(ch.loc[lo, "volume"]),
            mean_R=full["mean"], p_win=float((y > 0).mean()), p_boot=full["p_boot"], p_cluster=full["p_cluster"],
            ci90=f"[{full['ci90_lo']:+.3f}, {full['ci90_hi']:+.3f}]",
            disc_mean=d_["mean"], disc_p_boot=d_.get("p_boot", np.nan),
            val_mean=v_["mean"], val_p_boot=v_.get("p_boot", np.nan),
            oos_mean=float(Rl[k].mean()), oos_wins=f"{int((Rl[k] > 0).sum())}/{len(Rl[k])}",
            p_maxt_adj=float(sn.loc[k, "p_maxt_adj"]), unhedged_mean=float(Rraw[k].mean()),
            f_kelly=kelly(y), f_bayes_tau10=b10["f_bayes"], p_edge_pos_tau10=b10["p_edge_positive"],
            f_dd_hedged=float(f_vec[pdd <= 0.05].max()) if (pdd <= 0.05).any() else 0.0,
            f_dd_unhedged=float(f_vec[pddr <= 0.05].max()) if (pddr <= 0.05).any() else 0.0))
    out = pd.DataFrame(rows)
    out.to_csv(c.RESULTS / "edge_exploratory_near_money.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print("EXPLORATORY (not pre-registered): near-the-money spreads vs the pre-registered 90/60")
    print(out.round(3).to_string(index=False))


# ============================================================ EXPLORATORY: implementable delta hedge
def _bs_put_delta(S, K, tau, sig):
    """BS put value and delta, vectorised; tau in years (>0)."""
    tau = np.maximum(tau, 1e-6)
    d1 = (np.log(S / K) + (RF + 0.5 * sig ** 2) * tau) / (sig * np.sqrt(tau))
    d2 = d1 - sig * np.sqrt(tau)
    v = K * np.exp(-RF * tau) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return v, -stats.norm.cdf(-d1)


def delta_hedged(names=("95/85 spread", "90/80 spread", "90/85 spread", "100/85 spread", "105/85 spread",
                        "90/60 spread"), beta=1.05, rebalance=5):
    """Replay each historical T4 IPO's REAL daily path from day 2 to day 162:
    hold the spread (bought at ANT's quotes, scaled to the event's day-2 price),
    and a long-QQQ hedge sized to beta*|net delta| (BS delta at each strike's
    current ANT IV, sticky strike), rebalanced every `rebalance` sessions and
    financed at r. Return on premium = (payoff + hedge P&L, PV) / cost - 1."""
    chain = pd.read_csv(c.RESULTS / "options_chain_analytics.csv").set_index("strike")
    w = pd.read_csv(c.DATA / "cohort_prices_window.csv.gz", parse_dates=["Date"])
    b = pd.read_csv(c.DATA / "benchmarks.csv.gz", index_col=0, parse_dates=True)
    fw = load_forward()
    t4 = fw[(fw["role"] == "historical") & (fw["tier"] == "T4")].reset_index(drop=True)
    st = structures()
    paths = {}
    for t in t4["ticker"]:
        g = w[w["ticker"] == t].sort_values("day").set_index("Date")
        S = g["AdjClose"].to_numpy()[DAY_FROM:DAY_TO + 1]
        Q = b["QQQ"].reindex(g.index).ffill().to_numpy()[DAY_FROM:DAY_TO + 1]
        paths[t] = (S / S[0], Q)
    n_steps = DAY_TO - DAY_FROM
    tau = T * (1 - np.arange(n_steps + 1) / n_steps)
    out_rows, Rdh = [], {}
    for nm in names:
        k = int(np.where(st["name"] == nm)[0][0])
        hi, lo = (float(s_) for s_ in nm.split(" ")[0].split("/"))
        mh, ml, cost = hi / S0, lo / S0, st.loc[k, "cost"]
        sh, sl = chain.loc[hi, "iv_amer"], chain.loc[lo, "iv_amer"]
        R = []
        for t in t4["ticker"]:
            S, Q = paths[t]
            hedge_pnl = 0.0
            for j in range(0, n_steps, rebalance):
                j2 = min(j + rebalance, n_steps)
                _, dh = _bs_put_delta(S[j], mh, tau[j], sh)
                _, dl = _bs_put_delta(S[j], ml, tau[j], sl)
                units = beta * abs(dh - dl) * S[j] / Q[j]          # QQQ units per 1 share of spread
                hedge_pnl += units * (Q[j2] - Q[j] * np.exp(RF * (j2 - j) / 252))
            payoff = max(mh - S[-1], 0) - max(ml - S[-1], 0)
            R.append(np.exp(-RF * T) * (payoff + hedge_pnl) / cost - 1)
        R = np.array(R)
        Rdh[nm] = R
        qtr = t4["quarter"].to_numpy()
        disc = (t4["split"] == "discovery").to_numpy()
        full = test_mean(R, qtr, nm)
        x_mn = (1 + t4["rel"].to_numpy()) * np.exp(RF * T)
        x_raw = 1 + t4["s"].to_numpy()
        out_rows.append(dict(structure=nm, mean_delta_hedged=full["mean"], p_boot=full["p_boot"],
                             p_cluster=full["p_cluster"], ci90=f"[{full['ci90_lo']:+.3f}, {full['ci90_hi']:+.3f}]",
                             disc_mean=R[disc].mean(), val_mean=R[~disc].mean(), p_win=float((R > 0).mean()),
                             worst=R.min(), mean_perfect_hedge=returns(st.iloc[[k]], x_mn)[0].mean(),
                             mean_unhedged=returns(st.iloc[[k]], x_raw)[0].mean(), f_kelly=kelly(R)))
    res = pd.DataFrame(out_rows)
    res.to_csv(c.RESULTS / "edge_delta_hedged.csv", index=False, float_format="%.4f")
    pd.to_pickle(dict(Rdh=Rdh, t4=t4), c.DATA / "raw" / "_edge_dh_state.pkl")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("IMPLEMENTABLE DELTA HEDGE (weekly QQQ rebalancing, beta 1.05, financed at r) vs perfect hedge vs unhedged")
    print(res.round(3).to_string(index=False))
    return res, Rdh, t4
