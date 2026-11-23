# Pre-registration: formal test of the ANT put-spread edge, and the sizing rule

This document was written and committed **before** `src/edge_test.py` existed or was run. The git commit timestamp is the record. Everything below is binding: the result is reported against these rules whether it passes or fails.

**Disclosure: this is not a blind test.**
- The same 61 historical T4 events were already used in [OPTIONS_QUANT.md](OPTIONS_QUANT.md) to compute expected values.
- The 90/60 spread was singled out *because* it ranked well there.
- The design below corrects for that selection (Hansen's SPA test over all 91 structures). It does not pretend the selection didn't happen.
- The only truly out-of-sample evidence is the 2025–26 listings that have since matured, and future IPOs, ANT included.

## Strategy under test

- **Universe:** historical T4 IPOs (> $10B at listing, 2010–2025) from the parent universe, with prices on trading days 2 and 162. n = 61.
- **Entry and exit:** enter at the close of trading day 2. Exit at the close of day 162, holding to expiry: 160 sessions, ≈ 235 calendar days.
- **Instruments:** all 13 puts and the 78 vertical put spreads in the 2026-11-23 ANT Jul-16-2027 snapshot (`data/ant_put_chain_2027-07-16_snapshot.csv`).
  - Each is expressed as moneyness K/S₀ (S₀ = $87.48).
  - Each is costed at ANT's executable quotes: buy at the ask, sell at the bid, as a fraction of spot.
  - Every historical event is priced at ANT's *current* quotes. The question is whether ANT's options are cheap relative to how comparable IPOs actually traded.
- **Primary structure:** the 90/60 put spread. K₁/S = 1.0288, K₂/S = 0.6859, cost = (19.60 − 6.00)/87.48 = 15.55% of spot.
- **Return on premium (present-value basis):** R_i = e^(−rT) · payoff_i / cost − 1, with r = 4.1% and T = 235/365.
- **Primary terminal price (market-neutral):** S_T/S₂ = (1 + rel_i) · e^(rT). Here rel_i is the event's QQQ-neutral return from day 2 to day 162. This removes the 2010s bull market; it is equivalent to the market earning the risk-free rate.
- **Secondary terminal price (as actually traded, unhedged):** S_T/S₂ = 1 + s_i, the stock's actual return.

## Hypotheses and tests

The null is H0: E[R] ≤ 0 for the primary structure. The alternative is H1: E[R] > 0. All tests are one-sided at α = 0.05.

- **T1. Moving-block bootstrap over listing quarters.** Blocks of 4 consecutive quarters, 20,000 draws, seed 2026. The one-sided p-value comes from the centred bootstrap distribution of the mean: p = P*(R̄* − R̄ ≥ R̄). This is the same dependence-honest resampling the parent study used.
- **T2. Cluster-robust t.** Clusters are listing quarters. The one-sided p-value uses Student-t with G − 1 degrees of freedom.
- **T3. Split sample.** Discovery covers listings 2010–2018; validation covers 2019–2025. Each is tested with T1.
- **T4. Data-snooping.** Hansen's SPA_c and White's Reality Check over all 91 structures, using the same bootstrap. Also: the single-step max-t (Romano–Wolf / Westfall–Young) adjusted p for the 90/60, and the Deflated Sharpe Ratio with N = 91 trials.
- **Reported but not gating:**
  - the naive iid t-test;
  - the secondary (unhedged) version;
  - the T3+T4 pool;
  - the matured 2025–26 listings, both tiers, out-of-sample;
  - a power analysis: how many independent events a 5% test would need.

## Decision rule (verbatim)

The edge is **"statistically established"** if and only if ALL of the following hold:

1. T1 p < 0.05 **and** T2 p < 0.05 on the full T4 sample (primary, market-neutral, 90/60).
2. The validation-period mean R is > 0 with T1 p < 0.10, **and** the discovery-period mean R is > 0.
3. SPA_c p < 0.05 across the 91 structures.

Otherwise the verdict is **"edge not statistically established"**. The evidence is then reported as a probability, not a proof.

## Sizing rule (fixed now)

Everything is computed on the primary R_i of the 90/60 spread.

- **f_K: plug-in Kelly.** argmax_f mean ln(1 + f·R_i).
- **f_LCB: lower-confidence-bound Kelly.** Kelly after rescaling payoffs so that mean R equals the one-sided 90% bootstrap lower bound. f_LCB = 0 if that bound is ≤ 0.
- **f_B(τ): Bayesian Kelly with a skeptical prior.** The prior is μ ~ N(0, τ²), with τ = 10% of premium (sensitivities at 5% and 20%). The likelihood is normal, using the cluster-robust SE. Payoffs are rescaled to each posterior draw of μ, and f maximizes the posterior-expected log growth.
- **f_DD: drawdown-constrained fraction.** The largest f such that P(max drawdown > 25%) ≤ 5%. This comes from a calendar-time block-bootstrap backtest that trades every T4 IPO from 2010–2025 at day 2, with f of equity per trade.
- **If the edge is established:** recommended f = min(f_K/4, f_B(10%), f_DD).
- **If it is not established:** recommended f = min(f_B(10%), 1%), labelled **discretionary**. There is no statistical license for more.
