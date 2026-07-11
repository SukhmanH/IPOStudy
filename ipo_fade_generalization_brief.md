# Brief for Claude Code — Generalizing the IPO Fade Study

## Context

You are working in (or alongside) the existing `SpaceX_Thesis` repo, which contains a pre-listing event study on SPCX: Study 1 found traditional underwritten IPOs >$10B at listing give back a median −29.2% vs QQQ within 180 trading days of the day-1 high (n=63, t=−6.51, 79% fade rate); Study 2 showed the rebalance-day index-inclusion pop has been statistically zero since 2010; Study 3 mapped the SPCX lockup calendar. The pipeline crosses Ritter's IPO database with exchange calendars, verifies against EDGAR, and pulls prices via yfinance.

The new task: test whether the fade is a **general, implementable short edge across the broader IPO universe** — not just mega-caps — and if so, derive the optimal entry day, holding period, and volatility-aware position sizing. The output must sharply distinguish "a fade exists in the data" from "an edge exists after borrow costs, liquidity constraints, and tail risk."

## Objective — two questions, in order

1. **Existence:** Is the post-IPO fade robust across size tiers, or an artifact of the >$5B cohort?
2. **Capturability:** Where (size tier, entry day, horizon) does the fade survive realistic implementation frictions, and what sizing rule maximizes risk-adjusted capture given post-IPO volatility?

## Non-negotiable ground rules

1. **Regression test before extending.** Rerun the legacy pipeline first. You must reproduce the published headline within rounding: large-cap median −29.2%, mean −30.2%, t=−6.51, n=63, and the Table 2 robustness battery. If you can't, stop and reconcile before touching anything new.
2. **Pre-registration.** The Primary Spec below is the one confirmatory test. Everything else is exploratory/descriptive. Never promote the best-looking exploratory cell to the headline result.
3. **Executable prices only.** All entries and exits at daily closes. The day-1 intraday high remains a *measurement reference* for fade depth, never an entry price — nobody can systematically sell the high.
4. **Same exclusions as the legacy universe:** SPACs and SPAC units, de-SPAC mergers, direct listings, spinoffs, uplistings, closed-end funds, blank-check vehicles. Keep ADRs but tag them as a subgroup.
5. **n on every table**, and survivorship accounting per size tier and era (same standard as Appendix C of the paper).
6. **Out-of-sample split fixed now, before any results:** discovery = listings 2010-01-01 through 2018-12-31; validation = 2019-01-01 through the latest listing with a complete +180 trading-day history. All parameter choices justified on discovery only; the primary verdict is rendered on validation.
7. **Multiple-testing honesty:** count and report the total number of (entry × exit × tier × regime) cells examined, and say so in the limitations section.

## Data

- **Universe:** Ritter IPO database (Warrington/UF) crossed with NYSE/Nasdaq listing calendars, verified against EDGAR S-1/424B4 filings, as in the legacy pipeline — but lower the market-cap floor to **$300M at listing**. Expect on the order of 1,500–2,500 candidate names, 2010–present.
- **Prices:** yfinance primary. Delisted tickers are the known weakness — attempt a fallback source (e.g., Stooq) for names yfinance drops; log every unresolvable name with a reason code. Report retention by tier/era and state the bias direction both ways: missing post-collapse delistings understates the fade (works against the short thesis looking too good); missing acquired-at-a-premium names overstates it. Down-cap, both happen.
- **Benchmarks:** QQQ primary; SPY and IWM robustness (IWM is arguably the right benchmark for T1/T2 — report both). Daily VIX for regime splits.
- **The uploaded IPO calendar (Oct 2025–Jul 2026) is a universe seed for the newest cohort ONLY.** Do not ingest its return columns: (a) roughly two-thirds of its rows are SPAC units (tickers ending U/.U, $10.00 pricing) that the methodology excludes; (b) its Chg 1D/1W/1M columns are decimal fractions rendered with a % sign (e.g., SWMR shows "+1.48%" on a 12.50→31.00 move; SPCX shows "+0.07%" against a $150→$160.95 print) — whatever the convention, re-derive all returns from raw price history; (c) nothing listed after ~Oct 2025 has a +180 outcome yet, so these names enter the panel only for short-horizon cells.
- Pin the data-pull date, cache raw downloads, and commit the universe audit CSV (every candidate, every exclusion reason).

## Size tiers

- T1: $300M–$1B  ·  T2: $1B–$5B  ·  T3: $5B–$10B  ·  T4: >$10B
- Report anything under $300M only if it falls out trivially, and mark it untradeable-short.

## Primary confirmatory spec (committed before running anything exploratory)

- **Universe:** T3+T4 (≥$5B at listing), 2010–present, standard exclusions.
- **Trade:** short at the close of trading day **+20** (post index-inclusion window, per the legacy Figure 1 shape: medians pinned at −4.5% to −6.3% through week 4, decline arrives after); cover at the close of **+180**. Hedge: long QQQ, equal dollar, same timestamps (dollar-neutral spread).
- **Metric:** hedged return per event (short-leg P&L + hedge-leg P&L); median, mean, hit rate.
- **Inference:** moving-block bootstrap clustered by listing quarter (10,000 draws) for the median CI; report the naive t alongside, labeled as assuming independence.
- **Costs:** base case 3% annualized borrow over the ~160-trading-day hold; sensitivities at 0%, 10%, 30%.
- **Decision rule (state verbatim in the report):** the edge "exists and is plausibly capturable in T3/T4" iff the **validation-period** hedged mean is profitable to the short with the 95% bootstrap CI excluding zero **at the 10% borrow sensitivity**. Otherwise the honest conclusion is: no systematic edge after costs; the fade is real but only tradeable when listing structure is special (SPCX-style float/lockup calendars).

## Exploratory battery (report all cells, flag n<25, no cherry-picking)

**A. Entry × exit grid.** Entry ∈ {+1 close, +5, +10, +20, +30} × exit ∈ {+60, +90, +120, +180, +250} × tier {T1–T4}. Heatmap of median hedged return with n per cell. Purpose: robustness *around* the pre-registered cell, not a search for a better one.

**B. Conditioning.** (i) Listing-month VIX terciles (legacy found high-VIX listings fade −34.8% mean vs −16.5% low-VIX — test whether this holds down-cap and whether it's a usable entry filter). (ii) Day-1 pop terciles. (iii) Cross-sectional regression: fade(+20→+180, hedged) ~ realized vol days 1–20 + log(mktcap) + day-1 pop + listing-month VIX, SEs clustered by listing quarter. This answers whether early realized vol *predicts* fade depth or just scales it — the difference matters for sizing vs filtering.

**C. Calendar-time portfolio — the actual strategy test.** Each trading day 2010–present, hold a short in every name with IPO age ∈ [21, 180] trading days passing the tier/liquidity filter; equal weight; long QQQ equal notional. Report: annualized return, vol, Sharpe, Newey–West t (21 lags), max drawdown, per-year returns, average and 10th-percentile name count, % of days holding <3 names, monthly turnover. Variants: (i) T3+T4 only, (ii) all tiers, (iii) inverse-vol weights. If the T3/T4 book holds 1–5 names most days, say so plainly — that is a capacity/lumpiness fact the reader needs.

**D. Volatility-aware sizing.** Inverse 20-day-realized-vol weights, scaled to a 10% annualized portfolio vol target, 10% single-name cap. Compare Sharpe and worst single-event loss vs equal weight. Separately, using **discovery-half data only**, estimate the per-event edge and variance of the primary spec and report the implied full-Kelly and quarter-Kelly fractions for a standalone position — with the explicit caveat that the return distribution is left-skewed for the short (fat right tail in the stock), which makes Kelly estimates from the mean/variance overstate safe size.

**E. Right-tail protection.** Stop-loss overlay on the hedged position at −15% / −25% / −40% adverse moves (evaluated at closes only). Report the change in mean return, hit rate, and the 95th/99th percentile loss with vs without stops. Also count, per tier, events where the hedged position moved >+50% against the short (the ARM +71% problem). Expected finding worth confirming or refuting: stops cost EV on a skewed distribution but cap ruin — quantify the trade, don't assume it.

**F. Implementability filter.** "Tradeable set" = entry-day 20-day ADV ≥ $25M. Report the edge in the tradeable set vs the full set per tier. Add an optionability proxy flag (mktcap > ~$2B ≈ listed options likely) without attempting to price options.

**G. Lockup-anchored variant (generalizes Study 3).** For the broad universe, standard lockups expire ~180 *calendar* days ≈ trading day +124. Event study of hedged CAR in [expiry −20, expiry +40] around the estimated expiry date. If the fade clusters around lockup expiry rather than fixed trading-day offsets, the entry rule should be anchored to the lockup calendar, not the listing date — report which anchoring wins on discovery data.

## Deliverables

- `results/` — one CSV per table; `figs/` — median hedged paths by tier, entry×exit heatmap, calendar-time equity curve (log scale), retention table.
- `REPORT.md` — same structure and honesty standard as the original paper: methods, results, robustness, then a Limitations section that explicitly covers survivorship by tier, the multiple-testing count, borrow-cost realism, and cross-event dependence.
- `PLAYBOOK.md` — if the decision rule passes: entry rule, sizing rule, holding period, expected hedged-return distribution table, and per-event kill criteria (the analog of the paper's P3 falsification anchor). If it fails: state plainly that no systematic edge survives costs, and that the strategy reduces to structurally special situations.
- Reproducibility: one command per study, extending the existing repo conventions.

## Anti-goals

- Do not headline the best exploratory cell.
- Do not report unhedged returns as the primary metric.
- Do not enter at intraday extremes or the day-1 high.
- Do not silently drop failed price downloads — log and account for every one.
- Do not include SPACs, and do not trust the uploaded calendar's percentage columns.
- Do not optimize stops, weights, or filters on validation-period data.
