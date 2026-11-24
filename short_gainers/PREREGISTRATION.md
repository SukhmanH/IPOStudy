# Pre-registration — Shorting the Day's Top Gainers

**Written 2026-11-24, before any forward (post-signal) return was computed.**
This file is committed to git before `build_events.py` computes forward returns;
the commit timestamp is the evidence. Nothing below may be changed after results
are seen. Any later deviation must be listed in REPORT.md under "Deviations from
pre-registration", with the reason.

## 1. Question

1. **Existence / capturability (H1).** If you short the stocks that were the day's
   biggest gainers, entering the *next* trading day, is there a systematic profit
   after realistic costs?
2. **Rules (H2).** Can a simple rule, built only from information available at
   entry, and chosen using the discovery period only, pick out a subset where the
   edge is larger, and does that rule hold up out of sample?

## 2. Data

| Item | Source | Notes |
|---|---|---|
| Universe | Nasdaq screener, as of 2026-11-24 | Common/ordinary shares and ADRs on NASDAQ, NYSE, NYSE American. Excluded with reason codes: preferreds, warrants, rights, units, debt, funds/ETPs. SPAC commons kept but tagged. `data/universe_audit.csv` |
| Daily OHLCV and splits | Yahoo (yfinance), 2008-06-01 to 2026-11-23 | Split-adjusted OHLC plus split history. `data/price_fetch_log.csv` |
| Traded price level | Reconstructed | `raw_t = adj_t × Π(split ratios dated after t)`. Price floors use this. Without it, a 2014 penny stock that later reverse-split 1:50 would show as a "$25 stock" (look-ahead). |
| Hedge and regime | SPY (hedge), IWM (robustness), ^VIX | Yahoo |
| Earnings dates | Nasdaq earnings calendar, by date | `data/earnings_dates.csv.gz` |
| Independent cross-check | shibui DB (server-side SQL) | Separate vendor pipeline; used to replicate the headline |

**Survivorship (known, unfixable with available data).** Both price sources cover
only *currently listed* names. Spiked-then-collapsed names that were later delisted
are missing. So are takeover targets, which spike and then trade flat near the deal
price until they delist. The first group would most likely have *helped* a short
strategy and the second is roughly neutral the next day. So the expected bias runs
**against** the short thesis (conservative), most strongly for multi-day holds and
sub-$5 names. The size can't be measured. Coverage (names per year) will be
reported.

## 3. Definitions (all computed per ticker, consecutive trading rows)

* Signal day **t**. Gain `r_t = C_t / C_{t-1} − 1` (split-adjusted closes).
  `C_{t-1}` must be the previous *market* trading day (SPY calendar). Otherwise
  r_t spans a gap and the row is excluded.
* **Eligibility at t (main universe, "U5")**:
  * reconstructed traded close `C_t ≥ $5.00`
  * dollar volume `C_t × V_t ≥ $10M` on day t
  * `≥ 60` prior trading rows of history
  * `r_t ≥ +10%`
  * O, H, L, C > 0 and V > 0 on t and t−1
  * no split recorded within rows [t−2, t+2]
* **Probable data-error exclusion**: `r_t ≥ +100%` with `RVOL_t < 1.5` (a 2× move
  on non-elevated volume is the signature of an unrecorded reverse split or a bad
  print). Logged with a reason code.
* **Event set**: the **top 10** eligible names by r_t each day (ties broken by
  dollar volume). There are fewer events on days when fewer than 10 qualify.
* **Sub-$5 universe ("U1", exploratory only)**: traded close $1.00–$4.99, day-t
  dollar volume ≥ $2M, same rules otherwise. Flagged: *borrow typically
  unavailable or expensive; not tradeable evidence.*

## 4. Primary trade

* **Short at the open of t+1, cover at the close of t+1** (intraday; no overnight
  position, so no overnight borrow fee).
* Raw short return `s = 1 − C_{t+1} / O_{t+1}`.
* **Hedged** (primary metric) `s_h = s + (SPY_C_{t+1} / SPY_O_{t+1} − 1)`, an
  equal-dollar long SPY position over the same window.
* t+1 must be the next market trading day and the stock must trade that day
  (O > 0, V > 0). Otherwise the event is *untradeable*: excluded and counted.

## 5. Portfolio (calendar time)

* On each trading day d = t+1, hold that day's events with **fixed 10% slots**:
  `R_d = Σ_i 0.10 × (s_h,i − c)`, where c is the round-trip cost per event.
  Gross short exposure is ≤ 100%. Days without events have R_d = 0.

## 6. Costs

* Round-trip cost per event, `c ∈ {0, 0.10%, 0.25% (base), 0.50%, 1.00%}`.
  0.25% represents auction (MOO/MOC) or tight-limit execution plus fees in a
  ≥$5, ≥$10M-volume name.
* The **gross mean hedged return per event** is the break-even cost and is
  reported for every cell.
* Exploratory multi-day holds also pay borrow, per night held, at 10% annualized
  (base) with sensitivities at 30% and 100%. Recent top gainers are often
  hard-to-borrow.

## 7. Sample split (fixed now)

* **Discovery**: signal days 2010-01-04 to 2018-12-31.
* **Validation**: signal days 2019-01-02 to the last signal day with t+1 data.
* Every rule, threshold and parameter choice is made on discovery only.
  Validation is used once per hypothesis.

## 8. Inference

* Daily portfolio mean, with a 95% CI from a **moving-block bootstrap** (block =
  21 trading days, 10,000 resamples, seed 42, percentile interval).
* Annualized Sharpe (√252). The Newey–West t (5 lags) is reported alongside.
* Per event: n, mean, median, hit rate (s_h > 0), profit factor, 1st/99th
  percentiles, worst event.

## 9. H1 — confirmatory test (unconditional strategy)

**Decision rule (verbatim):** the "short the top gainers next day" edge *exists
and is plausibly capturable* **iff** in the **validation** period the calendar-time
portfolio's mean daily **hedged** return, net of a **0.50%** round-trip cost per
event, is positive and its 95% block-bootstrap CI excludes zero. Otherwise the
honest conclusion is: *no unconditional edge survives realistic costs.*

(The base case is 0.25%. As in the repo's IPO study, the verdict is rendered at
the stricter sensitivity.)

## 10. H2 — rule discovery (discovery period only), then one validation test

**Candidate features.** Every feature is known at the decision instant. Buckets
are fixed here.

| # | Feature | Buckets |
|---|---|---|
| F1 | gain r_t | [10,15), [15,25), [25,50), [50,100), ≥100 % |
| F2 | close location CLV_t = (C−L)/(H−L) | <0.50, [0.50,0.85), ≥0.85 |
| F3 | day-t opening gap O_t/C_{t−1} − 1 | <2%, [2%,10%), ≥10% |
| F4 | relative volume V_t / mean(V_{t−20..t−1}) | <3, [3,10), ≥10 |
| F5 | prior 20-day return C_{t−1}/C_{t−21} − 1 | <−20%, [−20%,+20%), ≥+20% |
| F6 | new closing high vs prior 252 rows (or all history if shorter) | yes / no |
| F7 | RSI(14), Wilder, at t | <70, [70,85), ≥85 |
| F8 | consecutive up closes ending at t | 1, 2, ≥3 |
| F9 | traded price C_t | [$5,$10), [$10,$30), ≥$30 |
| F10 | day-t dollar volume | [$10M,$30M), [$30M,$100M), ≥$100M |
| F11 | SPY close at t vs its 200-day SMA | above / below |
| F12 | VIX close at t | <15, [15,25), ≥25 |
| F13 | SPY return on day t | <0, ≥0 |
| F14 | entry-day gap O_{t+1}/C_t − 1 (observable at the entry instant) | <−5%, [−5%,0), [0,+5%), ≥+5% |
| F15 | earnings report dated t−1 or t | yes / no |
| F16 | sector = Health Care (current classification) | yes / no |
| F17 | SPAC common (tag) | yes / no |

**Candidate filters.** For ordered features, every contiguous range of buckets
except the full range. For binary features, each value. That gives 80
single-condition filters.

**Selection (forward stepwise, at most two conditions):**
1. For each candidate filter applied to the top-10 event set, compute the
   discovery calendar-time portfolio (hedged, base cost 0.25%). A filter is
   eligible only with **≥ 300 discovery events**. Choose the filter with the
   highest discovery annualized Sharpe.
2. Holding step 1 fixed, try adding each filter on a *different* feature (again
   ≥ 300 discovery events). Accept the best one only if it raises discovery
   Sharpe by **≥ 0.20**. Otherwise stop at one condition.
3. The selected rule is tested **once** on validation with the H1 decision rule
   (0.50% cost, CI excludes zero).

## 11. Exploratory battery (descriptive only; never promoted to the headline)

* **A. Entry × exit grid.** Entries: close of t (needs a ~3:50pm scan) and open
  of t+1. Exits: open of t+1 (overnight only), close of t+1, t+2, t+5, t+10,
  t+20. Borrow is applied for overnight holds.
* **B. Feature profiles.** Mean, median, hit rate and n per bucket, by split.
* **C. Signal variants.** Top-N ∈ {5, 10, 20, all}; gain floor ∈ {10, 20, 30}%.
* **D. Universe variants.** U1 (sub-$5, flagged); ≥$5 with ≥$2M dollar volume;
  ex-SPAC; ex-Health Care; raw (unhedged) and IWM-hedged.
* **E. Squeeze protection.** Buy-stop at +10/+20/+30% above entry, triggered
  when H_{t+1} reaches the level and filled at the level +1% slippage. LULD halts
  can fill far worse, so this is optimistic. Report mean, P1/P99, worst day and
  max drawdown with and without stops.
* **F. Per-year table**, with the 2020–21 retail-mania episode called out.
* **G. Second data source.** Replicate the H1 portfolio in the shibui DB. Its
  prices are split-adjusted only, so the $5 floor is approximate there. Compare
  monthly P&L.

## 12. Reporting rules

* n on every table; flag cells with fewer than 100 events.
* Count and report every cell examined (multiple-testing ledger).
* Report raw and hedged results. Never headline the best exploratory cell.
* Log every exclusion with a reason code. No silent drops.
