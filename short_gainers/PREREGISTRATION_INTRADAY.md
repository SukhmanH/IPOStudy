# Pre-registration addendum — intraday timing

**Written 2026-11-27, before any intraday return was computed** (the hourly download was still
running). Committed to git before `intraday_analysis.py` exists. Same rules as
`PREREGISTRATION.md`: nothing here changes after results are seen. Deviations go in the report.

## Question

Can the trade be done **within one session**, with no overnight position and so no overnight
borrow fee? Two ways:

* **Test I:** short a stock *while it is running* on the gain day (mid-day, or shortly before the
  close) and cover at that day's close.
* **Test II:** on the day after, delay the short from the open to later in the day, then cover at the
  close.

## Data

* Yahoo hourly bars, regular session only. Bars start at 09:30, 10:30, …, 15:30 ET. The window is
  2023-12-29 to 2026-11-25, about 730 trading days.
* Daily bars with split history, pulled at the same moment, give the official open, the official
  close and traded-price reconstruction.
* For tickers first listed after 2023-12-29, Yahoo's hourly limit means data start no earlier than
  2024-11-28. This is logged in `data/intraday_fetch_log.csv`.
* Universe: every ticker that had a day in the window with a high ≥ +10% over the prior close, a
  traded high ≥ $5 and ≥ $2M traded, plus SPY. This contains every stock that could have been an
  intraday top gainer under the filters below.
* Checkpoints: T ∈ {10:30, 11:30, 12:30, 13:30, 14:30, 15:30}. The price at T is the close of the
  hourly bar ending at T. Half-days use only the checkpoints that exist.

## Test I — short the running gainer, cover at the same day's close

At checkpoint T on day t, using only information available at T:

* **Gain so far:** `gain_T = P_T / C_{t−1} − 1`, against the official prior close with the same split
  adjustment.
* **Eligibility:**
  * gain_T ≥ +10%;
  * traded price P_T ≥ $5;
  * ≥ $5M traded from the open through T;
  * ≥ 60 trading days of history;
  * the previous row is the previous market day;
  * no split in rows [t−2, t+2].
* **Selection:** the **top 10 by gain_T** (ties broken by dollar volume through T). The ranking uses
  the gain at T, never the final close, to avoid look-ahead.
* **Trade:** short at P_T and cover at the official close C_t. Hedge: long SPY over the same window.
* **Returns:** `s = 1 − C_t / P_T`, hedged `s_h = s + (SPY_C_t / SPY_P_T − 1)`.
* **Variant I-NE:** the same, excluding names with an earnings report dated t−1 or t (the main study's
  H2 rule).

## Test II — delay the day-2 entry

* **Events:** the main study's top-10 gainers (gain ≥ +10%, traded close ≥ $5, ≥ $10M traded), with
  signal day t inside the hourly window.
* **Trade:** short on day t+1 at T ∈ {official open, 10:30, …, 15:30}, cover at the official close of
  t+1, SPY-hedged as above.
* **Variant II-NE:** excludes earnings gainers.

## Test III — short shortly before the gain-day close, hold overnight (descriptive only)

* **Entry:** the Test I selection at 15:30.
* **Exits:** the official open and the official close of t+1. Borrow is charged for one night at 10%,
  30% and 100% per year.
* This test pays an overnight fee, so it only shows what that fee buys.

## Portfolio, costs, inference

* **Portfolio:** 10% of capital per trade; daily P&L is the sum of that day's trades; days with no
  trades count as 0.
* **Costs:** round-trip cost per trade ∈ {0, 0.10%, 0.25% (base), 0.50%, 1.00%}. A mid-day entry
  crosses the spread rather than using an auction, so 0.25% is optimistic for fast-moving names. The
  report must say so.
* **Inference:** 95% moving-block bootstrap CI of the mean daily return (21-day blocks, 10,000
  draws, seed 42), with Newey–West t (5 lags).

## Primary cells and decision rule

The six primary cells are fixed now:

| Cell | Test | Checkpoint |
|---|---|---|
| I-1230 | I | 12:30 ("mid-day") |
| I-1530 | I | 15:30 ("right before the close") |
| I-NE-1230 | I-NE | 12:30 |
| I-NE-1530 | I-NE | 15:30 |
| II-1230 | II | 12:30 on day 2 |
| II-NE-1230 | II-NE | 12:30 on day 2 |

A cell **works** iff both of these hold:

* the mean daily hedged return at **0.25%** cost is positive, with a **Bonferroni-adjusted** bootstrap
  CI that excludes zero (99.17% = 1 − 0.05/6, for six primary cells);
* the mean is positive in **both halves** of the window (2023-12-29 to 2025-05-30, and 2025-06-02 to
  2026-11-25).

Every other checkpoint is reported as a profile and never promoted. There is no separate validation
period, because the window is ~3 years in one market regime. The two-halves requirement is the
stability check. Any cell that works is a lead for paper trading, not a proven edge.

Also reported:

* hit rate, worst trade, worst day;
* how often the stock went on to make a new high after entry ("pushed higher"), measured as the
  largest rise from P_T to the day's later high in the hourly bars;
* a 1.5% FX fee scenario for a Wealthsimple CAD account: about 3% per short-and-cover round trip,
  since the fee applies to each filled order. This is reported as a cost line, not a test.
