# Anthropic (ANT): the IPO-fade study, applied

This folder applies this repo's IPO-fade study to **Anthropic's IPO**. ANT priced at $83 on 2026-11-18, above its $70–78 range. It raised $81.3B, is valued at about $1.1T on basic shares (about $1.5T fully diluted), and started trading on Nasdaq on 2026-11-19. The folder answers two questions:

1. **Does the post-IPO fade apply to Anthropic?**
2. **What price targets does the study imply?**

**Full write-up: [REPORT.md](REPORT.md).**

## Answers

**Does it apply? Yes, as a base rate. ANT is also the "special structure" case where the parent study said the fade concentrates.**

- **In the population.** ANT is a traditional underwritten T4 IPO (> $10B). Of 61 comparable T4 IPOs, 74% trailed QQQ by day +180, measured from their day-1 close. The median give-back from the day-1 high was 29.2%, matching the legacy headline.
- **Size doesn't help.** There is no size gradient in-sample, and 7 of the 8 IPOs of $50B or more faded.
- **The pattern hasn't decayed.** The 2025–26 mega-IPO cohort faded more than history: 16 of 19 by day 60.
- **The structure is special.**
  - About 7% float.
  - A 23-date staggered lockup: up to 8.9× the IPO share count is sellable by May 17, 2027.
  - Projected Nasdaq-100 index buying at the **Dec 18 close**, which is exactly the parent's +20 short entry.
  - A listing into a high-VIX tape.
- **Not a proven edge.**
  - The parent's pre-registered test still **fails** under dependence-honest inference, including for T4 alone.
  - The tail is fat: a hedged short on ARM lost 122%.
  - In SpaceX, the closest analog, the stock fell 49% from peak to trough by day 36 and then recovered. A short opened at the +20 entry is −13.7% today.

**Price targets.** These are study-implied base rates, assume QQQ is flat, and are anchored on the $83 offer (ANT had not opened when this ran):

| Horizon | Date | Bear (P25) | **Base (median)** | Bull (P75) | P(below $83) |
|---|---|---|---|---|---|
| +20 | Dec 18, 2026 | $84 | **$92** | $115 | 25% |
| +60 | Feb 18, 2027 | $74 | **$86** | $105 | 41% |
| +120 | May 14, 2027 | $53 | **$73** | $98 | 57% |
| +180 | Aug 11, 2027 | $50 | **$73** | $100 | 59% |
| +250 | Nov 18, 2027 | $47 | **$65** | $87 | 66% |

Once ANT has a day-1 close **C** and intraday high **H**, the median base cases are:
- **+180 ≈ 0.77 × C ≈ 0.71 × H**
- **+250 ≈ 0.66 × C ≈ 0.63 × H**

The full scenario grid is in [REPORT §5.2](REPORT.md#52-scenario-grid--if-ant-closes-day-1-at-). These are base rates, not a fundamental valuation or investment advice.

## Layout

```
Anthropic/
  REPORT.md                     full methods, results, verdict, targets, limitations
  src/
    run_all.py                  ONE command: fetch -> panel -> applicability -> targets -> figs
    common.py                   paths, NYSE calendar (validated vs QQQ 2011-2026), cached fetch, anchors
    fetch_cohort.py             cohort audit + Yahoo fetch (every name reason-coded)
    build_paths.py              day-0-anchored event table + consistency gate vs parent panel
    applicability.py            parent-verdict replay, base rates, size test, conditioning, live check, SPCX
    price_targets.py            ANT status, event calendar, targets, scenario grid, trade view
    make_figs.py                figs/*.png
  data/
    ant_deal_terms.csv          every deal fact with its source (EDGAR FWP, CNBC, S-1/A readings)
    ant_lockup_schedule.csv     23 release tranches (dates, max shares, assumed-date flags)
    cohort_universe.csv         every candidate with role + reason (historical / live / excluded)
    price_fetch_log.csv         fetch status per name (survivorship accounting)
    cohort_prices_window.csv.gz day 0..259 bars per event (committed; offline reruns)
    cohort_rel_paths.csv.gz     daily QQQ-neutral paths per event
    benchmarks.csv.gz           QQQ / SPY / IWM / VIX
    ant_prices.csv              what Yahoo showed for ANT at run time
  results/                      one CSV per table (see REPORT for the index)
  figs/                         fig1 median path + ANT calendar, fig2 ANT price fan, fig3 live check
```

## Run

```bash
pip install yfinance pandas numpy scipy statsmodels matplotlib
python Anthropic/src/run_all.py             # refreshes ANT + benchmarks, cached cohort prices
python Anthropic/src/run_all.py --offline   # no network: recompute everything from committed data
```

Run it again after 16:15 ET on 2026-11-19, or on any later day. `results/ant_status.csv` then flips to `day0_final`, and every target re-anchors on ANT's actual day-1 close and high.
