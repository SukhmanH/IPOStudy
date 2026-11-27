# Doing it inside one day: intraday addendum

Pre-registered in [`PREREGISTRATION_INTRADAY.md`](PREREGISTRATION_INTRADAY.md) (commit `dc95305`) before
any intraday return was computed. Results are in [`results/intraday_*.csv`](results/).

<!--RESULTS-->

## 1. The fee question (Wealthsimple)

Wealthsimple's public pages were read on 2026-11-27. Three help-centre articles refused automated
access, so details that only they hold are marked as unconfirmed.

| Cost | When it applies | Does a same-day trade avoid it? |
|---|---|---|
| **Currency conversion, 1.5%** | Every filled order in a US-listed stock from a **CAD** account (fee schedule) | **No.** A short and its cover are two orders, so about 3% per round trip, five to seven times the edge measured here. A **USD account** removes it ($10/month; free for Premium and Generation clients). Moving CAD into it costs 1.5% under $10k, 1% for $10k–25k, 0.5% for $25k–100k and 0% above $100k. |
| **Stock borrow fee** | Industry practice: accrues daily on short positions held overnight. Wealthsimple's learn page says it "can spike to 50% or more" a year for hard-to-borrow names | **Normally yes:** a short opened and covered the same day holds no position overnight. Wealthsimple's fee schedule lists no borrow fee and doesn't describe same-day treatment, so confirm with them. |
| **Margin interest** | On debit balances, "calculated daily and charged monthly" | A short sale creates a credit, not a debit, so this normally doesn't apply. |
| **Pass-through fees** | ECN or regulatory fees on US orders "may be passed on" | Small per order. |
| **Ability to short at all** | Requires a non-registered margin account (no TFSA or RRSP) and a locate | Unknown per stock. Wealthsimple publishes no shortable list, and hot small caps are often unborrowable. |

Sources:
[Wealthsimple trade fee schedule](https://www.wealthsimple.com/en-ca/legal/fees/trade),
[What is short selling (Wealthsimple)](https://wealthsimple.com/en-ca/learn/what-is-short-selling),
[Margin trading resources](https://www.wealthsimple.com/en-ca/margin-trading-resources),
[Margin account FAQs](https://www.wealthsimple.com/en-ca/learn/margin-account-faqs),
[USD accounts subscription](https://www.wealthsimple.com/en-ca/magazine/usd-accounts-subscription),
[Questrade: cost of short selling](https://www.questrade.com/learning/investing-basics/cost-of-short-selling).

## 2. Data and method

* **Hourly bars.** Yahoo hourly bars, regular session (09:30, 10:30, …, 15:30 ET), for 2023-12-29 to
  2026-11-25 (730 trading days). Daily bars with split history were pulled at the same moment. The
  universe is every ticker that could have been an intraday top gainer: 4,190 tickers plus SPY
  (`data/intraday_tickers.csv`, `data/intraday_fetch_log.csv`).
* **The ranking uses only what was known at the time.** At 12:30 the candidates are ranked by their
  gain *at 12:30*, not by how the day ended. Using the end-of-day top-gainer list would quietly pick
  stocks already known to have held their gains.
* **Two data hazards found and fixed before any result was read:**
  * **Hourly volume is not split-adjusted, but hourly prices are.** Checked on NVDA's and SMCI's 10:1
    splits: pre-split hourly volume is one tenth of the adjusted daily volume, while prices match to
    the cent. Dollar volume to time T is therefore computed as adjusted price × split factor × raw
    volume.
  * **Missing hourly bars on halted squeeze days.** On ELPW's 2026-01-30 spike, Yahoo has only the
    first hourly bar of a 26M-share day. Forward-filling the last price would have created stale,
    untradeable entry prices. An entry at T therefore requires a real print in the hour before T.
    Days whose last hourly bar disagrees with the official close by more than 10% are counted, not
    dropped, because dropping them would remove exactly the halted squeezes.
* **Coverage gaps.** Stocks listed after 2023-12-29 only have hourly data from 2024-11-28, a Yahoo
  limit. Renamed tickers lose their hourly history before the rename. Both shrink the cross-section,
  mostly in the first half of the window; see the coverage audit in `results/intraday_summary.json`.
