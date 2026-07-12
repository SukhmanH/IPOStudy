"""
build_universe_t1t4.py — generalized IPO universe for the fade-generalization
study: traditional underwritten IPOs, 2010–present, market cap >= $300M at
listing, tiered T1–T4.

Reuses the legacy spcx-report builder (Ritter x Nasdaq x EDGAR/yfinance share
reconciliation, ADR/multi-class/recycled-ticker guards) with parameter
overrides — the pipeline logic is IDENTICAL to the one that passed the
regression test; only thresholds, output paths, and the newest-cohort seed
merge differ. Every deviation from the legacy build is listed below.

Deviations from legacy build_universe.py (all deliberate, per the brief):
  1. MIN_MCAP_B          5.0  -> 0.30   ($300M floor; tiers assigned below)
  2. NEAR_MISS_FLOOR_B   3.0  -> 0.15   ($150–300M logged 'considered'; below
                                         that dropped WITH a logged reason via
                                         the audit supplement — nothing silent)
  3. CANDIDATE_PRE_FLOOR 2.5  -> 0.15   (pre-gate ~= half the floor, same ratio
                                         convention as legacy 2.5 vs 5.0)
  4. RECENT_CUTOFF       2025-09-15 -> computed ~190 business days back from
                         today (listings after it lack a complete +180 td
                         history; they stay in the audit as 'considered' and
                         may enter SHORT-HORIZON exploratory cells only)
  5. Newest cohort (post-Ritter-coverage) seeded from
     general/data/ipo_seed_universe.csv (class == operating only; 'uncertain'
     rows are NOT ingested — they await adjudication and are logged as skipped).
     Seed rows ride the same EDGAR/yfinance reconciliation (match_method =
     'seed_calendar'); seed rows with offer < $8 cannot pass the legacy
     no-shares candidate gate and are logged as skipped, not silently lost.
  6. Audit supplement: matched-but-never-candidate rows (offer x Ritter shares
     below the pre-floor) are appended to the audit with an explicit reason.

Tiers: T1 $300M–1B · T2 $1–5B · T3 $5–10B · T4 >$10B  (sub-$300M -> 'sub')

Run:  python build_universe_t1t4.py     (first run fetches EDGAR facts for
      every new candidate — hours; fully cached + resumable after)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"c:\Users\brylan\Desktop\spaceX")
GEN = REPO / "general"
sys.path.insert(0, str(REPO / "spcx-report" / "src"))

import utils as u          # noqa: E402
import build_universe as bu  # noqa: E402

# ---------------------------- parameter overrides ----------------------------
bu.MIN_MCAP_B = 0.30
bu.NEAR_MISS_FLOOR_B = 0.15
bu.CANDIDATE_PRE_FLOOR_B = 0.15
# latest listing with a complete +180 td history (190 bdays ~ 180 NYSE tds)
RECENT_CUTOFF = (pd.Timestamp.today().normalize() - pd.offsets.BDay(190)).date().isoformat()
bu.RECENT_CUTOFF = RECENT_CUTOFF

SEED_CSV = GEN / "data" / "ipo_seed_universe.csv"
UNIVERSE_OUT = GEN / "data" / "ipo_universe_t1t4.csv"
AUDIT_OUT = GEN / "data" / "universe_audit_t1t4.csv"
bu.UNIVERSE_OUT = UNIVERSE_OUT   # not used directly (we orchestrate main ourselves)
bu.AUDIT_OUT = AUDIT_OUT

TIER_EDGES = [(10.0, "T4"), (5.0, "T3"), (1.0, "T2"), (0.30, "T1")]


def tier_of(mcap_b) -> str:
    if pd.isna(mcap_b):
        return ""
    for lo, name in TIER_EDGES:
        if mcap_b > lo:
            return name
    return "sub"


# ---------------------------------------------------------------------------
# Seed-calendar merge (newest cohort, post-Ritter coverage)
# ---------------------------------------------------------------------------
def inject_seed(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append operating-class seed-calendar names not already in the Ritter
    match, synthesized into the matched-row schema so assemble() treats them
    identically. Returns (matched_plus, skipped_log)."""
    seed = pd.read_csv(SEED_CSV, keep_default_na=False)
    seed["ipo_dt"] = pd.to_datetime(seed["ipo_date"])
    rit_max = matched["offer_dt"].max()
    have = set(matched["ticker"].astype(str))

    ops = seed[(seed["class"] == "operating") & (seed["ipo_dt"] > rit_max - pd.Timedelta(days=30))]
    skipped = []
    rows = []
    for r in ops.itertuples():
        if r.ticker in have:
            continue  # already covered by Ritter x Nasdaq
        offer = pd.to_numeric(r.offer_price, errors="coerce")
        if pd.isna(offer):
            skipped.append({"ticker": r.ticker, "reason": "seed: missing offer price"})
            continue
        if offer < 8:
            # legacy no-shares candidate gate requires offer >= $8; log, don't lose
            skipped.append({"ticker": r.ticker,
                            "reason": f"seed: offer ${offer:g} < $8 no-shares gate; shares unknown"})
            continue
        rows.append({
            "ritter_ticker": r.ticker, "ticker": r.ticker, "name": r.company,
            "offer_date": int(pd.Timestamp(r.ipo_dt).strftime("%Y%m%d")),
            "year": int(pd.Timestamp(r.ipo_dt).year), "offer_dt": pd.Timestamp(r.ipo_dt),
            "adr": False, "post_issue_shares": np.nan, "offer_price": float(offer),
            "match_method": "seed_calendar",
        })
    # uncertain-class rows: not ingested, but logged for the adjudication queue
    unc = seed[(seed["class"] == "uncertain") & (seed["ipo_dt"] > rit_max - pd.Timedelta(days=30))]
    for r in unc.itertuples():
        skipped.append({"ticker": r.ticker, "reason": "seed: class=uncertain, awaiting adjudication"})

    plus = pd.concat([matched, pd.DataFrame(rows)], ignore_index=True) if rows else matched
    print(f"[seed] injected {len(rows)} seed-calendar names (> {rit_max.date()} - 30d); "
          f"skipped+logged {len(skipped)}")
    return plus, pd.DataFrame(skipped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("BUILD T1-T4 UNIVERSE — $300M floor, 2010+, tiers T1-T4")
    print(f"  RECENT_CUTOFF (complete +180 td) = {RECENT_CUTOFF}")
    print("=" * 68)

    rit = bu.fetch_ritter()
    nasdaq = bu.build_nasdaq_index(range(bu.MIN_YEAR, bu.THIS_YEAR + 1))
    matched = bu.match_offer_price(rit, nasdaq)
    matched, seed_skips = inject_seed(matched)

    df = bu.assemble(matched)
    df["tier"] = df["mcap_listing_b"].map(tier_of)

    # ---- audit supplement: matched-but-never-candidate rows (pre-floor) ----
    m = matched.copy()
    m["free_pre_b"] = m["offer_price"] * m["post_issue_shares"] / 1e9
    verified = bu._load_verified()
    cand_mask = (m["offer_price"].notna()) & (
        (m["free_pre_b"] > bu.CANDIDATE_PRE_FLOOR_B)
        | (m["post_issue_shares"].isna() & (m["offer_price"] >= 8))
        | (m["ticker"].isin(verified.keys()))
    )
    noncand = m[~cand_mask].copy()
    supp = pd.DataFrame({
        "ticker": noncand["ticker"], "ritter_ticker": noncand["ritter_ticker"],
        "name": noncand["name"],
        "first_trade": noncand["offer_dt"].dt.date.astype(str),
        "year": noncand["year"], "offer_price": noncand["offer_price"],
        "adr": noncand["adr"], "ritter_shares": noncand["post_issue_shares"],
        "match_method": noncand["match_method"],
        "bucket": "pre_floor_drop",
        "reason": np.where(
            noncand["offer_price"].isna(),
            "no offer price matched (never priced / not in Nasdaq calendar)",
            "offer x Ritter shares < $" + str(int(bu.CANDIDATE_PRE_FLOOR_B * 1000))
            + "M pre-floor (well below $300M)"),
        "tier": "",
    })
    if len(seed_skips):
        seed_supp = pd.DataFrame({
            "ticker": seed_skips["ticker"], "bucket": "seed_skip",
            "reason": seed_skips["reason"], "tier": "",
        })
        supp = pd.concat([supp, seed_supp], ignore_index=True)

    audit_cols = ["ticker", "ritter_ticker", "name", "first_trade", "yf_first_trade",
                  "year", "offer_price", "adr", "ritter_shares", "edgar_shares",
                  "yf_shares", "est_edgar", "est_yf", "est_ritter", "mcap_listing_b",
                  "mcap_src", "mcap_conf", "match_method", "bucket", "reason", "tier"]
    audit = pd.concat([df, supp], ignore_index=True)
    for c in audit_cols:
        if c not in audit.columns:
            audit[c] = np.nan
    audit.sort_values(["bucket", "mcap_listing_b"], ascending=[True, False])[audit_cols] \
         .to_csv(AUDIT_OUT, index=False)

    # ---- universe output (headline + considered), tiered ----
    keep = df[df["bucket"].isin(["headline", "considered"])].copy()
    keep["_rank"] = (keep["bucket"] == "headline").astype(int)
    keep = (keep.sort_values(["_rank", "mcap_listing_b"], ascending=[False, False])
                .drop_duplicates(subset=["ticker"], keep="first").drop(columns="_rank"))
    out_cols = ["ticker", "bucket", "tier", "offer_price", "mcap_listing_b", "reason",
                "name", "first_trade", "mcap_src", "mcap_conf", "ritter_ticker",
                "match_method"]
    keep.sort_values(["bucket", "mcap_listing_b"], ascending=[True, False])[out_cols] \
        .to_csv(UNIVERSE_OUT, index=False)

    # ---- summary ----
    head = df[df["bucket"] == "headline"]
    print("\n" + "=" * 68)
    print(f"T1-T4 HEADLINE universe (traditional IPO, >$300M, 2010+): n={len(head)}")
    print("\nBy tier:")
    print(head["tier"].value_counts().reindex(["T1", "T2", "T3", "T4"]).to_string())
    print("\nBy era:")
    era = pd.cut(head["year"], [2009, 2014, 2019, 2026],
                 labels=["2010-2014", "2015-2019", "2020+"])
    print(era.value_counts().sort_index().to_string())
    print("\nmcap confidence mix:")
    print(head["mcap_conf"].value_counts().to_string())
    print(f"\nconsidered: n={len(df[df['bucket']=='considered'])}   "
          f"dropped: n={len(df[df['bucket']=='drop'])}   "
          f"pre-floor supplement: n={len(supp)}")
    print(f"\nWrote:\n  {UNIVERSE_OUT}\n  {AUDIT_OUT}")
    return df


if __name__ == "__main__":
    main()
