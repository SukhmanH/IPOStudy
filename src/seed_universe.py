"""
seed_universe.py — mechanical classification + cached price fetch for the
newest-cohort IPO seed (general\\tickersRaw.txt).

TASK 1: parse tickersRaw.txt, tag each row operating/spac/uncertain by
mechanical rules only (no lookups, no judgment calls).
TASK 2: fetch cached prices (via spcx-report/src/utils.py) for every
non-spac row, flag recycled-ticker mismatches.

No return math, no stats, no charts. Formatting + fetching + logging only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(r"c:\Users\brylan\Desktop\spaceX")
INPUT = REPO / "general" / "tickersRaw.txt"
OUT_CSV = REPO / "general" / "data" / "ipo_seed_universe.csv"

sys.path.insert(0, str(REPO / "spcx-report" / "src"))
import utils as u  # noqa: E402  (cached+retried fetch_history, resolve_first_trade)

SPAC_NAME_RE = re.compile(
    r"(Acquisition|Blank.?Check|Merger Corp|Capital Corp|Holdings? [IVX]+\b|Corp\.? [IVX]+$)",
    re.IGNORECASE,
)
RECYCLE_GAP_DAYS = 270


def parse_month(s: str) -> str:
    """'Jul 10, 2026' -> '2026-07-10' (ISO)."""
    return pd.to_datetime(s.strip(), format="%b %d, %Y").date().isoformat()


def clean_price(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").strip())


# ----------------------------------------------------------------------------
# TASK 1 — parse + classify
# ----------------------------------------------------------------------------
def load_and_classify() -> pd.DataFrame:
    rows = []
    with open(INPUT, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                print(f"  [WARN] line {lineno}: fewer than 4 tab fields, skipping raw text: {line!r}")
                continue
            date_raw, ticker, company, offer_raw = parts[0], parts[1], parts[2], parts[3]
            try:
                ipo_date = parse_month(date_raw)
            except Exception as e:
                print(f"  [WARN] line {lineno}: unparseable date {date_raw!r} ({e}); skipping")
                continue
            ticker = ticker.strip()
            company = company.strip()

            try:
                offer_price = clean_price(offer_raw)
            except Exception:
                # Missing offer price ("-") — keep the row (hard rule: nothing data-
                # shaped is dropped), classify uncertain; price rules can't fire.
                print(f"  [WARN] line {lineno}: no offer price ({offer_raw!r}) -> uncertain: {ticker}")
                rows.append({
                    "ipo_date": ipo_date, "ticker": ticker, "company": company,
                    "offer_price": None, "class": "uncertain",
                    "rule_hit": "missing_offer_price",
                })
                continue

            is_ten = abs(offer_price - 10.00) < 1e-9
            name_hits_spac_regex = bool(SPAC_NAME_RE.search(company))
            ticker_u_at_ten = is_ten and ticker.upper().endswith("U")

            spac_condition_a = is_ten and name_hits_spac_regex   # offer==10 AND name matches
            spac_condition_b = ticker_u_at_ten                   # ticker ends U AND offer==10

            if spac_condition_a or spac_condition_b:
                cls, rule = "spac", ("name_regex+offer10" if spac_condition_a else "ticker_U+offer10")
            elif not is_ten and not name_hits_spac_regex:
                cls, rule = "operating", "offer!=10+name_clean"
            else:
                # exactly one signal fired: offer==10 w/ clean name, OR offer!=10 w/ spac-like name
                cls = "uncertain"
                rule = "offer10_clean_name" if is_ten else "offer!=10_spac_name"

            rows.append({
                "ipo_date": ipo_date,
                "ticker": ticker,
                "company": company,
                "offer_price": offer_price,
                "class": cls,
                "rule_hit": rule,
            })

    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
# TASK 2 — cached fetch (non-spac only)
# ----------------------------------------------------------------------------
def fetch_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fetch_status"] = ""
    df["first_cached_date"] = ""

    to_fetch = df[df["class"] != "spac"]
    n = len(to_fetch)
    print(f"\nFetching {n} non-spac tickers (cached; slow on first run)...")

    for i, (idx, row) in enumerate(to_fetch.iterrows(), start=1):
        tkr = row["ticker"]
        if i % 50 == 0 or i == n:
            print(f"  progress {i}/{n}")
        try:
            hist = u.fetch_history(tkr, raw=True)
        except Exception as e:
            df.at[idx, "fetch_status"] = f"error:{str(e)[:60]}"
            continue

        if hist is None or len(hist) == 0:
            df.at[idx, "fetch_status"] = "no_data"
            continue

        first_dt = u.first_trade_date(hist)
        df.at[idx, "first_cached_date"] = first_dt.date().isoformat()

        expected = pd.Timestamp(row["ipo_date"])
        gap_days = abs((first_dt - expected).days)
        if gap_days > RECYCLE_GAP_DAYS:
            df.at[idx, "fetch_status"] = "recycled_mismatch"
        else:
            df.at[idx, "fetch_status"] = "ok"

    return df


def fetch_benchmarks() -> None:
    for b in ["QQQ", "SPY", "IWM", "^VIX"]:
        print(f"  benchmark {b}...")
        try:
            hist = u.fetch_history(b, raw=True)
            status = "ok" if hist is not None and len(hist) else "no_data"
        except Exception as e:
            status = f"error:{str(e)[:60]}"
        print(f"    {b}: {status}")


def main():
    print("SEED UNIVERSE — parse, classify, cached fetch (general/tickersRaw.txt)")
    print("=" * 68)

    df = load_and_classify()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nParsed {len(df)} rows.")
    counts = df["class"].value_counts()
    print("\nClass counts:")
    for cls, cnt in counts.items():
        print(f"  {cls:<10} {cnt}")

    df = fetch_and_flag(df)

    print("\nFetching benchmarks (QQQ, SPY, IWM, ^VIX)...")
    fetch_benchmarks()

    df.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 68)
    print("FINAL REPORT")
    print("=" * 68)
    print("\nCounts by class:")
    for cls, cnt in df["class"].value_counts().items():
        print(f"  {cls:<10} {cnt}")

    fetched = df[df["class"] != "spac"]
    print("\nFetch status counts (non-spac rows):")
    for status, cnt in fetched["fetch_status"].value_counts().items():
        print(f"  {status:<20} {cnt}")

    non_ok = fetched[fetched["fetch_status"] != "ok"]
    print(f"\nTickers with status != ok ({len(non_ok)}):")
    for _, r in non_ok.iterrows():
        print(f"  {r['ticker']:<8} {r['fetch_status']}")

    print("\nWrote:")
    print(f"  {OUT_CSV}")


if __name__ == "__main__":
    main()
