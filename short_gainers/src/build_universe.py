"""
build_universe.py -- ticker universe for the "short the top gainers" study.

Source: Nasdaq stock screener (all securities currently listed on NASDAQ,
NYSE and NYSE American), pulled once and cached.  Every row is kept in the
audit file with a reason code; only equity share classes go forward.

Reason codes (first match wins, in this order):
  preferred  -- '^' in symbol or preferred-stock wording in the name
  warrant    -- '.WS' symbol or 'Warrant(s)' in the name
  right      -- 'Right(s)' in the name
  unit       -- 'Unit(s)' in the name (SPAC / offering units)
  debt       -- notes, debentures, bonds
  fund       -- closed-end funds, ETFs/ETNs, physical-metal trusts
  kept       -- common / ordinary shares, ADRs (tagged), SPAC commons (tagged)

KNOWN LIMITATION (stated up front, quantified in the report): this is a
*currently listed* universe.  Names that were delisted, acquired or went
bankrupt before the pull date are absent.  The same is true of the shibui
database used as the independent cross-check.

Run:  python short_gainers/src/build_universe.py
Out:  short_gainers/data/universe_audit.csv, short_gainers/data/universe.csv
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)
SCREENER_CACHE = RAW / "nasdaq_screener.json"
URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"

PREF = re.compile(r"Preferred|Perpetual|\bPfd\b|Cumulative Redeemable|Depositary Shares?.{0,80}Series", re.I)
WARR = re.compile(r"\bWarrants?\b", re.I)
RIGHT = re.compile(r"\bRights?\b", re.I)
UNIT = re.compile(r"\bUnits?\b", re.I)
DEBT = re.compile(r"\bNotes?\b|Debentures?|\bBonds?\b|Subordinated", re.I)
FUND = re.compile(r"\bFund\b|\bETF\b|\bETN\b|Municipal|Physical (Gold|Silver|Platinum|Palladium)|Y'all", re.I)
ADR = re.compile(r"Depositary|\bADS\b|\bADR\b", re.I)
SPAC_NAME = re.compile(r"Acquisition (?:Corp|Co\b|Company|Limited|Ltd|Inc)|Blank Check", re.I)
CEF_INDUSTRIES = {"Finance/Investors Services", "Investment Managers", "Finance Companies"}


def fetch_screener() -> dict:
    if SCREENER_CACHE.exists():
        return json.loads(SCREENER_CACHE.read_text())
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    SCREENER_CACHE.write_text(r.text)
    return r.json()


def classify(sym: str, name: str, industry: str) -> str:
    if "^" in sym or PREF.search(name):
        return "preferred"
    if sym.endswith(".WS") or WARR.search(name):
        return "warrant"
    if RIGHT.search(name):
        return "right"
    if UNIT.search(name):
        return "unit"
    if DEBT.search(name) and not re.search(r"Common|Ordinary", name, re.I):
        return "debt"
    if FUND.search(name) or industry == "Trusts Except Educational Religious and Charitable":
        return "fund"
    if industry in CEF_INDUSTRIES and re.search(r"Shares of Beneficial Interest|\bTrust\b|\bTerm\b", name, re.I):
        return "fund"
    return "kept"


def to_yahoo(sym: str) -> str:
    return sym.replace("/", "-").replace(".", "-")


def main() -> None:
    js = fetch_screener()
    rows = pd.DataFrame(js["data"]["rows"])
    as_of = js["data"].get("asOf") or str(date.today())
    rows["name"] = rows["name"].fillna("")
    rows["industry"] = rows["industry"].fillna("")
    rows["reason"] = [classify(s, n, i) for s, n, i in zip(rows["symbol"], rows["name"], rows["industry"])]
    rows["yahoo"] = rows["symbol"].map(to_yahoo)
    rows["is_adr"] = rows["name"].str.contains(ADR)
    rows["is_spac"] = (rows["industry"] == "Blank Checks") | rows["name"].str.contains(SPAC_NAME)
    rows["is_healthcare"] = rows["sector"].fillna("") == "Health Care"
    rows["screener_as_of"] = as_of
    rows.to_csv(DATA / "universe_audit.csv", index=False)
    kept = rows[rows["reason"] == "kept"].copy()
    cols = ["symbol", "yahoo", "name", "sector", "industry", "country", "ipoyear",
            "marketCap", "is_adr", "is_spac", "is_healthcare", "screener_as_of"]
    kept[cols].to_csv(DATA / "universe.csv", index=False)
    print(f"screener as of {as_of}: {len(rows)} securities")
    print(rows["reason"].value_counts().to_string())
    print(f"kept {len(kept)}  (ADR {kept['is_adr'].sum()}, SPAC {kept['is_spac'].sum()}, "
          f"health care {kept['is_healthcare'].sum()})")


if __name__ == "__main__":
    main()
