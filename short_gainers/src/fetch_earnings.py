"""
fetch_earnings.py -- historical earnings-announcement dates from the Nasdaq
earnings calendar (api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD),
one request per weekday, 2009-12-01 .. 2026-11-23.

Used ONLY to tag whether a top-gainer day coincides with an earnings report
(report dated t-1 or t; Nasdaq's historical rows mostly lack the BMO/AMC time
stamp, so the two-day window is the conservative tag).  Like the price
universe, the calendar covers currently listed companies (survivor-only).

Cache: data/raw/earnings/<date>.json (git-ignored).  Output (committed):
data/earnings_dates.csv.gz with columns date, symbol, time, eps, eps_forecast.

Run:  python short_gainers/src/fetch_earnings.py
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
CACHE = DATA / "raw" / "earnings"
CACHE.mkdir(parents=True, exist_ok=True)
URL = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
HDRS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def get_day(d: str) -> str:
    f = CACHE / f"{d}.json"
    if f.exists():
        return "cached"
    for attempt in range(5):
        try:
            r = requests.get(URL.format(d=d), headers=HDRS, timeout=30)
            if r.status_code == 200:
                js = r.json()
                rows = ((js.get("data") or {}).get("rows")) or []
                f.write_text(json.dumps(rows))
                return f"ok:{len(rows)}"
            time.sleep(3 * (attempt + 1))
        except Exception:
            time.sleep(3 * (attempt + 1))
    return "failed"


def main() -> None:
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2009-12-01", "2026-11-23")]
    with ThreadPoolExecutor(max_workers=5) as ex:
        res = list(ex.map(get_day, days))
    print(pd.Series([r.split(":")[0] for r in res]).value_counts().to_dict(), flush=True)
    recs = []
    for d in days:
        f = CACHE / f"{d}.json"
        if not f.exists():
            continue
        for row in json.loads(f.read_text()):
            recs.append({"date": d, "symbol": row.get("symbol"), "time": row.get("time"),
                         "eps": row.get("eps"), "eps_forecast": row.get("epsForecast")})
    out = pd.DataFrame(recs)
    out.to_csv(DATA / "earnings_dates.csv.gz", index=False, compression="gzip")
    print(f"earnings rows {len(out)}, symbols {out['symbol'].nunique()}, "
          f"days with rows {out['date'].nunique()}/{len(days)}", flush=True)


if __name__ == "__main__":
    main()
