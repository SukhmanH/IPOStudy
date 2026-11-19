"""
run_all.py - one command for the whole Anthropic (ANT) study:

    python Anthropic/src/run_all.py              # cached prices; refreshes ANT + benchmarks
    python Anthropic/src/run_all.py --offline    # no network: recompute from committed data
    python Anthropic/src/run_all.py --refresh-all

Steps: fetch_cohort -> build_paths -> applicability -> price_targets -> make_figs.
Re-run after 16:15 ET on 2026-11-19 (or any later day) and the price targets
re-anchor automatically on ANT's actual day-0 close and day-0 high.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import applicability  # noqa: E402
import build_paths  # noqa: E402
import fetch_cohort  # noqa: E402
import make_figs  # noqa: E402
import price_targets  # noqa: E402


def main():
    if "--offline" not in sys.argv:
        fetch_cohort.main()
    for step in (build_paths, applicability, price_targets, make_figs):
        print("\n" + "#" * 80 + f"\n# {step.__name__}\n" + "#" * 80)
        step.main()


if __name__ == "__main__":
    main()
