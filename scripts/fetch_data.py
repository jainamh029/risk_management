"""Download the real price snapshot used by every analysis and test.

    python scripts/fetch_data.py                       # default window
    python scripts/fetch_data.py --start 2019-01-01 --end 2026-10-24

`--end` is exclusive (Yahoo Finance convention). The snapshot and a metadata
file are written to data/. The price file is git-ignored; the metadata file
is committed so the provenance of every result is recorded.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskengine import data  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2019-01-01")
ap.add_argument("--end", default="2026-10-24", help="exclusive end date")
args = ap.parse_args()

uni = data.load_universe()
tickers = uni["ticker"].tolist()
print(f"Downloading {len(tickers)} tickers from {args.start} to {args.end} (exclusive end)...")
prices, rep = data.fetch_prices(tickers, args.start, args.end)
rf = data.fetch_risk_free(args.start, args.end)
snap = data.snapshot_report(prices)

meta = {
    "source": "Yahoo Finance via yfinance, auto_adjust=True (split and dividend adjusted close)",
    "downloaded_utc": data.utc_now(),
    "requested_start": args.start,
    "requested_end_exclusive": args.end,
    "risk_free_series": data.RF_TICKER,
    "risk_free_note": "13-week T-bill yield in percent per year; analyses use its average over the analysis window",
    "tickers_requested": rep["requested"],
    "tickers_returned": rep["returned"],
    "tickers_failed": rep["failed"],
    **snap,
}
data.save_snapshot(prices, rf, meta)
print(f"returned {rep['returned']}/{rep['requested']}; failed {len(rep['failed'])}")
print(f"dates {snap['first_date']} to {snap['last_date']} ({snap['n_dates']} rows)")
print(f"listed after start: {len(snap['listed_after_start'])}; stopped before end: {len(snap['stopped_before_end'])}; tickers with internal gaps: {len(snap['internal_gap_counts'])}")
