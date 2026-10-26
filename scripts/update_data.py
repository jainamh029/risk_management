"""Bring the downloaded price snapshot up to date without downloading the whole history again.

    python scripts/update_data.py                      # add every finished trading day (including today after 5 pm New York time)
    python scripts/update_data.py --end 2026-11-02     # exclusive end date
    python scripts/update_data.py --rebuild            # then regenerate results, README tables and the web report
    python scripts/update_data.py --full               # download everything again (like fetch_data.py)

Only the days after the last stored date are downloaded, plus a one-week overlap that is compared with the stored prices.
Yahoo adjusts old prices after a split or dividend; a ticker whose overlap no longer matches is downloaded in full and
replaced, so the stored history always equals what a complete download would give, unless the new history covers
less than 90% of the stored dates (for example a symbol that now belongs to another company): then the old history
is kept and reported. The default end date is today, or tomorrow on a weekday after 5 pm New York time, so an unfinished
trading day is never stored. Tickers that failed before are not retried unless --retry-failed is given.
If nothing is new, no file is touched; weekends and market holidays simply give no new rows.
Create the first snapshot with scripts/fetch_data.py.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import data  # noqa: E402

REBUILD = ["scripts/run_pipeline.py", "scripts/check_sp500_membership.py", "scripts/make_readme_tables.py", "scripts/build_site.py"]


def default_end(now=None) -> str:
    """Exclusive end date that includes every finished trading day: tomorrow on a weekday from 5 pm New York time
    (the US market closes at 4 pm), otherwise today. Weekends and holidays return no rows, which is harmless."""
    ny = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    day = ny.date() + timedelta(days=1) if ny.weekday() < 5 and ny.hour >= 17 else ny.date()
    return day.isoformat()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="exclusive end date (default: see default_end: only finished trading days are stored)")
    ap.add_argument("--retry-failed", action="store_true", help="also retry the tickers Yahoo returned nothing for last time (slow: they are usually delisted)")
    ap.add_argument("--full", action="store_true", help="download the whole history again instead of only the new days")
    ap.add_argument("--rebuild", action="store_true", help="after a change, rerun the pipeline, the membership check, the README tables and the web report")
    args = ap.parse_args(argv)

    try:
        rep = data.update_snapshot(data.load_universe()["ticker"].tolist(), args.end or default_end(), full=args.full, retry_failed=args.retry_failed)
    except FileNotFoundError as e:
        print(e)
        return 1
    except ValueError as e:
        print(f"Not updated: {e}")
        return 1

    print(f"{rep['mode']} update: last stored date {rep['last_date_before']} -> {rep['last_date_after']} ({rep['new_rows']} new day(s))")
    print(f"  tickers extended with new days: {len(rep['appended'])}; downloaded in full again: {len(rep['refreshed'])}; added: {len(rep['added'])}")
    for t, why in sorted(rep["refreshed"].items())[:10]:
        print(f"    {t}: {why}")
    for t, why in sorted(rep["refused"].items()):
        print(f"  KEPT OLD HISTORY for {t}: {why}")
    if rep["skipped_known_failures"]:
        print(f"  {len(rep['skipped_known_failures'])} ticker(s) that failed before were not retried (use --retry-failed)")
    if rep["failed"]:
        print(f"  could not be downloaded: {', '.join(sorted(rep['failed']))}")
    print(f"  risk-free rate: {rep['risk_free']}")
    if not rep["written"]:
        print("Already up to date: no file was changed.")
        return 0
    print("Snapshot written.")
    if args.rebuild:
        for script in REBUILD:
            print(f"running {script} ...")
            if subprocess.run([sys.executable, script], cwd=ROOT).returncode != 0:
                print(f"{script} failed; the snapshot is updated but the outputs are not all regenerated.")
                return 1
    else:
        print("Next: python scripts/run_pipeline.py (or rerun this with --rebuild) to regenerate the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
