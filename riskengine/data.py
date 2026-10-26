"""Real market data: download, snapshot, and return calculation.

Rules enforced here:
- Prices come only from Yahoo Finance through yfinance. Nothing is generated.
- Missing values are never filled. A ticker that did not trade yet, or has
  stopped trading, has NaN prices on those dates and NaN returns.
- Every download writes a metadata file that records what was requested, what
  came back, and what failed.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PRICES_FILE = "prices.parquet"
RF_FILE = "tbill_13w.parquet"
META_FILE = "prices_meta.json"
RF_TICKER = "^IRX"  # 13-week Treasury bill yield, quoted in percent per year


def load_universe(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_DIR / "universe.csv"
    return pd.read_csv(path)


def _close_from_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract the Close column per ticker from a yfinance download."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        # group_by="ticker": level 0 is the ticker, level 1 the field
        out = {}
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                col = raw[t]["Close"]
                if not col.dropna().empty:
                    out[t] = col
        return pd.DataFrame(out)
    # single ticker download: flat columns
    if "Close" in raw.columns and len(tickers) == 1 and not raw["Close"].dropna().empty:
        return pd.DataFrame({tickers[0]: raw["Close"]})
    return pd.DataFrame()


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    batch_size: int = 20,
    retries: int = 3,
    pause: float = 2.0,
) -> tuple[pd.DataFrame, dict]:
    """Download adjusted close prices. Returns (prices, report).

    `end` is exclusive (yfinance convention). A ticker that returns no data
    after all retries is recorded in report["failed"] with the reason.
    """
    import yfinance as yf  # imported here so the rest of the package works without network

    warnings.filterwarnings("ignore", module=r"yfinance.*")
    frames: dict[str, pd.Series] = {}
    failed: dict[str, str] = {}

    def attempt(batch: list[str]) -> pd.DataFrame:
        last_err = None
        for k in range(retries):
            try:
                raw = yf.download(
                    batch if len(batch) > 1 else batch[0],
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=True,
                )
                got = _close_from_download(raw, batch)
                if not got.empty:
                    return got
                last_err = "empty response"
            except Exception as e:  # network or parsing error: record it, then retry
                last_err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(pause * (k + 1))
        raise RuntimeError(last_err or "unknown")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            got = attempt(batch)
        except RuntimeError:
            got = pd.DataFrame()
        for t in got.columns:
            frames[t] = got[t]
        missing = [t for t in batch if t not in frames]
        for t in missing:  # one more try, one ticker at a time
            try:
                one = attempt([t])
                frames[t] = one[t]
            except RuntimeError as e:
                failed[t] = str(e)

    prices = pd.DataFrame(frames).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices[[t for t in tickers if t in prices.columns]]
    report = {
        "requested": len(tickers),
        "returned": int(prices.shape[1]),
        "failed": failed,
    }
    return prices, report


def fetch_risk_free(start: str, end: str) -> pd.Series:
    """13-week T-bill yield (percent per year) from Yahoo Finance (^IRX)."""
    import yfinance as yf

    warnings.filterwarnings("ignore", module=r"yfinance.*")
    raw = yf.download(RF_TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for {RF_TICKER} between {start} and {end}")
    if "Close" not in raw.columns:
        raise ValueError(f"the Yahoo Finance response for {RF_TICKER} has no Close column (columns: {list(raw.columns)[:5]})")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):  # one column per symbol; there is exactly one symbol here
        close = close.iloc[:, 0]
    s = close.dropna()
    if s.empty:
        raise ValueError(f"the Yahoo Finance response for {RF_TICKER} has no observations between {start} and {end}")
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s.name = RF_TICKER
    return s


def save_snapshot(prices: pd.DataFrame, rf: pd.Series, meta: dict, data_dir: Path = DATA_DIR) -> None:
    """Write the three snapshot files. Each is written to a temporary name first and then renamed over the old file,
    so a failure part-way never leaves a half-written file where a good snapshot was."""
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = {name: data_dir / f".{name}.tmp" for name in (PRICES_FILE, RF_FILE, META_FILE)}
    try:
        prices.to_parquet(tmp[PRICES_FILE])
        rf.to_frame().to_parquet(tmp[RF_FILE])
        tmp[META_FILE].write_text(json.dumps(meta, indent=2, sort_keys=True))
        for name, path in tmp.items():
            os.replace(path, data_dir / name)
    finally:
        for path in tmp.values():
            if path.exists():
                path.unlink()


def load_snapshot(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.Series, dict]:
    p = data_dir / PRICES_FILE
    if not p.exists():
        raise FileNotFoundError(
            f"No price snapshot at {p}. Create it with: python scripts/fetch_data.py"
        )
    prices = pd.read_parquet(p)
    rf = pd.read_parquet(data_dir / RF_FILE).iloc[:, 0]
    meta = json.loads((data_dir / META_FILE).read_text())
    return prices, rf, meta


def snapshot_report(prices: pd.DataFrame) -> dict:
    """Describe gaps in the downloaded prices. Nothing is changed."""
    first = prices.apply(pd.Series.first_valid_index)
    last = prices.apply(pd.Series.last_valid_index)
    internal_gaps = {}
    for t in prices.columns:
        s = prices[t].loc[first[t] : last[t]]
        n = int(s.isna().sum())
        if n:
            internal_gaps[t] = n
    return {
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
        "n_dates": int(len(prices)),
        "listed_after_start": sorted(t for t in prices.columns if first[t] > prices.index.min()),
        "stopped_before_end": sorted(t for t in prices.columns if last[t] < prices.index.max()),
        "internal_gap_counts": internal_gaps,
        "fewer_than_250_valid_prices": {
            t: int(prices[t].notna().sum()) for t in prices.columns if prices[t].notna().sum() < 250
        },
    }


def validate_prices(prices: pd.DataFrame) -> None:
    """Documented policy for a price table. Raises ValueError if it is violated.

    - The index must be strictly increasing: sorted and without duplicate timestamps (returns are taken between
      adjacent rows, so an unsorted or duplicated index would silently pair the wrong days).
    - A missing price is NaN. Every observed (non-NaN) price must be finite and strictly positive; zero, negative
      and infinite prices are data errors, not missing values, and are never converted or dropped.
    """
    idx = prices.index
    if not idx.is_unique:
        raise ValueError(f"price index has duplicate timestamps (for example {idx[idx.duplicated()][0]!r}); remove or aggregate them explicitly")
    if not idx.is_monotonic_increasing:
        raise ValueError("price index is not sorted in increasing order; sort it explicitly before computing returns")
    values = prices.to_numpy(dtype=float)
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError(f"prices contain {int((~np.isfinite(observed)).sum())} infinite value(s); use NaN for a missing price")
    if (observed <= 0).any():
        raise ValueError(f"observed prices must be strictly positive; found {int((observed <= 0).sum())} value(s) <= 0 (use NaN for a missing price)")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns. NaN wherever today's or yesterday's price is NaN. No filling.

    The price table must satisfy validate_prices (strictly increasing index; observed prices finite and positive).
    """
    validate_prices(prices)
    return prices / prices.shift(1) - 1.0


def risk_free_annual(rf_percent: pd.Series, start=None, end=None) -> float:
    """Average 13-week T-bill yield over a window, as a decimal per year."""
    s = rf_percent
    if start is not None:
        s = s.loc[pd.Timestamp(start):]
    if end is not None:
        s = s.loc[: pd.Timestamp(end)]
    return float(s.mean() / 100.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- incremental update
OVERLAP_DAYS = 7  # calendar days before the last stored date that are downloaded again, to check the stored prices still agree
ADJUSTMENT_TOLERANCE = 1e-6  # largest relative difference between a stored and a re-downloaded price that counts as "the same"
MIN_HISTORY_COVERAGE = 0.9  # a re-downloaded history replaces the stored one only if it has a price on at least this share of the stored dates


def update_snapshot(
    tickers: list[str],
    end: str,
    data_dir: Path = DATA_DIR,
    full: bool = False,
    overlap_days: int = OVERLAP_DAYS,
    tolerance: float = ADJUSTMENT_TOLERANCE,
    retry_failed: bool = False,
    min_coverage: float = MIN_HISTORY_COVERAGE,
) -> dict:
    """Bring the stored snapshot up to date without downloading the whole history again.

    `end` is exclusive (yfinance convention). The rule, in plain words:

    1. Tickers already in the snapshot are downloaded only from `overlap_days` before the last stored date.
    2. The re-downloaded overlap is compared with the stored prices. Yahoo's prices are adjusted for splits and
       dividends after the fact, so a dividend or split since the last download changes every older price. If the overlap
       agrees (relative difference at most `tolerance`), only the rows after the last stored date are appended.
       If it disagrees, or there is no overlap to check, the stored history of that ticker is stale and its whole history
       is downloaded again and replaces it. Nothing is rescaled or patched.
    3. Tickers in `tickers` that are not in the snapshot are downloaded in full. Tickers that failed before (listed in
       the metadata, usually delisted names) are skipped unless `retry_failed` is set, so a daily run does not spend
       minutes retrying them.
    4. The risk-free series is handled the same way.
    5. `full=True` downloads everything again, like scripts/fetch_data.py.
    6. A re-downloaded history replaces a stored one only if it has a price on at least `min_coverage` of the stored
       dates. Otherwise the provider is describing something else (a symbol reassigned to another company, a
       truncated response): the stored history is kept, no new prices are attached to it, and the ticker is listed
       under `refused` in the report.

    The result is validated (sorted unique dates, positive finite prices) before anything is written, the files are
    replaced atomically, and the snapshot is not touched at all if there is nothing new. Returns a report.
    """
    prices, rf, meta = load_snapshot(data_dir)
    start_all = meta["requested_start"]
    last = prices.index.max()
    if pd.Timestamp(end) <= last:
        raise ValueError(f"end ({end}) must be after the last stored date ({last.date()}); end is exclusive")
    universe = list(dict.fromkeys(list(tickers)))
    stored = list(prices.columns)
    report = {"mode": "full" if full else "incremental", "last_date_before": str(last.date()), "appended": [], "refreshed": {}, "refused": {}, "added": [], "failed": {}, "kept_without_new_data": [], "skipped_known_failures": [], "risk_free": ""}

    if full:
        new_prices, rep = fetch_prices(list(dict.fromkeys(universe + stored)), start_all, end)
        refresh_frames, to_refresh = new_prices, list(new_prices.columns)
        report["refreshed"] = {t: "full download requested" for t in to_refresh}
        report["failed"] = dict(rep["failed"])
        appended_rows = pd.DataFrame(index=pd.DatetimeIndex([], name=prices.index.name))
    else:
        start = (last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        recent, rep_recent = fetch_prices(stored, start, end)
        to_refresh: list[str] = []
        append_cols: list[str] = []
        for t in stored:
            if t not in recent.columns:
                report["kept_without_new_data"].append(t)  # nothing came back: the stored history stays as it is
                continue
            new_s = recent[t].dropna()
            overlap = new_s.index[new_s.index <= last].intersection(prices[t].dropna().index)
            after = new_s[new_s.index > last]
            if len(overlap):
                rel = ((new_s.loc[overlap] - prices[t].loc[overlap]).abs() / prices[t].loc[overlap]).max()
                if rel > tolerance:
                    to_refresh.append(t)
                    report["refreshed"][t] = f"stored prices no longer match Yahoo (largest relative difference {rel:.2e}): a split, dividend or restatement"
                    continue
            elif len(after):
                to_refresh.append(t)  # new prices but nothing to compare them with
                report["refreshed"][t] = "no overlapping days to check the stored prices against"
                continue
            if len(after):
                append_cols.append(t)
        known_failed = set(meta.get("tickers_failed") or {})
        missing = [t for t in universe if t not in stored and (retry_failed or t not in known_failed)]
        report["skipped_known_failures"] = [t for t in universe if t not in stored and t in known_failed and not retry_failed]
        report["added"] = missing
        fetch_full = to_refresh + missing
        if fetch_full:
            refresh_frames, rep_full = fetch_prices(fetch_full, start_all, end)
            report["failed"] = dict(rep_full["failed"])
            report["added"] = [t for t in missing if t in refresh_frames.columns]
        else:
            refresh_frames = pd.DataFrame(index=pd.DatetimeIndex([], name=prices.index.name))
        for t in to_refresh:
            if t not in refresh_frames.columns:
                report["refreshed"][t] += "; the refresh failed, so the old history was kept"
        to_refresh = [t for t in to_refresh if t in refresh_frames.columns]
        appended_rows = recent.loc[recent.index > last, append_cols].dropna(how="all")
        report["appended"] = append_cols

    for t in [c for c in refresh_frames.columns if c in stored]:
        old_dates = prices[t].dropna().index
        covered = len(old_dates.intersection(refresh_frames[t].dropna().index)) / len(old_dates) if len(old_dates) else 1.0
        if covered < min_coverage:
            report["refused"][t] = (
                f"the new download has prices on only {covered:.0%} of the {len(old_dates)} stored dates (needs {min_coverage:.0%}); "
                "the symbol may now belong to another company or the response is incomplete, so the stored history was kept and no new prices were added"
            )
            report["refreshed"].pop(t, None)
            refresh_frames = refresh_frames.drop(columns=t)
    union = prices.index.union(appended_rows.index).union(refresh_frames.index).sort_values()
    result = prices.reindex(union)
    if len(appended_rows):
        result.loc[appended_rows.index, appended_rows.columns] = appended_rows
    for t in refresh_frames.columns:
        result[t] = refresh_frames[t].reindex(union)
    result = result[stored + [t for t in refresh_frames.columns if t not in stored]]
    result.index = pd.DatetimeIndex(result.index).as_unit(prices.index.unit)  # keep the stored time resolution so the file looks the same as a complete download
    validate_prices(result)

    # risk-free series: append the new days if the overlap agrees, otherwise download it all again
    rf_full = full
    if not full:
        rf_start = (last - pd.Timedelta(days=overlap_days)).strftime("%Y-%m-%d")
        new_rf = fetch_risk_free(rf_start, end)
        both = new_rf.index.intersection(rf.index)
        if len(both) and float((new_rf.loc[both] - rf.loc[both]).abs().max()) <= 1e-9:
            rf_new = new_rf[new_rf.index > rf.index.max()]
            rf_result = pd.concat([rf, rf_new]) if len(rf_new) else rf
            report["risk_free"] = f"{len(rf_new)} new day(s) appended"
        else:
            rf_full = True
    if rf_full:
        rf_result = fetch_risk_free(start_all, end)
        report["risk_free"] = "downloaded again in full" + ("" if full else " (the overlap did not match or could not be checked)")
    rf_result = rf_result[~rf_result.index.duplicated(keep="last")].sort_index()
    rf_result.index = pd.DatetimeIndex(rf_result.index).as_unit(rf.index.unit)
    rf_result.name = RF_TICKER

    changed = not result.equals(prices) or not rf_result.equals(rf) or set(result.columns) != set(prices.columns)
    report["new_rows"] = int(len(result) - len(prices))
    report["last_date_after"] = str(result.index.max().date())
    report["written"] = bool(changed)
    if not changed:
        return report
    snap = snapshot_report(result)
    failed = {t: r for t, r in (meta.get("tickers_failed") or {}).items() if t not in result.columns}
    failed.update({t: r for t, r in report["failed"].items() if t not in result.columns})
    new_meta = {
        **meta,
        "first_downloaded_utc": meta.get("first_downloaded_utc", meta["downloaded_utc"]),
        "downloaded_utc": utc_now(),
        "requested_end_exclusive": end,
        "tickers_requested": len(set(universe) | set(stored)),
        "tickers_returned": int(result.shape[1]),
        "tickers_failed": failed,
        **snap,
        "update_log": list(meta.get("update_log", []))
        + [{"utc": utc_now(), "mode": report["mode"], "last_date_before": report["last_date_before"], "last_date_after": report["last_date_after"], "new_rows": report["new_rows"],
            "tickers_appended": len(report["appended"]), "tickers_refreshed_in_full": sorted(report["refreshed"]), "tickers_refused": sorted(report["refused"]), "tickers_added": report["added"], "risk_free": report["risk_free"]}],
    }
    save_snapshot(result, rf_result, new_meta, data_dir)
    return report
