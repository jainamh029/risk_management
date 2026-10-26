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
    data_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(data_dir / PRICES_FILE)
    rf.to_frame().to_parquet(data_dir / RF_FILE)
    (data_dir / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))


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
