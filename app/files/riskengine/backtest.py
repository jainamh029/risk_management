"""Out-of-sample VaR backtesting with the Kupiec proportion-of-failures test.

For each day t the VaR forecast uses only the `window` returns strictly before
t. A breach is a day on which the realised loss is greater than the forecast.
Windows that contain a missing return produce no forecast for that day.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2, norm

from .var import METHODS, validate_confidence


def validate_window(window) -> int:
    """The rolling window must be an integer of at least 2 (a standard deviation needs two observations). Booleans are rejected."""
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, numbers.Integral):
        raise ValueError(f"window must be an integer number of trading days, got {window!r} ({type(window).__name__})")
    if window < 2:
        raise ValueError(f"window must be at least 2 trading days, got {window}")
    return int(window)


def _count(value, name: str) -> int:
    """A count must be an integer (an integer-valued float is accepted); booleans, NaN and fractions are rejected."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a whole number, not a boolean ({value!r})")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{name} must be a whole number, got {value!r}")


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec likelihood-ratio test of the breach rate against p.

    n: number of forecasts, x: number of breaches, p: expected breach rate.
    Returns (LR statistic, p-value) with a chi-square(1) reference.
    Computed in log space; x = 0 and x = n are handled exactly.
    """
    n, x = _count(n, "n"), _count(x, "x")
    if isinstance(p, (bool, np.bool_)) or not isinstance(p, numbers.Real) or not (0.0 < p < 1.0):
        raise ValueError(f"p must be a real number strictly between 0 and 1, got {p!r}")
    if n <= 0 or x < 0 or x > n:
        raise ValueError(f"need n > 0 and 0 <= x <= n, got n={n}, x={x}")
    ll_null = (n - x) * math.log1p(-p) + (x * math.log(p) if x > 0 else 0.0)
    ll_alt = 0.0
    if x > 0:
        ll_alt += x * math.log(x / n)
    if x < n:
        ll_alt += (n - x) * math.log1p(-x / n)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return lr, float(chi2.sf(lr, df=1))


def rolling_forecasts(returns: np.ndarray, window: int, conf: float, method: str) -> np.ndarray:
    """VaR forecast for each day using the previous `window` returns. NaN where unavailable.

    The confidence level and the method are validated before anything is calculated, with the same rules as the
    public VaR functions (a real scalar strictly between 0 and 1; booleans, strings, non-scalars, NaN and infinities
    raise ValueError). The history may contain missing returns (NaN): a window that contains one has no forecast, and the
    other windows of the same history are still forecast. It must not contain an infinite value (positive or negative):
    that raises ValueError, wherever it is, because infinity is not treated as missing and is never replaced.
    """
    c = validate_confidence(conf)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    window = validate_window(window)
    r = np.asarray(returns, dtype=float)
    infinite = np.flatnonzero(np.isinf(r))
    if infinite.size:  # infinity is neither a missing return nor a valid one; it is never converted to NaN
        raise ValueError(
            f"returns contain {infinite.size} infinite value(s) (first at position {int(infinite[0])}); a rolling VaR forecast cannot use them. "
            "Use NaN for a missing return (the windows containing it then have no forecast); an infinite return is a data error."
        )
    out = np.full(r.shape[0], np.nan)
    if r.shape[0] <= window:
        return out
    w = sliding_window_view(r, window)[:-1]  # window ending the day before each forecast day
    ok = ~np.isnan(w).any(axis=1)
    if method == "historical":
        f = np.quantile(-w[ok], c, axis=1)
    else:
        z = norm.ppf(c)
        f = -(w[ok].mean(axis=1) - z * w[ok].std(axis=1, ddof=1))
    idx = np.flatnonzero(ok) + window
    out[idx] = f
    return out


def backtest_series(returns: pd.Series, window: int = 250, confs=(0.95, 0.99), methods=METHODS) -> list[dict]:
    window = validate_window(window)
    rows = []
    r = returns.values.astype(float)
    for m in methods:
        for c in confs:
            f = rolling_forecasts(r, window, c, m)
            valid = ~np.isnan(f) & ~np.isnan(r)
            n = int(valid.sum())
            if n == 0:
                continue
            breaches = int((-r[valid] > f[valid]).sum())
            lr, pv = kupiec_pof(n, breaches, 1.0 - c)
            rows.append(
                {
                    "ticker": returns.name,
                    "method": m,
                    "confidence": c,
                    "n_forecasts": n,
                    "breaches": breaches,
                    "breach_rate": breaches / n,
                    "expected_rate": 1.0 - c,
                    "kupiec_lr": lr,
                    "kupiec_p": pv,
                }
            )
    return rows


def backtest_universe(returns: pd.DataFrame, window: int = 250, confs=(0.95, 0.99), min_forecasts: int = 250) -> pd.DataFrame:
    window = validate_window(window)
    rows = []
    for t in returns.columns:
        s = returns[t]
        if s.notna().sum() < window + min_forecasts:
            continue
        rows += backtest_series(s, window, confs)
    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """One row per method and confidence level.

    share_rejected: fraction of tickers whose Kupiec p-value is below alpha.
    Under a correct model about `alpha` of tickers would be rejected by chance;
    tickers are cross-sectionally dependent, so that figure is approximate.
    """
    if bt.empty:  # no ticker had enough history: an empty summary with the usual columns, not a KeyError
        return pd.DataFrame(columns=["method", "confidence", "n_tickers", "total_forecasts", "total_breaches", "mean_ticker_breach_rate", "share_rejected", "pooled_breach_rate", "expected_rate", "zero_breach_tickers"])
    g = bt.groupby(["method", "confidence"])
    out = g.agg(
        n_tickers=("ticker", "nunique"),
        total_forecasts=("n_forecasts", "sum"),
        total_breaches=("breaches", "sum"),
        mean_ticker_breach_rate=("breach_rate", "mean"),
        share_rejected=("kupiec_p", lambda p: float((p < alpha).mean())),
    ).reset_index()
    out["pooled_breach_rate"] = out["total_breaches"] / out["total_forecasts"]
    out["expected_rate"] = 1.0 - out["confidence"]
    out["zero_breach_tickers"] = g["breaches"].apply(lambda b: int((b == 0).sum())).values
    return out
