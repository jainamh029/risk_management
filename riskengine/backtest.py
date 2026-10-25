"""Out-of-sample VaR backtesting with the Kupiec proportion-of-failures test.

For each day t the VaR forecast uses only the `window` returns strictly before
t. A breach is a day on which the realised loss is greater than the forecast.
Windows that contain a missing return produce no forecast for that day.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import chi2, norm

from .var import METHODS


def kupiec_pof(n: int, x: int, p: float) -> tuple[float, float]:
    """Kupiec likelihood-ratio test of the breach rate against p.

    n: number of forecasts, x: number of breaches, p: expected breach rate.
    Returns (LR statistic, p-value) with a chi-square(1) reference.
    Computed in log space; x = 0 and x = n are handled exactly.
    """
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if n <= 0 or x < 0 or x > n:
        raise ValueError("need n > 0 and 0 <= x <= n")
    ll_null = (n - x) * math.log1p(-p) + (x * math.log(p) if x > 0 else 0.0)
    ll_alt = 0.0
    if x > 0:
        ll_alt += x * math.log(x / n)
    if x < n:
        ll_alt += (n - x) * math.log1p(-x / n)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return lr, float(chi2.sf(lr, df=1))


def rolling_forecasts(returns: np.ndarray, window: int, conf: float, method: str) -> np.ndarray:
    """VaR forecast for each day using the previous `window` returns. NaN where unavailable."""
    r = np.asarray(returns, dtype=float)
    out = np.full(r.shape[0], np.nan)
    if r.shape[0] <= window:
        return out
    w = sliding_window_view(r, window)[:-1]  # window ending the day before each forecast day
    ok = ~np.isnan(w).any(axis=1)
    if method == "historical":
        f = np.quantile(-w[ok], conf, axis=1)
    elif method == "normal":
        z = norm.ppf(conf)
        f = -(w[ok].mean(axis=1) - z * w[ok].std(axis=1, ddof=1))
    else:
        raise ValueError(f"unknown method {method!r}")
    idx = np.flatnonzero(ok) + window
    out[idx] = f
    return out


def backtest_series(returns: pd.Series, window: int = 250, confs=(0.95, 0.99), methods=METHODS) -> list[dict]:
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
