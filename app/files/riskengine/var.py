"""One-day Value at Risk and Expected Shortfall.

Conventions
- Inputs are daily simple returns.
- VaR and ES are reported as positive fractions of position value (a loss of
  0.03 means 3 percent). The horizon is one trading day. No annualisation and
  no square-root-of-time scaling is applied to VaR or ES. (The descriptive
  statistics in metrics.py are annualised with 252 trading days.)
- conf=0.99 means the loss that is exceeded on about 1 percent of days.

Methods
- historical: empirical quantile of losses; ES is the mean of losses at or
  beyond that quantile.
- normal: Gaussian with the sample mean and sample standard deviation (ddof=1).

Input contract (checked by every public function before any calculation)
- method: exactly "historical" or "normal".
- confidence: a real scalar, finite, strictly between 0 and 1. Booleans are rejected.
- returns: a real, numeric, one-dimensional sequence of at least two observations
  (list, tuple, NumPy array or pandas Series). NaN, +inf, -inf, complex values,
  scalars, multidimensional inputs and non-numeric values raise ValueError. Nothing
  is dropped, filled, clipped or flattened. Constant finite returns are valid.
- Windows with missing returns are handled by the callers (latest_var_table and the
  rolling backtest), which leave the forecast unavailable; they never reach here.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
from scipy.stats import norm

METHODS = ("historical", "normal")


def validate_confidence(conf) -> float:
    """Return conf as a float, or raise ValueError explaining why it is not a valid confidence level."""
    if isinstance(conf, (bool, np.bool_)):
        raise ValueError(f"confidence must be a real number strictly between 0 and 1, not a boolean ({conf!r})")
    if not isinstance(conf, numbers.Real):
        raise ValueError(f"confidence must be a real scalar number strictly between 0 and 1, got {type(conf).__name__}")
    c = float(conf)
    if not math.isfinite(c):
        raise ValueError(f"confidence must be finite, got {conf!r}")
    if not 0.0 < c < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1 (for example 0.95), got {conf!r}")
    return c


def validate_returns(returns) -> np.ndarray:
    """Return the returns as a float64 array, or raise ValueError explaining what is wrong with them."""
    try:
        arr = np.asarray(returns)
    except Exception as exc:  # ragged nested lists, objects that cannot be converted
        raise ValueError(f"returns could not be converted to a numeric array: {exc}") from exc
    if np.iscomplexobj(arr):
        raise ValueError("returns must be real numbers; complex values are not accepted")
    if arr.dtype.kind not in "iuf":
        raise ValueError(f"returns must be real numeric values, got dtype {arr.dtype} (booleans, strings, objects and missing-value markers are not accepted)")
    if arr.ndim == 0:
        raise ValueError("returns must be a one-dimensional sequence, got a scalar")
    if arr.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {arr.shape}; flatten or select a column explicitly")
    if arr.size < 2:
        raise ValueError(f"at least 2 returns are required, got {arr.size}")
    x = arr.astype(float)
    bad = ~np.isfinite(x)
    if bad.any():
        raise ValueError(
            f"returns must be finite: found {int(np.isnan(x).sum())} NaN, {int(np.isposinf(x).sum())} +inf and {int(np.isneginf(x).sum())} -inf; "
            "drop or exclude them explicitly before calling"
        )
    return x


def historical_var_es(returns, conf: float) -> tuple[float, float]:
    r = validate_returns(returns)
    c = validate_confidence(conf)
    losses = -r
    var = float(np.quantile(losses, c))  # linear interpolation
    tail = losses[losses >= var]
    return var, float(tail.mean())


def normal_var_es(returns, conf: float) -> tuple[float, float]:
    r = validate_returns(returns)
    c = validate_confidence(conf)
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm.ppf(c)
    var = float(-(mu - z * sigma))
    es = float(-mu + sigma * norm.pdf(z) / (1.0 - c))
    return var, es


def var_es(returns, conf: float, method: str) -> tuple[float, float]:
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if method == "historical":
        return historical_var_es(returns, conf)
    return normal_var_es(returns, conf)


def latest_var_table(
    returns: pd.DataFrame,
    window: int = 250,
    confs: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """VaR/ES per ticker from the most recent `window` trading days.

    A ticker is included only if every one of those `window` days has a
    return (no gaps, no stale data). Others are listed in `excluded`.
    """
    rows, excluded = [], {}
    tail = returns.iloc[-window:]
    if len(tail) < window:
        raise ValueError(f"need {window} rows of returns, have {len(tail)}")
    for t in returns.columns:
        x = tail[t]
        if x.isna().any():
            excluded[t] = int(x.isna().sum())
            continue
        row = {"ticker": t, "window_start": tail.index[0], "window_end": tail.index[-1]}
        for c in confs:
            for m in METHODS:
                v, e = var_es(x.values, c, m)
                row[f"{m}_var_{int(round(c * 100))}"] = v
                row[f"{m}_es_{int(round(c * 100))}"] = e
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["excluded_incomplete_window"] = excluded
    return out
