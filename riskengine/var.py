"""One-day Value at Risk and Expected Shortfall.

Conventions
- Inputs are daily simple returns.
- VaR and ES are reported as positive fractions of position value (a loss of
  0.03 means 3 percent). The horizon is one trading day. No annualisation and
  no square-root-of-time scaling is applied anywhere in this package.
- conf=0.99 means the loss that is exceeded on about 1 percent of days.

Methods
- historical: empirical quantile of losses; ES is the mean of losses at or
  beyond that quantile.
- normal: Gaussian with the sample mean and sample standard deviation (ddof=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

METHODS = ("historical", "normal")


def historical_var_es(returns, conf: float) -> tuple[float, float]:
    losses = -np.asarray(returns, dtype=float)
    if np.isnan(losses).any():
        raise ValueError("returns contain NaN; drop or exclude them explicitly")
    var = float(np.quantile(losses, conf))  # linear interpolation
    tail = losses[losses >= var]
    return var, float(tail.mean())


def normal_var_es(returns, conf: float) -> tuple[float, float]:
    r = np.asarray(returns, dtype=float)
    if np.isnan(r).any():
        raise ValueError("returns contain NaN; drop or exclude them explicitly")
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm.ppf(conf)
    var = float(-(mu - z * sigma))
    es = float(-mu + sigma * norm.pdf(z) / (1.0 - conf))
    return var, es


def var_es(returns, conf: float, method: str) -> tuple[float, float]:
    if method == "historical":
        return historical_var_es(returns, conf)
    if method == "normal":
        return normal_var_es(returns, conf)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


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
