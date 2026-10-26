"""Per-ticker descriptive risk and return statistics over a stated window.

All figures describe the window that is passed in. They are historical
statistics, not forecasts. Annualisation uses 252 trading days. The risk-free
rate is a parameter; callers pass the average 13-week T-bill yield from the
downloaded data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Annualised volatility at or below this value is treated as zero. Returns that are constant up to floating-point
# rounding have a computed volatility of about 1e-18, which would otherwise give a Sharpe ratio of about 1e18. Real
# daily return series have an annualised volatility of the order of 1e-1, so this floor never applies to them.
MIN_VOLATILITY = 1e-12


def sharpe_ratio(annual_return: float, rf_annual: float, annual_vol: float) -> float:
    """(annual return - risk-free rate) / annualised volatility; NaN (undefined) when the volatility is zero.

    The sign of the excess return does not matter: with zero volatility the ratio has no value (it is neither
    plus nor minus infinity nor zero), so it is undefined whether the excess return is positive, zero or negative.
    """
    if not np.isfinite(annual_vol) or annual_vol <= MIN_VOLATILITY:
        return float("nan")
    return float((annual_return - rf_annual) / annual_vol)


def max_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    peak = p.cummax()
    return float((p / peak - 1.0).min())


def current_drawdown(prices: pd.Series) -> float:
    p = prices.dropna()
    return float(p.iloc[-1] / p.cummax().iloc[-1] - 1.0)


def downside_deviation(daily_returns: np.ndarray, mar_daily: float) -> float:
    """Annualised downside deviation: root mean square of shortfalls below the
    minimum acceptable return, averaged over ALL observations."""
    shortfall = np.minimum(daily_returns - mar_daily, 0.0)
    return float(np.sqrt(np.mean(shortfall**2)) * np.sqrt(TRADING_DAYS))


def beta(stock: pd.Series, market: pd.Series) -> float:
    both = pd.concat([stock, market], axis=1, join="inner").dropna()
    if len(both) < 2:
        return float("nan")
    if np.all(both.iloc[:, 1].to_numpy() == both.iloc[0, 1]):
        return float("nan")  # a constant benchmark has zero variance: beta is undefined (float rounding would otherwise give a huge number)
    cov = np.cov(both.iloc[:, 0], both.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def compute_metrics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf_annual: float,
    market: str = "SPY",
    min_obs: int = 250,
) -> pd.DataFrame:
    """Statistics for every ticker with at least `min_obs` returns in the window.

    Period returns (1M/3M/6M/1Y) use the last 21/63/126/252 observations of the
    ticker's own price history and are NaN when that history is too short.
    """
    rows = []
    mkt = returns[market] if market in returns.columns else None
    for t in returns.columns:
        r = returns[t].dropna()
        if len(r) < min_obs:
            continue
        p = prices[t].dropna()
        mean_ann = r.mean() * TRADING_DAYS
        vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
        dd_dev = downside_deviation(r.values, rf_annual / TRADING_DAYS)
        years = (p.index[-1] - p.index[0]).days / 365.25
        cagr = (p.iloc[-1] / p.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
        mdd = max_drawdown(p)

        def period(k: int) -> float:
            return float(p.iloc[-1] / p.iloc[-1 - k] - 1.0) if len(p) > k else float("nan")

        rows.append(
            {
                "ticker": t,
                "n_returns": int(len(r)),
                "first_date": r.index[0],
                "last_date": r.index[-1],
                "mean_annual_return": mean_ann,
                "cagr": cagr,
                "annual_vol": vol,
                "downside_dev": dd_dev,
                "sharpe": sharpe_ratio(mean_ann, rf_annual, vol),
                "sortino": (mean_ann - rf_annual) / dd_dev if dd_dev > 0 else np.nan,
                "calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
                "max_drawdown": mdd,
                "current_drawdown": current_drawdown(p),
                "skew": float(r.skew()),
                "excess_kurtosis": float(r.kurt()),
                "beta_vs_" + market.lower(): beta(returns[t], mkt) if (mkt is not None and t != market) else np.nan,
                "return_1m": period(21),
                "return_3m": period(63),
                "return_6m": period(126),
                "return_1y": period(252),
            }
        )
    return pd.DataFrame(rows)
