"""Portfolio construction with a strict train / test split.

Selection and weights use only returns up to the training end date. Every
performance figure reported for the test period is realised, out-of-sample,
and uses constant weights (rebalanced daily to the same weights). Transaction
costs are not modelled.

A preset whose screen leaves fewer than MIN_ASSETS stocks is reported as not
constructed. Screens are never relaxed.

The Sharpe ratio is undefined when a portfolio's volatility is zero. In a reported performance figure it is NaN.
In the optimiser's objective it raises ZeroVolatilityError, which `fit` records as "not constructed" with the reason:
no objective value is invented, so a degenerate portfolio can never come back as a successful allocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config
from .metrics import MIN_VOLATILITY, TRADING_DAYS, compute_metrics, sharpe_ratio
from .var import historical_var_es


class ZeroVolatilityError(ValueError):
    """The Sharpe ratio is undefined because a portfolio's volatility is zero, so the objective cannot be evaluated."""


def screen(train_metrics: pd.DataFrame, train_var99: pd.Series, preset: dict) -> pd.DataFrame:
    m = train_metrics.set_index("ticker").copy()
    m["hist_var_99"] = train_var99
    keep = m["sharpe"] > preset["min_sharpe"]
    if preset["min_max_drawdown"] is not None:
        keep &= m["max_drawdown"] > preset["min_max_drawdown"]
    if preset["max_hist_var99_1d"] is not None:
        keep &= m["hist_var_99"] < preset["max_hist_var99_1d"]
    if preset["min_return_1y"] is not None:
        keep &= m["return_1y"] > preset["min_return_1y"]
    return m[keep].sort_values("sharpe", ascending=False)  # all stocks passing, best training Sharpe first


def max_sharpe_weights(train_returns: pd.DataFrame, rf_annual: float, bounds: tuple[float, float]) -> np.ndarray:
    n = train_returns.shape[1]
    lo, hi = bounds
    if not (n * lo <= 1.0 <= n * hi):
        raise ValueError(f"bounds {bounds} cannot sum to 1 with {n} assets")
    mu = train_returns.mean().values * TRADING_DAYS
    cov = train_returns.cov().values * TRADING_DAYS

    def neg_sharpe(w):
        variance = float(w @ cov @ w)
        if not np.isfinite(variance):
            raise ZeroVolatilityError("the Sharpe ratio cannot be evaluated: the portfolio variance is not a finite number")
        if variance <= MIN_VOLATILITY**2:  # zero volatility, whatever the sign of the excess return
            raise ZeroVolatilityError(
                "the Sharpe ratio is undefined at zero volatility: the optimiser reached a portfolio whose volatility is zero "
                "(constant returns, or holdings whose returns cancel exactly). No allocation is returned."
            )
        return -(w @ mu - rf_annual) / np.sqrt(variance)

    res = minimize(
        neg_sharpe,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[bounds] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    if not res.success:
        raise RuntimeError(f"optimizer did not converge: {res.message}")
    w = res.x
    if not np.isfinite(w).all():
        raise RuntimeError("optimizer returned non-finite weights")
    if abs(w.sum() - 1.0) > 1e-6 or w.min() < lo - 1e-8 or w.max() > hi + 1e-8:
        raise RuntimeError("optimizer returned weights outside the constraints")
    return w


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> tuple[pd.Series, int]:
    """Constant-weight portfolio returns. A missing return (ticker not trading that day)
    contributes 0, i.e. that weight is held as cash. The count of such ticker-days is returned."""
    r = returns[weights.index]
    missing = int(r.isna().sum().sum())
    return (r.fillna(0.0) * weights).sum(axis=1), missing


def performance(daily: pd.Series, rf_annual: float) -> dict:
    d = daily.dropna()
    ann_ret = float(d.mean() * TRADING_DAYS)
    vol = float(d.std(ddof=1) * np.sqrt(TRADING_DAYS))
    cum = (1.0 + d).cumprod()
    wealth = np.concatenate([[1.0], cum.values])  # starting capital counts as a peak
    out = {
        "n_days": int(len(d)),
        "start": str(d.index[0].date()),
        "end": str(d.index[-1].date()),
        "mean_annual_return": ann_ret,
        "annual_vol": vol,
        "sharpe": sharpe_ratio(ann_ret, rf_annual, vol),
        "total_return": float(cum.iloc[-1] - 1.0),
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
    }
    for c in config.CONFS:
        v, e = historical_var_es(d.values, c)
        out[f"hist_var_{int(round(c * 100))}_1d"] = v
        out[f"hist_es_{int(round(c * 100))}_1d"] = e
    return out


def fit(returns: pd.DataFrame, prices: pd.DataFrame, is_etf: pd.Series, rf_train: float, presets: dict | None = None) -> dict:
    """Choose holdings and weights using returns up to config.TRAIN_END only.

    Anything after TRAIN_END in the inputs is ignored (sliced off here).
    """
    config.validate()
    returns = returns.dropna(how="all")  # the first date has no return for any ticker by construction
    train = returns.loc[: config.TRAIN_END]
    if train.empty:
        raise ValueError("training window is empty")
    stocks = [t for t in returns.columns if not bool(is_etf.get(t, False))]
    complete = [t for t in stocks if train[t].notna().all()]  # full history over the whole training window
    if not complete:
        raise ValueError("no stock has a complete training-window history")
    tr = train[complete]
    m = compute_metrics(prices.loc[: config.TRAIN_END, complete], tr, rf_train, market=config.MARKET, min_obs=len(tr))
    from .var import latest_var_table

    vt = latest_var_table(tr, window=config.WINDOW, confs=(0.99,)).set_index("ticker")["historical_var_99"]

    out = {"train": train, "eligible": complete, "presets": {}}
    for name, preset in (presets or config.PRESETS).items():
        all_passed = screen(m, vt, preset)
        passed = all_passed.head(preset["max_stocks"])
        entry = {
            "preset": preset | {"bounds": list(preset["bounds"])},
            "n_passing_screen": int(len(all_passed)),
            "n_selected": int(len(passed)),
        }
        if len(all_passed) < config.MIN_ASSETS:
            entry["status"] = "not_constructed"
            entry["reason"] = f"{len(all_passed)} stocks passed the screen; at least {config.MIN_ASSETS} are required. The screen was not relaxed."
        else:
            tickers = passed.index.tolist()
            try:
                entry["weights"] = pd.Series(max_sharpe_weights(tr[tickers], rf_train, preset["bounds"]), index=tickers)
                entry["status"] = "constructed"
            except (ValueError, RuntimeError) as e:
                entry["status"] = "not_constructed"
                entry["reason"] = str(e)
        out["presets"][name] = entry
    return out


def build_and_evaluate(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    is_etf: pd.Series,
    rf_train: float,
    rf_test: float,
    presets: dict | None = None,
) -> dict:
    returns = returns.dropna(how="all")
    fitted = fit(returns, prices, is_etf, rf_train, presets)
    train = fitted["train"]
    test = returns.loc[config.TEST_START :]
    if test.empty:
        raise ValueError("test window is empty")
    spy_test = performance(test[config.MARKET].dropna(), rf_test) if config.MARKET in test else None
    results = {
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "risk_free_train": rf_train,
        "risk_free_test": rf_test,
        "eligible_stocks": len(fitted["eligible"]),
        "benchmark_spy_test": spy_test,
        "presets": {},
    }
    for name, entry in fitted["presets"].items():
        e = {k: v for k, v in entry.items() if k != "weights"}
        if entry["status"] == "constructed":
            w = entry["weights"]
            eq = pd.Series(1.0 / len(w), index=w.index)
            d_train, _ = portfolio_returns(train, w)
            d_test, miss_test = portfolio_returns(test, w)
            eq_test, _ = portfolio_returns(test, eq)
            e.update(
                tickers=w.index.tolist(),
                weights={t: float(x) for t, x in w.items()},
                in_sample_train=performance(d_train, rf_train),
                out_of_sample_test=performance(d_test, rf_test),
                equal_weight_same_holdings_test=performance(eq_test, rf_test),
                test_ticker_days_without_return=miss_test,
            )
        results["presets"][name] = e
    return results
