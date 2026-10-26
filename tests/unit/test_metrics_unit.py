"""Descriptive metrics on hand-constructed prices. Each expected value is worked out in the comments."""
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine import data, metrics

IDX = pd.bdate_range("2024-01-01", periods=4)  # Mon 1 Jan to Thu 4 Jan 2024: 3 calendar days apart


def frame(**cols):
    n = max(len(v) for v in cols.values())
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({k: pd.Series(v, index=idx[: len(v)]) for k, v in cols.items()}, index=idx)


def run(prices, rf=0.0252, **kw):
    r = data.simple_returns(prices)
    return metrics.compute_metrics(prices, r, rf, **kw).set_index("ticker")


# ----------------------------------------------------------------------------- the worked example [100, 120, 90, 108]
P = [100.0, 120.0, 90.0, 108.0]  # returns: 0.2, -0.25, 0.2


def test_drawdown_example_maximum_minus_25_percent_and_current_minus_10_percent():
    s = pd.Series(P, index=IDX)
    assert metrics.max_drawdown(s) == pytest.approx(-0.25, abs=1e-15)  # peak 120 then trough 90: 90/120 - 1
    assert metrics.current_drawdown(s) == pytest.approx(-0.10, abs=1e-15)  # latest 108 against the peak 120


def test_every_statistic_of_the_worked_example_by_hand():
    rf = 0.0252  # daily risk-free rate 0.0252 / 252 = 0.0001
    m = run(frame(A=P), rf, min_obs=3).loc["A"]
    r = np.array([0.2, -0.25, 0.2])
    assert m["n_returns"] == 3
    assert m["mean_annual_return"] == pytest.approx(0.05 * 252, rel=1e-12)  # arithmetic mean 0.05 times 252
    sd = math.sqrt(((0.15) ** 2 + (0.30) ** 2 + (0.15) ** 2) / 2)  # deviations 0.15, -0.30, 0.15; sample variance 0.0675
    assert sd == pytest.approx(0.25980762, abs=1e-8) and m["annual_vol"] == pytest.approx(sd * math.sqrt(252), rel=1e-12)
    years = 3 / 365.25  # 1 Jan to 4 Jan is 3 calendar days
    assert m["cagr"] == pytest.approx((108 / 100) ** (1 / years) - 1, rel=1e-12) and m["cagr"] != pytest.approx(m["mean_annual_return"])  # CAGR is not the arithmetic annualisation
    mar = rf / 252
    dd = math.sqrt(sum(min(x - mar, 0.0) ** 2 for x in r) / 3) * math.sqrt(252)  # only the -0.25 day is below the minimum return, divided by ALL 3 observations
    assert m["downside_dev"] == pytest.approx(dd, rel=1e-12)
    assert m["sharpe"] == pytest.approx((0.05 * 252 - rf) / (sd * math.sqrt(252)), rel=1e-12)
    assert m["sortino"] == pytest.approx((0.05 * 252 - rf) / dd, rel=1e-12)
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.25, rel=1e-12)  # CAGR over the absolute maximum drawdown
    assert m["max_drawdown"] == pytest.approx(-0.25) and m["current_drawdown"] == pytest.approx(-0.10)
    assert m["skew"] == pytest.approx(stats.skew(r, bias=False), rel=1e-12)  # the bias-adjusted sample skewness needs 3 observations
    # the bias-adjusted excess kurtosis has (n-3) in a denominator, so it is undefined for 3 observations; the project follows the
    # pandas convention (NaN) and reports undefined, not 0. (scipy returns -1.5 here, which is an artefact of its implementation.)
    assert math.isnan(m["excess_kurtosis"])


def test_skewness_and_excess_kurtosis_use_the_bias_adjusted_sample_conventions():
    prices = frame(A=[100.0, 101.0, 99.5, 102.0, 103.5, 101.0, 104.0])  # 6 returns, enough for both statistics
    r = data.simple_returns(prices)["A"].dropna().to_numpy()
    m = run(prices, min_obs=5).loc["A"]
    assert m["skew"] == pytest.approx(stats.skew(r, bias=False), rel=1e-12) and m["excess_kurtosis"] == pytest.approx(stats.kurtosis(r, fisher=True, bias=False), rel=1e-9)
    n = len(r)  # by hand: G1 = sqrt(n(n-1))/(n-2) * m3 / m2^1.5
    m2, m3 = np.mean((r - r.mean()) ** 2), np.mean((r - r.mean()) ** 3)
    assert m["skew"] == pytest.approx(math.sqrt(n * (n - 1)) / (n - 2) * m3 / m2**1.5, rel=1e-9)


def test_period_returns_are_compounded_price_ratios_over_the_ticker_own_observation_count():
    prices = frame(A=list(np.linspace(100, 160, 30)), B=[100.0] * 4 + [np.nan] * 0 + list(np.linspace(100, 150, 26)))
    m = run(prices, min_obs=5)
    a = prices["A"].to_numpy()
    assert m.loc["A", "return_1m"] == pytest.approx(a[-1] / a[-22] - 1, rel=1e-12)  # 21 observations back
    assert np.isnan(m.loc["A", "return_3m"]) and np.isnan(m.loc["A", "return_1y"])  # 30 prices are too few for 63 or 252: undefined, not zero
    gap = frame(A=list(np.linspace(100, 160, 40)))
    gap.iloc[10, 0] = np.nan  # a missing observation inside the history
    obs = gap["A"].dropna().to_numpy()
    assert run(gap, min_obs=5).loc["A", "return_1m"] == pytest.approx(obs[-1] / obs[-22] - 1, rel=1e-12)  # indexed by observations of that ticker, as documented


# ----------------------------------------------------------------------------- degenerate inputs stay undefined
def test_constant_prices_give_undefined_ratios_not_zeros_and_without_warnings():
    prices = frame(A=[50.0] * 30, SPY=list(np.linspace(100, 130, 30)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = run(prices, 0.0252, min_obs=10)
    a = m.loc["A"]
    assert a["annual_vol"] == 0.0 and a["max_drawdown"] == 0.0
    assert np.isnan(a["sharpe"]) and np.isnan(a["calmar"])  # zero volatility and zero drawdown: undefined
    assert a["beta_vs_spy"] == pytest.approx(0.0, abs=1e-15)  # a constant stock really has zero beta


def test_no_downside_observations_makes_sortino_undefined():
    prices = frame(A=list(np.linspace(100, 200, 30)))
    m = run(prices, 0.0, min_obs=10).loc["A"]  # every daily return is positive and rf is 0, so no shortfall exists
    assert m["downside_dev"] == 0.0 and np.isnan(m["sortino"]) and m["sharpe"] > 0


def test_insufficient_history_excludes_the_ticker():
    prices = frame(A=list(np.linspace(100, 130, 30)), B=[np.nan] * 25 + [10.0, 10.1, 10.2, 10.3, 10.4])
    assert list(run(prices, min_obs=10).index) == ["A"]  # B has 4 returns: excluded, not reported with fabricated numbers


def test_beta_uses_only_dates_present_in_both_series_and_is_undefined_for_constant_or_disjoint_benchmarks():
    a = pd.Series([0.01, 0.02, -0.01, 0.03, np.nan, 0.0], index=pd.bdate_range("2024-01-01", periods=6))
    m = pd.Series([0.02, 0.01, 0.0, 0.05, 0.01, 0.2], index=pd.bdate_range("2024-01-02", periods=6))  # calendar shifted by one day
    both = pd.concat([a, m], axis=1, join="inner").dropna()  # common dates Jan 2, 3, 4, 5, 8; the NaN on Jan 5 is removed, leaving 4
    x, y = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
    expected = sum((xi - x.mean()) * (yi - y.mean()) for xi, yi in zip(x, y)) / sum((yi - y.mean()) ** 2 for yi in y)
    assert len(both) == 4 and metrics.beta(a, m) == pytest.approx(expected, rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(metrics.beta(a, pd.Series([0.001] * 6, index=a.index)))  # constant benchmark: zero variance
        assert np.isnan(metrics.beta(a, pd.Series([0.01, 0.02], index=pd.bdate_range("2030-01-01", periods=2))))  # no common dates
    assert metrics.beta(a, a) == pytest.approx(1.0, abs=1e-12) and metrics.beta(2 * a, a) == pytest.approx(2.0, abs=1e-12)


def test_a_benchmark_missing_from_the_table_leaves_beta_undefined():
    prices = frame(A=list(np.linspace(100, 130, 30)))
    assert np.isnan(run(prices, min_obs=10, market="SPY").loc["A", "beta_vs_spy"])
