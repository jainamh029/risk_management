"""The two numerical edge cases left open by the module audit: infinite values in a rolling history, and zero volatility in
the portfolio objective. Hand-constructed deterministic fixtures only; expected values are computed independently of the code."""
import math

import numpy as np
import pandas as pd
import pytest

from riskengine import backtest, config, metrics, portfolio
from tests.unit.fixtures import wavy_returns

METHODS = ("historical", "normal")


# ----------------------------------------------------------------------------- 1. infinity in a rolling history
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [float("inf"), float("-inf")], ids=["+inf", "-inf"])
@pytest.mark.parametrize("position", [0, 7, 29, 39], ids=["first", "inside_first_window", "forecast_day", "last"])
def test_an_infinite_return_anywhere_in_the_history_is_rejected_with_an_explanation(method, bad, position):
    r = wavy_returns(40)
    r[position] = bad
    before = r.copy()
    with pytest.raises(ValueError, match=r"infinite value.*first at position " + str(position) + r".*NaN for a missing return"):
        backtest.rolling_forecasts(r, 30, 0.95, method)
    assert np.array_equal(r, before, equal_nan=True)  # the input was not modified, in particular not turned into NaN


@pytest.mark.parametrize("method", METHODS)
def test_infinity_is_rejected_even_when_the_history_is_too_short_to_forecast_and_when_it_also_contains_nan(method):
    short = wavy_returns(10)
    short[3] = float("inf")
    with pytest.raises(ValueError, match="infinite"):
        backtest.rolling_forecasts(short, 30, 0.95, method)
    mixed = wavy_returns(40)
    mixed[5], mixed[25] = float("nan"), float("-inf")
    with pytest.raises(ValueError, match="1 infinite value"):
        backtest.rolling_forecasts(mixed, 30, 0.95, method)


@pytest.mark.parametrize("method", METHODS)
def test_the_number_of_infinite_values_is_reported(method):
    r = wavy_returns(40)
    r[[2, 11, 12]] = [float("inf"), float("-inf"), float("inf")]
    with pytest.raises(ValueError, match="3 infinite value"):
        backtest.rolling_forecasts(r, 30, 0.95, method)


@pytest.mark.parametrize("method", METHODS)
def test_the_nan_policy_is_unchanged_affected_windows_have_no_forecast_and_the_rest_match_the_reference(method):
    r = wavy_returns(60)
    r[35] = float("nan")
    f = backtest.rolling_forecasts(r, 20, 0.95, method)
    for t in range(60):
        window_has_nan = t >= 20 and np.isnan(r[t - 20 : t]).any()
        if t < 20 or window_has_nan:
            assert np.isnan(f[t]), t
        else:
            w = r[t - 20 : t]
            if method == "historical":
                s = np.sort(-w)
                pos = 0.95 * (len(s) - 1)
                lo, hi = math.floor(pos), math.ceil(pos)
                ref = s[lo] + (pos - lo) * (s[hi] - s[lo])
            else:
                mu, sd = w.mean(), w.std(ddof=1)
                ref = -(mu - 1.6448536269514722 * sd)  # the 95% standard-normal quantile
            assert f[t] == pytest.approx(ref, abs=1e-12), t
    assert np.isnan(f[36:56]).all() and not np.isnan(f[56:]).any()  # the forecasts for days 36 to 55 inclusive use windows containing day 35


@pytest.mark.parametrize("method", METHODS)
def test_the_higher_level_backtest_functions_reject_infinity_too(method):
    r = pd.Series(wavy_returns(40), index=pd.bdate_range("2020-01-01", periods=40), name="X")
    r.iloc[33] = float("inf")  # on a day that would be a forecast day
    with pytest.raises(ValueError, match="infinite"):
        backtest.backtest_series(r, window=20, confs=(0.95,), methods=(method,))
    with pytest.raises(ValueError, match="infinite"):
        backtest.backtest_universe(pd.DataFrame({"X": r}), window=20, confs=(0.95,), min_forecasts=5)


# ----------------------------------------------------------------------------- 2. zero volatility in the portfolio objective
IDX = pd.bdate_range("2020-01-01", periods=300)


def constant_returns(value, n_assets=4):
    return pd.DataFrame({f"S{k}": np.full(300, value) for k in range(n_assets)}, index=IDX)


def hand_annual(value):
    return value * 252  # the annualised mean of a constant daily return


@pytest.mark.parametrize("excess_sign", ["positive", "zero", "negative"])
def test_zero_volatility_with_exactly_representable_returns_is_undefined_for_every_sign_of_the_excess_return(excess_sign):
    c = 0.0625  # 1/16: every mean and deviation is exact, so the volatility is exactly zero
    mu = hand_annual(c)  # 15.75, exact
    rf = {"positive": mu - 5.0, "zero": mu, "negative": mu + 5.0}[excess_sign]
    excess = mu - rf
    assert (excess > 0, excess == 0, excess < 0) == {"positive": (True, False, False), "zero": (False, True, False), "negative": (False, False, True)}[excess_sign]
    df = constant_returns(c)
    w = np.full(4, 0.25)
    assert w @ (df.cov().to_numpy() * 252) @ w == 0.0  # independent check that the fixture really has zero volatility
    with pytest.raises(portfolio.ZeroVolatilityError, match="undefined at zero volatility"):
        portfolio.max_sharpe_weights(df, rf, (0.02, 0.5))


@pytest.mark.parametrize("excess_sign", ["positive", "zero", "negative"])
def test_zero_volatility_that_is_only_zero_up_to_rounding_is_also_undefined(excess_sign):
    c = 0.001  # not exactly representable: the computed volatility is about 1e-19, not 0
    mu = hand_annual(c)
    rf = {"positive": mu - 5.0, "zero": mu, "negative": mu + 5.0}[excess_sign]
    df = constant_returns(c)
    w = np.full(4, 0.25)
    vol = math.sqrt(max(w @ (df.cov().to_numpy() * 252) @ w, 0.0))
    assert 0.0 <= vol < 1e-15  # rounding noise, far below any real volatility
    with pytest.raises(portfolio.ZeroVolatilityError):
        portfolio.max_sharpe_weights(df, rf, (0.02, 0.5))


@pytest.mark.parametrize("excess_sign,rf_offset", [("positive", -5.0), ("zero", 0.0), ("negative", 5.0)])
def test_holdings_whose_returns_cancel_exactly_are_refused_not_returned_as_a_successful_allocation(excess_sign, rf_offset):
    a = 0.01 * np.sin(0.3 * np.arange(300))
    df = pd.DataFrame({"A": 0.001 + a, "B": 0.001 - a}, index=IDX)  # A + B is constant: the 50/50 portfolio has no volatility
    w = np.array([0.5, 0.5])
    assert np.var(w @ df.to_numpy().T) < 1e-30  # independent: the daily return of the 50/50 portfolio is constant
    with pytest.raises(portfolio.ZeroVolatilityError):
        portfolio.max_sharpe_weights(df, hand_annual(0.001) + rf_offset, (0.0, 1.0))


def test_a_zero_volatility_portfolio_is_a_not_constructed_preset_with_the_reason_never_a_constructed_one():
    t = np.arange(560)
    a = 0.01 * np.sin(0.3 * t)
    idx = pd.bdate_range("2022-01-03", periods=560)  # ends in early 2024: a training period of about 520 days
    cols = {"S1": 0.0005 + a, "S2": 0.0005 - a, "S3": 0.0005 + a, "S4": 0.0005 - a}  # the equal-weight portfolio has constant returns
    returns = pd.DataFrame(cols, index=idx)
    prices = 100.0 * (1.0 + returns).cumprod()
    preset = dict(min_sharpe=-10.0, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=4, bounds=(0.25, 0.25))
    out = portfolio.fit(returns, prices, pd.Series(False, index=returns.columns), rf_train=0.0, presets={"degenerate": preset})
    entry = out["presets"]["degenerate"]
    assert entry["n_passing_screen"] == 4  # each stock passes the screen on its own: the problem is the combination
    assert entry["status"] == "not_constructed" and "weights" not in entry
    assert "undefined at zero volatility" in entry["reason"]


# ----------------------------------------------------------------------------- the same rule where Sharpe is reported
@pytest.mark.parametrize("rf_offset", [-5.0, 0.0, 5.0])
def test_a_reported_sharpe_ratio_at_zero_volatility_is_nan_for_every_sign_of_the_excess_return(rf_offset):
    d = pd.Series(np.full(100, 0.0625), index=pd.bdate_range("2024-01-02", periods=100))
    assert math.isnan(portfolio.performance(d, hand_annual(0.0625) + rf_offset)["sharpe"])
    noisy = pd.Series(np.full(100, 0.001), index=d.index)
    assert math.isnan(portfolio.performance(noisy, hand_annual(0.001) + rf_offset)["sharpe"])  # rounding-level volatility: was about 1e18


def test_sharpe_ratio_helper_values_and_boundaries():
    assert metrics.sharpe_ratio(0.12, 0.02, 0.2) == pytest.approx(0.5)
    assert metrics.sharpe_ratio(0.00, 0.02, 0.2) == pytest.approx(-0.1)  # a negative Sharpe is a valid value
    assert metrics.sharpe_ratio(0.02, 0.02, 0.2) == 0.0  # a zero excess return with positive volatility is a valid zero
    for vol in (0.0, 1e-12, 1e-18, float("nan"), float("inf"), -0.0):
        assert math.isnan(metrics.sharpe_ratio(0.12, 0.02, vol)), vol
    assert metrics.sharpe_ratio(0.12, 0.02, 2e-12) == pytest.approx(0.1 / 2e-12)  # just above the floor: a number


def test_metrics_table_reports_an_undefined_sharpe_for_a_constant_return_series():
    prices = pd.DataFrame({"C": 100.0 * (1.001 ** np.arange(300))}, index=IDX)
    table = metrics.compute_metrics(prices, prices / prices.shift(1) - 1.0, 0.0, market="SPY", min_obs=250)
    assert math.isnan(table.set_index("ticker").loc["C", "sharpe"])


def test_valid_inputs_give_the_same_weights_as_the_independent_two_asset_reference():
    """The repaired objective is the same function for every portfolio with positive volatility: the optimum of a
    two-asset problem still matches a brute-force grid, as in test_two_asset_optimum_matches_a_brute_force_grid_search."""
    t = np.arange(300)
    df = pd.DataFrame({"A": 0.0008 + 0.01 * np.sin(0.2 * t), "B": 0.0003 + 0.006 * np.cos(0.37 * t) + 0.004 * np.sin(0.2 * t)}, index=IDX)
    rf = 0.02
    w = portfolio.max_sharpe_weights(df, rf, (0.1, 0.9))
    mu, cov = df.mean().to_numpy() * 252, df.cov().to_numpy() * 252
    grid = np.linspace(0.1, 0.9, 80001)
    sharpe = [(g * mu[0] + (1 - g) * mu[1] - rf) / math.sqrt(np.array([g, 1 - g]) @ cov @ np.array([g, 1 - g])) for g in grid[::20]]
    best = grid[::20][int(np.argmax(sharpe))]
    assert w[0] == pytest.approx(best, abs=2e-3) and w.sum() == pytest.approx(1.0)
