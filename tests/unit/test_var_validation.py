"""Regression tests for the VaR/ES input contract (repair stage).

Every expected value is independent: scipy distributions or hand calculation. Nothing here uses market data.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskengine import backtest, var
from tests.unit.fixtures import wavy_returns

GOOD = [-0.01, 0.0, 0.01]
FUNCTIONS = [("historical", lambda r, c: var.historical_var_es(r, c)), ("normal", lambda r, c: var.normal_var_es(r, c))]


def via_entry(method):
    return lambda r, c: var.var_es(r, c, method)


# ----------------------------------------------------------------------------- invalid confidence
INVALID_CONF = [
    (0, "strictly between"), (1, "strictly between"), (-0.1, "strictly between"), (1.0000001, "strictly between"), (2, "strictly between"),
    (float("nan"), "finite"), (float("inf"), "finite"), (float("-inf"), "finite"),
    (True, "boolean"), (False, "boolean"), (np.True_, "boolean"),
    ([0.95], "scalar"), ((0.95,), "scalar"), (np.array([0.95]), "scalar"), (np.array(0.95), "scalar"), ({"c": 0.95}, "scalar"),
    ("0.95", "scalar"), (None, "scalar"), (0.95 + 0j, "scalar"),
]


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf,message", INVALID_CONF, ids=[repr(c)[:28] for c, _ in INVALID_CONF])
def test_invalid_confidence_raises_value_error_with_an_explanation(method, conf, message):
    with pytest.raises(ValueError, match=message):
        var.var_es(GOOD, conf, method)


@pytest.mark.parametrize("name,fn", FUNCTIONS)
@pytest.mark.parametrize("conf", [0, 1, float("nan"), True, "0.95"])
def test_the_direct_entry_points_validate_too(name, fn, conf):
    with pytest.raises(ValueError):
        fn(GOOD, conf)


# ----------------------------------------------------------------------------- invalid returns
S = pd.Series
INVALID_RETURNS = [
    ("empty list", [], "at least 2"),
    ("one observation", [0.01], "at least 2"),
    ("one-element array", np.array([0.01]), "at least 2"),
    ("python scalar", 0.01, "scalar"),
    ("numpy scalar", np.float64(0.01), "scalar"),
    ("two-dimensional list", [[0.01, 0.02], [0.03, 0.04]], "one-dimensional"),
    ("two-dimensional array", np.array([[0.01, 0.02, 0.03]]), "one-dimensional"),
    ("dataframe", pd.DataFrame({"a": [0.01, 0.02], "b": [0.03, 0.04]}), "one-dimensional"),
    ("NaN", [0.01, np.nan, -0.02], "finite"),
    ("positive infinity", [0.01, np.inf, -0.02], "finite"),
    ("negative infinity", [0.01, -np.inf, -0.02], "finite"),
    ("all NaN", [np.nan, np.nan], "finite"),
    ("series with NaN", S([0.01, np.nan, 0.02]), "finite"),
    ("strings", ["a", "b"], "numeric"),
    ("numeric strings", ["0.01", "0.02"], "numeric"),
    ("mixed string", [0.01, "x"], "numeric"),
    ("None inside", [0.01, None, 0.02], "numeric"),
    ("booleans", [True, False, True], "numeric"),
    ("series of strings", S(["a", "b"]), "numeric"),
    ("nullable series with NA", S([0.01, pd.NA, 0.02], dtype="Float64"), "finite"),  # the NA becomes NaN and is rejected
    ("complex list", [0.01 + 1j, 0.02], "complex"),
    ("complex array with zero imaginary part", np.array([0.01 + 0j, 0.02 + 0j]), "complex"),
    ("ragged nested list", [[0.01], [0.02, 0.03]], "numeric|convert|dimension"),
    ("dict", {"a": 0.01, "b": 0.02}, "numeric|scalar"),
]


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("label,returns,message", INVALID_RETURNS, ids=[x[0] for x in INVALID_RETURNS])
def test_invalid_returns_raise_value_error_with_an_explanation(method, label, returns, message):
    with pytest.raises(ValueError, match=message):
        var.var_es(returns, 0.95, method)


def test_unknown_methods_raise_value_error():
    for bad in ("monte_carlo", "Historical", "", None, 3, ["normal"]):
        with pytest.raises(ValueError, match="unknown method"):
            var.var_es(GOOD, 0.95, bad)


def test_the_returns_are_not_modified_dropped_clipped_or_filled():
    for container in (list(GOOD), np.array(GOOD), S(GOOD)):
        before = list(np.asarray(container))
        var.var_es(container, 0.95, "historical")
        var.var_es(container, 0.95, "normal")
        assert list(np.asarray(container)) == before
    with pytest.raises(ValueError):  # a bad observation is never silently removed
        var.var_es([0.01, 0.02, np.nan], 0.95, "historical")


# ----------------------------------------------------------------------------- valid inputs keep working, with independent references
CONTAINERS = {
    "list": lambda: [-0.01, 0.0, 0.01],
    "tuple": lambda: (-0.01, 0.0, 0.01),
    "float64 array": lambda: np.array([-0.01, 0.0, 0.01]),
    "float32 array": lambda: np.array([-0.01, 0.0, 0.01], dtype=np.float32),
    "series": lambda: S([-0.01, 0.0, 0.01]),
    "series with a date index": lambda: S([-0.01, 0.0, 0.01], index=pd.bdate_range("2024-01-01", periods=3)),
}


@pytest.mark.parametrize("name", list(CONTAINERS))
@pytest.mark.parametrize("c", [0.95, 0.99])
def test_ordinary_inputs_still_work_and_match_the_independent_references(name, c):
    r = CONTAINERS[name]()
    tol = 1e-6 if "float32" in name else 1e-12  # float32 input carries float32 rounding of the data itself
    z = norm.ppf(c)
    v, e = var.normal_var_es(r, c)  # mean 0, sample sd 0.01
    assert v == pytest.approx(0.01 * z, rel=tol, abs=1e-8 if "float32" in name else 1e-15)
    assert e == pytest.approx(0.01 * norm.pdf(z) / (1 - c), rel=tol, abs=1e-8 if "float32" in name else 1e-15)
    # historical by hand: losses sorted [-0.01, 0, 0.01]; position c*2; VaR = 0 + (c*2 - 1) * 0.01 ; ES = the one loss at or above it = 0.01
    v, e = var.historical_var_es(r, c)
    assert v == pytest.approx((c * 2 - 1) * 0.01, rel=tol, abs=1e-8 if "float32" in name else 1e-15)
    assert e == pytest.approx(0.01, rel=tol)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("c", [0.5, 0.95, 0.99])
def test_constant_finite_returns_are_valid_and_give_minus_the_constant(method, c):
    v, e = var.var_es([0.01, 0.01, 0.01], c, method)
    assert v == pytest.approx(-0.01, abs=1e-15) and e == pytest.approx(-0.01, abs=1e-15)  # loss of -1%: a gain; zero volatility is not an error


def test_integer_dtype_returns_are_accepted():
    v, e = var.normal_var_es(np.array([-1, 0, 1]), 0.95)
    assert v == pytest.approx(1.0 * norm.ppf(0.95), rel=1e-12) and e == pytest.approx(norm.pdf(norm.ppf(0.95)) / 0.05, rel=1e-12)


def test_the_existing_reference_results_are_unchanged():
    """B6 and B8 of the acceptance specification, repeated after the repair."""
    for c in (0.95, 0.99):
        v, e = var.normal_var_es([-0.01, 0.0, 0.01], c)
        assert v == pytest.approx(0.01 * norm.ppf(c), rel=1e-12) and e == pytest.approx(0.01 * norm.pdf(norm.ppf(c)) / (1 - c), rel=1e-12)
    r = [-0.04, -0.02, 0.0, 0.01, 0.03]
    assert var.historical_var_es(r, 0.75) == pytest.approx((0.02, 0.03), abs=1e-15)
    assert var.historical_var_es(r, 0.95) == pytest.approx((0.036, 0.04), abs=1e-15)


def test_negative_var_is_reported_not_clipped_to_zero():
    v, e = var.normal_var_es([x + 0.05 for x in GOOD], 0.95)
    assert v < 0 and e < 0
    assert v == pytest.approx(0.01 * norm.ppf(0.95) - 0.05, abs=1e-14)


# ----------------------------------------------------------------------------- missing windows stay in the backtest layer
@pytest.mark.parametrize("method", var.METHODS)
def test_a_missing_return_in_a_window_remains_an_unavailable_forecast_not_an_error(method):
    r = wavy_returns(20)
    r[8] = np.nan
    f = backtest.rolling_forecasts(r, 5, 0.9, method)  # must not raise: validation of finite input is the caller's contract
    # window of 5 ending the day before forecast day t contains index 8 for t = 9..13; day 14 is the first clean window
    assert np.isnan(f[:5]).all() and not np.isnan(f[5:9]).any() and np.isnan(f[9:14]).all() and not np.isnan(f[14:]).any()


def test_latest_var_table_excludes_an_incomplete_window_instead_of_raising_or_filling():
    returns = pd.DataFrame({"A": wavy_returns(12), "B": wavy_returns(12, phase=1.0)}, index=pd.bdate_range("2024-01-01", periods=12))
    returns.loc[returns.index[9], "B"] = np.nan
    table = var.latest_var_table(returns, window=5, confs=(0.95,))
    assert list(table["ticker"]) == ["A"] and table.attrs["excluded_incomplete_window"] == {"B": 1}
    assert np.isfinite(table.filter(like="_var_").to_numpy()).all()
