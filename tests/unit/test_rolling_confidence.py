"""Regression tests: backtest.rolling_forecasts validates its confidence like the public VaR functions.

Reference values come from loops written in this file (sorted losses with linear interpolation; the sample mean and
standard deviation with statistics.NormalDist). Nothing here uses market data.
"""
import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from riskengine import backtest, var
from tests.unit.fixtures import wavy_returns

W = 12


def independent(r, window, conf, method):
    """Forecast for every day from the previous `window` returns, written without the package."""
    out = np.full(len(r), np.nan)
    for t in range(window, len(r)):
        w = r[t - window : t]
        if np.isnan(w).any():
            continue
        if method == "historical":
            s = sorted(-x for x in w)
            pos = float(conf) * (len(s) - 1)
            lo, hi = int(math.floor(pos)), int(math.ceil(pos))
            out[t] = s[lo] + (pos - lo) * (s[hi] - s[lo])
        else:
            mu = sum(w) / len(w)
            sd = math.sqrt(sum((x - mu) ** 2 for x in w) / (len(w) - 1))
            out[t] = -(mu - NormalDist().inv_cdf(float(conf)) * sd)
    return out


# ----------------------------------------------------------------------------- valid confidence levels
VALID = [0.5, 0.9, 0.95, 0.975, 0.99, 0.999, np.float64(0.95), Fraction(19, 20)]


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf", VALID, ids=[repr(c) for c in VALID])
def test_valid_confidence_levels_give_the_independent_reference_forecasts(method, conf):
    r = wavy_returns(60)
    np.testing.assert_allclose(backtest.rolling_forecasts(r, W, conf, method), independent(r, W, conf, method), rtol=1e-12, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_breach_counts_are_unchanged_and_match_an_independent_loop(method, conf):
    r = wavy_returns(80, scale=0.01, phase=0.4)
    s = pd.Series(r, name="X", index=pd.bdate_range("2022-01-03", periods=len(r)))
    row = backtest.backtest_series(s, W, (conf,), (method,))[0]
    f = independent(r, W, conf, method)
    valid = ~np.isnan(f)
    assert row["n_forecasts"] == int(valid.sum()) and row["breaches"] == int((-r[valid] > f[valid]).sum())


# ----------------------------------------------------------------------------- invalid confidence levels
INVALID = [
    (0, "strictly between"), (1, "strictly between"), (0.0, "strictly between"), (1.0, "strictly between"),
    (-0.1, "strictly between"), (-5, "strictly between"), (1.0000001, "strictly between"), (2, "strictly between"),
    (float("nan"), "finite"), (float("inf"), "finite"), (float("-inf"), "finite"),
    (True, "boolean"), (False, "boolean"), (np.True_, "boolean"), (np.False_, "boolean"),
    ("0.95", "scalar"), (b"0.95", "scalar"), ([0.95], "scalar"), ((0.95,), "scalar"), (np.array([0.95]), "scalar"), (np.array(0.95), "scalar"),
    ({"c": 0.95}, "scalar"), (None, "scalar"), (0.95 + 0j, "scalar"),
]


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf,message", INVALID, ids=[repr(c)[:26] for c, _ in INVALID])
def test_invalid_confidence_raises_an_explanatory_value_error(method, conf, message):
    with pytest.raises(ValueError, match=message):
        backtest.rolling_forecasts(wavy_returns(40), W, conf, method)


def test_the_policy_is_the_shared_one_not_a_copy():
    assert backtest.validate_confidence is var.validate_confidence
    for conf, _ in INVALID:
        with pytest.raises(ValueError) as a:
            var.validate_confidence(conf)
        with pytest.raises(ValueError) as b:
            backtest.rolling_forecasts(wavy_returns(40), W, conf, "normal")
        assert str(a.value) == str(b.value)  # identical message: one rule, one place


@pytest.mark.parametrize("method", var.METHODS)
def test_validation_happens_before_any_calculation_starts(method, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a calculation started before the confidence was validated")

    monkeypatch.setattr(backtest, "sliding_window_view", forbidden)
    for conf in (1, float("nan"), True, "0.95", [0.95]):
        with pytest.raises(ValueError):
            backtest.rolling_forecasts(wavy_returns(40), W, conf, method)
    with pytest.raises(AssertionError, match="before the confidence"):  # the patch is effective: a valid call does reach the calculation
        backtest.rolling_forecasts(wavy_returns(40), W, 0.95, method)


@pytest.mark.parametrize("method", var.METHODS)
def test_a_short_history_still_validates_the_confidence(method):
    with pytest.raises(ValueError):
        backtest.rolling_forecasts(wavy_returns(5), 10, 1.0, method)  # shorter than the window: no forecast, but the setting is still checked
    assert np.isnan(backtest.rolling_forecasts(wavy_returns(5), 10, 0.95, method)).all()


def test_unknown_methods_raise_before_anything_is_calculated():
    for method in ("monte_carlo", "Historical", "", None):
        with pytest.raises(ValueError, match="unknown method"):
            backtest.rolling_forecasts(wavy_returns(5), 10, 0.95, method)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf", [0, 1, float("nan"), True, "0.95", [0.95]])
def test_the_higher_level_backtest_functions_inherit_the_validation(method, conf):
    s = pd.Series(wavy_returns(40), name="X", index=pd.bdate_range("2022-01-03", periods=40))
    with pytest.raises(ValueError):
        backtest.backtest_series(s, W, (conf,), (method,))
    frame = pd.DataFrame({"A": s, "B": s * 1.1})
    with pytest.raises(ValueError):
        backtest.backtest_universe(frame, W, (conf,), min_forecasts=1)


# ----------------------------------------------------------------------------- the missing-window policy is preserved
@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("conf", [0.9, 0.99])
def test_missing_observations_make_only_the_affected_windows_unavailable(method, conf):
    r = wavy_returns(40)
    r[15] = np.nan
    r[16] = np.nan  # two adjacent missing returns
    f = backtest.rolling_forecasts(r, W, conf, method)  # a usable history with gaps is not rejected
    unavailable = [t for t in range(len(r)) if np.isnan(f[t])]
    assert unavailable == list(range(0, W)) + list(range(16, 16 + W + 1))  # windows ending before day 15 are clean; day 16's window ends on 15
    np.testing.assert_allclose(f, independent(r, W, conf, method), rtol=1e-12, atol=1e-15, equal_nan=True)


@pytest.mark.parametrize("method", var.METHODS)
def test_an_invalid_confidence_is_rejected_even_when_the_history_has_gaps(method):
    r = wavy_returns(40)
    r[15] = np.nan
    with pytest.raises(ValueError):
        backtest.rolling_forecasts(r, W, 1.0, method)
