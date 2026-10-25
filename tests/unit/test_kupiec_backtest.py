"""C. Kupiec test and rolling-backtest mechanics on hand-constructed series."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from riskengine import backtest
from tests.unit.fixtures import wavy_returns


def loglik(n, x, p):
    """Direct binomial log likelihood without the binomial coefficient (it cancels in the ratio)."""
    ll = (n - x) * math.log(1 - p)
    if x > 0:
        ll += x * math.log(p)
    return ll


def reference_lr(n, x, p):
    ll_alt = 0.0
    if x > 0:
        ll_alt += x * math.log(x / n)
    if x < n:
        ll_alt += (n - x) * math.log(1 - x / n)
    return -2.0 * (loglik(n, x, p) - ll_alt)


def test_C12_exact_boundary_cases_from_the_specification():
    lr, p = backtest.kupiec_pof(100, 5, 0.05)  # breach rate equals the expected rate
    assert lr == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0, abs=1e-12)
    lr, p = backtest.kupiec_pof(100, 0, 0.05)
    assert lr == pytest.approx(-200 * math.log(0.95), rel=1e-12) and p == pytest.approx(chi2.sf(-200 * math.log(0.95), 1), rel=1e-12)
    lr, p = backtest.kupiec_pof(100, 100, 0.05)
    assert lr == pytest.approx(-200 * math.log(0.05), rel=1e-12) and p == pytest.approx(chi2.sf(-200 * math.log(0.05), 1), rel=1e-12)


@pytest.mark.parametrize("n,x,pp", [(250, 9, 0.05), (1000, 17, 0.01), (500, 40, 0.05), (60, 1, 0.01)])
def test_C12_ordinary_cases_against_direct_log_likelihoods(n, x, pp):
    lr, p = backtest.kupiec_pof(n, x, pp)
    assert lr == pytest.approx(reference_lr(n, x, pp), rel=1e-10, abs=1e-12)
    assert p == pytest.approx(chi2.sf(reference_lr(n, x, pp), df=1), rel=1e-10)


def test_C12_a_hand_calculated_value():
    # n = 100, x = 10, p = 0.05: LR = -2 * [ 90 ln .95 + 10 ln .05 - 90 ln .9 - 10 ln .1 ]
    expected = -2.0 * (90 * math.log(0.95) + 10 * math.log(0.05) - 90 * math.log(0.9) - 10 * math.log(0.1))
    # by hand: 90 ln .95 + 10 ln .05 = -34.5737 ; 90 ln .9 + 10 ln .1 = -32.5083 ; difference -2.0654 ; times -2 = 4.1308
    assert expected == pytest.approx(4.1308, abs=1e-3)
    assert backtest.kupiec_pof(100, 10, 0.05)[0] == pytest.approx(expected, rel=1e-12)


def series(values, name="X"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range("2021-01-04", periods=len(values)), name=name)


def test_C13_a_loss_exactly_equal_to_var_is_not_a_breach():
    # window of 5 returns, conf 0.75 -> position 0.75 * 4 = 3.0 exactly -> VaR is the 4th smallest loss
    window = [-0.04, -0.02, 0.0, 0.01, 0.03]  # losses sorted: -0.03 -0.01 0.0 0.02 0.04 -> VaR = 0.02
    at_var = backtest.backtest_series(series(window + [-0.02]), window=5, confs=(0.75,), methods=("historical",))[0]
    assert at_var["n_forecasts"] == 1 and at_var["breaches"] == 0  # loss 0.02 == VaR 0.02: strict > convention
    above = backtest.backtest_series(series(window + [-0.0200001]), window=5, confs=(0.75,), methods=("historical",))[0]
    assert above["breaches"] == 1  # a loss above the VaR is a breach
    below = backtest.backtest_series(series(window + [-0.0199999]), window=5, confs=(0.75,), methods=("historical",))[0]
    assert below["breaches"] == 0


@pytest.mark.parametrize("method", ["historical", "normal"])
def test_C14_forecast_for_t_ignores_the_return_at_t_and_every_later_return(method):
    base = wavy_returns(60, scale=0.01)
    w, c = 12, 0.95
    f_base = backtest.rolling_forecasts(base, w, c, method)
    for t in (w, 20, 33, 59):
        altered = base.copy()
        altered[t:] = 0.3 * np.cos(np.arange(len(base) - t)) - 0.2  # different return at t and after
        f_alt = backtest.rolling_forecasts(altered, w, c, method)
        assert f_alt[t] == f_base[t]  # exactly equal: the forecast for t used neither r[t] nor anything later
        np.testing.assert_array_equal(f_alt[: t + 1], f_base[: t + 1])
    # the breach outcome, unlike the forecast, does depend on r[t]
    calm = np.append(base[:30], 0.0)
    crash = np.append(base[:30], -0.5)
    rows = lambda r: backtest.backtest_series(series(r), window=w, confs=(c,), methods=(method,))[0]
    assert rows(calm)["n_forecasts"] == rows(crash)["n_forecasts"]
    assert rows(crash)["breaches"] == rows(calm)["breaches"] + 1


@pytest.mark.parametrize("method", ["historical", "normal"])
def test_C15_a_window_with_a_missing_return_gives_no_forecast_then_recovers(method):
    n, w, gap = 30, 5, 12
    r = wavy_returns(n)
    r[gap] = np.nan
    f = backtest.rolling_forecasts(r, w, 0.9, method)
    nan_days = [i for i in range(n) if np.isnan(f[i])]
    # no forecast before a full window (0..4); no forecast while the gap is inside the window (13..17); forecasts return at 18
    assert nan_days == list(range(0, w)) + list(range(gap + 1, gap + 1 + w))
    assert not np.isnan(f[gap])  # the window ending on day 11 is complete, so day 12 has a forecast
    assert all(not np.isnan(f[i]) for i in range(gap + 1 + w, n))
    # recovery: the forecast at 18 equals the one computed on the clean window r[13:18]
    clean = backtest.rolling_forecasts(r[gap + 1 :], w, 0.9, method)
    assert f[gap + 1 + w] == clean[w]
    row = backtest.backtest_series(series(r), window=w, confs=(0.9,), methods=(method,))[0]
    # days 5..29 = 25 candidate days; minus 5 days without a forecast; minus day 12 whose own return is missing
    assert row["n_forecasts"] == 25 - 5 - 1


def test_forecast_array_has_the_input_length_and_nothing_is_manufactured_for_short_input():
    assert backtest.rolling_forecasts(wavy_returns(5), 5, 0.95, "historical").tolist() == [np.nan] * 5 or np.isnan(backtest.rolling_forecasts(wavy_returns(5), 5, 0.95, "historical")).all()
    f = backtest.rolling_forecasts(wavy_returns(6), 5, 0.95, "normal")
    assert len(f) == 6 and np.isnan(f[:5]).all() and not np.isnan(f[5])
    with pytest.raises(ValueError):
        backtest.rolling_forecasts(wavy_returns(10), 5, 0.95, "monte_carlo")
