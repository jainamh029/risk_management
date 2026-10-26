"""Backtest module: window and count validation, no-forecast behaviour, consistent alignment between methods."""
import numpy as np
import pandas as pd
import pytest

from riskengine import backtest, var
from tests.unit.fixtures import wavy_returns


@pytest.mark.parametrize("bad", [0, 1, -3, 2.5, 12.0, True, False, np.True_, "12", None, [12], np.array([12]), float("nan")], ids=lambda v: repr(v)[:20])
@pytest.mark.parametrize("method", var.METHODS)
def test_invalid_windows_raise_value_error_everywhere(bad, method):
    s = pd.Series(wavy_returns(40), name="X", index=pd.bdate_range("2022-01-03", periods=40))
    with pytest.raises(ValueError, match="window"):
        backtest.rolling_forecasts(wavy_returns(40), bad, 0.95, method)
    with pytest.raises(ValueError, match="window"):
        backtest.backtest_series(s, bad, (0.95,), (method,))
    with pytest.raises(ValueError, match="window"):
        backtest.backtest_universe(pd.DataFrame({"A": s}), bad, (0.95,), min_forecasts=1)


def test_a_window_of_two_or_more_integers_is_valid_including_numpy_integers():
    for w in (2, 5, np.int64(5), np.int32(7)):
        f = backtest.rolling_forecasts(wavy_returns(20), w, 0.9, "normal")
        assert np.isnan(f[: int(w)]).all() and not np.isnan(f[int(w) :]).any()


# ----------------------------------------------------------------------------- Kupiec argument validation
@pytest.mark.parametrize("n,x,p", [(10.5, 1, 0.05), (100, 2.5, 0.05), (100, float("nan"), 0.05), (float("inf"), 1, 0.05), (True, 0, 0.05), (100, True, 0.05), ("100", 5, 0.05), (100, "5", 0.05), (None, 1, 0.05), (0, 0, 0.05), (-5, 0, 0.05), (10, -1, 0.05), (10, 11, 0.05)])
def test_kupiec_rejects_invalid_counts(n, x, p):
    with pytest.raises(ValueError):
        backtest.kupiec_pof(n, x, p)


@pytest.mark.parametrize("p", [0, 1, 0.0, 1.0, -0.1, 1.5, float("nan"), float("inf"), True, "0.05", None, [0.05], np.array([0.05])], ids=lambda v: repr(v)[:20])
def test_kupiec_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError, match="between 0 and 1"):
        backtest.kupiec_pof(100, 5, p)


def test_kupiec_accepts_numpy_integers_and_integer_valued_floats_with_identical_results():
    base = backtest.kupiec_pof(100, 7, 0.05)
    assert backtest.kupiec_pof(np.int64(100), np.int32(7), np.float64(0.05)) == base
    assert backtest.kupiec_pof(100.0, 7.0, 0.05) == base


# ----------------------------------------------------------------------------- no-forecast behaviour and aggregation
def test_a_series_with_no_possible_forecast_returns_no_rows_and_an_empty_summary():
    s = pd.Series(wavy_returns(8), name="X", index=pd.bdate_range("2022-01-03", periods=8))
    assert backtest.backtest_series(s, 10, (0.95,), ("historical",)) == []  # fewer observations than the window: nothing is manufactured
    empty = backtest.backtest_universe(pd.DataFrame({"A": s}), 10, (0.95,), min_forecasts=1)
    assert empty.empty
    summary = backtest.summarise(empty)
    assert summary.empty and list(summary.columns)[:3] == ["method", "confidence", "n_tickers"]


def test_a_series_that_is_all_missing_produces_no_rows():
    s = pd.Series(np.full(40, np.nan), name="X", index=pd.bdate_range("2022-01-03", periods=40))
    assert backtest.backtest_series(s, 10, (0.95, 0.99), var.METHODS) == []


def test_universe_excludes_tickers_with_too_little_history_and_the_counts_reconcile():
    idx = pd.bdate_range("2021-01-04", periods=120)
    frame = pd.DataFrame({"LONG": wavy_returns(120), "SHORT": np.where(np.arange(120) < 100, np.nan, wavy_returns(120)), "MID": wavy_returns(120, phase=2.0)}, index=idx)
    w, mf = 20, 50
    bt = backtest.backtest_universe(frame, w, (0.9,), min_forecasts=mf)
    eligible = [t for t in frame if frame[t].notna().sum() >= w + mf]  # independent rule
    assert sorted(bt["ticker"].unique()) == sorted(eligible) == ["LONG", "MID"]
    assert set(frame.columns) - set(bt["ticker"]) == {"SHORT"}  # the excluded ticker is accounted for, not silently averaged in
    s = backtest.summarise(bt)
    assert (s["n_tickers"] == len(eligible)).all() and (s["total_forecasts"] == bt.groupby(["method", "confidence"])["n_forecasts"].sum().to_numpy()).all()


@pytest.mark.parametrize("conf", [0.9, 0.99])
def test_both_methods_have_identical_forecast_availability_and_counting(conf):
    r = wavy_returns(60)
    r[[20, 21, 40]] = np.nan
    h = backtest.rolling_forecasts(r, 10, conf, "historical")
    n = backtest.rolling_forecasts(r, 10, conf, "normal")
    assert (np.isnan(h) == np.isnan(n)).all()  # the same windows are available whichever method is used
    s = pd.Series(r, name="X", index=pd.bdate_range("2022-01-03", periods=60))
    rows = {row["method"]: row for row in backtest.backtest_series(s, 10, (conf,), var.METHODS)}
    assert rows["historical"]["n_forecasts"] == rows["normal"]["n_forecasts"] == int((~np.isnan(h) & ~np.isnan(r)).sum())


def test_the_realised_return_on_the_forecast_day_can_change_that_days_breach_flag_only():
    base = wavy_returns(40)
    w = 10
    f0 = backtest.rolling_forecasts(base, w, 0.9, "normal")
    alt = base.copy()
    alt[25] = -0.5  # a crash on day 25
    f1 = backtest.rolling_forecasts(alt, w, 0.9, "normal")
    assert f1[25] == f0[25]
    np.testing.assert_array_equal(f1[:26], f0[:26])  # everything up to and including day 25 is unchanged (the first window days are NaN in both)
    assert not np.array_equal(f1[26:36], f0[26:36])  # later forecasts do use the crash (it is now inside their windows)
    s0 = pd.Series(base, name="X", index=pd.bdate_range("2022-01-03", periods=40))
    s1 = pd.Series(alt, name="X", index=s0.index)
    b0 = backtest.backtest_series(s0.iloc[:26], w, (0.9,), ("normal",))[0]["breaches"]
    b1 = backtest.backtest_series(s1.iloc[:26], w, (0.9,), ("normal",))[0]["breaches"]
    assert b1 == b0 + 1  # the crash day is a breach, the forecast for it was unchanged
