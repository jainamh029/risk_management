"""A. Data integrity on hand-constructed prices (no market data, no network)."""
import numpy as np
import pandas as pd
import pytest

from riskengine import data
from tests.unit.fixtures import smooth_prices


def frame(values, name="X"):
    return pd.DataFrame({name: values}, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_A1_simple_returns_of_100_110_99():
    r = data.simple_returns(frame([100.0, 110.0, 99.0]))["X"].to_numpy()
    assert np.isnan(r[0])
    assert r[1] == pytest.approx(110 / 100 - 1, abs=1e-15)  # +0.10
    assert r[2] == pytest.approx(99 / 110 - 1, abs=1e-15)  # -0.10
    assert r[1] == pytest.approx(0.10, abs=1e-12) and r[2] == pytest.approx(-0.10, abs=1e-12)


def test_A2_gaps_give_missing_returns_and_nothing_is_filled():
    prices = frame([np.nan, 100.0, np.nan, 110.0])
    r = data.simple_returns(prices)["X"].to_numpy()
    assert np.isnan(r).all()  # consecutive-price policy: a return needs two adjacent observed prices
    # what a forward fill would have produced is different, so the check is not vacuous
    assert not np.isnan(data.simple_returns(prices.ffill())["X"].to_numpy()).all()
    # the input frame is not modified
    assert prices["X"].isna().tolist() == [True, False, True, False]


def test_A2b_a_returned_gap_does_not_leak_into_neighbouring_returns():
    r = data.simple_returns(frame([100.0, 101.0, np.nan, 103.0, 104.0]))["X"].to_numpy()
    assert np.isnan(r[[0, 2, 3]]).all()
    assert r[1] == pytest.approx(0.01, abs=1e-15) and r[4] == pytest.approx(104 / 103 - 1, abs=1e-15)


def test_A5_changing_prices_after_t_does_not_change_returns_as_of_t():
    p = smooth_prices(3, 60)
    r = data.simple_returns(p)
    for t in (10, 30, 59):
        q = p.copy()
        q.iloc[t + 1 :] = q.iloc[t + 1 :] * 3.7 + 11.0  # arbitrary change of everything after t
        r2 = data.simple_returns(q)
        pd.testing.assert_frame_equal(r.iloc[: t + 1], r2.iloc[: t + 1], check_exact=True)


def test_snapshot_report_describes_gaps_without_changing_them():
    p = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0], "LATE": [np.nan, np.nan, 5.0, 6.0], "GAP": [1.0, np.nan, 3.0, 4.0], "STOP": [1.0, 2.0, np.nan, np.nan]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    before = p.copy()
    rep = data.snapshot_report(p)
    pd.testing.assert_frame_equal(p, before)
    assert rep["listed_after_start"] == ["LATE"]
    assert rep["stopped_before_end"] == ["STOP"]
    assert rep["internal_gap_counts"] == {"GAP": 1}
    assert rep["n_dates"] == 4


def test_risk_free_average_uses_only_the_requested_window():
    s = pd.Series([4.0, 5.0, 6.0, 100.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]))
    assert data.risk_free_annual(s, end="2020-01-03") == pytest.approx((4 + 5 + 6) / 3 / 100)
    assert data.risk_free_annual(s, start="2020-01-02", end="2020-01-03") == pytest.approx(5.5 / 100)
    assert data.risk_free_annual(s) == pytest.approx((4 + 5 + 6 + 100) / 4 / 100)
