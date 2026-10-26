"""Branches that the coverage report showed no other test reaches: each gets a test that states the behaviour it protects."""
import numpy as np
import pandas as pd
import pytest

from riskengine import data, ml, var
from tests.unit.fixtures import smooth_prices, wavy_returns


def test_latest_var_table_rejects_a_return_table_shorter_than_the_window_with_both_numbers_in_the_message():
    returns = pd.DataFrame({"A": wavy_returns(10)})
    with pytest.raises(ValueError, match=r"need 250 rows of returns, have 10"):
        var.latest_var_table(returns, window=250)


def test_ml_run_on_a_history_too_short_to_form_any_sample_says_so_instead_of_fitting_on_nothing():
    prices = smooth_prices(12, 100, start="2019-01-02")
    with pytest.raises(ValueError, match="empty train or test sample"):
        ml.run(prices, pd.Series(False, index=prices.columns))


def test_close_extraction_skips_tickers_whose_close_is_entirely_missing_and_rejects_unrecognised_shapes():
    idx = pd.bdate_range("2024-01-02", periods=3)
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "Close"]])
    raw = pd.DataFrame([[1.0, 2.0, np.nan, np.nan]] * 3, index=idx, columns=cols)
    out = data._close_from_download(raw, ["AAA", "BBB", "CCC"])
    assert list(out.columns) == ["AAA"]  # BBB has only missing values, CCC is absent: neither is invented
    assert data._close_from_download(pd.DataFrame({"Volume": [1.0]}, index=idx[:1]), ["AAA"]).empty  # no Close column
    assert data._close_from_download(pd.DataFrame({"Close": [1.0, 2.0]}, index=idx[:2]), ["AAA", "BBB"]).empty  # flat frame cannot be attributed to two tickers
    assert data._close_from_download(pd.DataFrame({"Close": [np.nan, np.nan]}, index=idx[:2]), ["AAA"]).empty  # a single ticker with nothing observed


def test_risk_free_response_with_one_close_column_per_symbol_is_accepted(monkeypatch):
    idx = pd.bdate_range("2024-01-02", periods=3)
    cols = pd.MultiIndex.from_product([["Close"], ["^IRX"]])
    frame = pd.DataFrame([[4.0], [4.1], [4.2]], index=idx, columns=cols)

    class Yahoo:
        @staticmethod
        def download(*a, **k):
            return frame

    import sys

    monkeypatch.setitem(sys.modules, "yfinance", Yahoo)
    s = data.fetch_risk_free("2024-01-01", "2024-02-01")
    assert s.tolist() == [4.0, 4.1, 4.2] and s.name == "^IRX"
