import numpy as np
import pandas as pd

from riskengine import data


def test_universe_is_the_original_list_deduplicated(universe):
    assert len(universe) == 483
    assert universe["ticker"].is_unique
    assert int(universe["is_etf"].sum()) == 10
    assert not universe["ticker"].str.contains(r"\.").any()  # Yahoo uses dashes: BRK-B, not BRK.B


def test_dash_tickers_are_present_in_the_downloaded_data(prices):
    # The original notebook converted '-' to '.', which returns nothing from Yahoo.
    assert "BRK-B" in prices.columns and "BF-B" in prices.columns
    assert prices["BRK-B"].notna().sum() > 1500


def test_snapshot_metadata_matches_the_files(prices, meta):
    assert meta["tickers_returned"] == prices.shape[1]
    assert meta["first_date"] == str(prices.index.min().date())
    assert meta["last_date"] == str(prices.index.max().date())
    assert meta["n_dates"] == len(prices)
    assert meta["tickers_requested"] == meta["tickers_returned"] + len(meta["tickers_failed"])


def test_every_ticker_comes_from_the_universe(prices, universe):
    assert set(prices.columns) <= set(universe["ticker"])


def test_late_listed_tickers_are_not_back_filled(prices, returns, meta):
    """The original code used bfill(), giving late-listed tickers a flat fake history."""
    checked = []
    for t in meta["listed_after_start"]:
        if prices[t].notna().sum() < 250:
            continue
        first = prices[t].first_valid_index()
        before = prices[t].loc[: first].iloc[:-1]
        assert len(before) > 0 and before.isna().all(), t
        # the first real price has no return (no previous price); it must not be a fake 0.0
        assert np.isnan(returns[t].loc[first]), t
        assert returns[t].loc[:first].isna().all(), t
        checked.append(t)
    assert "VLTO" in checked and "GEHC" in checked


def test_returns_match_an_independent_calculation(prices, returns):
    for t in ("SPY", "AAPL", "VLTO"):
        p = prices[t].dropna()
        expected = np.diff(p.values) / p.values[:-1]
        got = returns[t].dropna().values
        assert np.allclose(got, expected, rtol=0, atol=1e-12), t


def test_no_filled_values_in_downloaded_series(meta):
    # internal gaps would need an explicit decision; the real data has none
    assert meta["internal_gap_counts"] == {}


def test_short_series_are_recorded(prices, meta):
    short = {t: int(prices[t].notna().sum()) for t in prices.columns if prices[t].notna().sum() < 250}
    assert short == meta["fewer_than_250_valid_prices"]


def test_risk_free_is_the_real_tbill_series(rf):
    assert rf.name == "^IRX" and len(rf) > 1500
    avg = data.risk_free_annual(rf)
    assert avg == float(rf.mean() / 100.0)
    assert 0.0 < avg < 0.10
    # window averaging uses only dates inside the window
    w = data.risk_free_annual(rf, start="2024-01-01")
    assert w == float(rf.loc["2024-01-01":].mean() / 100.0)
