import math

import numpy as np
import pytest

from riskengine import config, data, metrics


def test_downside_deviation_matches_hand_formula(returns, rf):
    r = returns["SPY"].dropna().values
    mar = data.risk_free_annual(rf) / 252
    total = 0.0
    for x in r:
        s = min(x - mar, 0.0)
        total += s * s
    expected = math.sqrt(total / len(r)) * math.sqrt(252)
    assert metrics.downside_deviation(r, mar) == pytest.approx(expected, rel=1e-12)


def test_downside_deviation_is_not_the_std_of_negative_returns(returns):
    """The original used the standard deviation of the negative returns only."""
    r = returns["AAPL"].dropna().values
    original = np.std(r[r < 0], ddof=1) * math.sqrt(252)
    correct = metrics.downside_deviation(r, 0.0)
    assert correct != pytest.approx(original, rel=0.02)


@pytest.mark.parametrize("t", ["SPY", "AAPL", "XOM"])
def test_drawdowns_match_an_independent_loop(prices, t):
    p = prices[t].dropna().values
    peak, worst = p[0], 0.0
    for x in p:
        peak = max(peak, x)
        worst = min(worst, x / peak - 1.0)
    assert metrics.max_drawdown(prices[t]) == pytest.approx(worst, abs=1e-15)
    assert metrics.current_drawdown(prices[t]) == pytest.approx(p[-1] / max(p) - 1.0, abs=1e-15)


def test_beta_matches_least_squares_slope(returns):
    both = returns[["AAPL", "SPY"]].dropna()
    slope = np.polyfit(both["SPY"].values, both["AAPL"].values, 1)[0]
    assert metrics.beta(returns["AAPL"], returns["SPY"]) == pytest.approx(slope, rel=1e-9)
    assert metrics.beta(returns["SPY"], returns["SPY"]) == pytest.approx(1.0, abs=1e-12)


def test_compute_metrics_row_matches_hand_calculation(prices, returns, rf):
    rf_a = data.risk_free_annual(rf)
    m = metrics.compute_metrics(prices, returns, rf_a).set_index("ticker")
    r = returns["SPY"].dropna().values
    mean_ann = r.mean() * 252
    vol = r.std(ddof=1) * math.sqrt(252)
    row = m.loc["SPY"]
    assert row["mean_annual_return"] == pytest.approx(mean_ann, rel=1e-12)
    assert row["annual_vol"] == pytest.approx(vol, rel=1e-12)
    assert row["sharpe"] == pytest.approx((mean_ann - rf_a) / vol, rel=1e-12)
    p = prices["SPY"].dropna()
    assert row["return_1m"] == pytest.approx(p.iloc[-1] / p.iloc[-22] - 1, rel=1e-12)
    assert row["return_1y"] == pytest.approx(p.iloc[-1] / p.iloc[-253] - 1, rel=1e-12)
    yrs = (p.index[-1] - p.index[0]).days / 365.25
    assert row["cagr"] == pytest.approx((p.iloc[-1] / p.iloc[0]) ** (1 / yrs) - 1, rel=1e-12)
    assert row["max_drawdown"] <= 0 <= row["annual_vol"]
    # ratios and moments, each from an independent route
    from scipy import stats

    mar = rf_a / 252
    dd_dev = math.sqrt(sum(min(x - mar, 0.0) ** 2 for x in r) / len(r)) * math.sqrt(252)
    assert row["downside_dev"] == pytest.approx(dd_dev, rel=1e-12)
    assert row["sortino"] == pytest.approx((mean_ann - rf_a) / dd_dev, rel=1e-12)
    peak, worst = p.iloc[0], 0.0
    for x in p.values:
        peak = max(peak, x)
        worst = min(worst, x / peak - 1)
    assert row["calmar"] == pytest.approx(row["cagr"] / abs(worst), rel=1e-12)
    assert row["skew"] == pytest.approx(stats.skew(r, bias=False), rel=1e-9)
    assert row["excess_kurtosis"] == pytest.approx(stats.kurtosis(r, fisher=True, bias=False), rel=1e-9)


def test_short_histories_get_nan_not_a_substitute(prices, returns, rf):
    """The original replaced a missing 1-year return with the annualised mean return."""
    m = metrics.compute_metrics(prices, returns, data.risk_free_annual(rf))
    short = m[m["n_returns"] < 252]
    assert (short["return_1y"].isna()).all()
    assert (m["n_returns"] >= 250).all()


def test_stored_metrics_match_a_fresh_run(prices, returns, rf):
    import pandas as pd
    from pathlib import Path

    stored = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "metrics.csv").set_index("ticker")
    fresh = metrics.compute_metrics(prices, returns, data.risk_free_annual(rf)).set_index("ticker")
    assert set(stored.index) == set(fresh.index)
    for col in ("sharpe", "sortino", "annual_vol", "max_drawdown", "cagr"):
        assert np.allclose(stored.loc[fresh.index, col], fresh[col], equal_nan=True, atol=1e-9)
