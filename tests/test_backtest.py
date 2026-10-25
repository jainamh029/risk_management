import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2

from riskengine import backtest, config


def kupiec_reference(n, x, p):
    """Likelihood ratio from the binomial log-pmf, an independent route to the same statistic."""
    ll0 = binom.logpmf(x, n, p)
    ll1 = binom.logpmf(x, n, x / n)
    return -2.0 * (ll0 - ll1)


def real_backtest_rows(returns):
    rows = []
    for t in ("SPY", "AAPL", "JPM", "XOM", "VLTO", "NVDA"):
        rows += backtest.backtest_series(returns[t], config.WINDOW, config.CONFS)
    return rows


def test_kupiec_matches_binomial_reference_on_real_backtest_counts(returns):
    """n, x and p all come from real backtests of real tickers."""
    rows = real_backtest_rows(returns)
    assert len(rows) == 24
    for r in rows:
        lr, pv = backtest.kupiec_pof(r["n_forecasts"], r["breaches"], r["expected_rate"])
        assert lr == pytest.approx(kupiec_reference(r["n_forecasts"], r["breaches"], r["expected_rate"]), abs=1e-9)
        assert pv == pytest.approx(chi2.sf(lr, 1), rel=1e-12)
        assert (r["kupiec_lr"], r["kupiec_p"]) == (lr, pv)


def test_kupiec_reproduces_the_numbers_saved_in_the_original_notebook():
    # SO and TMO rows from the saved output of the original run on real data (n=1525, 99 percent).
    # The original formula was correct for x > 0.
    assert backtest.kupiec_pof(1525, 23, 0.01)[0] == pytest.approx(3.441928, abs=1e-6)
    assert backtest.kupiec_pof(1525, 24, 0.01)[0] == pytest.approx(4.317578, abs=1e-6)


def test_kupiec_boundaries_use_the_real_forecast_count(returns):
    """x = 0 and x = n are the logical boundaries; n is the real number of SPY forecasts."""
    n = backtest.backtest_series(returns["SPY"], config.WINDOW, (0.99,))[0]["n_forecasts"]
    for p in (0.01, 0.05):
        lr, pv = backtest.kupiec_pof(n, 0, p)
        assert lr == pytest.approx(-2.0 * n * math.log(1.0 - p), rel=1e-12)  # the original returned 0 here
        assert lr > 3.84 and pv < 0.05
        lr, pv = backtest.kupiec_pof(n, n, p)
        assert lr == pytest.approx(-2.0 * n * math.log(p), rel=1e-12) and pv < 1e-10
    lr, pv = backtest.kupiec_pof(n, 0.05 * n if float(0.05 * n).is_integer() else round(0.05 * n), 0.05)
    assert lr < 0.5 and pv > 0.4  # a breach count at the expected rate is not rejected


def test_kupiec_is_stable_at_the_pooled_real_sample_size(returns):
    """The original multiplied powers before taking logs. At the size of the whole real backtest that underflows."""
    bt = backtest.backtest_universe(returns, config.WINDOW, config.CONFS)
    g = bt[(bt["method"] == "historical") & (bt["confidence"] == 0.99)]
    n, x = int(g["n_forecasts"].sum()), int(g["breaches"].sum())
    assert (1 - 0.01) ** (n - x) * 0.01**x == 0.0  # the naive product is 0.0 in float64
    lr, pv = backtest.kupiec_pof(n, x, 0.01)
    assert math.isfinite(lr) and lr == pytest.approx(kupiec_reference(n, x, 0.01), rel=1e-9)


def test_kupiec_rejects_invalid_arguments():
    for args in [(0, 0, 0.05), (10, 11, 0.05), (10, -1, 0.05), (10, 1, 0.0), (10, 1, 1.0)]:
        with pytest.raises(ValueError):
            backtest.kupiec_pof(*args)


def independent_forecast(w, conf, method):
    """Hand-written forecast from one window, sharing no code with the package."""
    w = np.asarray(w)
    if method == "historical":
        losses = np.sort(-w)
        pos = conf * (len(losses) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return losses[lo] + (pos - lo) * (losses[hi] - losses[lo])
    from statistics import NormalDist

    mu = sum(w) / len(w)
    sd = math.sqrt(sum((x - mu) ** 2 for x in w) / (len(w) - 1))
    return -(mu - NormalDist().inv_cdf(conf) * sd)


@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "XOM", "VLTO"])
def test_backtest_breach_counts_match_an_independent_loop(returns, ticker):
    r = returns[ticker].values
    rows = backtest.backtest_series(returns[ticker], config.WINDOW, config.CONFS)
    assert len(rows) == 4
    for row in rows:
        n = x = 0
        for t in range(config.WINDOW, len(r)):
            win = r[t - config.WINDOW : t]
            if np.isnan(win).any() or np.isnan(r[t]):
                continue
            f = independent_forecast(win, row["confidence"], row["method"])
            n += 1
            x += int(-r[t] > f)
        assert row["n_forecasts"] == n and row["breaches"] == x, (ticker, row["method"], row["confidence"])


def test_forecasts_do_not_use_the_day_they_forecast(returns):
    """Forecast for day t must be identical when everything from t onward is removed."""
    r = returns["AAPL"].values
    for method in ("historical", "normal"):
        full = backtest.rolling_forecasts(r, config.WINDOW, 0.99, method)
        for T in (400, 900, 1500, 1900):
            cut = backtest.rolling_forecasts(r[:T], config.WINDOW, 0.99, method)
            assert np.isnan(cut[:config.WINDOW + 1]).sum() >= config.WINDOW  # nothing before a full window
            assert cut[T - 1] == full[T - 1] or (np.isnan(cut[T - 1]) and np.isnan(full[T - 1]))
            assert np.allclose(cut[config.WINDOW:T], full[config.WINDOW:T], equal_nan=True)


def test_late_listed_ticker_has_no_forecasts_before_a_full_window(returns):
    s = returns["VLTO"]
    n_valid = int(s.notna().sum())
    rows = backtest.backtest_series(s, config.WINDOW, config.CONFS)
    assert all(row["n_forecasts"] == n_valid - config.WINDOW for row in rows)


def test_universe_backtest_and_summary_are_consistent(returns):
    bt = backtest.backtest_universe(returns, config.WINDOW, config.CONFS)
    assert bt["ticker"].nunique() > 400
    assert set(bt["method"]) == {"historical", "normal"} and set(bt["confidence"]) == set(config.CONFS)
    assert (bt["breaches"] <= bt["n_forecasts"]).all() and (bt["kupiec_p"].between(0, 1)).all()
    s = backtest.summarise(bt)
    for _, row in s.iterrows():
        sub = bt[(bt["method"] == row["method"]) & (bt["confidence"] == row["confidence"])]
        assert row["total_breaches"] == sub["breaches"].sum()
        assert row["pooled_breach_rate"] == pytest.approx(sub["breaches"].sum() / sub["n_forecasts"].sum())
        assert row["share_rejected"] == pytest.approx((sub["kupiec_p"] < 0.05).mean())
        assert row["zero_breach_tickers"] == int((sub["breaches"] == 0).sum())


def test_stored_results_match_a_fresh_run(returns):
    stored = pd.read_csv(config_root() / "results" / "var_backtest_by_ticker.csv")
    fresh = backtest.backtest_universe(returns, config.WINDOW, config.CONFS)
    key = ["ticker", "method", "confidence"]
    a = stored.sort_values(key).reset_index(drop=True)
    b = fresh.sort_values(key).reset_index(drop=True)
    assert a[key].equals(b[key])
    assert (a["breaches"].values == b["breaches"].values).all()
    assert np.allclose(a["kupiec_lr"], b["kupiec_lr"], atol=1e-9)


def config_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]
