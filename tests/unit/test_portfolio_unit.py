"""E. Portfolio experiment on hand-constructed fixtures."""
import inspect
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine import config, data, portfolio
from tests.unit.fixtures import smooth_prices

ROOT = Path(__file__).resolve().parents[2]
CUT = 399
RF_TRAIN, RF_TEST = 0.02, 0.04


@pytest.fixture(scope="module")
def prices():
    p = smooth_prices(14, 560)
    p = p.rename(columns={"S00": "SPY"})
    return p


@pytest.fixture(scope="module")
def is_etf(prices):
    return pd.Series({c: c == "SPY" for c in prices.columns})


@pytest.fixture(autouse=True)
def cut(monkeypatch, prices):
    monkeypatch.setattr(config, "TRAIN_END", prices.index[CUT].strftime("%Y-%m-%d"))
    monkeypatch.setattr(config, "TEST_START", prices.index[CUT + 1].strftime("%Y-%m-%d"))


def preset(**kw):
    base = dict(min_sharpe=-100.0, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=6, bounds=(0.05, 0.40))
    base.update(kw)
    return {"p": base}


def train_sharpes(prices, is_etf, rf=RF_TRAIN):
    r = data.simple_returns(prices).dropna(how="all").loc[: config.TRAIN_END]
    out = {}
    for t in prices.columns:
        if is_etf[t]:
            continue
        x = r[t].to_numpy()
        out[t] = (x.mean() * 252 - rf) / (x.std(ddof=1) * math.sqrt(252))
    return pd.Series(out).sort_values(ascending=False)


# ------------------------------------------------------------------ E26 training data only
def test_E26_perturbing_every_test_period_price_leaves_holdings_and_weights_unchanged(prices, is_etf):
    a = portfolio.fit(data.simple_returns(prices), prices, is_etf, RF_TRAIN, preset())
    moved = prices.copy()
    moved.iloc[CUT + 1 :] = moved.iloc[CUT + 1 :] * 2.5 + 3.0  # every test-period price changes
    b = portfolio.fit(data.simple_returns(moved), moved, is_etf, RF_TRAIN, preset())
    assert a["eligible"] == b["eligible"]
    ea, eb = a["presets"]["p"], b["presets"]["p"]
    assert ea["status"] == eb["status"] == "constructed"
    assert list(ea["weights"].index) == list(eb["weights"].index)
    np.testing.assert_array_equal(ea["weights"].to_numpy(), eb["weights"].to_numpy())


def test_E26b_test_period_risk_free_rate_cannot_change_selection_or_weights(prices, is_etf):
    r = data.simple_returns(prices)
    x = portfolio.build_and_evaluate(r, prices, is_etf, RF_TRAIN, 0.0, presets=preset())["presets"]["p"]
    y = portfolio.build_and_evaluate(r, prices, is_etf, RF_TRAIN, 0.09, presets=preset())["presets"]["p"]
    assert x["tickers"] == y["tickers"] and x["weights"] == y["weights"]  # only evaluation figures may differ
    assert x["out_of_sample_test"]["sharpe"] != y["out_of_sample_test"]["sharpe"]
    assert x["in_sample_train"] == y["in_sample_train"]
    assert "rf_test" not in inspect.getsource(portfolio.fit)


# ------------------------------------------------------------------ E27 constraints, screens, no relaxation
def test_E27_a_constructed_portfolio_satisfies_weight_sum_bounds_and_screen(prices, is_etf):
    sh = train_sharpes(prices, is_etf)
    thr = float((sh.iloc[5] + sh.iloc[6]) / 2)  # independently: exactly six stocks have a higher training Sharpe
    f = portfolio.fit(data.simple_returns(prices), prices, is_etf, RF_TRAIN, preset(min_sharpe=thr, max_stocks=10))
    e = f["presets"]["p"]
    assert e["status"] == "constructed" and e["n_passing_screen"] == 6
    assert set(e["weights"].index) == set(sh.index[:6])  # selected = the stocks that pass the screen
    w = e["weights"].to_numpy()
    assert w.sum() == pytest.approx(1.0, abs=1e-9) and w.min() >= 0.05 - 1e-9 and w.max() <= 0.40 + 1e-9


def test_E27_infeasible_bounds_stay_unconstructed_and_are_not_relaxed(prices, is_etf):
    f = portfolio.fit(data.simple_returns(prices), prices, is_etf, RF_TRAIN, preset(bounds=(0.30, 0.40)))  # 6 assets x 0.30 = 1.8 > 1
    e = f["presets"]["p"]
    assert e["status"] == "not_constructed" and "weights" not in e and "cannot sum to 1" in e["reason"]


def test_E27_too_few_passing_stocks_stays_unconstructed(prices, is_etf):
    sh = train_sharpes(prices, is_etf)
    for k in (0, 1, config.MIN_ASSETS - 1):
        thr = float(sh.iloc[k] + 1e-9) if k == 0 else float((sh.iloc[k - 1] + sh.iloc[k]) / 2)
        e = portfolio.fit(data.simple_returns(prices), prices, is_etf, RF_TRAIN, preset(min_sharpe=thr))["presets"]["p"]
        assert e["n_passing_screen"] == k
        assert e["status"] == "not_constructed" and "weights" not in e and "not relaxed" in e["reason"]


# ------------------------------------------------------------------ E28 rebalancing convention
def test_E28_the_implementation_is_daily_constant_weight_rebalancing_not_buy_and_hold():
    prices = pd.DataFrame({"A": [100.0, 200.0, 100.0], "B": [100.0, 100.0, 100.0]}, index=pd.bdate_range("2022-01-03", periods=3))
    r = data.simple_returns(prices).iloc[1:]
    w = pd.Series({"A": 0.5, "B": 0.5})
    daily, missing = portfolio.portfolio_returns(r, w)
    wealth = float(np.prod(1.0 + daily))
    buy_and_hold = 0.5 * (100 / 100) + 0.5 * (100 / 100)  # shares bought on day 0 and never traded: 0.5 * A_end/A_0 + 0.5 * B_end/B_0
    constant_weight = (1 + (0.5 * 1.0 + 0.5 * 0.0)) * (1 + (0.5 * -0.5 + 0.5 * 0.0))  # 1.5 * 0.75
    assert buy_and_hold == pytest.approx(1.0) and constant_weight == pytest.approx(1.125)
    assert wealth == pytest.approx(constant_weight, abs=1e-12) and missing == 0
    assert wealth != pytest.approx(buy_and_hold)
    assert portfolio.performance(daily, 0.0)["total_return"] == pytest.approx(0.125, abs=1e-12)


def test_E28b_readme_describes_the_implemented_convention():
    text = (ROOT / "README.md").read_text()
    assert "constant weights, rebalanced daily" in text
    assert "buy-and-hold" not in text.lower()


# ------------------------------------------------------------------ E29 performance figures
def test_E29_performance_matches_hand_calculation():
    r = pd.Series([0.01, -0.02, 0.015, 0.0, -0.01, 0.02], index=pd.bdate_range("2023-01-02", periods=6))
    rf = 0.03
    out = portfolio.performance(r, rf)
    v = r.tolist()
    mean = sum(v) / 6
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / 5)
    wealth, peak, worst = 1.0, 1.0, 0.0
    for x in v:
        wealth *= 1 + x
        peak = max(peak, wealth)
        worst = min(worst, wealth / peak - 1)
    assert out["mean_annual_return"] == pytest.approx(mean * 252, rel=1e-12)
    assert out["annual_vol"] == pytest.approx(sd * math.sqrt(252), rel=1e-12)
    assert out["sharpe"] == pytest.approx((mean * 252 - rf) / (sd * math.sqrt(252)), rel=1e-12)
    assert out["total_return"] == pytest.approx(wealth - 1, rel=1e-12)
    assert out["max_drawdown"] == pytest.approx(worst, rel=1e-12)
    assert (out["start"], out["end"], out["n_days"]) == ("2023-01-02", "2023-01-09", 6)


def test_E29b_same_holdings_equal_weight_benchmark_uses_the_same_dates_and_convention(prices, is_etf):
    r = data.simple_returns(prices)
    sh = train_sharpes(prices, is_etf)
    res = portfolio.build_and_evaluate(r, prices, is_etf, RF_TRAIN, RF_TEST, presets=preset(min_sharpe=float((sh.iloc[5] + sh.iloc[6]) / 2)))
    e = res["presets"]["p"]
    test = r.dropna(how="all").loc[config.TEST_START :]
    t = e["tickers"]
    eq_daily = test[t].sum(axis=1) / len(t)  # direct algebra: the average of the holdings' returns each day
    own = pd.Series(e["weights"])
    own_daily = (test[t] * own[t]).sum(axis=1)
    for key, daily, rf in (("equal_weight_same_holdings_test", eq_daily, RF_TEST), ("out_of_sample_test", own_daily, RF_TEST)):
        ref = portfolio.performance(daily, rf)
        for k in ("total_return", "max_drawdown", "sharpe", "mean_annual_return", "annual_vol", "n_days", "start", "end"):
            assert e[key][k] == pytest.approx(ref[k], rel=1e-9) if not isinstance(ref[k], str) else e[key][k] == ref[k]
    assert e["out_of_sample_test"]["start"] == e["equal_weight_same_holdings_test"]["start"] == res["test_start"]
    assert e["out_of_sample_test"]["end"] == e["equal_weight_same_holdings_test"]["end"] == res["test_end"]
    assert res["risk_free_test"] == RF_TEST and res["risk_free_train"] == RF_TRAIN


# ------------------------------------------------------------------ E30 risk-free rate
def test_E30_the_rate_used_to_choose_weights_is_computed_from_training_dates_only():
    idx = pd.bdate_range("2020-01-01", periods=10)
    rate = pd.Series([2.0] * 6 + [9.0] * 4, index=idx)
    train_end = idx[5]
    assert data.risk_free_annual(rate, end=train_end) == pytest.approx(0.02)
    later = rate.copy()
    later.iloc[6:] = 50.0  # a different future
    assert data.risk_free_annual(later, end=train_end) == data.risk_free_annual(rate, end=train_end)
    assert data.risk_free_annual(rate, start=idx[6]) == pytest.approx(0.09)  # the evaluation rate is the later average, labelled as such


def test_E30b_the_pipeline_passes_the_training_rate_to_selection_and_the_test_rate_to_evaluation_only():
    src = (ROOT / "scripts" / "run_pipeline.py").read_text()
    assert re.search(r"rf_train\s*=\s*data\.risk_free_annual\(rf_series?,\s*end=config\.TRAIN_END\)", src) or re.search(r"rf_train\s*=\s*data\.risk_free_annual\(rf,\s*end=config\.TRAIN_END\)", src)
    assert re.search(r"rf_test\s*=\s*data\.risk_free_annual\(rf\w*,\s*start=config\.TEST_START\)", src)
    assert "portfolio.build_and_evaluate(returns, prices, is_etf, rf_train, rf_test)" in src
