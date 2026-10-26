"""Portfolio module: solver failure handling, independent optimisation check, missing holdings, costs, window errors."""
import inspect
import math
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine import config, data, portfolio
from tests.unit.fixtures import smooth_prices, wavy_returns

ROOT = Path(__file__).resolve().parents[2]


def four_assets():
    return pd.DataFrame({"A": wavy_returns(80), "B": wavy_returns(80, phase=1.0) + 0.001, "C": wavy_returns(80, phase=2.0), "D": wavy_returns(80, phase=3.0) - 0.0005})


def fake_solver(**attrs):
    return lambda *a, **k: types.SimpleNamespace(**attrs)


# ----------------------------------------------------------------------------- solver failure is explicit
def test_a_solver_that_does_not_converge_raises_with_its_message(monkeypatch):
    monkeypatch.setattr(portfolio, "minimize", fake_solver(success=False, message="iteration limit reached", x=np.full(4, 0.25)))
    with pytest.raises(RuntimeError, match="did not converge: iteration limit reached"):
        portfolio.max_sharpe_weights(four_assets(), 0.02, (0.0, 1.0))


@pytest.mark.parametrize(
    "weights,message",
    [([np.nan] * 4, "non-finite"), ([np.nan, 0.4, 0.3, 0.3], "non-finite"), ([np.inf, 0.0, 0.0, 0.0], "non-finite"), ([0.7, 0.7, -0.4, 0.0], "outside the constraints"), ([0.5, 0.5, 0.5, 0.5], "outside the constraints"), ([0.25, 0.25, 0.25, 0.2], "outside the constraints")],
    ids=["all NaN", "one NaN", "infinite", "negative weight", "sum above 1", "sum below 1"],
)
def test_weights_that_are_not_finite_or_violate_the_constraints_are_refused(monkeypatch, weights, message):
    monkeypatch.setattr(portfolio, "minimize", fake_solver(success=True, message="ok", x=np.array(weights)))
    with pytest.raises(RuntimeError, match=message):
        portfolio.max_sharpe_weights(four_assets(), 0.02, (0.0, 1.0))


def test_solver_failure_inside_fit_leaves_the_preset_unconstructed_with_the_reason(monkeypatch):
    prices = smooth_prices(8, 560).rename(columns={"S00": "SPY"})
    is_etf = pd.Series({c: c == "SPY" for c in prices.columns})
    monkeypatch.setattr(config, "TRAIN_END", prices.index[399].strftime("%Y-%m-%d"))
    monkeypatch.setattr(config, "TEST_START", prices.index[400].strftime("%Y-%m-%d"))
    preset = {"p": dict(min_sharpe=-100.0, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=6, bounds=(0.05, 0.4))}
    monkeypatch.setattr(portfolio, "minimize", fake_solver(success=False, message="line search failed", x=np.zeros(6)))
    e = portfolio.fit(data.simple_returns(prices), prices, is_etf, 0.02, preset)["presets"]["p"]
    assert e["status"] == "not_constructed" and "line search failed" in e["reason"] and "weights" not in e  # explicit, not relaxed, not equal-weighted


# ----------------------------------------------------------------------------- the objective, checked against a brute-force reference
def test_two_asset_optimum_matches_a_brute_force_grid_search():
    r = four_assets()[["A", "B"]]
    rf = 0.02
    mu, cov = r.mean().to_numpy() * 252, r.cov().to_numpy() * 252

    def sharpe(w1):
        w = np.array([w1, 1 - w1])
        return (w @ mu - rf) / math.sqrt(w @ cov @ w)

    grid = np.linspace(0.0, 1.0, 100001)
    best = grid[np.argmax([sharpe(g) for g in grid[::10]]) * 10]  # coarse-to-fine: every 10th point, then refine around it
    fine = np.linspace(max(0.0, best - 0.001), min(1.0, best + 0.001), 2001)
    ref = max(sharpe(g) for g in fine)
    w = portfolio.max_sharpe_weights(r, rf, (0.0, 1.0))
    assert w.sum() == pytest.approx(1.0, abs=1e-9) and (w >= -1e-9).all()
    assert sharpe(w[0]) == pytest.approx(ref, abs=1e-6) and abs(w[0] - best) < 2e-3


def test_bounds_that_cannot_hold_the_asset_count_fail_before_solving(monkeypatch):
    called = []
    monkeypatch.setattr(portfolio, "minimize", lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="cannot sum to 1"):
        portfolio.max_sharpe_weights(four_assets(), 0.02, (0.3, 0.4))  # 4 x 0.3 = 1.2 > 1
    with pytest.raises(ValueError, match="cannot sum to 1"):
        portfolio.max_sharpe_weights(four_assets(), 0.02, (0.0, 0.2))  # 4 x 0.2 = 0.8 < 1
    assert called == []


# ----------------------------------------------------------------------------- returns convention, missing holdings, costs
def test_a_holding_without_a_return_contributes_zero_and_is_counted():
    r = pd.DataFrame({"A": [0.10, np.nan, 0.02], "B": [0.00, 0.02, -0.01]}, index=pd.bdate_range("2024-01-02", periods=3))
    daily, missing = portfolio.portfolio_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    assert missing == 1
    assert daily.tolist() == pytest.approx([0.05, 0.01, 0.005], abs=1e-15)  # day 2: 0.5 * 0 (held as cash) + 0.5 * 0.02
    assert (daily.index == r.index).all()


def test_no_transaction_cost_is_modelled_and_the_documentation_says_so():
    for fn in (portfolio.performance, portfolio.portfolio_returns, portfolio.build_and_evaluate, portfolio.fit):
        assert not [p for p in inspect.signature(fn).parameters if "cost" in p.lower() or "fee" in p.lower() or "slippage" in p.lower()]
    assert "no transaction costs" in (ROOT / "README.md").read_text().lower()
    assert "transaction costs are not modelled" in " ".join(inspect.getdoc(portfolio).split()).lower()  # the module docstring, wrapped lines joined


def test_the_cumulative_wealth_drawdown_and_volatility_of_a_simple_path_by_hand():
    d = pd.Series([0.10, -0.20, 0.05, 0.0], index=pd.bdate_range("2024-01-02", periods=4))
    p = portfolio.performance(d, 0.0)
    wealth = 1.10 * 0.80 * 1.05 * 1.0  # 0.924
    assert p["total_return"] == pytest.approx(wealth - 1, abs=1e-15)
    assert p["max_drawdown"] == pytest.approx(0.80 - 1, abs=1e-15)  # 1.10 -> 0.88 is -20% from the peak
    mean = 0.35 / 4 - 0.20 / 4 * 0 if False else (0.10 - 0.20 + 0.05 + 0.0) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / 3)
    assert p["annual_vol"] == pytest.approx(sd * math.sqrt(252), rel=1e-12) and p["sharpe"] == pytest.approx(mean * 252 / (sd * math.sqrt(252)), rel=1e-12)


# ----------------------------------------------------------------------------- windows that cannot be used say so
def test_an_empty_training_or_test_window_raises_a_clear_error(monkeypatch):
    prices = smooth_prices(8, 560).rename(columns={"S00": "SPY"})
    is_etf = pd.Series({c: c == "SPY" for c in prices.columns})
    r = data.simple_returns(prices)
    monkeypatch.setattr(config, "TRAIN_END", "2001-01-01")
    monkeypatch.setattr(config, "TEST_START", "2001-06-01")
    with pytest.raises(ValueError, match="training window is empty"):
        portfolio.fit(r, prices, is_etf, 0.02)
    monkeypatch.setattr(config, "TRAIN_END", prices.index[399].strftime("%Y-%m-%d"))
    monkeypatch.setattr(config, "TEST_START", "2090-01-01")
    with pytest.raises(ValueError, match="test window is empty"):
        portfolio.build_and_evaluate(r, prices, is_etf, 0.02, 0.02, presets={"p": dict(min_sharpe=-100.0, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=6, bounds=(0.05, 0.4))})
    monkeypatch.setattr(config, "TEST_START", prices.index[400].strftime("%Y-%m-%d"))
    all_missing = prices.copy()
    all_missing.iloc[100:110, 1:] = np.nan  # a gap inside every stock's training history: no stock has a complete history
    with pytest.raises(ValueError, match="no stock has a complete training-window history"):
        portfolio.fit(data.simple_returns(all_missing), all_missing, is_etf, 0.02)
