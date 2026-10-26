import math

import numpy as np
import pandas as pd
import pytest

from riskengine import config, data, metrics, portfolio


@pytest.fixture(scope="session")
def rfs(rf):
    return data.risk_free_annual(rf, end=config.TRAIN_END), data.risk_free_annual(rf, start=config.TEST_START)


@pytest.fixture(scope="session")
def result(returns, prices, is_etf, rfs):
    return portfolio.build_and_evaluate(returns, prices, is_etf, rfs[0], rfs[1])


@pytest.fixture(scope="session")
def fitted(returns, prices, is_etf, rfs):
    return portfolio.fit(returns, prices, is_etf, rfs[0])


def test_train_and_test_windows_do_not_overlap(result):
    assert result["train_end"] <= config.TRAIN_END < config.TEST_START <= result["test_start"]
    assert result["train_start"] < result["train_end"] < result["test_start"] < result["test_end"]


def test_fit_ignores_everything_after_the_training_end(returns, prices, is_etf, rfs, fitted):
    """Removing all post-training data must not change the selection or the weights."""
    cut_r = returns.loc[: config.TRAIN_END]
    cut_p = prices.loc[: config.TRAIN_END]
    f2 = portfolio.fit(cut_r, cut_p, is_etf, rfs[0])
    for name, e in fitted["presets"].items():
        assert e["status"] == f2["presets"][name]["status"]
        if e["status"] == "constructed":
            assert list(e["weights"].index) == list(f2["presets"][name]["weights"].index)
            assert np.allclose(e["weights"].values, f2["presets"][name]["weights"].values, atol=1e-12)


def test_only_stocks_with_a_complete_training_history_are_eligible(fitted, returns, is_etf):
    train = returns.dropna(how="all").loc[: config.TRAIN_END]
    for t in fitted["eligible"]:
        assert not is_etf[t] and train[t].notna().all()
    assert not any(is_etf.get(t, False) for t in fitted["eligible"])


def test_constructed_portfolios_respect_every_constraint(result):
    n_constructed = 0
    for name, e in result["presets"].items():
        if e["status"] != "constructed":
            continue
        n_constructed += 1
        lo, hi = config.PRESETS[name]["bounds"]
        w = np.array(list(e["weights"].values()))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.min() >= lo - 1e-8 and w.max() <= hi + 1e-8
        assert config.MIN_ASSETS <= len(w) <= config.PRESETS[name]["max_stocks"]
        assert e["n_passing_screen"] >= config.MIN_ASSETS
    assert n_constructed >= 1


def test_screens_are_never_relaxed(result):
    for name, e in result["presets"].items():
        if e["status"] == "not_constructed":
            assert e["n_passing_screen"] < config.MIN_ASSETS or "optimizer" in e["reason"] or "cannot sum" in e["reason"]
            assert "tickers" not in e
        else:
            assert e["n_passing_screen"] >= config.MIN_ASSETS


def test_selected_stocks_meet_the_screen_by_independent_calculation(result, returns, prices, rfs):
    rf_tr = rfs[0]
    train = returns.dropna(how="all").loc[: config.TRAIN_END]
    for name, e in result["presets"].items():
        if e["status"] != "constructed":
            continue
        pr = config.PRESETS[name]
        for t in e["tickers"]:
            r = train[t].values
            sharpe = (r.mean() * 252 - rf_tr) / (r.std(ddof=1) * math.sqrt(252))
            assert sharpe > pr["min_sharpe"], (name, t)
            p = prices.loc[: config.TRAIN_END, t].dropna().values
            peak, worst = p[0], 0.0
            for x in p:
                peak = max(peak, x)
                worst = min(worst, x / peak - 1)
            if pr["min_max_drawdown"] is not None:
                assert worst > pr["min_max_drawdown"], (name, t)
            if pr["min_return_1y"] is not None:
                assert p[-1] / p[-253] - 1 > pr["min_return_1y"], (name, t)
            if pr["max_hist_var99_1d"] is not None:
                last = -r[-config.WINDOW :]
                assert np.quantile(last, 0.99) < pr["max_hist_var99_1d"], (name, t)


def test_optimiser_beats_equal_weight_on_the_training_sharpe(fitted, rfs):
    tr = fitted["train"]
    for name, e in fitted["presets"].items():
        if e["status"] != "constructed":
            continue
        w = e["weights"]
        eq = pd.Series(1 / len(w), index=w.index)

        def sh(x):
            d, _ = portfolio.portfolio_returns(tr, x)
            return (d.mean() * 252 - rfs[0]) / (d.std(ddof=1) * math.sqrt(252))

        assert sh(w) >= sh(eq) - 1e-9


def test_infeasible_bounds_raise(fitted, rfs):
    tr = fitted["train"][fitted["eligible"][:3]]
    with pytest.raises(ValueError):
        portfolio.max_sharpe_weights(tr, rfs[0], (0.05, 0.25))  # 3 assets cannot sum to 1 with a 0.25 cap


def test_out_of_sample_numbers_match_an_independent_recalculation(result, returns, prices, rfs):
    test = returns.loc[config.TEST_START :]
    for name, e in result["presets"].items():
        if e["status"] != "constructed":
            continue
        t = list(e["weights"])
        w = np.array([e["weights"][x] for x in t])
        missing = int(test[t].isna().sum().sum())
        assert e["test_ticker_days_without_return"] == missing
        daily = np.nan_to_num(test[t].values) @ w
        cum = np.cumprod(1 + daily)
        o = e["out_of_sample_test"]
        assert o["total_return"] == pytest.approx(cum[-1] - 1, rel=1e-9)
        assert o["mean_annual_return"] == pytest.approx(daily.mean() * 252, rel=1e-9)
        assert o["annual_vol"] == pytest.approx(daily.std(ddof=1) * math.sqrt(252), rel=1e-9)
        peak, worst = 1.0, 0.0
        for c in cum:
            peak = max(peak, c)
            worst = min(worst, c / peak - 1)
        assert o["max_drawdown"] == pytest.approx(worst, rel=1e-9, abs=1e-12)
        assert o["n_days"] == len(test)


def test_equal_weight_comparison_uses_the_same_holdings(result, returns):
    test = returns.loc[config.TEST_START :]
    for name, e in result["presets"].items():
        if e["status"] != "constructed":
            continue
        t = e["tickers"]
        daily = np.nan_to_num(test[t].values).mean(axis=1)  # weight 1/n each
        q = e["equal_weight_same_holdings_test"]
        assert q["total_return"] == pytest.approx(np.prod(1 + daily) - 1, rel=1e-9)
        assert q["mean_annual_return"] == pytest.approx(daily.mean() * 252, rel=1e-9)


def test_selected_stocks_are_the_best_training_sharpe_among_those_passing(result, returns, prices, is_etf, rfs):
    """Independent recomputation of the aggressive screen over every eligible stock."""
    train = returns.dropna(how="all").loc[: config.TRAIN_END]
    pr = config.PRESETS["aggressive"]
    passers = {}
    for t in returns.columns:
        if is_etf.get(t, False) or not train[t].notna().all():
            continue
        r = train[t].values
        sharpe = (r.mean() * 252 - rfs[0]) / (r.std(ddof=1) * math.sqrt(252))
        p = prices.loc[: config.TRAIN_END, t].dropna().values
        if sharpe > pr["min_sharpe"] and p[-1] / p[-253] - 1 > pr["min_return_1y"]:
            passers[t] = sharpe
    best = sorted(passers, key=passers.get, reverse=True)[: pr["max_stocks"]]
    e = result["presets"]["aggressive"]
    assert e["n_passing_screen"] == len(passers)
    assert e["tickers"] == best


def test_weights_are_a_local_optimum_of_the_training_sharpe(fitted, rfs):
    """Moving 1 percentage point of weight between any two holdings, in either direction, does not raise the Sharpe ratio."""
    tr = fitted["train"]
    for name, e in fitted["presets"].items():
        if e["status"] != "constructed":
            continue
        w = e["weights"]
        lo, hi = config.PRESETS[name]["bounds"]
        mu = tr[w.index].mean().values * 252
        cov = tr[w.index].cov().values * 252

        def sharpe(x):
            return (x @ mu - rfs[0]) / math.sqrt(x @ cov @ x)

        base = sharpe(w.values)
        for i in range(len(w)):
            for j in range(len(w)):
                if i == j:
                    continue
                x = w.values.copy()
                x[i] += 0.01
                x[j] -= 0.01
                if x.min() >= lo - 1e-12 and x.max() <= hi + 1e-12:
                    assert sharpe(x) <= base + 1e-7, (name, i, j)


def _worst(d, start_peak):
    wealth, peak, worst = 1.0, (1.0 if start_peak else None), 0.0
    for x in d:
        wealth *= 1 + x
        peak = wealth if peak is None else max(peak, wealth)
        worst = min(worst, wealth / peak - 1)
    return worst


def test_max_drawdown_counts_the_starting_capital_as_a_peak(returns, rfs):
    """Real test-period series for which the two definitions differ: a loss from the starting value is the worst drawdown."""
    test = returns.loc[config.TEST_START :]
    differ = [t for t in test.columns if test[t].notna().all() and abs(_worst(test[t].values, True) - _worst(test[t].values, False)) > 1e-9]
    assert len(differ) >= 3, "the real data should contain stocks whose worst fall began at the start of the test window"
    for t in differ[:8]:
        assert portfolio.performance(test[t], rfs[1])["max_drawdown"] == pytest.approx(_worst(test[t].values, True), abs=1e-12), t


def test_performance_var_and_es_match_the_var_module_formulas(returns, rfs):
    d = returns["SPY"].loc[config.TEST_START :].dropna()
    p = portfolio.performance(d, rfs[1])
    losses = np.sort(-d.values)
    for c, key in ((0.95, "95"), (0.99, "99")):
        pos = c * (len(losses) - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        v = losses[lo] + (pos - lo) * (losses[hi] - losses[lo])
        assert p[f"hist_var_{key}_1d"] == pytest.approx(v, abs=1e-15)
        assert p[f"hist_es_{key}_1d"] == pytest.approx(losses[losses >= v].mean(), abs=1e-15)


def test_spy_benchmark_matches_a_direct_calculation(result, returns, rfs):
    s = returns["SPY"].loc[config.TEST_START :].dropna().values
    b = result["benchmark_spy_test"]
    assert b["total_return"] == pytest.approx(np.prod(1 + s) - 1, rel=1e-9)
    assert b["sharpe"] == pytest.approx((s.mean() * 252 - rfs[1]) / (s.std(ddof=1) * math.sqrt(252)), rel=1e-9)


def test_stored_portfolio_results_match_a_fresh_run(result):
    import json
    from pathlib import Path

    stored = json.loads((Path(__file__).resolve().parents[2] / "results" / "portfolios.json").read_text())
    for name, e in result["presets"].items():
        s = stored["presets"][name]
        assert s["status"] == e["status"]
        if e["status"] == "constructed":
            assert s["tickers"] == e["tickers"]
            assert s["out_of_sample_test"]["total_return"] == pytest.approx(e["out_of_sample_test"]["total_return"], rel=1e-6)  # optimiser weights agree across operating systems to about 1e-7


def _training_sharpes(fitted, rf_train):
    tr = fitted["train"][fitted["eligible"]]
    sh = (tr.mean() * 252 - rf_train) / (tr.std(ddof=1) * math.sqrt(252))
    return sh.sort_values(ascending=False)


def test_a_screen_with_too_few_passers_is_refused_not_relaxed(monkeypatch, returns, prices, is_etf, rfs, fitted):
    """Uses the real training Sharpe ratios to place the threshold between the 2nd and 3rd best stock."""
    sh = _training_sharpes(fitted, rfs[0])
    thr = (sh.iloc[1] + sh.iloc[2]) / 2
    preset = dict(min_sharpe=float(thr), min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=10, bounds=(0.02, 0.30))
    monkeypatch.setitem(config.PRESETS, "balanced", preset)
    f = portfolio.fit(returns, prices, is_etf, rfs[0])
    e = f["presets"]["balanced"]
    assert e["n_passing_screen"] == 2
    assert e["status"] == "not_constructed" and "weights" not in e
    assert "2 stocks passed" in e["reason"] and "not relaxed" in e["reason"]


def test_a_screen_with_exactly_the_minimum_passers_is_built_from_those_stocks(monkeypatch, returns, prices, is_etf, rfs, fitted):
    sh = _training_sharpes(fitted, rfs[0])
    thr = (sh.iloc[config.MIN_ASSETS - 1] + sh.iloc[config.MIN_ASSETS]) / 2
    preset = dict(min_sharpe=float(thr), min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=None, max_stocks=10, bounds=(0.02, 0.50))
    monkeypatch.setitem(config.PRESETS, "balanced", preset)
    f = portfolio.fit(returns, prices, is_etf, rfs[0])
    e = f["presets"]["balanced"]
    assert e["n_passing_screen"] == config.MIN_ASSETS and e["status"] == "constructed"
    assert set(e["weights"].index) == set(sh.index[: config.MIN_ASSETS])
