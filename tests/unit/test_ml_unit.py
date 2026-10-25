"""D. ML correctness and leakage controls on a hand-constructed, deterministic price panel.

The panel has a negative drift before the training cutoff and a positive drift after it, so the
training-mean forecast is negative while most test targets are positive. That separates quantities
that coincide on a rising market (for example baseline accuracy and the positive-target share).
"""
import inspect
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine import config, ml
from tests.unit.fixtures import smooth_prices

N_TICKERS, N_DAYS, CUT = 55, 700, 399
SMALL_RF = dict(n_estimators=12, max_depth=4, min_samples_leaf=20, max_features="sqrt", n_jobs=1)
SMALL_GB = dict(n_estimators=12, max_depth=2, learning_rate=0.1, subsample=0.7)


@pytest.fixture(scope="module")
def prices():
    return smooth_prices(N_TICKERS, N_DAYS)


@pytest.fixture(scope="module")
def etf(prices):
    return pd.Series(False, index=prices.columns)


def patch_config(mp, prices):
    mp.setattr(config, "TRAIN_END", prices.index[CUT].strftime("%Y-%m-%d"))
    mp.setattr(config, "TEST_START", prices.index[CUT + 1].strftime("%Y-%m-%d"))
    mp.setattr(config, "ML_STEP", 3)
    mp.setattr(config, "ML_RF_PARAMS", SMALL_RF)
    mp.setattr(config, "ML_GB_PARAMS", SMALL_GB)


@pytest.fixture(autouse=True)
def small_config(monkeypatch, prices):
    patch_config(monkeypatch, prices)


@pytest.fixture(scope="module")
def run1(prices, etf):
    mp = pytest.MonkeyPatch()
    patch_config(mp, prices)
    try:
        return ml.run(prices, etf)
    finally:
        mp.undo()


# ------------------------------------------------------------------ D17 alignment and target
def test_D17_target_is_exactly_p_t_plus_63_over_p_t_minus_1_and_aligned_by_timestamp(prices):
    y = ml.forward_return(prices)
    p = prices.to_numpy()
    n = len(prices)
    for j in range(p.shape[1]):
        for i in (0, 1, 100, 250, n - 64, n - 63):
            if i + 63 < n:
                assert y.iloc[i, j] == p[i + 63, j] / p[i, j] - 1.0  # bitwise equal, not approximately
    assert y.iloc[n - 63 :].isna().all().all() and y.iloc[: n - 63].notna().all().all()
    assert y.index.equals(prices.index)
    panel = ml.build_panel(prices, list(prices.columns))
    d = prices.index[200]
    assert panel.at[(d, "S03"), "y"] == prices["S03"].iloc[263] / prices["S03"].iloc[200] - 1.0
    assert panel.at[(d, "S03"), "mom_20"] == prices["S03"].iloc[200] / prices["S03"].iloc[180] - 1.0  # feature at t uses P[t], P[t-20]


def test_D17b_feature_at_t_ignores_everything_after_t(prices):
    full = ml.build_features(prices)
    for t in (150, 300, 500):
        cut = ml.build_features(prices.iloc[: t + 1])
        for k in ml.FEATURES:
            np.testing.assert_array_equal(cut[k].iloc[t].to_numpy(), full[k].iloc[t].to_numpy())


# ------------------------------------------------------------------ D18 training labels end by the cutoff
def test_D18_every_training_label_finishes_by_the_cutoff_and_none_reaches_the_test_period(prices):
    panel = ml.build_panel(prices, list(prices.columns))
    sp = ml.split_samples(panel, prices.index)
    pos = pd.Series(np.arange(len(prices)), index=prices.index)
    tr = pos[sp["train"].index.get_level_values("date")].to_numpy()
    te = pos[sp["test"].index.get_level_values("date")].to_numpy()
    cutoff = pd.Timestamp(config.TRAIN_END)
    assert sp["train_end_pos"] == CUT
    assert len(tr) > 1000 and len(te) > 1000
    assert (tr + config.ML_HORIZON <= CUT).all()
    assert prices.index[tr.max() + config.ML_HORIZON] <= cutoff  # the last label's end date is on or before the cutoff
    assert prices.index[te.min()] >= pd.Timestamp(config.TEST_START)
    assert prices.index[tr.max() + config.ML_HORIZON] < prices.index[te.min()]  # no label window reaches a test feature date
    assert (tr % config.ML_STEP == 0).all()


# ------------------------------------------------------------------ D19 later prices cannot change training
def test_D19_changing_prices_after_the_cutoff_changes_nothing_that_is_fitted(prices, etf):
    altered = prices.copy()
    altered.iloc[CUT + 1 :] = altered.iloc[CUT + 1 :] * 1.37  # one jump in every test-period price
    p0 = ml.build_panel(prices, list(prices.columns))
    p1 = ml.build_panel(altered, list(prices.columns))
    s0, s1 = ml.split_samples(p0, prices.index), ml.split_samples(p1, prices.index)
    pd.testing.assert_frame_equal(s0["train"], s1["train"], check_exact=True)  # same features and same labels
    rf0, gb0 = ml.fit_models(s0["train"])
    rf1, gb1 = ml.fit_models(s1["train"])
    X = s0["train"][ml.FEATURES].to_numpy()[:400]
    np.testing.assert_array_equal(rf0.predict(X), rf1.predict(X))
    np.testing.assert_array_equal(gb0.predict(X), gb1.predict(X))
    assert float(s0["train"]["y"].mean()) == float(s1["train"]["y"].mean())  # the training-mean baseline is unchanged too


# ------------------------------------------------------------------ D20 latest inference
def test_D20_latest_inference_uses_the_latest_eligible_date_and_exclusions_reconcile(prices, etf):
    stopped = prices.copy()
    stopped.loc[stopped.index[-5:], ["S04", "S09"]] = np.nan  # two tickers stop trading five days before the end
    late = stopped.copy()
    late.loc[late.index[:-100], "S20"] = np.nan  # a ticker with only 100 prices: too short for the 126-day feature
    r = ml.run(late, etf)
    last = prices.index[-1]
    assert (r["latest"]["feature_date"] == last).all() and r["info"]["latest_feature_date"] == str(last.date())
    # independent eligibility rule: all 127 prices ending on the last date must be present (the 126-day return needs P[t-126])
    eligible = [t for t in late.columns if late[t].iloc[-127:].notna().all()]
    assert set(r["latest"]["ticker"]) == set(eligible) and len(eligible) == N_TICKERS - 3
    excluded = sorted(set(late.columns) - set(r["latest"]["ticker"]))
    assert excluded == ["S04", "S09", "S20"]
    # it is not the last date with an observable 63-day target
    assert r["info"]["final_model_last_feature_date"] != r["info"]["latest_feature_date"]
    assert prices.index.get_loc(pd.Timestamp(r["info"]["final_model_last_feature_date"])) + config.ML_HORIZON <= len(prices) - 1


# ------------------------------------------------------------------ D21 preprocessing and ensemble weights
def test_D21_nothing_is_fitted_other_than_the_models_and_the_training_mean():
    src = inspect.getsource(ml)
    for name in ("StandardScaler", "MinMaxScaler", "RobustScaler", "fit_transform", "Pipeline", "PCA", "SelectKBest", "SimpleImputer", "QuantileTransformer"):
        assert name not in src
    assert "baseline_mean = float(train[\"y\"].mean())" in src


def test_D21_evaluated_and_deployed_ensemble_use_the_same_configured_weights(run1, monkeypatch):
    class Stub:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    X = np.zeros((4, 11))
    original = config.ML_RF_WEIGHT
    assert run1["info"]["rf_weight"] == original  # the weight the evaluated run recorded is the configured one
    assert np.allclose(ml.ensemble_predict(Stub(1.0), Stub(0.0), X), original)
    monkeypatch.setattr(config, "ML_RF_WEIGHT", 0.8)
    assert np.allclose(ml.ensemble_predict(Stub(1.0), Stub(0.0), X), 0.8)
    o = run1["_objects"]
    Xt = o["split"]["test"][ml.FEATURES].to_numpy()[:300]
    np.testing.assert_allclose(o["preds"]["ensemble"].to_numpy()[:300], 0.5 * o["rf"].predict(Xt) + 0.5 * o["gb"].predict(Xt), atol=1e-15)
    src = inspect.getsource(ml.run)
    assert src.count("ensemble_predict(") == 2  # evaluation and the latest forecast both call the one blend function
    assert "0.5" not in src and "0.6" not in src  # no weight is hard-coded inside run()


# ------------------------------------------------------------------ D22 R squared
def test_D22_r_squared_matches_the_documented_formula(run1):
    o = run1["_objects"]
    y = o["split"]["test"]["y"].to_numpy()
    base = float(o["split"]["train"]["y"].mean())
    for name, pred in o["preds"].items():
        p = pred.to_numpy()
        expected = 1.0 - np.sum((y - p) ** 2) / np.sum((y - base) ** 2)
        assert run1["models"][name]["oos_r2_vs_training_mean"] == pytest.approx(expected, rel=1e-12)
    assert o["baseline_mean"] == base and base < 0  # the fixture's training mean is negative on purpose


def test_D22b_a_zero_denominator_is_handled_explicitly():
    """If every test target equals the training mean, R squared is undefined. It must be reported as NaN or
    refused with ValueError; an unhandled ZeroDivisionError is not an explicit handling."""
    dates = pd.bdate_range("2020-01-01", periods=3)
    idx = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(12)]], names=["date", "ticker"])
    y = pd.Series(0.05, index=idx)
    test = pd.DataFrame({"y": y})
    pred = pd.Series(np.linspace(-0.1, 0.1, len(idx)), index=idx)
    try:
        out, _ = ml.score(pred, test, 0.05, pd.bdate_range("2020-01-01", periods=50))
    except ValueError:
        return
    assert math.isnan(out["oos_r2_vs_training_mean"])


# ------------------------------------------------------------------ D23 direction accuracy and baselines
def test_D23_direction_accuracy_baseline_accuracy_and_positive_share_are_separate_quantities(run1):
    o = run1["_objects"]
    y = o["split"]["test"]["y"].to_numpy()
    base = o["baseline_mean"]
    pos_share = float(np.mean(y > 0))
    assert run1["baselines"]["share_of_test_samples_with_positive_return"] == pytest.approx(pos_share, abs=1e-15)
    # training mean is negative, so the baseline says "not up" every time and scores the share of non-positive targets
    assert base < 0
    assert run1["baselines"]["training_mean_forecast"]["direction_accuracy"] == pytest.approx(float(np.mean(y <= 0)), abs=1e-15)
    assert run1["baselines"]["training_mean_forecast"]["direction_accuracy"] != pytest.approx(pos_share, abs=1e-6)
    assert pos_share > 0.5  # the fixture has a rising test period
    for name, pred in o["preds"].items():
        assert run1["models"][name]["direction_accuracy"] == pytest.approx(float(np.mean((pred.to_numpy() > 0) == (y > 0))), abs=1e-15)


# ------------------------------------------------------------------ D24 rank IC
def test_D24_rank_ic_is_computed_across_stocks_on_each_date():
    dates = pd.bdate_range("2020-01-01", periods=4)
    tick = [f"T{i}" for i in range(12)]
    idx = pd.MultiIndex.from_product([dates, tick], names=["date", "ticker"])
    forecast = pd.Series(np.tile(np.arange(12.0), 4), index=idx)
    realised = forecast.copy()
    realised.loc[dates[1]] = -np.arange(12.0)  # reversed ranking on date 1
    realised.loc[dates[2]] = np.array([3, 1, 4, 1.5, 9, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4])  # an arbitrary ranking
    realised.loc[dates[3]] = np.array([5.0] * 6 + [6.0] * 6)  # heavily tied
    ic = ml.rank_ic_by_date(forecast, realised)
    assert ic[dates[0]] == pytest.approx(1.0, abs=1e-12) and ic[dates[1]] == pytest.approx(-1.0, abs=1e-12)
    for d in dates[2:]:
        assert ic[d] == pytest.approx(stats.spearmanr(forecast.loc[d], realised.loc[d])[0], abs=1e-12)
    # a date with fewer than 10 stocks has no IC
    small = ml.rank_ic_by_date(forecast.loc[dates[0]].iloc[:9].to_frame().T.stack() if False else forecast.iloc[:9], realised.iloc[:9])
    assert small.empty


def test_D24b_non_overlapping_dates_are_at_least_63_sessions_apart_and_statistics_are_hand_checkable(run1, prices):
    cal = prices.index
    pos = pd.Series(np.arange(len(cal)), index=cal)
    ic = run1["ic_by_date"]["ensemble"]
    sel = ml.nonoverlapping(ic, cal)
    gaps = np.diff(pos[sel.index].to_numpy())
    assert (gaps >= config.ML_HORIZON).all() and (gaps == config.ML_HORIZON).all()
    s = run1["models"]["ensemble"]["rank_ic_nonoverlapping"]
    n = len(sel)
    assert s["n"] == n and s["mean"] == pytest.approx(sel.mean(), abs=1e-15)
    assert s["std_error"] == pytest.approx(math.sqrt(sum((x - sel.mean()) ** 2 for x in sel) / (n - 1)) / math.sqrt(n), rel=1e-12)
    assert s["t_stat"] == pytest.approx(s["mean"] / s["std_error"], rel=1e-12)
    # a calendar with missing dates cannot bring two selected dates closer than 63 sessions
    holey = ic.drop(ic.index[[3, 4, 70, 71, 72]])
    gaps = np.diff(pos[ml.nonoverlapping(holey, cal).index].to_numpy())
    assert (gaps >= config.ML_HORIZON).all()


# ------------------------------------------------------------------ D25 determinism
def test_D25_a_refit_with_unchanged_inputs_reproduces_predictions(run1, prices, etf):
    again = ml.run(prices, etf)
    for name in run1["_objects"]["preds"]:
        np.testing.assert_allclose(again["_objects"]["preds"][name].to_numpy(), run1["_objects"]["preds"][name].to_numpy(), rtol=0, atol=1e-12)
    pd.testing.assert_frame_equal(again["latest"].reset_index(drop=True), run1["latest"].reset_index(drop=True), atol=1e-12, rtol=0)
    assert again["models"]["ensemble"]["oos_r2_vs_training_mean"] == pytest.approx(run1["models"]["ensemble"]["oos_r2_vs_training_mean"], abs=1e-12)
