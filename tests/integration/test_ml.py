import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine import config, ml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def mlr(prices, is_etf):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ml.run(prices, is_etf)


def hand_features(p: np.ndarray, i: int) -> dict:
    """Every feature at row i from raw prices, written independently of the package."""
    r = p[1 : i + 1] / p[0:i] - 1.0  # r[k] is the return on row k+1
    r20, r60 = r[i - 20 : i], r[i - 60 : i]
    return {
        "ret_5": p[i] / p[i - 5] - 1,
        "mom_20": p[i] / p[i - 20] - 1,
        "mom_60": p[i] / p[i - 60] - 1,
        "mom_126": p[i] / p[i - 126] - 1,
        "vol_20": np.std(r20, ddof=1),
        "vol_60": np.std(r60, ddof=1),
        "skew_60": stats.skew(r60, bias=False),
        "kurt_60": stats.kurtosis(r60, fisher=True, bias=False),
        "dist_ma20": p[i] / p[i - 19 : i + 1].mean() - 1,
        "dist_ma50": p[i] / p[i - 49 : i + 1].mean() - 1,
        "drawdown_60": p[i] / p[i - 59 : i + 1].max() - 1,
    }


@pytest.mark.parametrize("ticker", ["AAPL", "XOM", "JPM"])
def test_features_match_a_hand_calculation(prices, ticker):
    p = prices[ticker].to_numpy()
    feats = ml.build_features(prices[[ticker]])
    for i in (300, 900, 1500, 1900):
        expected = hand_features(p, i)
        for k, v in expected.items():
            assert feats[k][ticker].iloc[i] == pytest.approx(v, rel=1e-9, abs=1e-12), (ticker, i, k)


def test_features_use_only_data_up_to_the_day(prices):
    full = ml.build_features(prices[["AAPL", "NVDA", "VLTO"]])
    for pos in (200, 700, 1300, 1800):
        cut = ml.build_features(prices[["AAPL", "NVDA", "VLTO"]].iloc[: pos + 1])
        for k in ml.FEATURES:
            a, b = cut[k].iloc[pos].to_numpy(), full[k].iloc[pos].to_numpy()
            assert np.allclose(a, b, equal_nan=True), (k, pos)


def test_a_feature_is_missing_unless_its_whole_window_has_prices(prices):
    f = ml.build_features(prices[["VLTO"]])
    first = prices["VLTO"].notna().to_numpy().argmax()
    for k, need in {"mom_126": 126, "vol_60": 60, "dist_ma50": 49, "drawdown_60": 59}.items():
        col = f[k]["VLTO"].to_numpy()
        assert np.isnan(col[: first + need]).all(), k  # nothing before a full window of real prices
        assert not np.isnan(col[first + need]), k


@pytest.mark.parametrize("ticker", ["AAPL", "JPM", "VLTO"])
def test_target_is_the_return_to_63_trading_days_later(prices, ticker):
    p = prices[ticker].to_numpy()
    y = ml.forward_return(prices[[ticker]])[ticker].to_numpy()
    h = config.ML_HORIZON
    n = len(p)
    first = np.flatnonzero(~np.isnan(p))[0]
    checked = 0
    for i in (first, first + 5, 1000, 1500, n - h - 1):
        if np.isnan(p[i]):
            continue  # before the ticker was listed there is no price and no target
        assert y[i] == pytest.approx(p[i + h] / p[i] - 1, rel=1e-12)
        checked += 1
    assert checked >= 3
    assert np.isnan(y[n - h :]).all()  # the last 63 days have no realised target yet


def test_panel_contains_only_stocks_and_matches_the_wide_frames(prices, is_etf, mlr):
    panel = mlr["_objects"]["panel"]
    tickers = set(panel.index.get_level_values("ticker"))
    assert tickers and not any(is_etf.get(t, False) for t in tickers)
    feats = ml.build_features(prices[["AAPL"]])
    row = panel.xs("AAPL", level="ticker")
    assert np.allclose(row["mom_60"].to_numpy(), feats["mom_60"]["AAPL"].to_numpy(), equal_nan=True)


def test_training_set_is_purged_and_test_set_starts_after_it(prices, mlr):
    sp = mlr["_objects"]["split"]
    dates = prices.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    tr_dates = sp["train"].index.get_level_values("date")
    te_dates = sp["test"].index.get_level_values("date")
    h = config.ML_HORIZON
    tr_pos = pos[tr_dates].to_numpy()
    assert (tr_pos + h <= sp["train_end_pos"]).all()  # whole target window ends inside training
    assert (tr_pos % config.ML_STEP == 0).all()  # thinned
    last_target_end = dates[tr_pos.max() + h]
    assert last_target_end <= pd.Timestamp(config.TRAIN_END)
    assert te_dates.min() >= pd.Timestamp(config.TEST_START) > last_target_end
    assert sp["train"]["y"].notna().all() and sp["test"]["y"].notna().all()
    assert sp["train"][ml.FEATURES].notna().all().all() and sp["test"][ml.FEATURES].notna().all().all()


def test_baseline_and_scores_match_independent_formulas(mlr):
    o = mlr["_objects"]
    train, test = o["split"]["train"], o["split"]["test"]
    assert o["baseline_mean"] == pytest.approx(train["y"].mean(), rel=1e-12)
    y = test["y"].to_numpy()
    for name, pred in o["preds"].items():
        p = pred.to_numpy()
        sse = sum((a - b) ** 2 for a, b in zip(y, p))
        sse0 = sum((a - o["baseline_mean"]) ** 2 for a in y)
        s = mlr["models"][name]
        assert s["oos_r2_vs_training_mean"] == pytest.approx(1 - sse / sse0, rel=1e-9)
        assert s["direction_accuracy"] == pytest.approx(np.mean((p > 0) == (y > 0)), rel=1e-12)
        assert s["n_test_samples"] == len(y)
    pos_share = float((y > 0).mean())
    assert mlr["baselines"]["share_of_test_samples_with_positive_return"] == pytest.approx(pos_share)
    assert o["baseline_mean"] > 0
    assert mlr["baselines"]["training_mean_forecast"]["direction_accuracy"] == pytest.approx(pos_share)


def test_rank_ic_matches_scipy_spearman(mlr):
    o = mlr["_objects"]
    pred, y = o["preds"]["ensemble"], o["split"]["test"]["y"]
    ic = mlr["ic_by_date"]["ensemble"]
    for d in ic.index[[0, 50, 200, 400, 600]]:
        a, b = pred.xs(d, level="date"), y.xs(d, level="date")
        assert ic[d] == pytest.approx(stats.spearmanr(a, b)[0], abs=1e-9)


def test_non_overlapping_dates_are_a_full_horizon_apart(prices, mlr):
    pos = pd.Series(np.arange(len(prices.index)), index=prices.index)
    ic = mlr["ic_by_date"]["ensemble"]
    sel = ml.nonoverlapping(ic, prices.index)
    gaps = np.diff(pos[sel.index].to_numpy())
    assert (gaps == config.ML_HORIZON).all() and len(sel) == mlr["models"]["ensemble"]["rank_ic_nonoverlapping"]["n"]
    s = mlr["models"]["ensemble"]["rank_ic_nonoverlapping"]
    assert s["mean"] == pytest.approx(sel.mean()) and s["std_error"] == pytest.approx(sel.std(ddof=1) / np.sqrt(len(sel)))
    assert s["t_stat"] == pytest.approx(s["mean"] / s["std_error"])


def test_the_evaluated_ensemble_is_the_blend_that_is_deployed(mlr):
    o = mlr["_objects"]
    X = o["split"]["test"][ml.FEATURES].to_numpy()[:500]
    expected = config.ML_RF_WEIGHT * o["rf"].predict(X) + (1 - config.ML_RF_WEIGHT) * o["gb"].predict(X)
    assert np.allclose(ml.ensemble_predict(o["rf"], o["gb"], X), expected)
    assert np.allclose(o["preds"]["ensemble"].to_numpy()[:500], expected)


def test_latest_forecast_uses_features_from_the_last_date(prices, is_etf, mlr):
    last = prices.index[-1]
    latest = mlr["latest"]
    assert (latest["feature_date"] == last).all()
    assert mlr["info"]["latest_feature_date"] == str(last.date())
    stocks = [t for t in prices.columns if not is_etf.get(t, False)]
    feats = ml.build_features(prices[stocks])
    complete = [t for t in stocks if all(not np.isnan(feats[k][t].iloc[-1]) for k in ml.FEATURES)]
    assert set(latest["ticker"]) == set(complete) and len(latest) == mlr["info"]["latest_forecasts"]
    assert np.isfinite(latest["predicted_63d_return"]).all()
    # the final model only used samples whose 63-day target had already ended
    dates = prices.index
    last_used = pd.Timestamp(mlr["info"]["final_model_last_feature_date"])
    assert dates.get_loc(last_used) + config.ML_HORIZON <= len(dates) - 1


def test_no_intervals_or_trading_signals_are_produced(mlr):
    cols = " ".join(mlr["latest"].columns).lower()
    for bad in ("lower", "upper", "confidence", "signal", "buy", "sell"):
        assert bad not in cols


def test_stored_ml_results_match_a_fresh_run(mlr):
    stored = json.loads((ROOT / "results" / "ml_summary.json").read_text())
    for name, m in mlr["models"].items():
        s = stored["models"][name]
        assert s["oos_r2_vs_training_mean"] == pytest.approx(m["oos_r2_vs_training_mean"], rel=1e-9)
        assert s["rank_ic_nonoverlapping"]["mean"] == pytest.approx(m["rank_ic_nonoverlapping"]["mean"], rel=1e-9)
    f = pd.read_csv(ROOT / "results" / "ml_latest_forecasts.csv").sort_values("ticker").reset_index(drop=True)
    g = mlr["latest"].sort_values("ticker").reset_index(drop=True)
    assert list(f["ticker"]) == list(g["ticker"])
    assert np.allclose(f["predicted_63d_return"], g["predicted_63d_return"], atol=1e-9)
