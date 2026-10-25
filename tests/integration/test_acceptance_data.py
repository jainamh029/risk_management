"""A, D and E acceptance checks on the cached real-price snapshot."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine import config, data, ml, portfolio

ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- A3, A4, A5
def test_A3_price_timestamps_are_valid_unique_sorted_and_observed_prices_are_positive_and_finite(prices, rf):
    for name, idx in (("prices", prices.index), ("rf", rf.index)):
        assert isinstance(idx, pd.DatetimeIndex), name
        assert not idx.isna().any() and idx.is_unique and idx.is_monotonic_increasing, name
        assert idx.tz is None and (idx == idx.normalize()).all(), name
        assert (idx.day_of_week < 5).all(), f"{name}: a weekend timestamp"
    v = prices.to_numpy()
    observed = v[~np.isnan(v)]
    assert np.isfinite(observed).all() and (observed > 0).all()
    assert np.isfinite(rf.dropna().to_numpy()).all()
    assert prices.dtypes.map(lambda d: np.issubdtype(d, np.floating)).all()


def test_A4_requested_returned_missing_equity_and_etf_counts_reconcile(prices, meta, universe, is_etf):
    assert meta["tickers_requested"] == len(universe) == universe["ticker"].nunique()
    assert meta["tickers_returned"] == prices.shape[1]
    assert meta["tickers_requested"] == meta["tickers_returned"] + len(meta["tickers_failed"])
    failed, returned = set(meta["tickers_failed"]), set(prices.columns)
    assert failed.isdisjoint(returned) and failed | returned == set(universe["ticker"])
    assert returned <= set(universe["ticker"])
    info = json.loads((ROOT / "results" / "run_info.json").read_text())["data"]
    etf_returned = sum(bool(is_etf[t]) for t in returned)
    assert info["etfs_returned"] == etf_returned and info["stocks_returned"] == len(returned) - etf_returned
    assert info["tickers_requested"] == meta["tickers_requested"] and info["tickers_returned"] == meta["tickers_returned"] and info["tickers_failed"] == len(failed)
    mem = json.loads((ROOT / "results" / "universe_membership_summary.json").read_text())
    assert mem["tickers_in_list"] == len(universe) and mem["etfs_in_list"] == int(universe["is_etf"].sum())
    assert mem["stocks_in_list"] == len(universe) - int(universe["is_etf"].sum())
    assert mem["stocks_returned_data"] == info["stocks_returned"]


def test_A5_later_prices_cannot_change_returns_or_features_as_of_t(prices):
    cols = ["AAPL", "XOM", "JPM", "VLTO", "NVDA"]
    base = prices[cols]
    r0, f0 = data.simple_returns(base), ml.build_features(base)
    for t in (300, 900, 1500, 1900):
        moved = base.copy()
        moved.iloc[t + 1 :] = moved.iloc[t + 1 :] * 0.41 + 7.0
        pd.testing.assert_frame_equal(r0.iloc[: t + 1], data.simple_returns(moved).iloc[: t + 1], check_exact=True)
        f1 = ml.build_features(moved)
        for k in ml.FEATURES:
            np.testing.assert_array_equal(f0[k].iloc[t].to_numpy(), f1[k].iloc[t].to_numpy())


def test_A5b_the_snapshot_is_split_and_dividend_adjusted_so_it_is_not_strictly_point_in_time(meta):
    """Recorded limitation, asserted so that it cannot silently change: adjusted closes are rewritten retroactively by the
    vendor whenever a dividend or split occurs, so a price series downloaded today is not what an investor saw on that date."""
    assert "adjusted" in meta["source"].lower() and "auto_adjust=True" in meta["source"]


# ----------------------------------------------------------------------------- D17, D20, D24 on real data
def test_D17_real_target_is_exactly_p_t_plus_63_over_p_t_minus_1(prices):
    y = ml.forward_return(prices[["AAPL", "JPM", "VLTO", "XOM"]])
    p = prices[["AAPL", "JPM", "VLTO", "XOM"]].to_numpy()
    n = len(p)
    checked = 0
    for i in (1, 50, 400, 1000, 1500, n - 64):
        for j in range(4):
            if not (np.isnan(p[i, j]) or np.isnan(p[i + 63, j])):
                assert y.iloc[i, j] == p[i + 63, j] / p[i, j] - 1.0
                checked += 1
    assert checked >= 18 and y.iloc[n - 63 :].isna().all().all()


def test_D20_real_latest_forecast_exclusions_reconcile_with_an_independent_rule(snapshot):
    r = json.loads((ROOT / "results" / "ml_summary.json").read_text())["info"]
    latest = pd.read_csv(ROOT / "results" / "ml_latest_forecasts.csv")
    prices = pd.read_parquet(ROOT / "data" / "prices.parquet")
    uni = data.load_universe().set_index("ticker")
    stocks = [t for t in prices.columns if not bool(uni.at[t, "is_etf"])]
    eligible = [t for t in stocks if prices[t].iloc[-127:].notna().all()]  # the 126-day return needs P[t-126]
    assert set(latest["ticker"]) == set(eligible) and len(latest) == r["latest_forecasts"]
    excluded = sorted(set(stocks) - set(eligible))
    assert len(stocks) == len(eligible) + len(excluded) and r["n_eligible_stocks"] == len(stocks)
    for t in excluded:  # every exclusion has a visible reason: a missing price in the last 127 sessions
        assert prices[t].iloc[-127:].isna().any()
    assert (pd.to_datetime(latest["feature_date"]) == prices.index[-1]).all()
    # the final model's last training feature date lies 63 or more sessions before the latest date
    assert prices.index.get_loc(pd.Timestamp(r["final_model_last_feature_date"])) + 63 <= len(prices) - 1


def test_D24_real_nonoverlapping_dates_and_reported_statistics(prices):
    ic = pd.read_csv(ROOT / "results" / "ml_rank_ic_by_date.csv", parse_dates=["date"]).set_index("date")["ensemble"]
    pos = pd.Series(np.arange(len(prices.index)), index=prices.index)
    sel = ml.nonoverlapping(ic, prices.index)
    assert (np.diff(pos[sel.index].to_numpy()) >= 63).all()
    rep = json.loads((ROOT / "results" / "ml_summary.json").read_text())["models"]["ensemble"]["rank_ic_nonoverlapping"]
    n = len(sel)
    sd = math.sqrt(sum((x - sel.mean()) ** 2 for x in sel) / (n - 1))
    assert rep["n"] == n and rep["mean"] == pytest.approx(sel.mean(), abs=1e-12)
    assert rep["std_error"] == pytest.approx(sd / math.sqrt(n), rel=1e-9) and rep["t_stat"] == pytest.approx(sel.mean() / (sd / math.sqrt(n)), rel=1e-9)
    assert n <= 15, "the number of independent periods is small; a larger n would need a longer test window"


# ----------------------------------------------------------------------------- E26, E30 on real data
def test_E26_real_perturbing_every_test_period_price_leaves_holdings_and_weights_unchanged(prices, returns, is_etf, rf):
    rf_train = data.risk_free_annual(rf, end=config.TRAIN_END)
    a = portfolio.fit(returns, prices, is_etf, rf_train)
    moved = prices.copy()
    moved.loc[config.TEST_START :] = moved.loc[config.TEST_START :] * 2.0 + 5.0
    b = portfolio.fit(data.simple_returns(moved), moved, is_etf, rf_train)
    assert a["eligible"] == b["eligible"]
    for name, ea in a["presets"].items():
        eb = b["presets"][name]
        assert ea["status"] == eb["status"] and ea["n_passing_screen"] == eb["n_passing_screen"]
        if ea["status"] == "constructed":
            assert list(ea["weights"].index) == list(eb["weights"].index)
            np.testing.assert_array_equal(ea["weights"].to_numpy(), eb["weights"].to_numpy())


def test_E30_real_risk_free_rates_are_the_training_and_test_window_averages(rf):
    pf = json.loads((ROOT / "results" / "portfolios.json").read_text())
    assert pf["risk_free_train"] == pytest.approx(float(rf.loc[: config.TRAIN_END].mean()) / 100, rel=1e-12)
    assert pf["risk_free_test"] == pytest.approx(float(rf.loc[config.TEST_START :].mean()) / 100, rel=1e-12)
    changed = rf.copy()
    changed.loc[config.TEST_START :] = 99.0
    assert data.risk_free_annual(changed, end=config.TRAIN_END) == data.risk_free_annual(rf, end=config.TRAIN_END)
