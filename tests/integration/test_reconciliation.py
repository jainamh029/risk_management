"""Counts and denominators in the checked-in results, reconciled with independent calculations on the frozen real snapshot.

Where a full independent recomputation is already done (tests/integration/test_acceptance_outputs.py::test_C16_*) this file adds
the reconciliations that were not covered: exclusion counts, ETF and stock splits, eligibility rules, and membership-filtered denominators.
"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from riskengine import backtest, config

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


def simple(prices, t):
    p = prices[t].to_numpy(dtype=float)
    r = np.full(len(p), np.nan)
    r[1:] = p[1:] / p[:-1] - 1.0
    return r


def test_ticker_counts_in_run_info_equal_the_snapshot_and_the_universe(prices, universe, snapshot):
    info = json.loads((RES / "run_info.json").read_text())["data"]
    meta = snapshot[2]
    etf = set(universe.loc[universe["is_etf"] == 1, "ticker"])
    assert info["tickers_requested"] == len(universe) == meta["tickers_requested"]
    assert info["tickers_returned"] == prices.shape[1] == info["etfs_returned"] + info["stocks_returned"]
    assert info["etfs_returned"] == len(set(prices.columns) & etf) and info["stocks_returned"] == len(set(prices.columns) - etf)
    assert info["tickers_failed"] == len(meta["tickers_failed"]) == info["tickers_requested"] - info["tickers_returned"]
    assert set(meta["tickers_failed"]) == set(universe["ticker"]) - set(prices.columns)
    assert info["n_dates"] == len(prices) and info["first_date"] == str(prices.index[0].date()) and info["last_date"] == str(prices.index[-1].date())


def test_latest_var_table_and_exclusions_reconcile_with_the_last_250_returns(prices):
    info = json.loads((RES / "run_info.json").read_text())
    vl = pd.read_csv(RES / "var_latest.csv")
    complete = [t for t in prices.columns if not np.isnan(simple(prices, t)[-config.WINDOW :]).any()]
    incomplete = [t for t in prices.columns if t not in complete]
    assert sorted(vl["ticker"]) == sorted(complete)
    assert info["var_latest"]["n_tickers"] == len(complete) and info["var_latest"]["excluded_incomplete_window"] == len(incomplete)
    assert len(complete) + len(incomplete) == prices.shape[1]
    # every excluded ticker really has a gap or a missing recent price: nothing is excluded for a different reason
    for t in incomplete:
        assert np.isnan(simple(prices, t)[-config.WINDOW :]).any()
    assert vl["window_end"].eq(str(prices.index[-1].date())).all() and vl["window_start"].eq(str(prices.index[-config.WINDOW].date())).all()


def test_backtest_and_metrics_eligibility_follow_the_documented_rules(prices):
    info = json.loads((RES / "run_info.json").read_text())
    bt = pd.read_csv(RES / "var_backtest_by_ticker.csv")
    mt = pd.read_csv(RES / "metrics.csv")
    n_ret = {t: int((~np.isnan(simple(prices, t))).sum()) for t in prices.columns}
    # backtest: at least window + 250 returns; metrics: at least 250
    expect_bt = sorted(t for t, n in n_ret.items() if n >= config.WINDOW + 250)
    expect_mt = sorted(t for t, n in n_ret.items() if n >= 250)
    assert sorted(bt["ticker"].unique()) == expect_bt and info["backtest"]["n_tickers"] == len(expect_bt)
    assert sorted(mt["ticker"]) == expect_mt and info["metrics"]["n_tickers"] == len(expect_mt)
    assert dict(zip(mt["ticker"], mt["n_returns"])) == {t: n_ret[t] for t in expect_mt}
    # every backtested ticker has all four method and confidence rows
    assert (bt.groupby("ticker").size() == 4).all()


def test_etf_and_stock_splits_reconcile_across_results(prices, universe):
    etf = set(universe.loc[universe["is_etf"] == 1, "ticker"])
    mt = pd.read_csv(RES / "metrics.csv")
    assert set(mt.loc[mt["is_etf"], "ticker"]) == etf & set(mt["ticker"])
    assert (~mt["is_etf"]).sum() + mt["is_etf"].sum() == len(mt)
    bt = pd.read_csv(RES / "var_backtest_by_ticker.csv")
    summ = pd.DataFrame(json.loads((RES / "var_backtest_summary.json").read_text()))
    # all-ticker totals equal stocks-only totals plus ETF totals, so the two published denominators cannot silently diverge
    for (m, c), s in summ.set_index(["method", "confidence"]).iterrows():
        sub = bt[(bt["method"] == m) & (bt["confidence"] == c)]
        stocks, etfs = sub[~sub["ticker"].isin(etf)], sub[sub["ticker"].isin(etf)]
        assert s["total_forecasts"] == stocks["n_forecasts"].sum() + etfs["n_forecasts"].sum()
        assert s["total_breaches"] == stocks["breaches"].sum() + etfs["breaches"].sum()
        assert s["n_tickers"] == len(stocks) + len(etfs) == sub["ticker"].nunique()
    mem = json.loads((RES / "universe_membership_summary.json").read_text())
    assert mem["backtest"]["stocks_tested"] == bt[~bt["ticker"].isin(etf)]["ticker"].nunique()


def test_member_only_denominators_equal_an_independent_count_on_a_stock_sample(prices, universe):
    """Plain-loop forecasts and breaches, restricted to dates on which the stock was an index member, against the
    package-function calculation the membership script uses, on a deterministic sample (alphabetical every 12th backtested stock).
    The full-universe membership results are reproduced exactly by test_membership.py::test_stored_membership_results_equal_a_fresh_analysis."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_sp500_membership", ROOT / "scripts" / "check_sp500_membership.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cur = pd.read_csv(ROOT / "data" / "sp500_current.csv")
    ch = pd.read_csv(ROOT / "data" / "sp500_changes.csv", parse_dates=["date"])
    M = mod.membership_matrix(prices.index, set(cur["ticker"]), ch)
    etf = set(universe.loc[universe["is_etf"] == 1, "ticker"])
    tested = sorted(set(pd.read_csv(RES / "var_backtest_by_ticker.csv")["ticker"]) - etf)
    sample = [t for t in tested[::12] if t in M.columns]
    assert len(sample) >= 30
    nd = NormalDist()
    for method, conf in (("historical", 0.99), ("normal", 0.95)):
        n_all = x_all = n_mem = x_mem = 0
        n_pkg = x_pkg = 0
        for t in sample:
            r = simple(prices, t)
            member = M[t].to_numpy()
            for i in range(config.WINDOW, len(r)):
                w = r[i - config.WINDOW : i]
                if np.isnan(w).any() or np.isnan(r[i]):
                    continue
                if method == "historical":
                    s = np.sort(-w)
                    pos = conf * (len(s) - 1)
                    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
                    f = s[lo] + (pos - lo) * (s[hi] - s[lo])
                else:
                    mu = w.sum() / len(w)
                    f = -(mu - nd.inv_cdf(conf) * math.sqrt(((w - mu) ** 2).sum() / (len(w) - 1)))
                b = -r[i] > f
                n_all += 1
                x_all += int(b)
                if member[i]:
                    n_mem += 1
                    x_mem += int(b)
            fp = backtest.rolling_forecasts(r, config.WINDOW, conf, method)
            valid = ~np.isnan(fp) & ~np.isnan(r)
            n_pkg += int((valid & member).sum())
            x_pkg += int((valid & (-r > fp) & member).sum())
        assert (n_mem, x_mem) == (n_pkg, x_pkg), (method, conf)
        assert 0 < n_mem <= n_all and x_mem <= x_all
