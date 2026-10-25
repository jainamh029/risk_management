"""Tests for the S&P 500 membership cross-check. They read the stored source tables (data/sp500_*.csv),
the real price snapshot and the stored results; no network is used."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mod():
    spec = importlib.util.spec_from_file_location("check_sp500_membership", ROOT / "scripts" / "check_sp500_membership.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="session")
def tables():
    cur = pd.read_csv(ROOT / "data" / "sp500_current.csv")
    ch = pd.read_csv(ROOT / "data" / "sp500_changes.csv", parse_dates=["date"])
    meta = json.loads((ROOT / "data" / "sp500_membership_meta.json").read_text())
    return cur, ch, meta


@pytest.fixture(scope="session")
def matrix(mod, tables, prices):
    cur, ch, _ = tables
    return mod.membership_matrix(prices.index, set(cur["ticker"]), ch)


def test_source_tables_are_stored_with_provenance(tables):
    cur, ch, meta = tables
    assert len(cur) == meta["current_constituents_rows"] == 503 and cur["ticker"].is_unique
    assert len(ch) == meta["changes_rows"]
    assert all(s["revision_id"].isdigit() and s["url"].startswith("https://en.wikipedia.org/wiki/") for s in meta["sources"])
    assert meta["fetched_utc"].endswith("Z") and "not the official" in meta["note"]


def test_no_change_row_was_dropped_while_parsing(tables):
    """An earlier version of the parser discarded the first table row (the 2026-10-06 WBD and TWLO change)."""
    _, ch, _ = tables
    last = ch.sort_values("date").iloc[-3:]
    assert str(ch["date"].max().date()) >= "2026-10-06"
    assert ((ch["added"] == "TWLO") & (ch["removed"] == "WBD")).any()
    assert (ch[["added", "removed"]].notna().any(axis=1)).all()  # every row adds or removes something
    assert ch["date"].notna().all()


def test_every_dated_change_flips_membership_on_its_effective_date(mod, tables, matrix, prices):
    """Independent check of the reconstruction: for every change inside the price window, the added ticker is a
    member on the first trading day on or after the date and not on the day before; a removed ticker is the reverse."""
    cur, ch, _ = tables
    dates = prices.index
    skip = set(mod.inconsistent_tickers(set(cur["ticker"]), ch, dates[0]))  # the two source tables disagree about these
    assert skip, "expected at least one known inconsistency (SATS) in the source tables"
    n = 0
    for row in ch.itertuples(index=False):
        if not (dates[0] < row.date <= dates[-1]):
            continue
        later = dates[dates >= row.date][0]
        earlier = dates[dates < row.date][-1]
        simultaneous = ch[(ch["date"] == row.date)]
        for t, kind in ((row.added, "added"), (row.removed, "removed")):
            if not isinstance(t, str) or t in skip:
                continue
            # a ticker that is both added and removed on the same date, or twice on the same date, has no net flip
            if ((simultaneous["added"] == t).sum() + (simultaneous["removed"] == t).sum()) != 1:
                continue
            after, before = bool(matrix.at[later, t]), bool(matrix.at[earlier, t])
            if kind == "added":
                assert after and not before, (t, row.date, kind)
            else:
                assert before and not after, (t, row.date, kind)
            n += 1
    assert n >= 150  # roughly 160 changes since 2019


def test_last_day_membership_is_exactly_the_current_list(tables, matrix):
    cur, _, _ = tables
    last = set(matrix.columns[matrix.iloc[-1].to_numpy()])
    assert last == set(cur["ticker"])


def test_rebuilt_index_size_stays_near_500_on_every_date(matrix):
    size = matrix.sum(axis=1)
    assert size.min() >= 500 and size.max() <= 510  # a table with missing rows would drift away from about 500


def test_known_membership_facts_from_the_stored_tables(matrix):
    d = matrix.index
    assert matrix.loc[d[0], "AAPL"] and matrix.loc[d[-1], "AAPL"]
    wbd = matrix["WBD"]
    assert not wbd.loc[: "2022-04-08"].any() and wbd.loc["2022-04-11":"2026-10-05"].all() and not wbd.loc["2026-10-06":].any()
    assert not matrix["TWLO"].loc[:"2026-10-05"].any() and matrix["TWLO"].loc["2026-10-06":].all()


def test_stored_membership_results_equal_a_fresh_analysis(mod, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mod, "RES", tmp_path)
    for f in ("var_backtest_by_ticker.csv",):
        (tmp_path / f).write_bytes((ROOT / "results" / f).read_bytes())
    mod.analyse()
    capsys.readouterr()
    fresh = json.loads((tmp_path / "universe_membership_summary.json").read_text())
    stored = json.loads((ROOT / "results" / "universe_membership_summary.json").read_text())
    assert fresh == stored
    a = pd.read_csv(tmp_path / "universe_membership.csv")
    b = pd.read_csv(ROOT / "results" / "universe_membership.csv")
    assert a.equals(b)


def test_summary_counts_are_consistent(universe):
    s = json.loads((ROOT / "results" / "universe_membership_summary.json").read_text())
    assert s["tickers_in_list"] == len(universe) == s["etfs_in_list"] + s["stocks_in_list"]
    assert sum(s["categories_stocks"].values()) == s["stocks_in_list"]
    assert s["current_members_in_list"] + len(s["current_members_not_in_list"]) == s["current_index_size"]
    assert 0 < s["backtest"]["share_while_member"] <= 1
    t = pd.read_csv(ROOT / "results" / "universe_membership.csv")
    assert (t["member_days"] <= t["window_days"]).all()
    assert int(t[~t["is_etf"]]["current_member"].sum()) == s["current_members_in_list"]


def test_source_inconsistencies_are_reported_not_hidden(mod, tables, prices):
    cur, ch, _ = tables
    s = json.loads((ROOT / "results" / "universe_membership_summary.json").read_text())
    found = mod.inconsistent_tickers(set(cur["ticker"]), ch, prices.index[0])
    assert s["source_inconsistencies"]["tickers"] == found
    assert "SATS" in found  # added 2026-03-23 per the history table, absent from today's list, no removal recorded
