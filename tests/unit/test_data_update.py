"""Incremental update of the price snapshot, against a stand-in provider that answers from a known table of prices.

The stand-in plays Yahoo Finance: `download` returns the requested tickers and dates from its table, so the table is also the
"what a complete download would give" reference. Every expected value is that table (hand-constructed, deterministic, no random
numbers); the real provider and the network are never used (sockets are blocked).
"""
import json
import socket
import sys

import numpy as np
import pandas as pd
import pytest

from riskengine import data
from tests.unit.fixtures import smooth_prices

START = "2024-01-02"
REQUESTED_START = "2024-01-01"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def block(*args, **kwargs):
        attempts.append(args)
        raise OSError("real network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", block)
    monkeypatch.setattr(socket, "getaddrinfo", block)
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    yield
    assert attempts == []


class Provider:
    """Answers `download` from `self.prices` (adjusted closes) and `self.rf`; records every request."""

    def __init__(self, prices, rf):
        self.prices, self.rf, self.calls = prices.copy(), rf.copy(), []

    def download(self, tickers, start=None, end=None, **kw):
        names = tickers if isinstance(tickers, list) else [tickers]
        self.calls.append((tuple(names), start, end))
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        if names == ["^IRX"] or tickers == "^IRX":
            s = self.rf[(self.rf.index >= lo) & (self.rf.index < hi)]
            return pd.DataFrame({"Close": s})
        have = [t for t in names if t in self.prices.columns]
        sl = self.prices.loc[(self.prices.index >= lo) & (self.prices.index < hi), have]
        if sl.empty or not have:
            return pd.DataFrame()
        cols = pd.MultiIndex.from_product([have, ["Close"]])
        return pd.DataFrame(sl.to_numpy(), index=sl.index, columns=cols)

    def ticker_calls(self):
        return [c for c in self.calls if c[0] != ("^IRX",)]


N_DAYS, N_STORED = 40, 25


@pytest.fixture()
def truth():
    prices = smooth_prices(4, N_DAYS, start=START)
    rf = pd.Series(4.0 + 0.01 * np.arange(N_DAYS), index=prices.index, name="^IRX")
    return prices, rf


def make_snapshot(tmp_path, prices, rf, n=N_STORED, failed=None):
    head = prices.iloc[:n]
    meta = {"source": "stand-in provider", "downloaded_utc": "2024-02-06T00:00:00Z", "requested_start": REQUESTED_START, "requested_end_exclusive": str((head.index[-1] + pd.Timedelta(days=1)).date()),
            "risk_free_series": "^IRX", "tickers_requested": head.shape[1] + len(failed or {}), "tickers_returned": head.shape[1], "tickers_failed": failed or {}, **data.snapshot_report(head)}
    data.save_snapshot(head, rf.iloc[:n], meta, tmp_path)
    return head


def install(monkeypatch, prices, rf):
    p = Provider(prices, rf)
    monkeypatch.setitem(sys.modules, "yfinance", p)
    return p


END = "2024-12-31"
UNIVERSE = ["S00", "S01", "S02", "S03"]


def test_only_the_new_days_are_downloaded_and_the_result_equals_a_complete_download(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    p = install(monkeypatch, prices, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    got, got_rf, meta = data.load_snapshot(tmp_path)
    pd.testing.assert_frame_equal(got, prices, check_freq=False)  # identical to what a download of the whole history gives
    assert got_rf.tolist() == rf.tolist() and rep["new_rows"] == N_DAYS - N_STORED and rep["written"]
    last_stored = prices.index[N_STORED - 1]
    expected_start = (last_stored - pd.Timedelta(days=data.OVERLAP_DAYS)).strftime("%Y-%m-%d")
    assert all(c[1] == expected_start for c in p.calls)  # never the 2024-01-01 start of the history: nothing old is downloaded again
    assert len(p.ticker_calls()) == 1 and set(p.ticker_calls()[0][0]) == set(UNIVERSE)
    assert sorted(rep["appended"]) == UNIVERSE and rep["refreshed"] == {} and rep["added"] == []


def test_metadata_records_the_update_and_keeps_the_history_of_downloads(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    install(monkeypatch, prices, rf)
    data.update_snapshot(UNIVERSE, END, tmp_path)
    meta = json.loads((tmp_path / data.META_FILE).read_text())
    assert meta["first_downloaded_utc"] == "2024-02-06T00:00:00Z" and meta["downloaded_utc"] != meta["first_downloaded_utc"]
    assert meta["requested_start"] == REQUESTED_START and meta["requested_end_exclusive"] == END
    assert meta["last_date"] == str(prices.index[-1].date()) and meta["n_dates"] == N_DAYS and meta["tickers_returned"] == 4
    (entry,) = meta["update_log"]
    assert entry["mode"] == "incremental" and entry["new_rows"] == N_DAYS - N_STORED and entry["tickers_refreshed_in_full"] == [] and entry["tickers_appended"] == 4
    assert entry["last_date_before"] == str(prices.index[N_STORED - 1].date()) and entry["last_date_after"] == str(prices.index[-1].date())


def test_a_dividend_or_split_after_the_last_download_makes_only_that_ticker_download_in_full(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    adjusted = prices.copy()
    adjusted.loc[adjusted.index < prices.index[N_STORED + 3], "S01"] *= 0.99  # an ex-dividend date 4 days after the last stored day lowers every older adjusted price
    p = install(monkeypatch, adjusted, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    got, _, meta = data.load_snapshot(tmp_path)
    pd.testing.assert_frame_equal(got, adjusted, check_freq=False)  # the stored S01 history was replaced, not patched, and equals a complete download
    assert list(rep["refreshed"]) == ["S01"] and "no longer match" in rep["refreshed"]["S01"]
    assert sorted(rep["appended"]) == ["S00", "S02", "S03"]
    full_calls = [c for c in p.ticker_calls() if c[1] == REQUESTED_START]
    assert len(full_calls) == 1 and full_calls[0][0] == ("S01",)  # one ticker, from the original start
    assert meta["update_log"][0]["tickers_refreshed_in_full"] == ["S01"]


def test_a_difference_just_below_the_tolerance_is_not_an_adjustment_and_just_above_is(tmp_path, truth, monkeypatch):
    prices, rf = truth
    for factor, expect_refresh in ((1 + 0.5 * data.ADJUSTMENT_TOLERANCE, False), (1 + 2 * data.ADJUSTMENT_TOLERANCE, True)):
        d = tmp_path / f"f{expect_refresh}"
        make_snapshot(d, prices, rf)
        changed = prices.copy()
        changed.loc[changed.index < prices.index[N_STORED], "S02"] *= factor
        install(monkeypatch, changed, rf)
        rep = data.update_snapshot(UNIVERSE, END, d)
        assert (list(rep["refreshed"]) == ["S02"]) is expect_refresh


def test_a_ticker_with_new_prices_but_nothing_in_the_overlap_to_check_is_downloaded_in_full(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    gap = prices.copy()
    gap.loc[gap.index[N_STORED - 6] : gap.index[N_STORED - 1], "S03"] = np.nan  # no S03 price in the week before the last stored day
    # the stored snapshot has the same gap, then S03 returns after the last stored day
    head = gap.iloc[:N_STORED]
    data.save_snapshot(head, rf.iloc[:N_STORED], {**json.loads((tmp_path / data.META_FILE).read_text()), **data.snapshot_report(head)}, tmp_path)
    install(monkeypatch, gap, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    assert "no overlapping days" in rep["refreshed"]["S03"]
    pd.testing.assert_frame_equal(data.load_snapshot(tmp_path)[0], gap, check_freq=False)


def test_a_ticker_that_stopped_trading_keeps_its_history_and_gets_no_invented_prices(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    stopped = prices.copy()
    stopped.loc[stopped.index >= stopped.index[N_STORED - 8], "S02"] = np.nan  # delisted a week and a half before the last stored day
    head = stopped.iloc[:N_STORED]
    data.save_snapshot(head, rf.iloc[:N_STORED], {**json.loads((tmp_path / data.META_FILE).read_text()), **data.snapshot_report(head)}, tmp_path)
    install(monkeypatch, stopped, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    got = data.load_snapshot(tmp_path)[0]
    assert "S02" in rep["kept_without_new_data"] and got["S02"].isna().loc[got.index >= stopped.index[N_STORED - 8]].all()
    pd.testing.assert_series_equal(got["S02"], stopped["S02"], check_freq=False)
    assert "S02" in json.loads((tmp_path / data.META_FILE).read_text())["stopped_before_end"]


def test_a_new_ticker_is_downloaded_in_full_and_a_failed_one_is_recorded_not_invented(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices.drop(columns="S03"), rf)
    head = prices.drop(columns="S03").iloc[:N_STORED]
    meta = json.loads((tmp_path / data.META_FILE).read_text())
    meta.update(tickers_requested=4, tickers_failed={"S03": "empty response"})
    data.save_snapshot(head, rf.iloc[:N_STORED], meta, tmp_path)
    p = install(monkeypatch, prices.assign(), rf)
    rep = data.update_snapshot(UNIVERSE + ["NOPE"], END, tmp_path, retry_failed=True)  # S03 failed before; retried on request
    got, _, m = data.load_snapshot(tmp_path)
    assert rep["added"] == ["S03"] and "NOPE" in rep["failed"] and "NOPE" not in got.columns
    pd.testing.assert_frame_equal(got, prices, check_freq=False)
    assert list(m["tickers_failed"]) == ["NOPE"] and m["tickers_requested"] == 5 and m["tickers_returned"] == 4  # S03 is no longer listed as failed
    assert any(c[1] == REQUESTED_START and "S03" in c[0] for c in p.ticker_calls())


def test_nothing_new_changes_no_file(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf, n=N_DAYS)
    before = {f: (tmp_path / f).read_bytes() for f in (data.PRICES_FILE, data.RF_FILE, data.META_FILE)}
    install(monkeypatch, prices, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    assert rep["written"] is False and rep["new_rows"] == 0
    assert {f: (tmp_path / f).read_bytes() for f in before} == before


def test_a_second_update_straight_after_the_first_is_a_no_op(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    install(monkeypatch, prices, rf)
    assert data.update_snapshot(UNIVERSE, END, tmp_path)["written"]
    snapshot = {f: (tmp_path / f).read_bytes() for f in (data.PRICES_FILE, data.RF_FILE, data.META_FILE)}
    assert not data.update_snapshot(UNIVERSE, END, tmp_path)["written"]
    assert {f: (tmp_path / f).read_bytes() for f in snapshot} == snapshot


def test_the_risk_free_series_is_appended_and_a_restated_one_is_downloaded_again(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    install(monkeypatch, prices, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    assert rep["risk_free"] == f"{N_DAYS - N_STORED} new day(s) appended"
    d2 = tmp_path / "restated"
    make_snapshot(d2, prices, rf)
    restated = rf.copy()
    restated.iloc[N_STORED - 3] += 0.5  # an old value inside the overlap was revised by the provider
    install(monkeypatch, prices, restated)
    rep = data.update_snapshot(UNIVERSE, END, d2)
    assert "downloaded again in full" in rep["risk_free"]
    assert data.load_snapshot(d2)[1].tolist() == restated.tolist()


def test_full_mode_downloads_everything_again_and_records_it(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    p = install(monkeypatch, prices, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path, full=True)
    pd.testing.assert_frame_equal(data.load_snapshot(tmp_path)[0], prices, check_freq=False)
    assert rep["mode"] == "full" and all(c[1] == REQUESTED_START for c in p.calls)
    assert json.loads((tmp_path / data.META_FILE).read_text())["update_log"][0]["mode"] == "full"


# ----------------------------------------------------------------------------- failure handling
def snapshot_bytes(path):
    return {f: (path / f).read_bytes() for f in (data.PRICES_FILE, data.RF_FILE, data.META_FILE)}


def test_a_provider_that_is_down_leaves_the_snapshot_untouched(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    before = snapshot_bytes(tmp_path)

    def down(*a, **k):
        raise ConnectionError("provider unavailable")

    monkeypatch.setitem(sys.modules, "yfinance", type("Down", (), {"download": staticmethod(down)}))
    with pytest.raises((ValueError, ConnectionError)):
        data.update_snapshot(UNIVERSE, END, tmp_path)
    assert snapshot_bytes(tmp_path) == before and not list(tmp_path.glob(".*.tmp"))


def test_invalid_new_prices_are_rejected_before_anything_is_written(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    before = snapshot_bytes(tmp_path)
    bad = prices.copy()
    bad.iloc[N_STORED + 2, 1] = -5.0
    install(monkeypatch, bad, rf)
    with pytest.raises(ValueError, match="strictly positive"):
        data.update_snapshot(UNIVERSE, END, tmp_path)
    assert snapshot_bytes(tmp_path) == before


def test_a_failure_while_writing_leaves_the_old_files_in_place_and_no_temporary_files(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    before = snapshot_bytes(tmp_path)
    original = pd.Series.to_frame

    def explode(self, *a, **k):
        raise OSError("disk full")

    install(monkeypatch, prices, rf)
    monkeypatch.setattr(pd.Series, "to_frame", explode)
    with pytest.raises(OSError, match="disk full"):
        data.update_snapshot(UNIVERSE, END, tmp_path)
    monkeypatch.setattr(pd.Series, "to_frame", original)
    assert snapshot_bytes(tmp_path) == before and not list(tmp_path.glob(".*.tmp"))


def test_an_end_date_that_is_not_after_the_last_stored_date_is_refused(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    install(monkeypatch, prices, rf)
    last = prices.index[N_STORED - 1]
    for end in (str(last.date()), "2024-01-05"):
        with pytest.raises(ValueError, match="must be after the last stored date"):
            data.update_snapshot(UNIVERSE, end, tmp_path)


def test_updating_without_a_snapshot_says_how_to_create_one_and_downloads_nothing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "yfinance", type("Y", (), {"download": staticmethod(lambda *a, **k: calls.append(a))}))
    with pytest.raises(FileNotFoundError, match="scripts/fetch_data.py"):
        data.update_snapshot(UNIVERSE, END, tmp_path)
    assert calls == []


def test_the_updated_snapshot_gives_the_same_returns_as_a_complete_download(tmp_path, truth, monkeypatch):
    prices, rf = truth
    make_snapshot(tmp_path, prices, rf)
    install(monkeypatch, prices, rf)
    data.update_snapshot(UNIVERSE, END, tmp_path)
    pd.testing.assert_frame_equal(data.simple_returns(data.load_snapshot(tmp_path)[0]), data.simple_returns(prices), check_freq=False)


def test_if_the_full_download_of_a_changed_ticker_fails_the_old_history_is_kept_and_the_failure_is_reported(tmp_path, truth, monkeypatch):
    prices, rf = truth
    head = make_snapshot(tmp_path, prices, rf)
    adjusted = prices.copy()
    adjusted.loc[adjusted.index < prices.index[N_STORED + 3], "S01"] *= 0.99
    p = install(monkeypatch, adjusted, rf)
    original = p.download

    def flaky(tickers, start=None, end=None, **kw):
        names = tickers if isinstance(tickers, list) else [tickers]
        if start == REQUESTED_START and names == ["S01"]:
            return pd.DataFrame()  # the refresh of S01 returns nothing
        return original(tickers, start=start, end=end, **kw)

    p.download = flaky
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    got = data.load_snapshot(tmp_path)[0]
    assert "the refresh failed, so the old history was kept" in rep["refreshed"]["S01"] and "S01" in rep["failed"]
    pd.testing.assert_series_equal(got["S01"].iloc[: len(head)], head["S01"], check_freq=False)  # the stored history was not touched
    assert got["S01"].iloc[len(head) :].isna().all()  # and no new prices were attached to a history that no longer matches
    assert sorted(rep["appended"]) == ["S00", "S02", "S03"]


def test_tickers_that_failed_before_are_not_retried_unless_asked(tmp_path, truth, monkeypatch):
    prices, rf = truth
    head = prices.drop(columns="S03").iloc[:N_STORED]
    meta = {"source": "stand-in", "downloaded_utc": "2024-02-06T00:00:00Z", "requested_start": REQUESTED_START, "requested_end_exclusive": "2024-02-07", "risk_free_series": "^IRX",
            "tickers_requested": 4, "tickers_returned": 3, "tickers_failed": {"S03": "empty response"}, **data.snapshot_report(head)}
    data.save_snapshot(head, rf.iloc[:N_STORED], meta, tmp_path)
    p = install(monkeypatch, prices, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    assert rep["skipped_known_failures"] == ["S03"] and rep["added"] == [] and not any("S03" in c[0] for c in p.ticker_calls())  # not even requested
    assert "S03" not in data.load_snapshot(tmp_path)[0].columns and list(json.loads((tmp_path / data.META_FILE).read_text())["tickers_failed"]) == ["S03"]


def reassigned(prices, ticker, n_rows):
    """The provider now returns, for `ticker`, only a short history of some other security."""
    other = prices.copy()
    other[ticker] = np.nan
    other.loc[other.index[-n_rows:], ticker] = 7.0 + np.arange(n_rows)
    return other


def test_a_symbol_that_now_means_something_else_keeps_the_old_history_and_adds_nothing(tmp_path, truth, monkeypatch):
    """Real case found on 2026-10-26: Yahoo returned one day of another company for a symbol that had 1,431 stored days.
    The stored history of that ticker has no overlap with the answer, so it is flagged for a full download, and the full
    download is then refused because it covers almost none of the stored dates."""
    prices, rf = truth
    head = make_snapshot(tmp_path, prices, rf)
    p = install(monkeypatch, reassigned(prices, "S01", 3), rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path)
    got = data.load_snapshot(tmp_path)[0]
    assert list(rep["refused"]) == ["S01"] and "only 0%" in rep["refused"]["S01"] and "S01" not in rep["refreshed"]
    pd.testing.assert_series_equal(got["S01"].iloc[: len(head)], head["S01"], check_freq=False)
    assert got["S01"].iloc[len(head) :].isna().all()  # the other company's prices are not attached to the old history
    pd.testing.assert_frame_equal(got[["S00", "S02", "S03"]], prices[["S00", "S02", "S03"]], check_freq=False)  # everything else updated normally
    assert json.loads((tmp_path / data.META_FILE).read_text())["update_log"][0]["tickers_refused"] == ["S01"]


@pytest.mark.parametrize("covered,expect_refused", [(25, False), (23, False), (22, True), (0, True)])
def test_the_history_coverage_rule_applies_to_a_full_download_too_and_has_a_sharp_boundary(tmp_path, truth, monkeypatch, covered, expect_refused):
    prices, rf = truth
    head = make_snapshot(tmp_path, prices, rf)  # 25 stored dates; 90% of 25 is 22.5, so at least 23 must be covered
    thin = prices.copy()
    thin.iloc[: N_STORED - covered, 1] = np.nan  # S01 loses its first (25 - covered) stored dates
    install(monkeypatch, thin, rf)
    rep = data.update_snapshot(UNIVERSE, END, tmp_path, full=True)
    assert (list(rep["refused"]) == ["S01"]) is expect_refused
    if expect_refused:
        pd.testing.assert_series_equal(data.load_snapshot(tmp_path)[0]["S01"].iloc[: len(head)], head["S01"], check_freq=False)  # the old history is intact
    else:
        pd.testing.assert_series_equal(data.load_snapshot(tmp_path)[0]["S01"], thin["S01"], check_freq=False)  # accepted: equal to the new download


@pytest.mark.parametrize(
    "utc,expected",
    [
        ("2026-10-26 23:00", "2026-10-27"),  # Monday 7 pm New York (daylight time): the day just finished is included
        ("2026-10-26 20:30", "2026-10-26"),  # Monday 4:30 pm New York: the market has only just closed, so wait
        ("2026-10-26 04:50", "2026-10-26"),  # Monday 12:50 am New York: today has not traded yet
        ("2026-10-24 23:00", "2026-10-24"),  # Saturday: nothing new, today is excluded
        ("2026-10-25 23:00", "2026-10-25"),  # Sunday
        ("2026-12-01 22:00", "2026-12-02"),  # Tuesday 5 pm New York (standard time): included
        ("2026-12-01 21:59", "2026-12-01"),  # one minute before 5 pm New York in winter
        ("2026-10-30 23:00", "2026-10-31"),  # Friday evening: Saturday as the exclusive end
    ],
)
def test_the_default_end_date_includes_a_day_only_after_the_us_market_has_closed(utc, expected):
    import importlib.util
    from datetime import datetime, timezone
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("update_data", Path(__file__).resolve().parents[2] / "scripts" / "update_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.default_end(datetime.strptime(utc, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)) == expected
