"""Data module: price policy, symbol handling, snapshot round trip, and the download function with a MOCKED provider.

The mocked tests verify the control flow of our code (retries, fallbacks, failure metadata). They say nothing about the
accuracy of Yahoo Finance's data. A socket guard makes any real network access fail the test.
"""
import json
import socket
import sys
import types

import numpy as np
import pandas as pd
import pytest

from riskengine import data

IDX = pd.bdate_range("2024-01-02", periods=5)


# ----------------------------------------------------------------------------- price policy (validate_prices / simple_returns)
def table(values, index=IDX, name="X"):
    return pd.DataFrame({name: values}, index=index[: len(values)] if len(values) <= len(index) else index)


def test_simple_returns_of_100_110_99_with_hand_calculation():
    r = data.simple_returns(table([100.0, 110.0, 99.0]))["X"].to_numpy()  # 110/100 - 1 = 0.10 ; 99/110 - 1 = -0.10
    assert np.isnan(r[0]) and r[1] == pytest.approx(0.10, abs=1e-12) and r[2] == pytest.approx(-0.10, abs=1e-12)


def test_pre_listing_history_stays_missing_and_gaps_are_not_bridged():
    p = table([np.nan, np.nan, 100.0, np.nan, 110.0])
    r = data.simple_returns(p)["X"].to_numpy()
    assert np.isnan(r).all()  # no return spans the gap; the pre-listing rows are not filled
    p2 = table([np.nan, np.nan, 100.0, 101.0, 102.0])
    r2 = data.simple_returns(p2)["X"].to_numpy()
    assert np.isnan(r2[:3]).all() and r2[3] == pytest.approx(0.01, abs=1e-15) and r2[4] == pytest.approx(102 / 101 - 1, abs=1e-15)
    assert p2["X"].isna().tolist() == [True, True, False, False, False]  # the input was not modified


@pytest.mark.parametrize(
    "label,index,message",
    [
        ("duplicate timestamps", IDX[[0, 1, 1, 3]], "duplicate timestamps"),
        ("unsorted timestamps", IDX[[0, 2, 1, 3]], "not sorted"),
        ("descending timestamps", IDX[[3, 2, 1, 0]], "not sorted"),
    ],
)
def test_unsorted_or_duplicate_timestamps_are_rejected_not_silently_paired(label, index, message):
    with pytest.raises(ValueError, match=message):
        data.simple_returns(pd.DataFrame({"X": [100.0, 110.0, 99.0, 105.0]}, index=index))


@pytest.mark.parametrize(
    "bad,message",
    [(0.0, "strictly positive"), (-5.0, "strictly positive"), (np.inf, "infinite"), (-np.inf, "infinite")],
    ids=["zero", "negative", "+inf", "-inf"],
)
def test_observed_prices_must_be_finite_and_positive(bad, message):
    with pytest.raises(ValueError, match=message):
        data.simple_returns(pd.DataFrame({"X": [100.0, bad, 99.0, 105.0]}, index=IDX[:4]))


def test_nan_is_the_only_way_to_say_missing_and_a_valid_table_passes():
    p = pd.DataFrame({"A": [100.0, np.nan, 101.0, 102.0], "B": [5.0, 5.5, 5.4, np.nan]}, index=IDX[:4])
    data.validate_prices(p)  # no exception
    r = data.simple_returns(p)
    assert np.isnan(r["A"].to_numpy()[:3]).all() and r["A"].iloc[3] == pytest.approx(102 / 101 - 1, abs=1e-15)
    assert r["B"].iloc[1] == pytest.approx(0.1, abs=1e-12) and np.isnan(r["B"].iloc[3])


def test_share_class_symbols_keep_their_yahoo_form_in_the_universe():
    u = data.load_universe()
    assert {"BRK-B", "BF-B"} <= set(u["ticker"]) and not u["ticker"].str.contains(r"\.").any()
    assert set(u["is_etf"].unique()) <= {0, 1} and u["ticker"].is_unique


# ----------------------------------------------------------------------------- risk-free window averaging
def test_risk_free_annual_boundaries_are_inclusive_and_empty_windows_are_nan():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    assert data.risk_free_annual(s, start="2020-01-02", end="2020-01-02") == pytest.approx(0.02)
    assert np.isnan(data.risk_free_annual(s, start="2021-01-01"))  # nothing is invented for an empty window


# ----------------------------------------------------------------------------- snapshot round trip
def test_snapshot_save_and_load_preserve_values_timestamps_columns_and_metadata(tmp_path):
    prices = pd.DataFrame({"AAA": [10.0, 11.0, np.nan, 12.5], "BRK-B": [np.nan, 200.0, 201.5, 199.25]}, index=pd.bdate_range("2023-05-01", periods=4))
    rf = pd.Series([4.1, 4.2, 4.3, 4.4], index=prices.index, name="^IRX")
    meta = {"source": "unit-test fixture", "tickers_requested": 2, "nested": {"a": [1, 2, 3]}}
    data.save_snapshot(prices, rf, meta, data_dir=tmp_path)
    p2, rf2, meta2 = data.load_snapshot(tmp_path)
    pd.testing.assert_frame_equal(prices, p2, check_exact=True, check_freq=False)
    assert list(p2.columns) == ["AAA", "BRK-B"] and p2.index.equals(prices.index) and p2.dtypes.eq(float).all()
    assert np.isnan(p2.loc[prices.index[2], "AAA"]) and np.isnan(p2.loc[prices.index[0], "BRK-B"])  # missing stays missing
    pd.testing.assert_series_equal(rf, rf2, check_exact=True, check_freq=False)
    assert meta2 == meta and set(p.name for p in tmp_path.iterdir()) == {"prices.parquet", "tbill_13w.parquet", "prices_meta.json"}


def test_loading_a_missing_snapshot_says_how_to_create_it_and_does_not_download(tmp_path, monkeypatch):
    called = []
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: called.append(a)))
    with pytest.raises(FileNotFoundError, match="scripts/fetch_data.py"):
        data.load_snapshot(tmp_path)
    assert called == []


def test_utc_now_is_an_iso_timestamp_in_utc():
    assert pd.Timestamp(data.utc_now()).tzinfo is not None and data.utc_now().endswith("Z")


# ----------------------------------------------------------------------------- mocked download
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def block(*args, **kwargs):
        attempts.append(args)
        raise OSError("real network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", block)
    monkeypatch.setattr(socket, "getaddrinfo", block)
    yield attempts
    assert attempts == [], "a real network connection was attempted"


class FakeYahoo:
    """Stands in for the yfinance module: records every call and answers from a script."""

    def __init__(self, behaviour):
        self.behaviour, self.calls = behaviour, []

    def download(self, tickers, **kwargs):
        names = tickers if isinstance(tickers, list) else [tickers]
        self.calls.append((tuple(names) if isinstance(tickers, list) else tickers, kwargs))
        return self.behaviour(names, len(self.calls), kwargs)


def ok_frame(names, tz=None):
    idx = pd.bdate_range("2024-01-02", periods=3, tz=tz)
    cols = pd.MultiIndex.from_product([names, ["Open", "Close", "Volume"]])
    values = np.array([[1.0 + i, 100.0 + 10 * i + d, 5.0] for i, _ in enumerate(names) for d in range(3)]).reshape(len(names), 3, 3).transpose(1, 0, 2).reshape(3, -1)
    return pd.DataFrame(values, index=idx, columns=cols)


@pytest.fixture()
def sleeps(monkeypatch):
    log = []
    monkeypatch.setattr(data.time, "sleep", lambda s: log.append(s))
    return log


def install(monkeypatch, behaviour):
    fake = FakeYahoo(behaviour)
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


def test_download_success_keeps_symbols_and_makes_one_call(monkeypatch, sleeps):
    fake = install(monkeypatch, lambda names, n, kw: ok_frame(names))
    prices, rep = data.fetch_prices(["AAA", "BRK-B", "CCC"], "2024-01-01", "2024-02-01", batch_size=20)
    assert list(prices.columns) == ["AAA", "BRK-B", "CCC"]  # share-class symbol reaches the provider and comes back unchanged
    assert fake.calls[0][0] == ("AAA", "BRK-B", "CCC") and len(fake.calls) == 1 and sleeps == []
    assert rep == {"requested": 3, "returned": 3, "failed": {}}
    assert prices.loc[prices.index[1], "BRK-B"] == 111.0  # the Close column, not Open or Volume
    kw = fake.calls[0][1]
    assert kw["auto_adjust"] is True and kw["start"] == "2024-01-01" and kw["end"] == "2024-02-01" and kw["progress"] is False


def test_download_partial_success_retries_the_missing_ticker_alone_and_records_it(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        keep = [t for t in names if t != "BAD"]
        return ok_frame(keep) if keep else pd.DataFrame()

    fake = install(monkeypatch, behaviour)
    prices, rep = data.fetch_prices(["AAA", "BAD", "CCC"], "2024-01-01", "2024-02-01", retries=3, pause=2.0)
    assert list(prices.columns) == ["AAA", "CCC"]
    assert rep["failed"] == {"BAD": "empty response"} and rep["returned"] == 2 and rep["requested"] == 3 == rep["returned"] + len(rep["failed"])
    singles = [c for c in fake.calls if c[0] == "BAD"]
    assert len(singles) == 3  # the documented retry count, for the one missing ticker
    assert sleeps == [2.0, 4.0, 6.0]  # the delay grows with each failed try


def test_download_empty_response_for_everything_exhausts_the_retries_and_returns_nothing(monkeypatch, sleeps):
    fake = install(monkeypatch, lambda names, n, kw: pd.DataFrame())
    prices, rep = data.fetch_prices(["AAA", "BBB"], "2024-01-01", "2024-02-01", retries=3)
    assert prices.shape[1] == 0 and rep["returned"] == 0 and set(rep["failed"]) == {"AAA", "BBB"} and set(rep["failed"].values()) == {"empty response"}
    assert len(fake.calls) == 3 + 3 * 2  # one batch attempt of 3 tries, then 3 tries for each ticker alone
    assert len(sleeps) == len(fake.calls)


@pytest.mark.parametrize("retries", [1, 2, 5])
def test_retry_count_is_honoured(monkeypatch, sleeps, retries):
    fake = install(monkeypatch, lambda names, n, kw: pd.DataFrame())
    data.fetch_prices(["AAA"], "2024-01-01", "2024-02-01", retries=retries, pause=1.0)
    assert len(fake.calls) == 2 * retries  # the batch of one, then the ticker alone
    assert sleeps == [1.0 * (k + 1) for k in range(retries)] * 2


def test_download_malformed_response_is_recorded_with_its_error_and_not_used(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        idx = pd.bdate_range("2024-01-02", periods=3)
        return pd.DataFrame(np.ones((3, 2)), index=idx, columns=pd.MultiIndex.from_product([names[:1], ["Open", "Volume"]]))  # no Close column

    install(monkeypatch, behaviour)
    prices, rep = data.fetch_prices(["AAA"], "2024-01-01", "2024-02-01", retries=2)
    assert prices.shape[1] == 0 and "KeyError" in rep["failed"]["AAA"] and "Close" in rep["failed"]["AAA"]


def test_download_provider_exception_is_recorded_and_the_batch_survives(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        if "BOOM" in names:
            raise RuntimeError("provider unavailable")
        return ok_frame(names)

    install(monkeypatch, behaviour)
    prices, rep = data.fetch_prices(["AAA", "BOOM", "CCC"], "2024-01-01", "2024-02-01", retries=2)
    assert list(prices.columns) == ["AAA", "CCC"] or list(prices.columns) == ["AAA", "CCC"][: prices.shape[1]]
    assert rep["failed"]["BOOM"] == "RuntimeError: provider unavailable"


def test_a_transient_failure_followed_by_success_is_not_recorded_as_failed(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        if n == 1:
            raise ConnectionError("temporary")
        return ok_frame(names)

    fake = install(monkeypatch, behaviour)
    prices, rep = data.fetch_prices(["AAA"], "2024-01-01", "2024-02-01", retries=3, pause=2.0)
    assert rep["failed"] == {} and list(prices.columns) == ["AAA"] and len(fake.calls) == 2 and sleeps == [2.0]


def test_flat_single_ticker_frames_and_timezone_aware_indexes_are_handled(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        idx = pd.bdate_range("2024-01-02 09:30", periods=3, tz="America/New_York")  # tz-aware, not midnight
        return pd.DataFrame({"Open": [1.0, 2, 3], "Close": [10.0, 11.0, 12.0]}, index=idx)

    install(monkeypatch, behaviour)
    prices, rep = data.fetch_prices(["ZZZ"], "2024-01-01", "2024-02-01")
    assert prices.index.tz is None and (prices.index == prices.index.normalize()).all() and prices["ZZZ"].tolist() == [10.0, 11.0, 12.0]


def test_duplicate_timestamps_from_the_provider_are_caught_when_returns_are_computed(monkeypatch, sleeps):
    def behaviour(names, n, kw):
        f = ok_frame(names)
        return pd.concat([f, f.iloc[[1]]]).sort_index()  # a repeated day

    install(monkeypatch, behaviour)
    prices, _ = data.fetch_prices(["AAA"], "2024-01-01", "2024-02-01")
    with pytest.raises(ValueError, match="duplicate timestamps"):
        data.simple_returns(prices)


def test_risk_free_download_success_and_failure_modes(monkeypatch):
    install(monkeypatch, lambda names, n, kw: pd.DataFrame({"Close": [4.0, 4.1, np.nan, 4.3]}, index=pd.bdate_range("2024-01-02", periods=4, tz="UTC")))
    s = data.fetch_risk_free("2024-01-01", "2024-02-01")
    assert s.name == "^IRX" and s.tolist() == [4.0, 4.1, 4.3] and s.index.tz is None  # the missing value is dropped, not filled
    for frame, message in ((pd.DataFrame(), "no data"), (pd.DataFrame({"Open": [1.0]}, index=pd.bdate_range("2024-01-02", periods=1)), "no Close"), (pd.DataFrame({"Close": [np.nan]}, index=pd.bdate_range("2024-01-02", periods=1)), "no observations")):
        install(monkeypatch, lambda names, n, kw, f=frame: f)
        with pytest.raises(ValueError, match=message):
            data.fetch_risk_free("2024-01-01", "2024-02-01")
    install(monkeypatch, lambda names, n, kw: (_ for _ in ()).throw(RuntimeError("provider unavailable")))
    with pytest.raises(RuntimeError, match="provider unavailable"):  # a provider exception is not swallowed
        data.fetch_risk_free("2024-01-01", "2024-02-01")
