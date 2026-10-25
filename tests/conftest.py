"""Shared fixtures and the BLOCKED reporting rule.

tests/unit         deterministic hand-constructed fixtures; no market data, no network.
tests/integration  the cached real-price snapshot in data/.
tests/statistical  simulation experiments on explicitly synthetic fixtures (marker: statistical).

If the price snapshot is missing, every test that needs it is reported as BLOCKED
(a skip whose reason starts with "BLOCKED") and the whole run exits with status 10,
so a missing snapshot can never look like a pass. Nothing is downloaded.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import data  # noqa: E402

_blocked = []


def pytest_runtest_logreport(report):
    if report.skipped and isinstance(report.longrepr, tuple) and "BLOCKED" in str(report.longrepr[-1]):
        _blocked.append(report.nodeid)


def pytest_terminal_summary(terminalreporter):
    if _blocked:
        terminalreporter.section("BLOCKED")
        terminalreporter.write_line(f"{len(_blocked)} test(s) could not run because the price snapshot is missing. Nothing was downloaded. Create it with: python scripts/fetch_data.py")


def pytest_sessionfinish(session, exitstatus):
    if _blocked:
        session.exitstatus = 10


@pytest.fixture(scope="session")
def snapshot():
    try:
        prices, rf, meta = data.load_snapshot()
    except FileNotFoundError as e:
        pytest.skip(f"BLOCKED: {e}")
    return prices, rf, meta


@pytest.fixture(scope="session")
def prices(snapshot):
    return snapshot[0]


@pytest.fixture(scope="session")
def rf(snapshot):
    return snapshot[1]


@pytest.fixture(scope="session")
def meta(snapshot):
    return snapshot[2]


@pytest.fixture(scope="session")
def returns(prices):
    return data.simple_returns(prices)


@pytest.fixture(scope="session")
def universe():
    return data.load_universe()


@pytest.fixture(scope="session")
def is_etf(universe):
    return universe.set_index("ticker")["is_etf"].astype(bool)
