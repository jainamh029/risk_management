"""All tests run on the real downloaded snapshot in data/. No data is generated,
simulated or filled in. Create the snapshot with: python scripts/fetch_data.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import data  # noqa: E402


@pytest.fixture(scope="session")
def snapshot():
    try:
        prices, rf, meta = data.load_snapshot()
    except FileNotFoundError as e:
        pytest.exit(f"{e}", returncode=2)
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
