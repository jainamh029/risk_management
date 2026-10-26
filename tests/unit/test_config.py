"""Configuration: defaults match the documentation, invalid settings fail clearly, imports have no side effects."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from riskengine import config, ml, portfolio
from tests.unit.fixtures import smooth_prices

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()


def test_defaults_match_what_the_documentation_states():
    assert config.WINDOW == 250 and "250 returns before that day" in README
    assert config.CONFS == (0.95, 0.99)
    assert config.TRAIN_END == "2023-12-31" and config.TEST_START == "2024-01-01" and config.MARKET == "SPY" and config.MIN_ASSETS == 4
    assert config.ML_HORIZON == 63 and "63 trading days later" in README
    assert config.ML_STEP == 5 and "every fifth day" in README
    assert config.ML_RF_WEIGHT == 0.5 and "equal blend" in README and config.ML_SEED == 0
    assert set(config.PRESETS) == {"conservative", "balanced", "aggressive"}
    assert config.PRESETS["conservative"]["bounds"] == (0.05, 0.25) and config.PRESETS["balanced"]["max_stocks"] == 12
    config.validate()  # the checked-in settings are themselves valid


@pytest.mark.parametrize(
    "setting,value,message",
    [
        ("WINDOW", 1, "WINDOW"), ("WINDOW", True, "WINDOW"), ("WINDOW", "250", "WINDOW"), ("WINDOW", 250.5, "WINDOW"),
        ("CONFS", (), "CONFS"), ("CONFS", (1.0,), "CONFS"), ("CONFS", (0.95, True), "CONFS"), ("CONFS", (1.5,), "CONFS"), ("CONFS", (float("nan"),), "CONFS"),
        ("TRAIN_END", "2024-06-01", "TEST_START"), ("TRAIN_END", "2024-01-01", "TEST_START"), ("TEST_START", "2023-12-31", "TEST_START"),
        ("TRAIN_END", "not a date", "TRAIN_END"), ("MARKET", "", "MARKET"), ("MARKET", None, "MARKET"),
        ("MIN_ASSETS", 0, "MIN_ASSETS"), ("MIN_ASSETS", 2.5, "MIN_ASSETS"),
        ("ML_HORIZON", 0, "ML_HORIZON"), ("ML_STEP", 0, "ML_STEP"), ("ML_STEP", True, "ML_STEP"),
        ("ML_RF_WEIGHT", 1.5, "ML_RF_WEIGHT"), ("ML_RF_WEIGHT", -0.1, "ML_RF_WEIGHT"), ("ML_RF_WEIGHT", True, "ML_RF_WEIGHT"),
        ("ML_RF_PARAMS", [1, 2], "ML_RF_PARAMS"), ("ML_GB_PARAMS", None, "ML_GB_PARAMS"),
    ],
)
def test_invalid_settings_fail_with_a_message_that_names_the_setting(monkeypatch, setting, value, message):
    monkeypatch.setattr(config, setting, value)
    with pytest.raises(ValueError, match=message):
        config.validate()


@pytest.mark.parametrize("bounds,max_stocks,message", [((0.5, 0.1), 6, "bounds"), ((-0.1, 0.3), 6, "bounds"), ((0.0, 1.5), 6, "bounds"), ((0.3, 0.4), 6, "cannot sum to 1"), ((0.0, 0.1), 6, "cannot sum to 1"), ((0.05, 0.25), 3, "max_stocks")])
def test_invalid_preset_bounds_and_counts_fail_clearly(monkeypatch, bounds, max_stocks, message):
    monkeypatch.setitem(config.PRESETS, "balanced", {**config.PRESETS["balanced"], "bounds": bounds, "max_stocks": max_stocks})
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_a_test_period_that_overlaps_training_is_refused_by_every_consumer(monkeypatch):
    prices = smooth_prices(6, 560).rename(columns={"S00": "SPY"})
    is_etf = pd.Series({c: c == "SPY" for c in prices.columns})
    from riskengine import data

    monkeypatch.setattr(config, "TRAIN_END", prices.index[400].strftime("%Y-%m-%d"))
    monkeypatch.setattr(config, "TEST_START", prices.index[300].strftime("%Y-%m-%d"))  # the test period would start inside the training data
    with pytest.raises(ValueError, match="TEST_START"):
        portfolio.fit(data.simple_returns(prices), prices, is_etf, 0.02)
    with pytest.raises(ValueError, match="TEST_START"):
        ml.run(prices, is_etf)
    assert "config.validate()" in (ROOT / "scripts" / "run_pipeline.py").read_text()


def test_seeds_and_paths_are_explicit_not_hidden():
    ml_source = (ROOT / "riskengine" / "ml.py").read_text()
    assert ml_source.count("random_state=config.ML_SEED") == 2  # both models take the one configured seed
    import inspect

    from riskengine import data

    assert inspect.signature(data.load_snapshot).parameters["data_dir"].default == data.DATA_DIR  # the default path is one named constant
    assert "data_dir" in inspect.signature(data.save_snapshot).parameters  # and every caller can redirect it, which the tests do


def digest_tree(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode() + f.read_bytes())
    return h.hexdigest()


def test_importing_the_package_has_no_network_file_or_output_side_effects(tmp_path):
    guard = tmp_path / "sitecustomize.py"
    guard.write_text("import socket\n\ndef _no(*a, **k):\n    raise OSError('network blocked during import')\n\nsocket.socket.connect = _no\nsocket.getaddrinfo = _no\n")
    before = {d: digest_tree(ROOT / d) for d in ("results", "data")}
    code = "import riskengine, riskengine.config, riskengine.data, riskengine.var, riskengine.backtest, riskengine.metrics, riskengine.portfolio, riskengine.ml, riskengine.serialize; print('imported')"
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{ROOT}"}, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == "imported" and out.stderr.strip() == ""
    assert {d: digest_tree(ROOT / d) for d in ("results", "data")} == before  # nothing was written
