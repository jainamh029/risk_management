"""Serialisation: round trips, the JSON-safe policy for non-finite numbers, explicit errors, no partial output files."""
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine import serialize

ROOT = Path(__file__).resolve().parents[2]


def strict_loads(text):
    return json.loads(text, parse_constant=lambda c: pytest.fail(f"nonstandard JSON constant {c}"))


def test_nested_structures_and_supported_scalar_types_round_trip_with_finite_values_unchanged():
    obj = {
        "ints": [1, np.int32(2), np.int64(3)], "floats": [0.1 + 0.2, np.float64(1 / 3), np.float32(0.25), -0.0, 1e-300, 1.7976931348623157e308],
        "text": "ünïcode", "none": None, "flag": True, "nested": {"a": {"b": [{"c": 1.5}, [2.5, [3.5]]]}}, 5: "int key", "tuple": (1, 2),
        "stamp": pd.Timestamp("2024-03-05"),
    }
    back = strict_loads(serialize.dumps(obj))
    assert back["ints"] == [1, 2, 3] and back["floats"][0] == 0.1 + 0.2 and back["floats"][1] == 1 / 3 and back["floats"][3] == 0.0
    assert back["floats"][4] == 1e-300 and back["floats"][5] == 1.7976931348623157e308  # extreme finite values survive exactly
    assert back["nested"] == obj["nested"] and back["5"] == "int key" and back["tuple"] == [1, 2] and back["flag"] is True and back["none"] is None
    assert back["stamp"] == "2024-03-05" and back["text"] == "ünïcode"


def test_non_finite_numbers_become_null_everywhere_and_never_a_nonstandard_token():
    obj = {"a": float("nan"), "b": float("inf"), "c": -float("inf"), "d": [np.nan, np.float32("inf"), np.float64("-inf")], "e": {"f": np.nan}}
    text = serialize.dumps(obj, indent=2)
    assert not re.search(r"NaN|Infinity", text)
    assert strict_loads(text) == {"a": None, "b": None, "c": None, "d": [None, None, None], "e": {"f": None}}


def test_timestamps_are_written_as_calendar_dates_by_an_explicit_documented_rule():
    assert serialize.clean(pd.Timestamp("2024-03-05 15:30:00")) == "2024-03-05"  # the time of day is deliberately dropped: results are daily
    assert serialize.clean({pd.Timestamp("2024-03-05"): 1}) == {"2024-03-05 00:00:00": 1}  # keys are simply str(); values get the date rule


@pytest.mark.parametrize("unsupported", [dt.datetime(2024, 3, 5), dt.date(2024, 3, 5), np.datetime64("2024-03-05"), pd.NaT, {1, 2}, np.array([1, 2]), np.True_, object()], ids=lambda v: type(v).__name__)
def test_unsupported_types_raise_type_error_instead_of_being_guessed(unsupported):
    with pytest.raises(TypeError):
        serialize.dumps({"x": unsupported})


def test_a_failed_serialisation_cannot_leave_a_misleading_output_file(tmp_path):
    target = tmp_path / "result.json"
    target.write_text('{"previous": "good result"}')
    with pytest.raises(TypeError):
        target.write_text(serialize.dumps({"x": object()}))  # the exact pattern the pipeline uses: serialise fully, then write
    assert target.read_text() == '{"previous": "good result"}'  # the old file is untouched
    fresh = tmp_path / "new.json"
    with pytest.raises(TypeError):
        fresh.write_text(serialize.dumps({"x": {1, 2}}))
    assert not fresh.exists()  # and no empty or half-written file appears
    assert "write_text(serialize.dumps(obj" in (ROOT / "scripts" / "run_pipeline.py").read_text()  # the pipeline really does use that pattern


def test_dumps_never_mutates_its_input():
    obj = {"a": [float("nan"), 1.0], "b": {"c": np.float64("inf")}}
    snapshot = repr(obj)
    serialize.dumps(obj)
    assert repr(obj) == snapshot
