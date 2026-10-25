"""Regression tests for the documented training-mean-reference score (repair stage):

    R2 = 1 - sum((y_test - prediction)^2) / sum((y_test - mean(y_train))^2)

Expected values come from this formula worked by hand. Nothing here uses market data.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from riskengine import ml, serialize

DATES = pd.bdate_range("2024-01-01", periods=80)
N_T = 12


def frame(y, pred, n_dates=3):
    idx = pd.MultiIndex.from_product([DATES[:n_dates], [f"T{i}" for i in range(N_T)]], names=["date", "ticker"])
    n = len(idx)
    y = pd.Series(np.broadcast_to(np.asarray(y, dtype=float), (n,)).copy(), index=idx)
    p = pd.Series(np.broadcast_to(np.asarray(pred, dtype=float), (n,)).copy(), index=idx)
    return p, pd.DataFrame({"y": y})


def score(pred, test, baseline):
    out, _ = ml.score(pred, test, baseline, DATES)
    return out


def tiny(y, pred, baseline):
    idx = pd.MultiIndex.from_tuples([(DATES[0], "A"), (DATES[0], "B")], names=["date", "ticker"])
    test = pd.DataFrame({"y": pd.Series(y, index=idx, dtype=float)})
    return score(pd.Series(pred, index=idx, dtype=float), test, baseline)


def test_the_independent_nonzero_denominator_example_gives_exactly_zero():
    out = tiny([2.0, 2.0], [1.0, 1.0], 1.0)  # model SSE = 1 + 1 = 2 ; baseline SSE = (2-1)^2 * 2 = 2 ; R2 = 1 - 2/2
    assert out["oos_r2_vs_training_mean"] == 0.0 and out["oos_r2_status"] == "defined"


def test_the_score_uses_the_training_mean_not_the_test_mean_as_its_reference():
    # test targets [1, 3] have mean 2; the training mean is 0. Test-mean reference would give 1 - 2/2 = 0 for pred [2, 2];
    # the documented training-mean reference gives 1 - ((1-2)^2 + (3-2)^2) / ((1-0)^2 + (3-0)^2) = 1 - 2/10 = 0.8
    assert tiny([1.0, 3.0], [2.0, 2.0], 0.0)["oos_r2_vs_training_mean"] == pytest.approx(0.8, abs=1e-15)


@pytest.mark.parametrize("label,pred", [("perfect predictions", None), ("imperfect predictions", np.linspace(-0.1, 0.1, 3 * N_T))])
def test_zero_denominator_is_undefined_and_does_not_crash(label, pred):
    y = 0.05  # every test target equals the training mean exactly
    p, test = frame(y, y if pred is None else pred)
    out = score(p, test, 0.05)
    assert math.isnan(out["oos_r2_vs_training_mean"])  # not 0, not 1, not a finite substitute
    assert out["oos_r2_status"].startswith("undefined") and "training mean" in out["oos_r2_status"]
    # the other metrics from the same evaluation are still produced and are correct
    assert out["n_test_samples"] == 3 * N_T
    expected_direction = 1.0 if pred is None else float(np.mean(np.asarray(pred) > 0))  # every target is positive
    assert out["direction_accuracy"] == pytest.approx(expected_direction, abs=1e-15)
    for key in ("rank_ic_all_dates", "rank_ic_nonoverlapping", "quintile_spread_all_dates", "quintile_spread_nonoverlapping"):
        assert set(out[key]) == {"n", "mean", "std", "std_error", "t_stat"}


def test_constant_targets_that_differ_from_the_training_mean_give_a_valid_score():
    p, test = frame(2.0, 1.5)  # baseline SSE = n * (2-1)^2 = n ; model SSE = n * 0.25 ; R2 = 1 - 0.25
    out = score(p, test, 1.0)
    assert out["oos_r2_vs_training_mean"] == pytest.approx(0.75, abs=1e-15) and out["oos_r2_status"] == "defined"


def test_a_small_positive_denominator_is_calculated_not_classified_as_zero():
    # the targets are 1e-6 above the training mean: baseline SSE = n * 1e-12 > 0 ; predictions halfway: model SSE = n * 0.25e-12
    p, test = frame(1.0 + 1e-6, 1.0 + 5e-7)
    out = score(p, test, 1.0)
    assert out["oos_r2_status"] == "defined" and out["oos_r2_vs_training_mean"] == pytest.approx(0.75, rel=1e-6)
    # far smaller still: denominators of 1e-300 per observation are positive and must be used
    p, test = frame(1e-150, 0.5e-150)
    out = score(p, test, 0.0)
    assert out["oos_r2_status"] == "defined" and out["oos_r2_vs_training_mean"] == pytest.approx(0.75, rel=1e-12)


def test_the_public_return_structure_is_preserved_plus_one_status_field():
    p, test = frame(np.linspace(-0.1, 0.2, 3 * N_T), np.linspace(-0.05, 0.1, 3 * N_T))
    out = score(p, test, 0.01)
    assert list(out) == ["n_test_samples", "oos_r2_vs_training_mean", "oos_r2_status", "direction_accuracy", "rank_ic_all_dates", "rank_ic_nonoverlapping", "quintile_spread_all_dates", "quintile_spread_nonoverlapping"]
    y = test["y"].to_numpy()
    expected = 1 - np.sum((y - p.to_numpy()) ** 2) / np.sum((y - 0.01) ** 2)
    assert out["oos_r2_vs_training_mean"] == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------------- JSON
def reject_nonstandard(constant):
    raise AssertionError(f"nonstandard JSON constant {constant!r} in the output")


def test_an_undefined_score_serialises_to_json_null_and_never_to_a_nonstandard_literal():
    p, test = frame(0.05, np.linspace(-0.1, 0.1, 3 * N_T))
    out = score(p, test, 0.05)
    text = serialize.dumps({"models": {"ensemble": out}}, indent=2)
    assert '"oos_r2_vs_training_mean": null' in text and "NaN" not in text and "Infinity" not in text
    back = json.loads(text, parse_constant=reject_nonstandard)  # strict: a NaN or Infinity literal would raise here
    assert back["models"]["ensemble"]["oos_r2_vs_training_mean"] is None
    assert back["models"]["ensemble"]["oos_r2_status"].startswith("undefined")
    assert back["models"]["ensemble"]["direction_accuracy"] == out["direction_accuracy"]


def test_the_serialiser_leaves_valid_numbers_unchanged_and_refuses_to_write_nan():
    obj = {"a": 0.1 + 0.2, "b": np.float64(1.5), "c": np.float32(0.25), "d": np.int64(3), "e": [float("nan"), float("inf"), -float("inf"), 2.0], "t": pd.Timestamp("2024-03-05")}
    back = json.loads(serialize.dumps(obj), parse_constant=reject_nonstandard)
    assert back == {"a": 0.30000000000000004, "b": 1.5, "c": 0.25, "d": 3, "e": [None, None, None, 2.0], "t": "2024-03-05"}
    with pytest.raises(ValueError):
        json.dumps({"x": float("nan")}, allow_nan=False)  # the safeguard dumps() relies on
