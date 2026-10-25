"""Regression tests: a rank IC that cannot be defined is reported as undefined, silently and honestly.

Undefined means: fewer than 10 paired finite observations, or constant predictions, or constant targets on that date.
The date is omitted (as dates with too few stocks always were), the reason is recorded, no value is invented,
no warning is emitted, and every defined, non-constant date is computed exactly as before.
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from riskengine import ml, serialize

TICKERS = [f"T{i}" for i in range(12)]
DATES = pd.bdate_range("2024-01-01", periods=5)
BASE_P = np.array([3.0, 1.0, 4.0, 1.5, 9.0, 2.6, 5.3, 5.8, 9.7, 9.3, 2.3, 8.4])
BASE_Y = np.array([2.0, 7.0, 1.0, 8.0, 2.8, 1.8, 2.5, 3.1, 4.1, 5.9, 0.2, 6.5])


def panel(rows):
    """rows: list of (pred vector, target vector) for consecutive dates; returns (pred, y) as MultiIndex series."""
    idx = pd.MultiIndex.from_product([DATES[: len(rows)], TICKERS], names=["date", "ticker"])
    p = pd.Series(np.concatenate([np.asarray(a, dtype=float) for a, _ in rows]), index=idx)
    y = pd.Series(np.concatenate([np.asarray(b, dtype=float) for _, b in rows]), index=idx)
    return p, y


def quiet(fn, *args):
    """Call fn with every warning turned into an error, so a RuntimeWarning fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args)


def test_a_constant_prediction_date_is_undefined_and_the_others_are_unchanged():
    p, y = panel([(BASE_P, BASE_Y), (np.full(12, 0.3), BASE_Y), (BASE_P[::-1], BASE_Y)])
    ic = quiet(ml.rank_ic_by_date, p, y)
    assert list(ic.index) == [DATES[0], DATES[2]]  # the constant date is omitted, not set to 0
    assert ic[DATES[0]] == pytest.approx(stats.spearmanr(BASE_P, BASE_Y)[0], abs=1e-12)
    assert ic[DATES[2]] == pytest.approx(stats.spearmanr(BASE_P[::-1], BASE_Y)[0], abs=1e-12)
    assert ic.attrs["undefined_dates"] == {DATES[1]: "constant predictions across 12 stocks"}
    assert (ic != 0.0).all() and ic.dtype == float


def test_a_constant_target_date_is_undefined():
    p, y = panel([(BASE_P, np.full(12, 0.05)), (BASE_P, BASE_Y)])
    ic = quiet(ml.rank_ic_by_date, p, y)
    assert list(ic.index) == [DATES[1]]
    assert ic.attrs["undefined_dates"] == {DATES[0]: "constant targets across 12 stocks"}


def test_constant_predictions_and_constant_targets_are_both_reported():
    p, y = panel([(np.full(12, 1.0), np.full(12, 2.0))])
    ic = quiet(ml.rank_ic_by_date, p, y)
    assert ic.empty and ic.dtype == float
    assert ic.attrs["undefined_dates"][DATES[0]] == "constant predictions and constant targets across 12 stocks"


def test_every_date_constant_gives_an_empty_result_with_every_reason_recorded():
    p, y = panel([(np.full(12, 0.0), BASE_Y)] * 3)
    ic = quiet(ml.rank_ic_by_date, p, y)
    assert ic.empty and set(ic.attrs["undefined_dates"]) == set(DATES[:3])


def test_fewer_than_ten_paired_finite_observations_is_undefined_for_missing_and_infinite_values():
    p = BASE_P.copy()
    y = BASE_Y.copy()
    p[:2] = np.nan  # 10 pairs remain: defined
    ok = quiet(ml.rank_ic_by_date, *panel([(p, y)]))
    assert list(ok.index) == [DATES[0]] and ok.iloc[0] == pytest.approx(stats.spearmanr(BASE_P[2:], BASE_Y[2:])[0], abs=1e-12)
    p[2] = np.inf  # now 9 finite pairs
    y[3] = -np.inf  # and 8
    bad = quiet(ml.rank_ic_by_date, *panel([(p, y)]))
    assert bad.empty and "fewer than 10 paired finite observations (8)" in bad.attrs["undefined_dates"][DATES[0]]


def test_nonconstant_inputs_with_ties_are_still_defined_and_unchanged():
    y_ties = np.array([1.0] * 6 + [2.0] * 6)
    ic = quiet(ml.rank_ic_by_date, *panel([(BASE_P, y_ties)]))
    assert ic.iloc[0] == pytest.approx(stats.spearmanr(BASE_P, y_ties)[0], abs=1e-12) and not ic.attrs["undefined_dates"]
    perfect = quiet(ml.rank_ic_by_date, *panel([(BASE_P, BASE_P * 2), (BASE_P, -BASE_P)]))
    assert list(perfect.round(12)) == [1.0, -1.0]


def test_no_warning_is_emitted_for_any_constant_case_and_nothing_is_suppressed_globally():
    p, y = panel([(np.full(12, 1.0), BASE_Y), (BASE_P, np.full(12, 3.0)), (BASE_P, BASE_Y)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ml.rank_ic_by_date(p, y)
    assert [str(w.message) for w in caught] == []
    # the global filter state was not altered by the call: a deliberate warning still gets through afterwards
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("still visible", RuntimeWarning)
    assert len(caught) == 1


def test_score_with_constant_targets_returns_undefined_statistics_without_warnings():
    idx = pd.MultiIndex.from_product([DATES[:3], TICKERS], names=["date", "ticker"])
    pred = pd.Series(np.tile(np.linspace(-0.1, 0.1, 12), 3), index=idx)
    test = pd.DataFrame({"y": pd.Series(0.05, index=idx)})
    out, ic = quiet(ml.score, pred, test, 0.05, pd.bdate_range("2024-01-01", periods=80))
    assert ic.empty and math.isnan(out["oos_r2_vs_training_mean"]) and out["oos_r2_status"].startswith("undefined")
    for key in ("rank_ic_all_dates", "rank_ic_nonoverlapping"):
        assert out[key]["n"] == 0 and all(math.isnan(out[key][k]) for k in ("mean", "std", "std_error", "t_stat"))
    text = serialize.dumps(out)
    assert "NaN" not in text and '"mean": null' in text  # the undefined statistics are JSON null, not zero


def test_score_with_ordinary_data_is_unchanged_by_the_repair():
    idx = pd.MultiIndex.from_product([DATES[:3], TICKERS], names=["date", "ticker"])
    p = pd.Series(np.concatenate([BASE_P, BASE_P[::-1], np.roll(BASE_P, 3)]), index=idx)
    y = pd.Series(np.concatenate([BASE_Y, BASE_Y, BASE_Y[::-1]]), index=idx)
    out, ic = quiet(ml.score, p, pd.DataFrame({"y": y}), 0.01, pd.bdate_range("2024-01-01", periods=80))
    expected = [stats.spearmanr(p.loc[d], y.loc[d])[0] for d in DATES[:3]]
    assert list(ic) == pytest.approx(expected, abs=1e-12) and out["rank_ic_all_dates"]["n"] == 3
    assert out["rank_ic_all_dates"]["mean"] == pytest.approx(np.mean(expected), abs=1e-12)
