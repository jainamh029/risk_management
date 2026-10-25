"""B. VaR and ES against independent references (scipy distributions, direct algebra, hand examples)."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from riskengine import var
from tests.unit.fixtures import wavy_returns

CONFS = (0.95, 0.99)


@pytest.mark.parametrize("c", CONFS)
def test_B6_normal_method_matches_the_closed_form_for_returns_minus1_0_plus1_percent(c):
    r = [-0.01, 0.00, 0.01]  # independently: mean 0; sample variance (0.0001 + 0 + 0.0001) / 2, so sd 0.01
    v, es = var.normal_var_es(r, c)
    z = norm.ppf(c)
    assert v == pytest.approx(0.01 * z, rel=1e-12, abs=1e-15)
    assert es == pytest.approx(0.01 * norm.pdf(z) / (1 - c), rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("c", CONFS)
def test_B7_a_constant_shift_in_returns_moves_loss_var_and_es_by_exactly_the_shift(c):
    base = [-0.01, 0.00, 0.01]
    shifted = [x + 0.001 for x in base]
    v0, e0 = var.normal_var_es(base, c)
    v1, e1 = var.normal_var_es(shifted, c)
    assert v0 - v1 == pytest.approx(0.001, abs=1e-15)
    assert e0 - e1 == pytest.approx(0.001, abs=1e-15)
    # large shift: loss VaR becomes negative and is NOT clipped to zero
    big = [x + 0.05 for x in base]
    vb, eb = var.normal_var_es(big, c)
    assert vb == pytest.approx(0.01 * norm.ppf(c) - 0.05, abs=1e-14) and vb < 0
    assert eb == pytest.approx(0.01 * norm.pdf(norm.ppf(c)) / (1 - c) - 0.05, abs=1e-14)
    vh, eh = var.historical_var_es(big, c)
    assert vh < 0 and eh < 0  # historical method also reports a negative loss quantile rather than clipping


def test_B8_historical_method_hand_example():
    r = [-0.04, -0.02, 0.00, 0.01, 0.03]
    # losses sorted: [-0.03, -0.01, 0.00, 0.02, 0.04]
    v, es = var.historical_var_es(r, 0.75)  # position 0.75 * 4 = 3 -> 0.02 ; ES = mean(0.02, 0.04)
    assert v == pytest.approx(0.02, abs=1e-15) and es == pytest.approx(0.03, abs=1e-15)
    v, es = var.historical_var_es(r, 0.95)  # position 3.8 -> 0.02 + 0.8 * (0.04 - 0.02) = 0.036 ; ES = 0.04
    assert v == pytest.approx(0.036, abs=1e-15) and es == pytest.approx(0.04, abs=1e-15)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("c", CONFS)
@pytest.mark.parametrize("k", [0.5, 2.0, 37.0])
def test_B9_positive_scaling_scales_var_and_es(method, c, k):
    r = wavy_returns(40)
    v, e = var.var_es(r, c, method)
    vk, ek = var.var_es(r * k, c, method)
    assert vk == pytest.approx(k * v, rel=1e-12) and ek == pytest.approx(k * e, rel=1e-12)


@pytest.mark.parametrize("method", var.METHODS)
def test_B10_es_is_at_least_var_and_both_are_ordered_by_confidence(method):
    for phase in (0.0, 1.0, 2.5):
        r = wavy_returns(60, phase=phase)
        prev_v = prev_e = -np.inf
        for c in (0.50, 0.75, 0.90, 0.95, 0.975, 0.99):
            v, e = var.var_es(r, c, method)
            assert e >= v - 1e-15
            assert v >= prev_v - 1e-15 and e >= prev_e - 1e-15
            prev_v, prev_e = v, e


def test_B10b_constant_returns_have_var_equal_to_es_equal_to_minus_the_constant():
    for method in var.METHODS:
        for c in CONFS:
            v, e = var.var_es([0.002] * 20, c, method)
            assert v == pytest.approx(-0.002, abs=1e-15) and e == pytest.approx(-0.002, abs=1e-15)


# ---- B11: invalid inputs must raise rather than return NaN, inf or a silently manufactured number ----
@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_B11a_invalid_confidence_is_rejected(method, bad):
    with pytest.raises(ValueError):
        var.var_es(wavy_returns(30), bad, method)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_B11b_missing_or_nonfinite_returns_are_rejected(method, bad):
    r = wavy_returns(30)
    r[7] = bad
    with pytest.raises(ValueError):
        var.var_es(r, 0.95, method)


@pytest.mark.parametrize("method", var.METHODS)
@pytest.mark.parametrize("n", [0, 1])
def test_B11c_insufficient_data_is_rejected(method, n):
    with pytest.raises(ValueError):
        var.var_es(wavy_returns(30)[:n], 0.95, method)


def test_B11d_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        var.var_es(wavy_returns(30), 0.95, "monte_carlo")
