"""EXPERIMENT A (explicitly synthetic): independent observations from a fully known distribution.

Purpose: check the Kupiec implementation where theory is exact. The true VaR is known, not estimated, so each day's
breach is an independent Bernoulli(1 - confidence) trial, and the exact probability that the 5% test rejects can be
computed by enumerating the binomial distribution with a separately written likelihood ratio.

What this does NOT show: that rolling-window estimated forecasts have this rejection rate. Those forecasts share data
from day to day and their VaR is estimated, which is a different experiment (EXPERIMENT B in test_statistical.py, where
the rejection rate was observed to be lower than nominal). Passing A is evidence about the Kupiec function, not about B.
"""
import math

import numpy as np
import pytest
from scipy.stats import binom, chi2, norm

from riskengine import backtest

pytestmark = pytest.mark.statistical

N_SERIES, N_OBS, MU, SD = 2000, 1200, 0.0004, 0.012
SEED_A = 20261101  # fixed in advance; never changed to obtain a pass
CRITICAL = chi2.ppf(0.95, df=1)


def reference_lr(n, x, p):
    """Likelihood ratio written directly from the binomial likelihoods (no use of the function under test)."""
    ll0 = (n - x) * math.log(1 - p) + (x * math.log(p) if x > 0 else 0.0)
    ll1 = ((n - x) * math.log(1 - x / n) if x < n else 0.0) + (x * math.log(x / n) if x > 0 else 0.0)
    return max(0.0, -2.0 * (ll0 - ll1))


def exact_size(n, p):
    """P(reject at 5%) when the breach count is exactly Binomial(n, p): the sum of pmf over counts whose LR exceeds the cutoff."""
    return float(sum(binom.pmf(x, n, p) for x in range(n + 1) if reference_lr(n, x, p) > CRITICAL))


@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_A1_kupiec_statistic_and_pvalue_equal_the_reference_for_every_possible_breach_count(conf):
    p = 1 - conf
    worst_lr = worst_p = 0.0
    for x in range(N_OBS + 1):
        lr, pv = backtest.kupiec_pof(N_OBS, x, p)
        worst_lr = max(worst_lr, abs(lr - reference_lr(N_OBS, x, p)))
        worst_p = max(worst_p, abs(pv - chi2.sf(reference_lr(N_OBS, x, p), 1)))
        assert (pv < 0.05) == (reference_lr(N_OBS, x, p) > CRITICAL)
    # chi2.sf has infinite slope at 0 (sf(e) ~ 1 - 0.8*sqrt(e)), so an LR rounding difference of 5e-13 near a breach rate equal to p moves the
    # p-value by about 1e-7; the p-value tolerance reflects that sensitivity, while the accept/reject decision above is compared exactly
    assert worst_lr < 1e-9 and worst_p < 1e-6


@pytest.mark.parametrize("conf", [0.95, 0.99])
def test_A2_with_independent_trials_the_rejection_rate_matches_the_exact_theoretical_size(conf, record_property):
    rng = np.random.default_rng(SEED_A)
    true_var = norm.ppf(conf) * SD - MU  # the known loss quantile: nothing is estimated
    losses = -rng.normal(MU, SD, (N_SERIES, N_OBS))
    breaches = (losses > true_var).sum(axis=1)
    rejected = np.array([backtest.kupiec_pof(N_OBS, int(x), 1 - conf)[1] < 0.05 for x in breaches])
    size = exact_size(N_OBS, 1 - conf)
    se = math.sqrt(size * (1 - size) / N_SERIES)
    record_property("observed_rejection_rate", float(rejected.mean()))
    record_property("exact_theoretical_size", size)
    record_property("difference_in_standard_errors", float((rejected.mean() - size) / se))
    assert abs(rejected.mean() - size) < 4 * se
    # the breach rate itself is unbiased and its variance is the independent-Bernoulli variance
    assert abs(breaches.mean() / N_OBS - (1 - conf)) < 4 * math.sqrt((1 - conf) * conf / (N_OBS * N_SERIES))
    assert breaches.var(ddof=1) == pytest.approx(N_OBS * (1 - conf) * conf, rel=0.15)
