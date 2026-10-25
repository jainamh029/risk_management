import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from riskengine import config, var


def last(returns, t, n=250):
    return returns[t].iloc[-n:].values


def test_normal_var_satisfies_its_defining_equation(returns):
    """VaR is the loss L with P(R <= -L) = 1 - conf under Normal(mu, sigma). Checked without the closed form."""
    for t in ("SPY", "AAPL", "XOM"):
        r = last(returns, t)
        mu, sd = r.mean(), r.std(ddof=1)
        for conf in (0.95, 0.99):
            v, _ = var.normal_var_es(r, conf)
            assert norm.cdf(-v, loc=mu, scale=sd) == pytest.approx(1 - conf, abs=1e-12)
            assert v > 0


def test_normal_es_matches_numerical_integration(returns):
    for t in ("SPY", "JPM"):
        r = last(returns, t)
        mu, sd = r.mean(), r.std(ddof=1)
        for conf in (0.95, 0.99):
            v, es = var.normal_var_es(r, conf)
            tail, _ = integrate.quad(lambda x: (-x) * norm.pdf(x, mu, sd), -np.inf, -v)
            assert es == pytest.approx(tail / (1 - conf), rel=1e-6)


def test_the_original_sign_error_is_real_on_real_data(returns):
    """The original code used -(mu + z*sigma). On real SPY returns that is negative."""
    r = last(returns, "SPY")
    mu, sd, z = r.mean(), r.std(ddof=1), norm.ppf(0.99)
    original = -(mu + z * sd)
    fixed, _ = var.normal_var_es(r, 0.99)
    assert original < 0 < fixed
    assert fixed == pytest.approx(z * sd - mu, rel=1e-12)


def test_historical_var_is_the_empirical_quantile(returns):
    for t in ("SPY", "AAPL", "MSFT"):
        r = last(returns, t)
        losses = np.sort(-r)
        for conf in (0.95, 0.99):
            v, es = var.historical_var_es(r, conf)
            pos = conf * (len(losses) - 1)  # linear interpolation between order statistics
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            expected = losses[lo] + (pos - lo) * (losses[hi] - losses[lo])
            assert v == pytest.approx(expected, rel=0, abs=1e-15)
            n = len(losses)
            # with linear interpolation the exceedance share is within one observation of the target
            assert np.mean(losses > v) <= 1 - conf + 1 / n + 1e-12
            assert np.mean(losses >= v) >= 1 - conf - 1 / n
            assert es == pytest.approx(losses[losses >= v].mean(), abs=1e-15)


def test_var_and_es_orderings_hold_for_every_ticker(returns):
    tbl = var.latest_var_table(returns, config.WINDOW, config.CONFS)
    assert len(tbl) > 400
    for m in var.METHODS:
        assert (tbl[f"{m}_es_95"] >= tbl[f"{m}_var_95"] - 1e-12).all()
        assert (tbl[f"{m}_es_99"] >= tbl[f"{m}_var_99"] - 1e-12).all()
        assert (tbl[f"{m}_var_99"] >= tbl[f"{m}_var_95"]).all()
        assert (tbl[f"{m}_es_99"] >= tbl[f"{m}_es_95"]).all()


def test_one_day_var_is_a_plausible_fraction_not_a_scaled_number(returns):
    """The original code multiplied by sqrt(252) and printed VaR above 100 percent."""
    tbl = var.latest_var_table(returns, config.WINDOW, config.CONFS)
    cols = [c for c in tbl.columns if "_var_" in c or "_es_" in c]
    assert tbl[cols].max().max() < 0.5
    spy = tbl.set_index("ticker").loc["SPY"]
    assert 0.003 < spy["historical_var_99"] < 0.10


def test_incomplete_windows_are_excluded_not_filled(returns, meta):
    tbl = var.latest_var_table(returns, config.WINDOW, config.CONFS)
    excluded = tbl.attrs["excluded_incomplete_window"]
    for t in meta["fewer_than_250_valid_prices"]:  # EA, WBD, AVB, EQR stopped trading
        assert t not in set(tbl["ticker"]) and t in excluded


def test_nan_input_is_rejected(returns):
    r = last(returns, "SPY").copy()
    r[3] = np.nan
    with pytest.raises(ValueError):
        var.historical_var_es(r, 0.99)
    with pytest.raises(ValueError):
        var.normal_var_es(r, 0.99)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        var.var_es([0.01, -0.02, 0.0], 0.95, "monte_carlo")
