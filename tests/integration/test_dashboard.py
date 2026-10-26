"""Drives the real dashboard headlessly (Streamlit's AppTest) on the real snapshot and results."""
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "dashboard" / "app.py")


@pytest.fixture()
def app(snapshot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = AppTest.from_file(APP, default_timeout=240)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at, label_prefix):
    return [m for m in at.metric if m.label.startswith(label_prefix)]


def rendered_text(at):
    parts = [e.value for e in at.markdown] + [e.value for e in at.caption] + [e.value for e in at.warning] + [e.value for e in at.info] + [e.value for e in at.success]
    parts += [m.label + " " + str(m.value) + " " + str(m.delta) for m in at.metric]
    return "\n".join(map(str, parts))


def test_loads_without_errors_and_has_every_tab(app):
    assert [t.label for t in app.tabs] == ["Overview", "Stock explorer", "VaR explorer", "Backtest comparison", "Portfolio experiment", "Forecasting model", "Glossary and limits"]
    assert not app.error
    assert len(app.get("plotly_chart")) >= 5


def test_every_chart_tab_has_a_data_table_behind_it(app):
    labels = [e.label for e in app.expander]
    assert sum(l.startswith("Data table") for l in labels) >= 3  # explorer, portfolio, forecasting
    # the backtest tab shows its summary and per-ticker tables directly
    cols = [set(d.value.columns) for d in app.dataframe]
    assert any("Pooled breach rate" in c for c in cols) and any("kupiec_p" in c for c in cols)
    # the forecasting data table holds the same rank-IC series that is plotted
    ic = next(d.value for d in app.dataframe if "ensemble" in d.value.columns and len(d.value) > 100)
    stored = pd.read_csv(ROOT / "results" / "ml_rank_ic_by_date.csv")
    assert len(ic) == len(stored) and np.allclose(ic["ensemble"].to_numpy(), stored["ensemble"].to_numpy())


def test_overview_numbers_equal_the_results_files(app):
    s = json.loads((ROOT / "results" / "var_backtest_summary.json").read_text())
    h99 = next(r for r in s if r["method"] == "historical" and r["confidence"] == 0.99)
    ml = json.loads((ROOT / "results" / "ml_summary.json").read_text())["models"]["ensemble"]
    assert metric(app, "Historical 99%")[0].value == f"{100 * h99['pooled_breach_rate']:.2f}%"
    assert metric(app, "Forecast model R squared")[0].value == f"{100 * ml['oos_r2_vs_training_mean']:.2f}%"
    assert metric(app, "Tickers backtested")[0].value == str(h99["n_tickers"])


def independent_breaches(r, window, conf, method):
    from statistics import NormalDist

    n = x = 0
    for t in range(window, len(r)):
        w = r[t - window : t]
        if np.isnan(w).any() or np.isnan(r[t]):
            continue
        if method == "historical":
            s = np.sort(-w)
            pos = conf * (len(s) - 1)
            lo, hi = int(math.floor(pos)), int(math.ceil(pos))
            f = s[lo] + (pos - lo) * (s[hi] - s[lo])
        else:
            mu = w.mean()
            sd = w.std(ddof=1)
            f = -(mu - NormalDist().inv_cdf(conf) * sd)
        n += 1
        x += int(-r[t] > f)
    return n, x


def test_var_explorer_recomputes_when_controls_change(app, returns):
    app.selectbox(key="var_ticker").set_value("AAPL")
    app.radio(key="var_conf").set_value("95%")
    app.slider(key="var_window").set_value(120)
    app.multiselect(key="var_methods").set_value(["normal"])
    app.run()
    assert not app.exception
    breaches = [m for m in app.metric if m.label == "Breaches"]
    assert len(breaches) == 1
    n, x = independent_breaches(returns["AAPL"].values, 120, 0.95, "normal")
    assert breaches[0].value == f"{x} of {n}"
    rate = [m for m in app.metric if m.label == "Breach rate"][0]
    assert rate.value == f"{100 * x / n:.2f}%"


def test_var_explorer_both_methods_and_dollar_var(app, returns):
    app.number_input(key="var_position").set_value(250_000)
    app.run()
    vt = [m for m in app.metric if m.label.endswith("VaR today")]
    assert {m.label for m in vt} == {"historical: VaR today", "normal: VaR today"}
    r = returns["SPY"].iloc[-250:].values
    losses = np.sort(-r)
    pos = 0.99 * (len(losses) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    v = losses[lo] + (pos - lo) * (losses[hi] - losses[lo])
    hist = [m for m in vt if m.label.startswith("historical")][0]
    assert hist.value == f"{100 * v:.2f}%"
    assert f"${v * 250_000:,.0f}" in hist.delta


def test_var_explorer_warns_instead_of_failing(app):
    app.multiselect(key="var_methods").set_value([])
    app.run()
    assert not app.exception and any("at least one method" in w.value for w in app.warning)
    app.multiselect(key="var_methods").set_value(["historical"])
    app.slider(key="var_window").set_value(60)
    app.run()
    assert not app.exception and any("noisy" in w.value for w in app.warning)
    app.selectbox(key="var_ticker").set_value("AVB")  # one valid price in the real data
    app.run()
    assert not app.exception and any("too few" in w.value for w in app.warning)


def test_backtest_filters_change_what_is_shown(app):
    app.radio(key="bt_conf").set_value("99%")
    app.multiselect(key="bt_methods").set_value(["normal"])
    app.run()
    assert not app.exception
    tables = [d.value for d in app.dataframe]
    summary = next(t for t in tables if "Pooled breach rate" in t.columns)
    assert list(summary["Method"]) == ["normal 99%"]
    app.text_input(key="bt_search").set_value("AAPL, msft")
    app.run()
    by_ticker = next(d.value for d in app.dataframe if "kupiec_p" in d.value.columns)
    assert set(by_ticker["ticker"]) == {"AAPL", "MSFT"} and set(by_ticker["method"]) == {"normal"} and set(by_ticker["confidence"]) == {0.99}
    app.checkbox(key="bt_rejected").set_value(True)
    app.run()
    by_ticker = next(d.value for d in app.dataframe if "kupiec_p" in d.value.columns)
    assert (by_ticker["kupiec_p"] < 0.05).all()


def test_portfolio_preset_switch_and_not_constructed_message(app):
    app.radio(key="pf_preset").set_value("conservative")
    app.run()
    assert not app.exception and any("not constructed" in w.value for w in app.warning)
    app.radio(key="pf_preset").set_value("balanced")
    app.run()
    pf = json.loads((ROOT / "results" / "portfolios.json").read_text())["presets"]["balanced"]
    stats = next(d.value for d in app.dataframe if "Test period (out-of-sample)" in d.value.columns)
    assert stats.loc["Total return", "Test period (out-of-sample)"] == f"{100 * pf['out_of_sample_test']['total_return']:.1f}%"


def test_custom_screen_with_the_aggressive_settings_reproduces_the_stored_portfolio(app):
    """Defaults of the custom screen equal the aggressive preset, so the live rebuild must match the stored result."""
    assert any("tuning on the test data" in w.value for w in app.warning)
    app.button(key="pf_build").click()
    app.run()
    assert not app.exception
    stored = json.loads((ROOT / "results" / "portfolios.json").read_text())["presets"]["aggressive"]
    custom = app.session_state["pf_custom"]
    assert custom["status"] == "constructed" and custom["tickers"] == stored["tickers"]
    assert custom["out_of_sample_test"]["total_return"] == pytest.approx(stored["out_of_sample_test"]["total_return"], rel=1e-9)
    app.slider(key="pf_sharpe").set_value(2.0)
    app.button(key="pf_build").click()
    app.run()
    assert not app.exception
    assert app.session_state["pf_custom"]["status"] == "not_constructed"  # nothing passes, and the screen is not relaxed


def test_forecast_table_filters_and_orders(app):
    full = pd.read_csv(ROOT / "results" / "ml_latest_forecasts.csv")
    app.slider(key="ml_topn").set_value(10)
    app.run()
    t = next(d.value for d in app.dataframe if "predicted_63d_return" in d.value.columns)
    assert len(t) == 10 and t["predicted_63d_return"].is_monotonic_decreasing
    assert t.iloc[0]["ticker"] == full.sort_values("predicted_63d_return").iloc[-1]["ticker"]
    app.radio(key="ml_order").set_value("lowest forecast first")
    app.run()
    t = next(d.value for d in app.dataframe if "predicted_63d_return" in d.value.columns)
    assert t["predicted_63d_return"].is_monotonic_increasing and t.iloc[0]["ticker"] == full.sort_values("predicted_63d_return").iloc[0]["ticker"]
    app.text_input(key="ml_search").set_value("aapl")
    app.run()
    t = next(d.value for d in app.dataframe if "predicted_63d_return" in d.value.columns)
    assert list(t["ticker"]) == ["AAPL"]
    assert any("not recommendations" in w.value for w in app.warning)


def test_every_control_option_runs_without_an_exception(app):
    for key, opts in (("var_conf", ["95%", "99%"]), ("bt_conf", ["95%", "99%", "both"])):
        for o in opts:
            app.radio(key=key).set_value(o)
            app.run()
            assert not app.exception, (key, o)
    for m in (["historical"], ["normal"], ["historical", "normal"]):
        app.multiselect(key="ml_models").set_value(["ensemble"])
        app.multiselect(key="var_methods").set_value(m)
        app.run()
        assert not app.exception, m
    for m in (["random_forest"], ["gradient_boosting", "ensemble"], ["random_forest", "gradient_boosting", "ensemble"]):
        app.multiselect(key="ml_models").set_value(m)
        app.run()
        assert not app.exception, m


def test_dashboard_text_makes_no_unsupported_claims(app):
    text = (rendered_text(app) + (ROOT / "dashboard" / "app.py").read_text()).lower()
    for bad in ("strong buy", "sell signal", "guaranteed", "production", "institutional", "robust", "state-of-the-art", "live data", "real-time"):
        assert bad not in text, bad
    emoji = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
    assert not emoji.search(text)
    assert "not investment advice" in text


def test_missing_snapshot_shows_instructions_instead_of_a_traceback(monkeypatch):
    """The loader raises the same FileNotFoundError the real one raises when data/prices.parquet is absent."""
    import streamlit as st

    from riskengine import data

    def missing(*args, **kwargs):
        raise FileNotFoundError("No price snapshot at data/prices.parquet. Create it with: python scripts/fetch_data.py")

    monkeypatch.setattr(data, "load_snapshot", missing)
    st.cache_data.clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert at.error and any("scripts/fetch_data.py" in m.value for m in at.markdown)
        assert not at.tabs  # nothing else is rendered
    finally:
        st.cache_data.clear()
