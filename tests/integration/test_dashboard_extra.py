"""Dashboard behaviour beyond the happy path: chart contents, files that are missing or malformed, undefined values,
exploratory labelling. The damaged-file tests run a copy of the app in a temporary tree (symlinks to the package and the
snapshot, a private copy of results/), so the checked-in results are never modified.
"""
import base64
import json
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from riskengine import backtest, config, data

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "dashboard" / "app.py")


def run_app(path=APP, timeout=240):
    st.cache_data.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AppTest.from_file(path, default_timeout=timeout).run()


@pytest.fixture()
def app(snapshot):
    at = run_app()
    yield at
    st.cache_data.clear()


def decode(a):
    """Plotly serialises numeric arrays either as lists or as {dtype, bdata}."""
    if isinstance(a, dict) and "bdata" in a:
        return np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"])).astype(float)
    return np.asarray(a)


def figure(at, key):
    el = next(e for e in at.get("plotly_chart") if e.key == key)
    return json.loads(el.proto.spec)


def title(fig):
    t = fig["layout"]["title"]
    return t["text"] if isinstance(t, dict) else t


# ----------------------------------------------------------------------------- tabs are all present; there is no upload feature
def test_the_dashboard_has_no_upload_control_so_invalid_uploads_cannot_occur(app):
    """The only input channels are widgets with fixed option lists, bounded sliders and two text filters. If an upload control
    is ever added, this test fails and the matrix row for 'invalid uploads' must be filled in."""
    assert "file_uploader" not in (ROOT / "dashboard" / "app.py").read_text()
    assert not app.get("file_uploader")


# ----------------------------------------------------------------------------- chart contents against independent values
def test_var_chart_forecast_lines_and_breach_markers_equal_an_independent_computation(app, snapshot):
    prices = snapshot[0]
    r = (prices["SPY"] / prices["SPY"].shift(1) - 1).to_numpy()
    fig = figure(app, "var_chart")
    names = [t["name"] for t in fig["data"]]
    assert title(fig) == "SPY: daily loss, 99% VaR forecast and breaches" and names[0].startswith("Daily loss")
    for m, (fi, bi) in zip(("historical", "normal"), ((1, 2), (3, 4))):
        f = backtest.rolling_forecasts(r, config.WINDOW, 0.99, m)
        got = decode(fig["data"][fi]["y"])
        assert np.allclose(got[~np.isnan(got)], f[~np.isnan(f)], atol=1e-12) and (~np.isnan(f)).sum() == (~np.isnan(got)).sum()
        breaches = (~np.isnan(f)) & (-r > f)
        assert len(decode(fig["data"][bi]["y"])) == int(breaches.sum()) and f"({int(breaches.sum())})" in fig["data"][bi]["name"]
        assert np.allclose(np.sort(decode(fig["data"][bi]["y"])), np.sort((-r)[breaches]), atol=1e-12)
    assert fig["layout"]["yaxis"]["tickformat"] == ".1%"  # losses as percentages


def test_var_chart_follows_the_controls(app):
    app.selectbox(key="var_ticker").set_value("AAPL")
    app.radio(key="var_conf").set_value("95%")
    app.multiselect(key="var_methods").set_value(["normal"])
    app.slider(key="var_window").set_value(120)
    app.run()
    assert not app.exception
    fig = figure(app, "var_chart")
    assert title(fig) == "AAPL: daily loss, 95% VaR forecast and breaches" and [t["name"] for t in fig["data"]][1].startswith("normal 95%")
    assert len(fig["data"]) == 3  # loss line plus one forecast line and one breach marker series
    first_valid = int(np.argmax(~np.isnan(decode(fig["data"][1]["y"]))))
    assert first_valid >= 120  # no forecast before a full 120-day window exists


def test_backtest_bar_chart_values_equal_the_stored_summary(app):
    summary = pd.DataFrame(json.loads((ROOT / "results" / "var_backtest_summary.json").read_text()))
    fig = figure(app, "bt_bar")
    bars = fig["data"][0]
    assert list(bars["x"]) == [f"{m} {int(c * 100)}%" for m, c in zip(summary["method"], summary["confidence"])]
    assert np.allclose(decode(bars["y"]), summary["pooled_breach_rate"].to_numpy(), atol=1e-15)
    assert np.allclose(decode(fig["data"][1]["y"]), summary["expected_rate"].to_numpy(), atol=1e-15)


def test_both_filters_on_the_backtest_tab_narrow_the_table_and_an_empty_selection_is_explained(app):
    app.multiselect(key="bt_methods").set_value([])
    app.run()
    assert not app.exception and any("Choose at least one method" in w.value for w in app.warning)
    app.multiselect(key="bt_methods").set_value(["normal"])
    app.radio(key="bt_conf").set_value("95%")
    app.text_input(key="bt_search").set_value("nonexistent_ticker")
    app.run()
    assert not app.exception
    assert any(m.value.startswith("**0 results**") for m in app.markdown)


def test_the_download_button_exists_and_the_page_data_behind_it_is_the_full_forecast_file(app):
    assert len(app.get("download_button")) == 1
    full = pd.read_csv(ROOT / "results" / "ml_latest_forecasts.csv")
    assert len(full) > 25  # the table on screen is limited to 25 rows by default while the download carries every row


# ----------------------------------------------------------------------------- exploratory labelling
def test_custom_portfolio_results_are_labelled_exploratory_before_and_after_building(app):
    assert any(w.value.startswith("Exploratory only.") for w in app.warning)
    app.button(key="pf_build").click()
    app.run()
    assert not app.exception and any(w.value.startswith("Exploratory only.") for w in app.warning)
    assert "pf_custom" in app.session_state


# ----------------------------------------------------------------------------- files that are missing, malformed, or have the wrong shape
def damaged_tree(tmp_path, mutate):
    t = tmp_path / "tree"
    (t / "dashboard").mkdir(parents=True)
    shutil.copy(ROOT / "dashboard" / "app.py", t / "dashboard" / "app.py")
    os.symlink(ROOT / "riskengine", t / "riskengine")
    os.symlink(ROOT / "data", t / "data")
    shutil.copytree(ROOT / "results", t / "results")
    mutate(t)
    return run_app(str(t / "dashboard" / "app.py"), timeout=120)


RESULT_FILES = ["run_info.json", "var_backtest_summary.json", "var_backtest_by_ticker.csv", "portfolios.json", "ml_summary.json", "ml_rank_ic_by_date.csv", "ml_latest_forecasts.csv", "ml_feature_importance.csv", "universe_membership_summary.json"]


@pytest.mark.parametrize("name", RESULT_FILES)
def test_every_missing_results_file_gives_an_actionable_message_not_a_traceback(snapshot, tmp_path, name):
    at = damaged_tree(tmp_path, lambda t: (t / "results" / name).unlink())
    assert not at.exception, [e.value[:120] for e in at.exception]
    assert at.error and name in at.error[0].value and any("scripts/run_pipeline.py" in m.value for m in at.markdown)
    assert not at.tabs


@pytest.mark.parametrize(
    "name,content",
    [
        ("run_info.json", "{truncated"),
        ("portfolios.json", ""),
        ("ml_summary.json", "[1, 2, 3]"),
        ("var_backtest_by_ticker.csv", ""),
        ("ml_latest_forecasts.csv", "ticker\nAAPL\n"),
        ("var_backtest_summary.json", '[{"unexpected": 1}]'),
        ("ml_rank_ic_by_date.csv", "date,ensemble\n2024-01-02,0.1\n"),
        ("universe_membership_summary.json", "{}"),
    ],
)
def test_malformed_or_wrongly_shaped_results_files_are_reported_by_name(snapshot, tmp_path, name, content):
    at = damaged_tree(tmp_path, lambda t: (t / "results" / name).write_text(content))
    assert not at.exception, [e.value[:120] for e in at.exception]
    assert at.error and "required file is missing or unreadable" in at.error[0].value and not at.tabs


# ----------------------------------------------------------------------------- undefined values display as 'undefined'
def edit_json(path_in_results, fn):
    def go(t):
        p = t / "results" / path_in_results
        d = json.loads(p.read_text())
        fn(d)
        p.write_text(json.dumps(d))

    return go


def test_an_undefined_r_squared_is_displayed_as_undefined_everywhere(snapshot, tmp_path):
    def null_r2(d):
        for m in d["models"].values():
            m["oos_r2_vs_training_mean"] = None
            m["oos_r2_status"] = "undefined: every test target equals the training average"

    at = damaged_tree(tmp_path, edit_json("ml_summary.json", null_r2))
    assert not at.exception
    assert [m.value for m in at.metric if m.label.startswith("Forecast model R squared")] == ["undefined"]
    text = "\n".join([str(e.value) for e in at.markdown if not e.value.lstrip().startswith("<style>")] + [str(e.value) for e in at.info])
    assert "out-of-sample R squared is undefined" in text and not re.search(r"\bnan\b|\bNone\b", text)
    table = next(d.value for d in at.dataframe if "R squared vs training mean" in d.value.columns)
    assert set(table["R squared vs training mean"].iloc[:3]) == {"undefined"}


def test_a_positive_r_squared_shows_the_success_message_with_the_caveat_and_the_matching_overview_sentence(snapshot, tmp_path):
    def positive(d):
        d["models"]["ensemble"]["oos_r2_vs_training_mean"] = 0.05

    at = damaged_tree(tmp_path, edit_json("ml_summary.json", positive))
    assert not at.exception
    assert any("better than always predicting the training average. See the rank IC statistics" in s.value for s in at.success)
    assert any("out-of-sample R squared of 5.00% is better than always predicting" in m.value for m in at.markdown)
    assert [m.value for m in at.metric if m.label.startswith("Forecast model R squared")] == ["5.00%"]


def test_an_undefined_rank_ic_and_sharpe_are_displayed_as_undefined_not_as_a_crash(snapshot, tmp_path):
    def null_ic(d):
        for m in d["models"].values():
            m["rank_ic_nonoverlapping"] = {"mean": None, "t_stat": None, "n": 0}

    at = damaged_tree(tmp_path, edit_json("ml_summary.json", null_ic))
    assert not at.exception, [e.value[:150] for e in at.exception]
    assert any("rank IC undefined with t statistic undefined over 0 independent periods" in m.value for m in at.markdown)

    def null_sharpe(d):
        for e in d["presets"].values():
            if e["status"] == "constructed":
                e["out_of_sample_test"]["sharpe"] = None
                e["in_sample_train"]["sharpe"] = None
                e["equal_weight_same_holdings_test"]["sharpe"] = None
        d["benchmark_spy_test"]["sharpe"] = None

    (tmp_path / "second").mkdir()
    at = damaged_tree(tmp_path / "second", edit_json("portfolios.json", null_sharpe))
    assert not at.exception, [e.value[:150] for e in at.exception]
    stats = next(d.value for d in at.dataframe if "Test period (out-of-sample)" in d.value.columns)
    assert set(stats.loc["Sharpe ratio"]) == {"undefined"}
    assert any("test Sharpe undefined against undefined" in m.value for m in at.markdown)


def test_percentage_and_number_helpers_use_the_word_undefined_for_null_and_nan():
    ns = {}
    src = (ROOT / "dashboard" / "app.py").read_text()
    start = src.index("def pct(")
    end = src.index("def line_chart_layout")
    exec("import numpy as np\n" + src[start:end], ns)
    assert ns["pct"](None) == ns["pct"](float("nan")) == ns["num"](None) == ns["num"](float("nan")) == ns["r2txt"](None) == "undefined"
    assert ns["pct"](0.01234) == "1.23%" and ns["pct"](-0.5, 0) == "-50%" and ns["num"](1.23456, ".2f") == "1.23" and ns["r2txt"](-0.0123) == "-1.23%"
