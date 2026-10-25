"""An undefined R squared (null in ml_summary.json) must render as the word "undefined" in the README and the dashboard.

The results files are copied to a temporary directory and edited there; the checked-in results are not touched.
"""
import importlib.util
import json
import shutil
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
REASON = "undefined: every test target equals the training mean, so the reference sum of squares is zero"


@pytest.fixture()
def undefined_results(tmp_path):
    shutil.copytree(ROOT / "results", tmp_path / "results")
    p = tmp_path / "results" / "ml_summary.json"
    ml = json.loads(p.read_text())
    for m in ml["models"].values():
        m["oos_r2_vs_training_mean"] = None
        m["oos_r2_status"] = REASON
    p.write_text(json.dumps(ml, indent=2))
    return tmp_path


def test_readme_tables_show_undefined_instead_of_a_number_or_an_exception(undefined_results):
    spec = importlib.util.spec_from_file_location("make_readme_tables_u", ROOT / "scripts" / "make_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = undefined_results
    block = mod.build()
    rows = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells[0] in ("random forest", "gradient boosting", "ensemble", "baseline: training-mean forecast"):
            rows[cells[0]] = cells
    assert {k: v[1] for k, v in rows.items() if k != "baseline: training-mean forecast"} == {"random forest": "undefined", "gradient boosting": "undefined", "ensemble": "undefined"}
    assert rows["baseline: training-mean forecast"][1] == "0.00%"  # the reference forecast scores zero against itself by definition
    for cells in rows.values():
        assert "nan" not in " ".join(cells).lower() and "None" not in " ".join(cells)
        assert len(cells) == 6  # the table structure is intact


def run_dashboard(root: Path):
    import streamlit as st

    shutil.copytree(ROOT / "dashboard", root / "dashboard")
    st.cache_data.clear()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return AppTest.from_file(str(root / "dashboard" / "app.py"), default_timeout=240).run()
    finally:
        st.cache_data.clear()


def test_dashboard_renders_undefined_without_an_exception_or_a_misleading_number(snapshot, undefined_results):
    at = run_dashboard(undefined_results)
    assert not at.exception, [e.value for e in at.exception]
    assert [m for m in at.metric if m.label.startswith("Forecast model R squared")][0].value == "undefined"
    text = "\n".join(str(e.value) for e in list(at.markdown) + list(at.info) + list(at.warning) + list(at.success))
    assert "out-of-sample R squared is undefined" in text  # overview finding
    assert any("R squared is undefined on the test period" in i.value for i in at.info)  # forecasting tab
    assert not any("worse than always predicting" in w.value for w in at.warning)  # no verdict is invented for an undefined score
    assert not any("better than always predicting" in s.value for s in at.success)
    table = next(d.value for d in at.dataframe if "R squared vs training mean" in d.value.columns)
    assert list(table["R squared vs training mean"].iloc[:3]) == ["undefined"] * 3
    assert table["R squared vs training mean"].iloc[3] == "0.00%"
    assert table["Direction accuracy"].iloc[:3].notna().all()  # the other measures are still shown


def test_dashboard_with_the_real_results_is_unchanged_by_the_repair(snapshot):
    """The valid-data rendering: numbers, not 'undefined'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=240).run()
    assert not at.exception
    ml = json.loads((ROOT / "results" / "ml_summary.json").read_text())["models"]["ensemble"]
    assert [m for m in at.metric if m.label.startswith("Forecast model R squared")][0].value == f"{100 * ml['oos_r2_vs_training_mean']:.2f}%"
    table = next(d.value for d in at.dataframe if "R squared vs training mean" in d.value.columns)
    assert all(v.endswith("%") and v != "undefined" for v in table["R squared vs training mean"].iloc[:3])
