"""The static report served by GitHub Pages (docs/index.html) is built from the real results, loads nothing from the network,
and cannot go stale unnoticed."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = (ROOT / "docs" / "index.html").read_text()


def test_the_checked_in_page_equals_a_fresh_build(snapshot):
    out = subprocess.run([sys.executable, "scripts/build_site.py", "--check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_page_states_the_real_headline_numbers(snapshot):
    s = json.loads((ROOT / "results" / "var_backtest_summary.json").read_text())
    h99 = next(r for r in s if r["method"] == "historical" and r["confidence"] == 0.99)
    ml = json.loads((ROOT / "results" / "ml_summary.json").read_text())["models"]["ensemble"]
    assert f"{100 * h99['pooled_breach_rate']:.2f}%" in PAGE and f">{h99['n_tickers']}<" in PAGE
    assert f"{100 * ml['oos_r2_vs_training_mean']:.2f}%" in PAGE
    info = json.loads((ROOT / "results" / "run_info.json").read_text())["data"]
    assert info["first_date"] in PAGE and info["last_date"] in PAGE


def test_the_page_loads_nothing_from_the_network_and_ships_its_own_plotting_library():
    external = re.findall(r"""(?:src|href)=["'](https?://[^"']+)""", PAGE)
    assert all(u.startswith("https://github.com/jainamh029/risk_management") for u in external), external  # only links to the repository
    assert 'src="plotly.min.js"' in PAGE and (ROOT / "docs" / "plotly.min.js").stat().st_size > 1_000_000
    assert not re.search(r"@import|fonts\.googleapis|cdn\.", PAGE)
    assert (ROOT / "docs" / ".nojekyll").exists()


def test_the_page_has_every_chart_a_table_and_the_required_disclosures():
    for cid in ("c-growth", "c-hist", "c-bars", "c-map", "c-ic"):
        assert f'id="{cid}"' in PAGE
    assert PAGE.count("<table>") == 3 and "<caption>" in PAGE
    for needle in ("Not investment advice", "not investment advice", "fixed report", "hand-typed and is not the S&amp;P 500", "No transaction costs", "survivorship"):
        assert needle.lower() in PAGE.lower(), needle
    assert '<html lang="en">' in PAGE and 'name="viewport"' in PAGE
    for bad in ("strong buy", "sell signal", "guaranteed", "robust", "state-of-the-art", "institutional", "production ready"):
        assert bad not in PAGE.lower()
    emoji = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
    assert not emoji.search(PAGE)


def test_the_build_writes_only_into_the_output_folder_and_is_deterministic(tmp_path, snapshot):
    copy = tmp_path / "repo"
    for d in ("riskengine", "scripts", "results", "dashboard", ".streamlit"):
        shutil.copytree(ROOT / d, copy / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data", copy / "data")
    outs = []
    for _ in range(2):
        subprocess.run([sys.executable, "scripts/build_site.py"], cwd=copy, capture_output=True, text=True, check=True, timeout=300)
        outs.append((copy / "docs" / "index.html").read_bytes())
    assert outs[0] == outs[1] == (ROOT / "docs" / "index.html").read_bytes()
    assert (copy / "docs" / "app" / "index.html").read_bytes() == (ROOT / "docs" / "app" / "index.html").read_bytes()
    assert sorted(p.name for p in (copy / "docs").iterdir()) == [".nojekyll", "app", "index.html", "plotly.min.js"]


# ----------------------------------------------------------------------------- the interactive dashboard published next to the report (stlite)
APP = ROOT / "docs" / "app"


def test_the_published_app_files_are_byte_copies_of_the_repository_files():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "dashboard" / "app.py").read_bytes() == (APP / "files" / "dashboard" / "app.py").read_bytes()  # the dashboard itself is not forked or edited for the web
    for rel in mod.APP_FILES:
        assert (APP / "files" / rel).read_bytes() == (ROOT / rel).read_bytes(), rel
    assert {p.relative_to(APP).as_posix() for p in APP.rglob("*") if p.is_file()} == {"index.html"} | {f"files/{p.as_posix()}" for p in mod.APP_FILES}


def test_the_app_page_pins_stlite_and_loads_only_from_the_cdn_and_its_own_files():
    page = (APP / "index.html").read_text()
    urls = set(re.findall(r"https?://[^\"'\s)<>]+", page))
    assert urls and all(u.startswith(("https://cdn.jsdelivr.net/npm/@stlite/browser@", "https://github.com/jainamh029/risk_management")) for u in urls), urls
    assert not re.search(r"@stlite/browser(/|@latest)", page) and re.search(r"@stlite/browser@\d+\.\d+\.\d+/", page)  # an exact version, never "latest"
    config = json.loads(re.search(r"const config = (\{.*?\n\});", page, re.S).group(1))
    assert config["entrypoint"] == "dashboard/app.py" and "plotly==7.1.0" in config["requirements"]
    assert all(v["url"] == f"./files/{k}" for k, v in config["files"].items()) and {"dashboard/app.py", "data/prices.parquet", "results/metrics.csv"} <= set(config["files"])
    assert "Not investment advice".lower() in page.lower() or "not investment advice" in page.lower()


def test_the_report_links_to_the_interactive_dashboard():
    assert 'href="app/"' in PAGE and "Open the interactive dashboard" in PAGE


def test_the_dashboard_starts_and_shows_every_tab_from_only_the_files_that_are_published(tmp_path, snapshot):
    """The browser gets exactly the files in the app's mount list. Copy only those into an empty folder and run the dashboard there:
    a file the dashboard reads but the page does not ship would raise here."""
    import importlib.util
    import warnings

    from streamlit.testing.v1 import AppTest

    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for rel in mod.APP_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(APP / "files" / rel, tmp_path / rel)
    import streamlit as st

    st.cache_data.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = AppTest.from_file(str(tmp_path / "dashboard" / "app.py"), default_timeout=240).run()
    st.cache_data.clear()
    assert not at.exception, [e.value[:150] for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Stock explorer", "VaR explorer", "Backtest comparison", "Portfolio experiment", "Forecasting model", "Glossary and limits"]
    assert len(at.get("plotly_chart")) >= 10
