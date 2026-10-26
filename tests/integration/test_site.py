"""The web pages published with GitHub Pages: a static report and the interactive dashboard (stlite).

They are built by scripts/build_site.py and are not kept on the main branch (the daily workflow builds and publishes them), so these tests
build them into a temporary folder from the real results and the price snapshot and check that build."""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def site(tmp_path_factory, snapshot):
    out = tmp_path_factory.mktemp("site")
    run = subprocess.run([sys.executable, "scripts/build_site.py", "--out", str(out)], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    return out


@pytest.fixture(scope="module")
def page(site):
    return (site / "index.html").read_text()


def test_the_build_is_deterministic_and_check_mode_agrees_with_it(site, tmp_path):
    again = tmp_path / "again"
    subprocess.run([sys.executable, "scripts/build_site.py", "--out", str(again)], cwd=ROOT, capture_output=True, text=True, check=True, timeout=300)
    names = sorted(p.relative_to(site).as_posix() for p in site.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(again).as_posix() for p in again.rglob("*") if p.is_file())
    assert all((site / n).read_bytes() == (again / n).read_bytes() for n in names)
    ok = subprocess.run([sys.executable, "scripts/build_site.py", "--out", str(site), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert ok.returncode == 0 and "up to date" in ok.stdout
    (site / "index.html").write_text((site / "index.html").read_text() + " ")
    stale = subprocess.run([sys.executable, "scripts/build_site.py", "--out", str(site), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert stale.returncode == 1 and "stale" in stale.stdout
    (site / "index.html").write_text((site / "index.html").read_text()[:-1])


def test_the_page_states_the_real_headline_numbers(page):
    s = json.loads((ROOT / "results" / "var_backtest_summary.json").read_text())
    h99 = next(r for r in s if r["method"] == "historical" and r["confidence"] == 0.99)
    ml = json.loads((ROOT / "results" / "ml_summary.json").read_text())["models"]["ensemble"]
    assert f"{100 * h99['pooled_breach_rate']:.2f}%" in page and f">{h99['n_tickers']}<" in page
    assert f"{100 * ml['oos_r2_vs_training_mean']:.2f}%" in page
    info = json.loads((ROOT / "results" / "run_info.json").read_text())["data"]
    assert info["first_date"] in page and info["last_date"] in page


def test_the_report_loads_nothing_from_the_network_and_ships_its_own_plotting_library(site, page):
    external = re.findall(r"""(?:src|href)=["'](https?://[^"']+)""", page)
    assert all(u.startswith("https://github.com/jainamh029/risk_management") for u in external), external  # only links to the repository
    assert 'src="plotly.min.js"' in page and (site / "plotly.min.js").stat().st_size > 1_000_000
    assert not re.search(r"@import|fonts\.googleapis|cdn\.", page)
    assert (site / ".nojekyll").exists()


def test_the_report_has_every_chart_a_table_and_the_required_disclosures(page):
    for cid in ("c-growth", "c-hist", "c-bars", "c-map", "c-ic"):
        assert f'id="{cid}"' in page
    assert page.count("<table>") == 3 and "<caption>" in page
    for needle in ("Not investment advice", "fixed report", "hand-typed and is not the S&amp;P 500", "No transaction costs", "survivorship"):
        assert needle.lower() in page.lower(), needle
    assert '<html lang="en">' in page and 'name="viewport"' in page
    for bad in ("strong buy", "sell signal", "guaranteed", "robust", "state-of-the-art", "institutional", "production ready"):
        assert bad not in page.lower()
    emoji = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
    assert not emoji.search(page)


def test_the_report_links_to_the_interactive_dashboard(page):
    assert 'href="app/"' in page and "Open the interactive dashboard" in page


def test_the_build_writes_only_into_the_output_folder(site):
    assert sorted(p.name for p in site.iterdir()) == [".nojekyll", "app", "index.html", "plotly.min.js"]


# ----------------------------------------------------------------------------- the interactive dashboard (stlite)
def test_the_published_app_files_are_byte_copies_of_the_repository_files(site):
    mod = _module()
    app = site / "app"
    assert (ROOT / "dashboard" / "app.py").read_bytes() == (app / "files" / "dashboard" / "app.py").read_bytes()  # the dashboard itself is not forked or edited for the web
    for rel in mod.APP_FILES:
        assert (app / "files" / rel).read_bytes() == (ROOT / rel).read_bytes(), rel
    assert {p.relative_to(app).as_posix() for p in app.rglob("*") if p.is_file()} == {"index.html"} | {f"files/{p.as_posix()}" for p in mod.APP_FILES}


def test_the_app_page_pins_stlite_and_loads_only_from_the_cdn_and_its_own_files(site):
    html = (site / "app" / "index.html").read_text()
    urls = set(re.findall(r"https?://[^\"'\s)<>]+", html))
    assert urls and all(u.startswith(("https://cdn.jsdelivr.net/npm/@stlite/browser@", "https://github.com/jainamh029/risk_management")) for u in urls), urls
    assert not re.search(r"@stlite/browser(/|@latest)", html) and re.search(r"@stlite/browser@\d+\.\d+\.\d+/", html)  # an exact version, never "latest"
    config = json.loads(re.search(r"const config = (\{.*?\n\});", html, re.S).group(1))
    assert config["entrypoint"] == "dashboard/app.py" and "plotly==7.1.0" in config["requirements"]
    assert all(v["url"] == f"./files/{k}" for k, v in config["files"].items()) and {"dashboard/app.py", "data/prices.parquet", "results/metrics.csv"} <= set(config["files"])
    assert "not investment advice" in html.lower()


def test_the_dashboard_starts_and_shows_every_tab_from_only_the_files_that_are_published(site, tmp_path):
    """The browser gets exactly the files in the app's mount list. Copy only those into an empty folder and run the dashboard there:
    a file the dashboard reads but the page does not ship would raise here."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    for rel in _module().APP_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(site / "app" / "files" / rel, tmp_path / rel)
    st.cache_data.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = AppTest.from_file(str(tmp_path / "dashboard" / "app.py"), default_timeout=240).run()
    st.cache_data.clear()
    assert not at.exception, [e.value[:150] for e in at.exception]
    assert [t.label for t in at.tabs] == ["Overview", "Stock explorer", "VaR explorer", "Backtest comparison", "Portfolio experiment", "Forecasting model", "Glossary and limits"]
    assert len(at.get("plotly_chart")) >= 10
