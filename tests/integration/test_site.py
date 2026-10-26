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
    for needle in ("Not investment advice", "not investment advice", "fixed snapshot", "hand-typed and is not the S&amp;P 500", "No transaction costs", "survivorship"):
        assert needle.lower() in PAGE.lower(), needle
    assert '<html lang="en">' in PAGE and 'name="viewport"' in PAGE
    for bad in ("strong buy", "sell signal", "guaranteed", "robust", "state-of-the-art", "institutional", "production ready"):
        assert bad not in PAGE.lower()
    emoji = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + "]")
    assert not emoji.search(PAGE)


def test_the_build_writes_only_into_the_output_folder_and_is_deterministic(tmp_path, snapshot):
    copy = tmp_path / "repo"
    for d in ("riskengine", "scripts", "results"):
        shutil.copytree(ROOT / d, copy / d, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "data", copy / "data")
    outs = []
    for _ in range(2):
        subprocess.run([sys.executable, "scripts/build_site.py"], cwd=copy, capture_output=True, text=True, check=True, timeout=300)
        outs.append((copy / "docs" / "index.html").read_bytes())
    assert outs[0] == outs[1] == (ROOT / "docs" / "index.html").read_bytes()
    assert sorted(p.name for p in (copy / "docs").iterdir()) == [".nojekyll", "index.html", "plotly.min.js"]
