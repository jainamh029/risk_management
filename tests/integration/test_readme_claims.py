"""Guards against claims the code cannot back."""
import importlib.util
import re
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = (ROOT / "README.md").read_text()

FORBIDDEN = [
    "production", "institutional", "professional", "robust", "state-of-the-art", "alpha generation",
    "cornish", "risk parity", "min vol", "selectkbest", "monte carlo",
    "efficient frontier", "heatmap", "time-series cross-validation", "badge",
]


def _tables_module():
    spec = importlib.util.spec_from_file_location("make_readme_tables", ROOT / "scripts" / "make_readme_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_results_block_equals_what_the_results_files_produce():
    mod = _tables_module()
    assert mod.render(README, mod.build()) == README, "README results block is stale: run scripts/make_readme_tables.py"


def test_no_forbidden_claims_in_readme_or_code():
    texts = {"README.md": README}
    for p in list((ROOT / "riskengine").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        texts[str(p.relative_to(ROOT))] = p.read_text()
    for name, t in texts.items():
        low = t.lower()
        for bad in FORBIDDEN:
            assert bad not in low, f"{name} contains {bad!r}"


def test_no_emoji_anywhere_in_documentation_or_code():
    pat = re.compile("[" + chr(0x1F300) + "-" + chr(0x1FAFF) + chr(0x2600) + "-" + chr(0x27BF) + chr(0x2705) + chr(0x274C) + "]")
    files = [ROOT / "README.md"] + list((ROOT / "riskengine").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for f in files:
        assert not pat.search(f.read_text()), f


def test_every_path_named_in_the_readme_exists():
    generated = {"data/prices.parquet"}  # git-ignored vendor data, created by scripts/fetch_data.py and documented as such
    for p in set(re.findall(r"`((?:[\w.-]+/)*[\w.-]+\.(?:py|md|csv|json|ipynb|parquet|txt))`", README)) - generated:
        assert (ROOT / p).exists(), f"README mentions {p} which does not exist"
    assert "data/prices.parquet" in (ROOT / ".gitignore").read_text() or "data/*.parquet" in (ROOT / ".gitignore").read_text()
    assert "`data/prices.parquet` (git-ignored)" in README or "data/prices.parquet` is git-ignored" in README
    for p in re.findall(r"python scripts/([\w]+\.py)", README):
        assert (ROOT / "scripts" / p).exists()


def test_ticker_count_in_readme_matches_the_universe(universe):
    n = int(re.search(r"hand-maintained list of (\d+) tickers", README).group(1))
    assert n == len(universe)


def test_no_data_is_generated_or_simulated_in_code_or_in_the_unit_and_integration_tests():
    """Results come from real data only. Random-number generation is allowed in exactly one place: tests/statistical,
    which is labelled synthetic and must never write to data/ or results/."""
    banned = ["numpy.random", "np.random", "default_rng", "random.seed", "import random", "from random", "randn", "make_regression"]
    files = list((ROOT / "riskengine").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "dashboard").glob("*.py"))
    files += list((ROOT / "tests" / "unit").glob("*.py")) + list((ROOT / "tests" / "integration").glob("*.py")) + [ROOT / "tests" / "conftest.py"]
    for f in files:
        if f.name == "test_readme_claims.py":
            continue  # this file contains the list above
        text = f.read_text()
        for b in banned:
            assert b not in text, f"{f.relative_to(ROOT)} contains {b!r}"
    for f in (ROOT / "tests" / "statistical").glob("*.py"):
        text = f.read_text()
        assert "pytestmark = pytest.mark.statistical" in text or f.name == "__init__.py"
        for sink in (".to_csv(", ".to_parquet(", ".write_text(", ".write_bytes(", "open("):
            assert sink not in text, f"{f.name} writes files ({sink})"


def test_requirements_pin_the_versions_that_ran_the_tests():
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, ver = line.split("==")
        assert metadata.version(name) == ver, f"{name}: pinned {ver}, installed {metadata.version(name)}"


def test_every_import_in_the_package_is_in_requirements():
    req = {l.split("==")[0].lower().replace("_", "-") for l in (ROOT / "requirements.txt").read_text().splitlines() if "==" in l}
    third = {"numpy", "pandas", "scipy", "yfinance", "pyarrow", "matplotlib", "scikit-learn", "streamlit", "plotly"}
    assert third <= req
    imported = set()
    for p in list((ROOT / "riskengine").glob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        imported |= set(re.findall(r"^\s*(?:import|from)\s+(\w+)", p.read_text(), re.M))
    assert {"numpy", "pandas", "scipy"} <= imported
    assert "seaborn" not in imported


def test_every_test_named_in_the_documents_exists():
    """Documents cite tests by name; a renamed or deleted test must not leave a dangling citation."""
    import fnmatch

    defined = set()
    for f in (ROOT / "tests").rglob("test_*.py"):
        defined |= set(re.findall(r"^def (test_\w+)", f.read_text(), re.M))
    for doc in ("AUDIT.md", "CLAIMS.md", "FORMULAS.md", "DECISIONS.md", "PROGRESS.md"):
        text = (ROOT / doc).read_text()
        for name in set(re.findall(r"`(test_[\w*]+)`", text)):
            assert any(fnmatch.fnmatch(d, name) for d in defined), f"{doc} cites {name}, which does not exist"
