"""Guards against claims the code cannot back."""
import importlib.util
import re
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    for p in set(re.findall(r"`((?:[\w.-]+/)*[\w.-]+\.(?:py|md|csv|json|ipynb|parquet|txt))`", README)):
        assert (ROOT / p).exists(), f"README mentions {p} which does not exist"
    for p in re.findall(r"python scripts/([\w]+\.py)", README):
        assert (ROOT / "scripts" / p).exists()


def test_ticker_count_in_readme_matches_the_universe(universe):
    n = int(re.search(r"hand-maintained list of (\d+) tickers", README).group(1))
    assert n == len(universe)


def test_no_data_is_generated_or_simulated_anywhere():
    """The package, the scripts and the tests contain no random-number generation or hand-typed price data."""
    banned = ["numpy.random", "np.random", "default_rng", "random.seed", "import random", "from random", "randn", "make_regression"]
    files = list((ROOT / "riskengine").glob("*.py")) + list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    for f in files:
        if f.name == "test_readme_claims.py":
            continue  # this file contains the list above
        text = f.read_text()
        for b in banned:
            assert b not in text, f"{f.name} contains {b!r}"


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
    for f in (ROOT / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_\w+)", f.read_text(), re.M))
    for doc in ("AUDIT.md", "CLAIMS.md", "FORMULAS.md", "DECISIONS.md", "PROGRESS.md"):
        text = (ROOT / doc).read_text()
        for name in set(re.findall(r"`(test_[\w*]+)`", text)):
            assert any(fnmatch.fnmatch(d, name) for d in defined), f"{doc} cites {name}, which does not exist"
