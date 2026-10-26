"""Every script's documented interface, run in temporary directory trees so the checked-in data and results are never written.

Real network access is impossible in these tests: fetch_data.py runs against a stand-in provider module, and the others
run under a sitecustomize that makes every socket operation fail and logs the attempt.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from riskengine import data
from tests.integration.test_acceptance_outputs import independent_counts
from tests.unit.fixtures import smooth_prices

ROOT = Path(__file__).resolve().parents[2]
BLOCK = "import os, socket\nLOG = os.environ['NET_LOG']\n\ndef _no(*a, **k):\n    open(LOG, 'a').write(repr(a)[:80] + '\\n')\n    raise OSError('network blocked')\n\nsocket.socket.connect = _no\nsocket.getaddrinfo = _no\nif os.environ.get('COVERAGE_PROCESS_START'):  # opt-in: lets a coverage run measure these subprocesses\n    import coverage\n    coverage.process_startup()\n"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_tree(tmp: Path, dirs=("riskengine", "scripts"), data_files=()):
    for d in dirs:
        shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp / "data").mkdir(exist_ok=True)
    for f in data_files:
        shutil.copy(ROOT / "data" / f, tmp / "data" / f)
    guard = tmp / "_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(BLOCK)
    return guard


def run(tmp: Path, args, guard: Path, extra_path=(), timeout=900):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(guard), *map(str, extra_path)]), "NET_LOG": str(tmp / "net.log")}
    return subprocess.run([sys.executable, *args], cwd=tmp, capture_output=True, text=True, env=env, timeout=timeout)


def no_network_attempts(tmp: Path):
    log = tmp / "net.log"
    return (log.read_text().splitlines() if log.exists() else [])


# ----------------------------------------------------------------------------- build_universe.py
def test_build_universe_regenerates_the_identical_universe_file_from_git_history():
    p = ROOT / "data" / "universe.csv"
    before = sha(p)
    out = subprocess.run([sys.executable, "scripts/build_universe.py"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "483 unique tickers, 10 ETFs" in out.stdout
    assert sha(p) == before  # deterministic: same bytes


# ----------------------------------------------------------------------------- fetch_data.py against a stand-in provider
FAKE_YF = '''
import time
import numpy as np
import pandas as pd
time.sleep = lambda s: None            # no real retry delays
FAIL = {"ABC", "ANSS", "DISH"}         # tickers the stand-in provider has no data for
CALLS = []

def download(tickers, **kw):
    CALLS.append(tickers)
    if tickers == "^IRX":
        return pd.DataFrame({"Close": [4.0, 4.1, 4.2, 4.3, 4.4]}, index=pd.bdate_range(kw["start"], periods=5))
    names = tickers if isinstance(tickers, list) else [tickers]
    keep = [t for t in names if t not in FAIL]
    if not keep:
        return pd.DataFrame()
    idx = pd.bdate_range(kw["start"], periods=5)
    cols = pd.MultiIndex.from_product([keep, ["Open", "Close", "Volume"]])
    vals = np.column_stack([[100.0 + i + d, 100.0 + i + d + 0.5, 1e6] for i, _ in enumerate(keep) for d in range(5)]).T if False else None
    data = {}
    for i, t in enumerate(keep):
        base = 50.0 + (sum(map(ord, t)) % 100)
        data[(t, "Open")] = base + np.arange(5)
        data[(t, "Close")] = base + 0.5 + np.arange(5)
        data[(t, "Volume")] = np.full(5, 1e6)
    return pd.DataFrame(data, index=idx)
'''


def test_fetch_data_script_runs_end_to_end_against_a_standin_provider_and_records_failures(tmp_path):
    guard = make_tree(tmp_path, data_files=("universe.csv",))
    fake = tmp_path / "_fake"
    fake.mkdir()
    (fake / "yfinance.py").write_text(FAKE_YF)
    out = run(tmp_path, ["scripts/fetch_data.py", "--start", "2024-01-02", "--end", "2024-01-20"], guard, extra_path=[fake])
    assert out.returncode == 0, out.stderr[-1500:]
    prices, rf, meta = data.load_snapshot(tmp_path / "data")
    uni = pd.read_csv(tmp_path / "data" / "universe.csv")
    assert meta["tickers_requested"] == len(uni) == 483 and meta["tickers_returned"] == 480 == prices.shape[1]
    assert meta["tickers_failed"] == {"ABC": "empty response", "ANSS": "empty response", "DISH": "empty response"}
    assert meta["tickers_requested"] == meta["tickers_returned"] + len(meta["tickers_failed"])
    assert set(prices.columns) == set(uni["ticker"]) - {"ABC", "ANSS", "DISH"} and "BRK-B" in prices.columns  # share-class symbols untouched
    assert meta["requested_start"] == "2024-01-02" and meta["requested_end_exclusive"] == "2024-01-20" and meta["first_date"] == "2024-01-02"
    assert rf.name == "^IRX" and rf.tolist() == [4.0, 4.1, 4.2, 4.3, 4.4] and meta["risk_free_series"] == "^IRX"
    assert (prices.to_numpy() > 0).all() and meta["internal_gap_counts"] == {} and "adjusted" in meta["source"]
    assert no_network_attempts(tmp_path) == []


# ----------------------------------------------------------------------------- run_pipeline.py on a hand-constructed snapshot
def build_synthetic_snapshot(tmp: Path):
    """40 stocks + SPY over 1,700 business days with one late listing and one stopped ticker. Test fixture only; the
    pipeline's checked-in results are untouched and this table is never presented as market data."""
    prices = smooth_prices(41, 1700, start="2019-01-02", switch=700).rename(columns={"S00": "SPY"})
    prices["S39"] = prices["S39"].where(np.arange(1700) >= 900)  # listed late: leading missing values
    prices["S40"] = prices["S40"].where(np.arange(1700) < 1600)  # stopped trading 100 days before the end
    uni = pd.DataFrame({"ticker": list(prices.columns), "group_in_original_list": "fixture", "is_etf": [int(c == "SPY") for c in prices.columns]})
    uni.to_csv(tmp / "data" / "universe.csv", index=False)
    rf = pd.Series(2.0, index=prices.index, name="^IRX")
    meta = {"source": "hand-constructed fixture", "downloaded_utc": "2000-01-01T00:00:00Z", "tickers_requested": 41, "tickers_returned": 41, "tickers_failed": {}, **data.snapshot_report(prices)}
    data.save_snapshot(prices, rf, meta, data_dir=tmp / "data")
    return prices


@pytest.fixture(scope="module")
def smoke_tree(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    guard = make_tree(tmp)
    prices = build_synthetic_snapshot(tmp)
    first = run(tmp, ["scripts/run_pipeline.py"], guard)
    assert first.returncode == 0, first.stderr[-2000:]
    return tmp, guard, prices, first


RESULT_FILES = ["run_info.json", "var_latest.csv", "var_backtest_by_ticker.csv", "var_backtest_summary.json", "metrics.csv", "portfolios.json", "ml_summary.json", "ml_rank_ic_by_date.csv", "ml_latest_forecasts.csv", "ml_feature_importance.csv"]


def strict(p: Path):
    return json.loads(p.read_text(), parse_constant=lambda c: pytest.fail(f"nonstandard constant {c} in {p.name}"))


def test_smoke_pipeline_writes_every_output_with_strict_json_and_the_documented_schema(smoke_tree):
    tmp, guard, prices, _ = smoke_tree
    assert sorted(p.name for p in (tmp / "results").iterdir()) == sorted(RESULT_FILES)
    info = strict(tmp / "results" / "run_info.json")
    assert info["data"]["tickers_requested"] == 41 and info["data"]["stocks_returned"] == 40 and info["data"]["etfs_returned"] == 1
    assert info["data"]["first_date"] == str(prices.index[0].date()) and info["data"]["last_date"] == str(prices.index[-1].date()) and info["data"]["n_dates"] == 1700
    assert info["config"]["window"] == 250 and info["config"]["train_end"] == "2023-12-31"
    summary = strict(tmp / "results" / "var_backtest_summary.json")
    assert {(r["method"], r["confidence"]) for r in summary} == {(m, c) for m in ("historical", "normal") for c in (0.95, 0.99)}
    pf = strict(tmp / "results" / "portfolios.json")
    assert set(pf["presets"]) == {"conservative", "balanced", "aggressive"} and pf["train_end"] <= "2023-12-31" < pf["test_start"]
    ml = strict(tmp / "results" / "ml_summary.json")
    assert set(ml["models"]) == {"random_forest", "gradient_boosting", "ensemble"} and all("oos_r2_status" in m for m in ml["models"].values())


def test_smoke_pipeline_counts_exclusions_and_numbers_reconcile_with_independent_calculations(smoke_tree):
    tmp, _, prices, _ = smoke_tree
    info = strict(tmp / "results" / "run_info.json")
    vl = pd.read_csv(tmp / "results" / "var_latest.csv")
    # S40 stopped trading inside the last 250 days, so its latest window is incomplete: excluded, not filled
    assert info["var_latest"]["excluded_incomplete_window"] == 1 and "S40" not in set(vl["ticker"]) and len(vl) == 40
    returns = (prices / prices.shift(1) - 1).to_numpy()
    spy = returns[-250:, list(prices.columns).index("SPY")]
    losses = np.sort(-spy)
    pos = 0.99 * 249
    hand = losses[int(np.floor(pos))] + (pos - np.floor(pos)) * (losses[int(np.ceil(pos))] - losses[int(np.floor(pos))])
    assert vl.set_index("ticker").loc["SPY", "historical_var_99"] == pytest.approx(hand, abs=1e-14)
    bt = pd.read_csv(tmp / "results" / "var_backtest_by_ticker.csv")
    ind = independent_counts(prices, ["SPY", "S05", "S39"])
    for row in bt[bt["ticker"].isin(["SPY", "S05", "S39"])].itertuples(index=False):
        assert (row.n_forecasts, row.breaches) == ind[(row.ticker, row.method, row.confidence)]
    assert sorted(bt["ticker"].unique()) == sorted(t for t in prices.columns if (~np.isnan(returns[:, list(prices.columns).index(t)])).sum() >= 500)  # eligibility by the documented 250 + 250 rule


def test_smoke_pipeline_rerun_with_unchanged_inputs_reproduces_every_file_byte_for_byte(smoke_tree):
    tmp, guard, _, _ = smoke_tree
    before = {f: sha(tmp / "results" / f) for f in RESULT_FILES}
    again = run(tmp, ["scripts/run_pipeline.py"], guard)
    assert again.returncode == 0 and {f: sha(tmp / "results" / f) for f in RESULT_FILES} == before
    assert no_network_attempts(tmp) == []


def test_pipeline_without_a_snapshot_stops_with_the_command_to_create_it_and_never_touches_the_network(tmp_path):
    guard = make_tree(tmp_path, data_files=("universe.csv",))
    out = run(tmp_path, ["scripts/run_pipeline.py"], guard)
    assert out.returncode != 0 and "scripts/fetch_data.py" in out.stderr and "FileNotFoundError" in out.stderr
    assert not (tmp_path / "results" / "run_info.json").exists()  # no partial output
    assert no_network_attempts(tmp_path) == []


# ----------------------------------------------------------------------------- make_readme_tables.py
def test_readme_generator_check_mode_detects_drift_and_the_plain_run_repairs_it(tmp_path):
    guard = make_tree(tmp_path, dirs=("scripts",))
    shutil.copytree(ROOT / "results", tmp_path / "results")
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    ok = run(tmp_path, ["scripts/make_readme_tables.py", "--check"], guard)
    assert ok.returncode == 0
    original = (tmp_path / "README.md").read_text()
    (tmp_path / "README.md").write_text(original.replace("| historical | 99% | 458 | 779,137 | 1.59%", "| historical | 99% | 458 | 779,137 | 9.99%", 1))
    assert run(tmp_path, ["scripts/make_readme_tables.py", "--check"], guard).returncode == 1  # a hand-edited number is detected
    assert run(tmp_path, ["scripts/make_readme_tables.py"], guard).returncode == 0 and (tmp_path / "README.md").read_text() == original
    (tmp_path / "results" / "ml_summary.json").unlink()
    gone = run(tmp_path, ["scripts/make_readme_tables.py"], guard)
    assert gone.returncode != 0 and "ml_summary.json" in gone.stderr and (tmp_path / "README.md").read_text() == original  # README not clobbered


# ----------------------------------------------------------------------------- check_sp500_membership.py (offline analysis)
@pytest.fixture()
def membership_tree(tmp_path, snapshot):
    guard = make_tree(tmp_path, data_files=("universe.csv", "prices.parquet", "tbill_13w.parquet", "prices_meta.json", "sp500_current.csv", "sp500_changes.csv", "sp500_membership_meta.json"))
    (tmp_path / "results").mkdir()
    shutil.copy(ROOT / "results" / "var_backtest_by_ticker.csv", tmp_path / "results")
    return tmp_path, guard


def test_membership_analysis_reproduces_the_checked_in_results_offline(membership_tree):
    tmp, guard = membership_tree
    out = run(tmp, ["scripts/check_sp500_membership.py"], guard)
    assert out.returncode == 0, out.stderr[-1000:]
    for f in ("universe_membership.csv", "universe_membership_summary.json"):
        assert sha(tmp / "results" / f) == sha(ROOT / "results" / f)
    assert no_network_attempts(tmp) == []


@pytest.mark.parametrize(
    "damage,message",
    [
        ("delete current", "missing.*sp500_current.csv"),
        ("delete changes", "missing.*sp500_changes.csv"),
        ("delete meta", "missing.*sp500_membership_meta.json"),
        ("rename a column", "malformed.*columns"),
        ("garble a date", "unparseable date"),
        ("truncate changes", "do not match their provenance"),
        ("break meta json", "malformed"),
        ("empty current", "malformed|Empty"),
    ],
)
def test_missing_or_malformed_cached_membership_sources_are_reported_honestly(membership_tree, damage, message):
    tmp, guard = membership_tree
    d = tmp / "data"
    if damage == "delete current":
        (d / "sp500_current.csv").unlink()
    elif damage == "delete changes":
        (d / "sp500_changes.csv").unlink()
    elif damage == "delete meta":
        (d / "sp500_membership_meta.json").unlink()
    elif damage == "rename a column":
        t = (d / "sp500_changes.csv").read_text().replace("date,added,removed,reason", "when,added,removed,reason", 1)
        (d / "sp500_changes.csv").write_text(t)
    elif damage == "garble a date":
        lines = (d / "sp500_changes.csv").read_text().splitlines()
        lines[3] = "not-a-date" + lines[3][lines[3].index(","):]
        (d / "sp500_changes.csv").write_text("\n".join(lines) + "\n")
    elif damage == "truncate changes":
        lines = (d / "sp500_changes.csv").read_text().splitlines()
        (d / "sp500_changes.csv").write_text("\n".join(lines[:50]) + "\n")
    elif damage == "break meta json":
        (d / "sp500_membership_meta.json").write_text("{not json")
    elif damage == "empty current":
        (d / "sp500_current.csv").write_text("")
    out = run(tmp, ["scripts/check_sp500_membership.py"], guard)
    assert out.returncode != 0 and not (tmp / "results" / "universe_membership_summary.json").exists()  # no results from damaged sources
    import re

    assert re.search(message, out.stderr, re.S), out.stderr[-600:]
    assert no_network_attempts(tmp) == []


# ----------------------------------------------------------------------------- notebook generation and execution
def test_notebook_generation_reproduces_the_checked_in_notebook_apart_from_cell_ids(tmp_path):
    guard = make_tree(tmp_path, dirs=("scripts",))
    (tmp_path / "notebooks").mkdir()
    out = run(tmp_path, ["scripts/build_notebook.py"], guard)
    assert out.returncode == 0, out.stderr[-800:]
    import nbformat

    new = nbformat.read(tmp_path / "notebooks" / "walkthrough.ipynb", as_version=4)
    old = nbformat.read(ROOT / "notebooks" / "walkthrough.ipynb", as_version=4)
    assert [(c.cell_type, c.source) for c in new.cells] == [(c.cell_type, c.source) for c in old.cells]
    assert all(not c.get("outputs") for c in new.cells if c.cell_type == "code")  # generated without stored outputs, so no stale numbers


def test_the_walkthrough_notebook_executes_top_to_bottom_without_errors_using_this_interpreter(tmp_path, monkeypatch, snapshot):
    import nbformat
    from nbclient import NotebookClient

    spec = tmp_path / "kernels" / "audit-kernel"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "audit", "language": "python"}))
    monkeypatch.delenv("MPLBACKEND", raising=False)  # importing Streamlit in this process sets MPLBACKEND=Agg, which would suppress the kernel's inline figures
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))  # first in the search order: the kernel is exactly this interpreter, not a user-level one
    nb = nbformat.read(ROOT / "notebooks" / "walkthrough.ipynb", as_version=4)
    before = sha(ROOT / "notebooks" / "walkthrough.ipynb")
    NotebookClient(nb, kernel_name="audit-kernel", timeout=300, resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    outputs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs]
    assert not [o for o in outputs if o.output_type == "error"] and sum(1 for o in outputs if o.output_type == "display_data" and "image/png" in o.data) == 4
    assert sha(ROOT / "notebooks" / "walkthrough.ipynb") == before  # the checked-in notebook was not modified
    first = next(o for o in outputs if o.output_type == "stream")
    assert f"{snapshot[0].shape[1]} tickers" in first.text and snapshot[2]["last_date"] in first.text


# ----------------------------------------------------------------------------- the independent verifier must be able to fail
def verifier_tree(tmp: Path, mutate=None):
    """A copy of the verifier next to a symlink to the real snapshot and a reduced results/ (6 tickers) that is internally
    consistent, optionally corrupted by `mutate(stored_frame)` before the summary is rebuilt from the result."""
    from riskengine import backtest

    (tmp / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "verify_backtest_independent.py", tmp / "scripts")
    os.symlink(ROOT / "data", tmp / "data")
    (tmp / "results").mkdir()
    stored = pd.read_csv(ROOT / "results" / "var_backtest_by_ticker.csv")
    stored = stored[stored["ticker"].isin(sorted(stored["ticker"].unique())[:6])].reset_index(drop=True)
    if mutate is not None:
        mutate(stored)  # the summary below is rebuilt from the corrupted rows, so only the per-ticker comparison can notice
    stored.to_csv(tmp / "results" / "var_backtest_by_ticker.csv", index=False)
    (tmp / "results" / "var_backtest_summary.json").write_text(json.dumps(backtest.summarise(stored).to_dict("records")))
    guard = tmp / "_guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(BLOCK)
    return guard


def test_the_verifier_passes_on_consistent_results_for_a_reduced_ticker_set(tmp_path, snapshot):
    guard = verifier_tree(tmp_path)
    out = run(tmp_path, ["scripts/verify_backtest_independent.py"], guard)
    assert out.returncode == 0 and "ALL MATCH" in out.stdout and "compared 24 ticker x method x confidence" in out.stdout


def test_the_verifier_exits_nonzero_and_names_the_ticker_when_one_breach_count_is_wrong(tmp_path, snapshot):
    def corrupt(df):
        df.loc[3, "breaches"] += 1

    guard = verifier_tree(tmp_path, corrupt)
    out = run(tmp_path, ["scripts/verify_backtest_independent.py"], guard)
    assert out.returncode == 1 and "DIFFERENCES" in out.stdout and "ALL MATCH" not in out.stdout
    assert "stored n=" in out.stdout


def test_the_verifier_exits_nonzero_when_a_row_is_missing_or_a_kupiec_p_value_is_wrong(tmp_path, snapshot):
    def corrupt(df):
        df.loc[0, "kupiec_p"] = 0.5

    out = run(tmp_path, ["scripts/verify_backtest_independent.py"], verifier_tree(tmp_path, corrupt))
    assert out.returncode == 1 and "DIFFERENCES" in out.stdout
    other = tmp_path / "missing"
    other.mkdir()
    guard = verifier_tree(other)
    frame = pd.read_csv(other / "results" / "var_backtest_by_ticker.csv").iloc[1:]
    frame.to_csv(other / "results" / "var_backtest_by_ticker.csv", index=False)
    out = run(other, ["scripts/verify_backtest_independent.py"], guard)
    assert out.returncode == 1 and "missing row" in out.stdout


def test_the_verifier_exits_nonzero_when_the_summary_does_not_match_the_per_ticker_rows(tmp_path, snapshot):
    guard = verifier_tree(tmp_path)  # per-ticker rows are correct; only the summary is altered below
    path = tmp_path / "results" / "var_backtest_summary.json"
    rows = json.loads(path.read_text())
    rows[0]["total_breaches"] += 1
    path.write_text(json.dumps(rows))
    out = run(tmp_path, ["scripts/verify_backtest_independent.py"], guard)
    assert out.returncode == 1 and "summary" in out.stdout and "does not match the per-ticker rows" in out.stdout


# ----------------------------------------------------------------------------- update_data.py (incremental update)
FAKE_YF_DATED = '''
import time
import numpy as np
import pandas as pd
time.sleep = lambda s: None
FAIL = {"ABC", "ANSS", "DISH"}

def _price(t, day_number):
    return 50.0 + (sum(map(ord, t)) % 100) + 0.5 * day_number

def download(tickers, start=None, end=None, **kw):
    idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    n = np.array([len(pd.bdate_range("2024-01-01", d)) - 1 for d in idx])
    if tickers == "^IRX":
        return pd.DataFrame({"Close": 4.0 + 0.01 * n}, index=idx) if len(idx) else pd.DataFrame()
    names = tickers if isinstance(tickers, list) else [tickers]
    keep = [t for t in names if t not in FAIL]
    if not keep or not len(idx):
        return pd.DataFrame()
    return pd.DataFrame({(t, "Close"): [_price(t, k) for k in n] for t in keep}, index=idx)
'''


def dated_tree(tmp: Path):
    guard = make_tree(tmp, data_files=("universe.csv",))
    fake = tmp / "_fake"
    fake.mkdir()
    (fake / "yfinance.py").write_text(FAKE_YF_DATED)
    return guard, fake


def test_update_script_adds_only_the_new_days_and_equals_a_complete_download(tmp_path):
    inc, full = tmp_path / "inc", tmp_path / "full"
    inc.mkdir(), full.mkdir()
    g1, f1 = dated_tree(inc)
    g2, f2 = dated_tree(full)
    assert run(inc, ["scripts/fetch_data.py", "--start", "2024-01-02", "--end", "2024-02-01"], g1, extra_path=[f1]).returncode == 0
    first = run(inc, ["scripts/update_data.py", "--end", "2024-02-15"], g1, extra_path=[f1])
    assert first.returncode == 0 and "2024-01-31 -> 2024-02-14 (10 new day(s))" in first.stdout and "downloaded in full again: 0" in first.stdout
    assert run(full, ["scripts/fetch_data.py", "--start", "2024-01-02", "--end", "2024-02-15"], g2, extra_path=[f2]).returncode == 0
    a, rf_a, meta_a = data.load_snapshot(inc / "data")
    b, rf_b, meta_b = data.load_snapshot(full / "data")
    pd.testing.assert_frame_equal(a, b, check_freq=False)  # the updated snapshot is what a complete download gives
    assert rf_a.tolist() == rf_b.tolist() and meta_a["last_date"] == meta_b["last_date"] == "2024-02-14"
    assert meta_a["update_log"][0]["new_rows"] == 10 and "update_log" not in meta_b
    again = run(inc, ["scripts/update_data.py", "--end", "2024-02-15"], g1, extra_path=[f1])
    assert again.returncode == 0 and "Already up to date" in again.stdout
    assert no_network_attempts(inc) == [] and no_network_attempts(full) == []


def test_update_script_without_a_snapshot_or_with_an_old_end_date_stops_with_a_clear_message(tmp_path):
    guard, fake = dated_tree(tmp_path)
    none = run(tmp_path, ["scripts/update_data.py", "--end", "2024-02-15"], guard, extra_path=[fake])
    assert none.returncode == 1 and "scripts/fetch_data.py" in none.stdout and not (tmp_path / "data" / "prices.parquet").exists()
    assert run(tmp_path, ["scripts/fetch_data.py", "--start", "2024-01-02", "--end", "2024-02-01"], guard, extra_path=[fake]).returncode == 0
    before = (tmp_path / "data" / "prices.parquet").read_bytes()
    old = run(tmp_path, ["scripts/update_data.py", "--end", "2024-01-10"], guard, extra_path=[fake])
    assert old.returncode == 1 and "must be after the last stored date" in old.stdout
    assert (tmp_path / "data" / "prices.parquet").read_bytes() == before
