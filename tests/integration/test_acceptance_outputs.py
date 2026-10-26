"""C16 and F: independent recomputation of cached results, temporary-directory regeneration, README reconciliation."""
import ast
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom, chi2

ROOT = Path(__file__).resolve().parents[2]
WINDOW = 250


# ----------------------------------------------------------------------------- C16
def independent_counts(prices: pd.DataFrame, tickers) -> dict:
    """Plain-loop recomputation of every (ticker, method, confidence) forecast and breach count. Imports nothing from riskengine."""
    nd = NormalDist()
    out = {}
    for t in tickers:
        p = prices[t].to_numpy(dtype=float)
        r = np.full(len(p), np.nan)
        r[1:] = p[1:] / p[:-1] - 1.0
        for method in ("historical", "normal"):
            for conf in (0.95, 0.99):
                n = x = 0
                for i in range(WINDOW, len(r)):
                    w = r[i - WINDOW : i]
                    if np.isnan(w).any() or np.isnan(r[i]):
                        continue
                    if method == "historical":
                        s = np.sort(-w)
                        pos = conf * (len(s) - 1)
                        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
                        f = s[lo] + (pos - lo) * (s[hi] - s[lo])
                    else:
                        mu = w.sum() / len(w)
                        sd = math.sqrt(((w - mu) ** 2).sum() / (len(w) - 1))
                        f = -(mu - nd.inv_cdf(conf) * sd)
                    n += 1
                    x += int(-r[i] > f)
                out[(t, method, conf)] = (n, x)
    return out


@pytest.fixture(scope="module")
def recomputed(prices):
    stored = pd.read_csv(ROOT / "results" / "var_backtest_by_ticker.csv")
    return stored, independent_counts(prices, sorted(stored["ticker"].unique()))


def test_C16_every_cached_per_ticker_count_matches_an_independent_recomputation(recomputed):
    stored, ind = recomputed
    assert len(stored) == len(ind)
    for row in stored.itertuples(index=False):
        n, x = ind[(row.ticker, row.method, row.confidence)]
        assert (row.n_forecasts, row.breaches) == (n, x), (row.ticker, row.method, row.confidence)
        lr = -2.0 * (binom.logpmf(x, n, 1 - row.confidence) - binom.logpmf(x, n, x / n))
        assert row.kupiec_lr == pytest.approx(max(lr, 0.0), abs=1e-6) and row.kupiec_p == pytest.approx(chi2.sf(max(lr, 0.0), 1), abs=1e-9)
        assert row.breach_rate == x / n


def test_C16_pooled_rates_shares_and_every_denominator_reconcile(recomputed, universe):
    stored, ind = recomputed
    etf = set(universe.loc[universe["is_etf"] == 1, "ticker"])
    summ = pd.DataFrame(json.loads((ROOT / "results" / "var_backtest_summary.json").read_text()))
    mem = json.loads((ROOT / "results" / "universe_membership_summary.json").read_text())
    tickers = sorted({k[0] for k in ind})
    stocks = [t for t in tickers if t not in etf]
    assert len(tickers) - len(stocks) == len(set(tickers) & etf) == 10  # the headline table counts 10 ETFs among its tickers
    for (method, conf), s in summ.set_index(["method", "confidence"]).iterrows() if False else [(k, r) for k, r in summ.set_index(["method", "confidence"]).iterrows()]:
        n_all = sum(ind[(t, method, conf)][0] for t in tickers)
        x_all = sum(ind[(t, method, conf)][1] for t in tickers)
        assert s["n_tickers"] == len(tickers) and s["total_forecasts"] == n_all and s["total_breaches"] == x_all
        assert s["pooled_breach_rate"] == pytest.approx(x_all / n_all, abs=1e-15)
        rej = np.mean([chi2.sf(max(-2.0 * (binom.logpmf(ind[(t, method, conf)][1], ind[(t, method, conf)][0], 1 - conf) - binom.logpmf(ind[(t, method, conf)][1], ind[(t, method, conf)][0], ind[(t, method, conf)][1] / ind[(t, method, conf)][0])), 0.0), 1) < 0.05 for t in tickers])
        assert s["share_rejected"] == pytest.approx(rej, abs=1e-12)
        # the member-only table is built from stocks only (ETFs have no index membership): its "all days" column is the stocks-only rate
        row = next(r for r in mem["pooled_breach_rate_members_only"] if r["method"] == method and r["confidence"] == conf)
        n_st = sum(ind[(t, method, conf)][0] for t in stocks)
        x_st = sum(ind[(t, method, conf)][1] for t in stocks)
        assert row["forecast_days_all"] == n_st and row["breach_rate_all_days"] == pytest.approx(x_st / n_st, abs=1e-15)
        assert row["forecast_days_member_only"] <= n_st
    assert mem["backtest"]["forecast_days"] == sum(ind[(t, "historical", 0.95)][0] for t in stocks) and mem["backtest"]["stocks_tested"] == len(stocks)


def test_C16c_the_readme_labels_the_denominators_of_its_two_breach_rate_tables():
    text = (ROOT / "README.md").read_text()
    assert "stocks only" in text and "ETFs" in text
    block = text[text.index("### Ticker list compared with S&P 500 membership") :]
    assert "stocks only" in block[block.index("Pooled breach rate over all backtested days") : block.index("Matching is by ticker symbol")]


# ----------------------------------------------------------------------------- F31
RESULT_FILES = ["run_info.json", "var_latest.csv", "var_backtest_by_ticker.csv", "var_backtest_summary.json", "metrics.csv", "portfolios.json", "ml_summary.json", "ml_rank_ic_by_date.csv", "ml_latest_forecasts.csv", "ml_feature_importance.csv"]


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same_result(new: Path, stored: Path, rtol: float = 1e-6, atol: float = 1e-12) -> bool:
    """Equal in content: identical text, dates, names and integers; floating-point numbers equal to `rtol`.

    The checked-in results are regenerated by a daily workflow on Linux and by developers on macOS. The two differ in the last
    digits (measured on identical data: 1e-15 to 5e-11 relative for the VaR and statistics tables, up to 1.4e-7 for optimiser
    weights, which an iterative solver stops at slightly different points); counts and every non-numeric field are identical.
    Byte equality on one machine is still checked, by running the pipeline twice (test_smoke_pipeline_rerun_with_unchanged_inputs_reproduces_every_file_byte_for_byte)."""
    import numpy as np

    if stored.suffix == ".json":
        def walk(x, y):
            if isinstance(x, dict):
                return x.keys() == y.keys() and all(walk(x[k], y[k]) for k in x)
            if isinstance(x, list):
                return len(x) == len(y) and all(walk(p, q) for p, q in zip(x, y))
            if isinstance(x, float) and isinstance(y, float):
                return (np.isnan(x) and np.isnan(y)) or abs(x - y) <= atol + rtol * abs(y)
            return x == y and type(x) is type(y)

        return walk(json.loads(new.read_text()), json.loads(stored.read_text()))
    a, b = pd.read_csv(new), pd.read_csv(stored)
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    for c in a.columns:
        if pd.api.types.is_float_dtype(a[c]) or pd.api.types.is_float_dtype(b[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=rtol, atol=atol, equal_nan=True):
                return False
        elif not a[c].equals(b[c]):
            return False
    return True


def test_F31_the_pipeline_regenerates_every_result_in_a_temporary_directory_without_touching_the_checked_in_results(tmp_path, snapshot):
    before = {f: digest(ROOT / "results" / f) for f in RESULT_FILES}
    work = tmp_path / "repo"
    for d in ("riskengine", "scripts", "data"):
        shutil.copytree(ROOT / d, work / d, ignore=shutil.ignore_patterns("__pycache__"))
    proc = subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=work, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert sorted(p.name for p in (work / "results").iterdir()) == sorted(RESULT_FILES)
    for f in RESULT_FILES:
        assert same_result(work / "results" / f, ROOT / "results" / f), f"{f} differs between the temporary run and the checked-in results"
    assert {f: digest(ROOT / "results" / f) for f in RESULT_FILES} == before  # the checked-in results were not touched


# ----------------------------------------------------------------------------- F32
def md_tables(text: str) -> dict:
    """heading (or the last non-table line before the table) -> list of rows"""
    out, current, rows = {}, None, []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        else:
            if rows:
                out[current] = rows
                rows = []
            if line.startswith("###") or line.strip():
                current = line.strip()
    if rows:
        out[current] = rows
    return out


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def test_F32_readme_tables_equal_independently_formatted_values_from_the_result_files(prices, universe):
    text = (ROOT / "README.md").read_text()
    res = ROOT / "results"
    tabs = md_tables(text)
    by = pd.read_csv(res / "var_backtest_by_ticker.csv")

    # data table: counts and date range
    meta = json.loads((ROOT / "data" / "prices_meta.json").read_text())
    data_rows = dict((r[0], r[1]) for r in next(v for k, v in tabs.items() if k.startswith("### Data used"))[1:])
    assert data_rows["Date range"] == f"{prices.index.min().date()} to {prices.index.max().date()} ({len(prices)} trading days)"
    assert data_rows["Tickers requested"] == str(len(universe)) and data_rows["Tickers Yahoo returned nothing for"] == str(len(meta["tickers_failed"]))
    rfm = pd.read_parquet(ROOT / "data" / "tbill_13w.parquet").iloc[:, 0]
    assert data_rows["Average 13-week T-bill yield, whole range"] == f"{float(rfm.mean()) / 100 * 100:.3f}%"

    # backtest table: rebuilt from the per-ticker file with explicit formatting
    rows = next(v for k, v in tabs.items() if k.startswith("### VaR backtest"))
    assert rows[0][:5] == ["Method", "Confidence", "Tickers", "Forecasts", "Pooled breach rate"]
    seen = set()
    for r in rows[1:]:
        method, conf = r[0], int(r[1].rstrip("%")) / 100
        g = by[(by.method == method) & (by.confidence == conf)]
        assert r[2] == str(g.ticker.nunique()) and r[3] == f"{int(g.n_forecasts.sum()):,}"
        assert r[4] == pct(g.breaches.sum() / g.n_forecasts.sum()) and r[5] == pct(1 - conf)
        assert r[6] == pct((g.kupiec_p < 0.05).mean(), 1) and r[7] == str(int((g.breaches == 0).sum()))
        seen.add((method, conf))
    assert seen == {(m, c) for m in ("historical", "normal") for c in (0.95, 0.99)}

    # SPY VaR table
    vl = pd.read_csv(res / "var_latest.csv").set_index("ticker").loc["SPY"]
    heading = next(k for k in tabs if k.startswith("### Latest one-day VaR and ES for SPY"))
    assert f"ending {pd.Timestamp(vl['window_end']).date()})" in heading
    for r in tabs[heading][1:]:
        m = r[0]
        assert r[1:] == [pct(vl[f"{m}_var_95"]), pct(vl[f"{m}_es_95"]), pct(vl[f"{m}_var_99"]), pct(vl[f"{m}_es_99"])]

    # portfolio table
    pf = json.loads((res / "portfolios.json").read_text())
    heading = next(k for k in tabs if k.startswith("### Portfolio experiment"))
    assert heading == f"### Portfolio experiment: chosen with data to {pf['train_end']}, evaluated on {pf['test_start']} to {pf['test_end']}"
    for r in tabs[heading][1:]:
        if r[0].startswith("SPY"):
            b = pf["benchmark_spy_test"]
            assert r[5:8] == [f"{b['sharpe']:.2f}", pct(b["total_return"], 1), pct(b["max_drawdown"], 1)]
            continue
        e = pf["presets"][r[0]]
        assert r[1] == ("constructed" if e["status"] == "constructed" else "not constructed") and r[2] == str(e["n_passing_screen"])
        if e["status"] == "constructed":
            assert r[3] == str(len(e["tickers"])) and r[4] == f"{e['in_sample_train']['sharpe']:.2f}"
            assert r[5:8] == [f"{e['out_of_sample_test']['sharpe']:.2f}", pct(e["out_of_sample_test"]["total_return"], 1), pct(e["out_of_sample_test"]["max_drawdown"], 1)]
            assert r[8:10] == [f"{e['equal_weight_same_holdings_test']['sharpe']:.2f}", pct(e["equal_weight_same_holdings_test"]["total_return"], 1)]
    assert f"{pct(pf['risk_free_train'], 3)} (training) and {pct(pf['risk_free_test'], 3)} (test)" in text

    # forecasting table
    ml = json.loads((res / "ml_summary.json").read_text())
    heading = next(k for k in tabs if k.startswith("| Model") or k.startswith("Pooled models trained on"))
    for r in tabs[heading][1:4]:
        m = ml["models"][r[0].replace(" ", "_")]
        ic = m["rank_ic_nonoverlapping"]
        assert r[1] == pct(m["oos_r2_vs_training_mean"]) and r[2] == pct(m["direction_accuracy"], 1)
        assert r[3] == f"{ic['mean']:.3f} (se {ic['std_error']:.3f}, t {ic['t_stat']:.2f}, n {ic['n']})" and r[4] == f"{m['rank_ic_all_dates']['mean']:.3f}"
    assert f"Share of test samples with a positive 63-day return: {pct(ml['baselines']['share_of_test_samples_with_positive_return'], 1)}." in text
    info = ml["info"]
    assert f"{info['train_samples']:,} samples (feature dates {info['train_first_feature_date']} to {info['train_last_feature_date']}" in text and f"{info['test_samples']:,} samples" in text

    # membership table
    mem = json.loads((res / "universe_membership_summary.json").read_text())
    t = pd.read_csv(res / "universe_membership.csv")
    rows = dict((r[0], r[1]) for r in next(v for k, v in tabs.items() if k.startswith("Source: Wikipedia") or "Item" in v[0][0])[1:]) if False else None
    assert f"{int(t[~t.is_etf].current_member.sum())} of {int((~t.is_etf).sum())}" in text
    assert f"| Stocks that were members on every date of the price window | {int((t.category == 'member on every date').sum())} |" in text
    assert f"| Stocks that were members on some dates only | {int((t.category == 'member on some dates').sum())} |" in text
    for r in mem["pooled_breach_rate_members_only"]:
        row = f"| {r['method']} | {pct(r['confidence'], 0)} | {pct(r['breach_rate_all_days'], 3)} | {pct(r['breach_rate_member_days_only'], 3)} | {pct(r['expected_rate'])} |"
        assert row in text


# ----------------------------------------------------------------------------- F33
def test_F33_the_independent_verifier_imports_nothing_from_the_package_and_passes(snapshot):
    path = ROOT / "scripts" / "verify_backtest_independent.py"
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"json", "math", "sys", "pathlib", "statistics", "numpy", "pandas", "scipy"}, imported
    identifiers = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not identifiers & {"riskengine", "importlib", "exec", "eval", "__import__"}  # code only; the docstring may name the package
    strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    doc = tree.body[0].value.value  # the module docstring, which is allowed to name the package
    assert not any("riskengine" in v for v in strings if v != doc)
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
    n_results = len(pd.read_csv(ROOT / "results" / "var_backtest_by_ticker.csv"))  # tickers x methods x confidence levels: follows the data, not a fixed number
    assert "ALL MATCH" in proc.stdout and f"compared {n_results} ticker" in proc.stdout
