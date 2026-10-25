"""Create notebooks/walkthrough.ipynb (no stored outputs).

The notebook only calls the riskengine package and reads results/; it holds no
analysis logic of its own.

    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
C = nbf.v4.new_code_cell
M = nbf.v4.new_markdown_cell

nb.cells = [
    M("# Walkthrough\n\nReads the downloaded real data and the files in `results/`. Run `python scripts/fetch_data.py` and `python scripts/run_pipeline.py` first. All logic lives in the `riskengine` package; this notebook only calls it and plots."),
    C("""import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from riskengine import backtest, config, data, var

prices, rf, meta = data.load_snapshot()
returns = data.simple_returns(prices)
res = ROOT / "results"
print(f"{prices.shape[1]} tickers, {meta['first_date']} to {meta['last_date']}, downloaded {meta['downloaded_utc']}")"""),
    M("## 1. One ticker: rolling one-day 99% VaR forecast and the days it was breached\n\nChange `TICKER` to any ticker in the downloaded data. The forecast for each day uses only the 250 returns before it."),
    C("""TICKER = "SPY"
r = returns[TICKER]
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, method in zip(axes, var.METHODS):
    f = pd.Series(backtest.rolling_forecasts(r.values, config.WINDOW, 0.99, method), index=r.index)
    breach = (-r > f)
    ax.plot(r.index, -r, lw=0.5, color="0.6", label="daily loss")
    ax.plot(f.index, f, lw=1.2, color="C0", label=f"{method} 99% VaR forecast")
    ax.scatter(r.index[breach], (-r)[breach], s=14, color="C3", zorder=3, label=f"breach ({int(breach.sum())} of {int(f.notna().sum())} days)")
    ax.set_ylabel("loss, fraction of position"); ax.legend(loc="upper left"); ax.set_title(f"{TICKER}: {method}")
plt.tight_layout(); plt.show()"""),
    M("## 2. Backtest across all tickers\n\nEach bar is the pooled share of days with a breach; the dashed line is the rate the model claims."),
    C("""summ = pd.DataFrame(json.loads((res / "var_backtest_summary.json").read_text()))
fig, ax = plt.subplots(figsize=(8, 4))
labels = [f"{m}\\n{int(c*100)}%" for m, c in zip(summ["method"], summ["confidence"])]
ax.bar(labels, summ["pooled_breach_rate"] * 100, color="C0")
for i, e in enumerate(summ["expected_rate"] * 100):
    ax.hlines(e, i - 0.4, i + 0.4, colors="k", linestyles="--")
ax.set_ylabel("breach rate, percent of forecast days"); ax.set_title("Pooled breach rate versus expected (dashed)")
plt.tight_layout(); plt.show()
summ[["method", "confidence", "n_tickers", "pooled_breach_rate", "expected_rate", "share_rejected", "zero_breach_tickers"]]"""),
    M("## 3. Distribution of Kupiec p-values by method\n\nIf a model were correct, p-values would be spread roughly evenly between 0 and 1. A pile-up near 0 means many tickers are rejected."),
    C("""bt = pd.read_csv(res / "var_backtest_by_ticker.csv")
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
for ax, ((m, c), g) in zip(axes.ravel(), bt.groupby(["method", "confidence"])):
    ax.hist(g["kupiec_p"], bins=20, range=(0, 1), color="C0"); ax.set_title(f"{m}, {int(c*100)}%"); ax.set_xlabel("Kupiec p-value")
plt.tight_layout(); plt.show()"""),
    M("## 4. Portfolio experiment: realised results on the test period\n\nCumulative growth of 1 for portfolios whose weights were fixed using data up to the training end."),
    C("""pf = json.loads((res / "portfolios.json").read_text())
test = returns.dropna(how="all").loc[config.TEST_START:]
fig, ax = plt.subplots(figsize=(10, 5))
(1 + test[config.MARKET].fillna(0)).cumprod().plot(ax=ax, color="k", label="SPY")
for name, e in pf["presets"].items():
    if e["status"] == "constructed":
        w = pd.Series(e["weights"])
        (1 + (test[w.index].fillna(0) * w).sum(axis=1)).cumprod().plot(ax=ax, label=name)
    else:
        print(f"{name}: {e['reason']}")
ax.set_ylabel("growth of 1"); ax.legend(); ax.set_title("Test period only (no transaction costs)")
plt.tight_layout(); plt.show()"""),
    M("Holdings and weights of the constructed portfolios (fixed using training data only)."),
    C("""tables = {name: pd.Series(e["weights"]).sort_values(ascending=False).rename(name) for name, e in pf["presets"].items() if e["status"] == "constructed"}
pd.concat(tables.values(), axis=1).fillna(0).round(3)"""),
]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = ROOT / "notebooks" / "walkthrough.ipynb"
nbf.write(nb, out)
print("wrote", out)
