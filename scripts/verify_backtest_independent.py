"""Independent re-computation of the whole VaR backtest.

Shares no code with riskengine: it reads the price snapshot with pandas, then
recomputes every ticker's one-day VaR forecasts, breaches and Kupiec statistic
with plain loops and `statistics.NormalDist`, and compares them with
results/var_backtest_by_ticker.csv and results/var_backtest_summary.json.

    python scripts/verify_backtest_independent.py

Exits with status 1 if any number differs.
"""
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

ROOT = Path(__file__).resolve().parents[1]
WINDOW, CONFS = 250, (0.95, 0.99)

prices = pd.read_parquet(ROOT / "data" / "prices.parquet")
stored = pd.read_csv(ROOT / "results" / "var_backtest_by_ticker.csv")
nd = NormalDist()
bad, checked = [], 0


def forecast(win, conf, method):
    if method == "historical":
        s = np.sort(-win)
        pos = conf * (len(s) - 1)
        lo, hi = int(math.floor(pos)), int(math.ceil(pos))
        return s[lo] + (pos - lo) * (s[hi] - s[lo])
    mu = win.sum() / len(win)
    sd = math.sqrt(((win - mu) ** 2).sum() / (len(win) - 1))
    return -(mu - nd.inv_cdf(conf) * sd)


tickers = sorted(stored["ticker"].unique())
for k, t in enumerate(tickers):
    p = prices[t].to_numpy(dtype=float)
    r = np.full(len(p), np.nan)
    r[1:] = p[1:] / p[:-1] - 1.0
    for method in ("historical", "normal"):
        for conf in CONFS:
            n = x = 0
            for i in range(WINDOW, len(r)):
                win = r[i - WINDOW : i]
                if np.isnan(win).any() or np.isnan(r[i]):
                    continue
                n += 1
                x += int(-r[i] > forecast(win, conf, method))
            row = stored[(stored.ticker == t) & (stored.method == method) & (stored.confidence == conf)]
            checked += 1
            if len(row) != 1:
                bad.append((t, method, conf, "missing row"))
                continue
            row = row.iloc[0]
            lr = -2.0 * (binom.logpmf(x, n, 1 - conf) - binom.logpmf(x, n, x / n))
            ok = int(row.n_forecasts) == n and int(row.breaches) == x and abs(row.kupiec_lr - lr) < 1e-6 and abs(row.kupiec_p - chi2.sf(max(lr, 0.0), 1)) < 1e-9
            if not ok:
                bad.append((t, method, conf, f"stored n={row.n_forecasts}, x={row.breaches}; independent n={n}, x={x}"))
    if (k + 1) % 100 == 0:
        print(f"  {k + 1}/{len(tickers)} tickers checked", flush=True)

# the summary table, recomputed from the independent counts is the same as from the stored per-ticker rows
summ = pd.DataFrame(json.loads((ROOT / "results" / "var_backtest_summary.json").read_text()))
for _, s in summ.iterrows():
    g = stored[(stored.method == s["method"]) & (stored.confidence == s["confidence"])]
    if int(g.breaches.sum()) != int(s["total_breaches"]) or int(g.n_forecasts.sum()) != int(s["total_forecasts"]) or abs((g.kupiec_p < 0.05).mean() - s["share_rejected"]) > 1e-12:
        bad.append((s["method"], s["confidence"], "summary", "does not match the per-ticker rows"))

print(f"compared {checked} ticker x method x confidence results against an independent implementation")
if bad:
    print(f"{len(bad)} DIFFERENCES, first few: {bad[:5]}")
    sys.exit(1)
print("ALL MATCH: forecast counts, breach counts, Kupiec statistics, p-values and the summary table")
