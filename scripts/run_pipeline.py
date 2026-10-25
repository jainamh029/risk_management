"""Run every analysis on the downloaded real data and write results/.

    python scripts/run_pipeline.py

Reads data/prices.parquet (create it with scripts/fetch_data.py). Writes
CSV/JSON files to results/. Nothing is printed except neutral summaries of
numbers that were just computed.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import backtest, config, data, metrics, portfolio, var  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def clean(o):
    """JSON-safe: NaN/inf -> None, numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def dump(name, obj):
    (OUT / name).write_text(json.dumps(clean(obj), indent=2))


prices, rf, meta = data.load_snapshot()
uni = data.load_universe().set_index("ticker")
returns = data.simple_returns(prices)
is_etf = uni["is_etf"].astype(bool)
fetched = list(prices.columns)
rf_all = data.risk_free_annual(rf)
rf_train = data.risk_free_annual(rf, end=config.TRAIN_END)
rf_test = data.risk_free_annual(rf, start=config.TEST_START)

# 1. latest VaR/ES table
vt = var.latest_var_table(returns, config.WINDOW, config.CONFS)
vt.to_csv(OUT / "var_latest.csv", index=False)
excluded = vt.attrs["excluded_incomplete_window"]

# 2. backtest
bt = backtest.backtest_universe(returns, config.WINDOW, config.CONFS)
bt.to_csv(OUT / "var_backtest_by_ticker.csv", index=False)
summ = backtest.summarise(bt)
dump("var_backtest_summary.json", summ.to_dict(orient="records"))

# 3. descriptive metrics over the whole downloaded window
mt = metrics.compute_metrics(prices, returns, rf_all, market=config.MARKET)
mt["is_etf"] = mt["ticker"].map(is_etf)
mt.to_csv(OUT / "metrics.csv", index=False)

# 4. portfolios with train/test split
pf = portfolio.build_and_evaluate(returns, prices, is_etf, rf_train, rf_test)
dump("portfolios.json", pf)

# 5. run information
info = {
    "config": {
        "window": config.WINDOW,
        "confidences": list(config.CONFS),
        "train_end": config.TRAIN_END,
        "test_start": config.TEST_START,
        "market": config.MARKET,
        "min_assets": config.MIN_ASSETS,
    },
    "data": {
        "downloaded_utc": meta["downloaded_utc"],
        "first_date": meta["first_date"],
        "last_date": meta["last_date"],
        "n_dates": meta["n_dates"],
        "tickers_requested": meta["tickers_requested"],
        "tickers_returned": meta["tickers_returned"],
        "tickers_failed": len(meta["tickers_failed"]),
        "etfs_returned": int(sum(is_etf.get(t, False) for t in fetched)),
        "stocks_returned": int(sum(not is_etf.get(t, False) for t in fetched)),
        "risk_free_all": rf_all,
        "risk_free_train": rf_train,
        "risk_free_test": rf_test,
    },
    "var_latest": {
        "n_tickers": int(len(vt)),
        "excluded_incomplete_window": len(excluded),
        "window_start": str(vt["window_start"].iloc[0].date()),
        "window_end": str(vt["window_end"].iloc[0].date()),
    },
    "backtest": {"n_tickers": int(bt["ticker"].nunique())},
    "metrics": {"n_tickers": int(len(mt))},
}
dump("run_info.json", info)

print(f"data: {info['data']['tickers_returned']} tickers ({info['data']['stocks_returned']} stocks, {info['data']['etfs_returned']} ETFs), {meta['first_date']} to {meta['last_date']}")
print(f"latest VaR table: {len(vt)} tickers with a complete {config.WINDOW}-day window; {len(excluded)} excluded")
print(f"backtest: {bt['ticker'].nunique()} tickers")
print(summ[["method", "confidence", "n_tickers", "pooled_breach_rate", "expected_rate", "share_rejected", "zero_breach_tickers"]].to_string(index=False))
for k, v in pf["presets"].items():
    print(k, v["status"], v.get("reason", ""))
