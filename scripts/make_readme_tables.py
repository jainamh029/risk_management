"""Write the generated results block of README.md from results/*.json and results/*.csv.

    python scripts/make_readme_tables.py          # rewrite the block
    python scripts/make_readme_tables.py --check  # exit 1 if README.md differs from the results

Every number in the block comes from a results file. Nothing is typed by hand.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def r2txt(x):
    """R squared as a percentage, or 'undefined' when it is null in the results file."""
    return "undefined" if x is None else pct(x)


def num(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build() -> str:
    res = ROOT / "results"
    info = json.loads((res / "run_info.json").read_text())
    summ = json.loads((res / "var_backtest_summary.json").read_text())
    pf = json.loads((res / "portfolios.json").read_text())
    vt = pd.read_csv(res / "var_latest.csv")
    d, cfg = info["data"], info["config"]

    parts = []
    parts.append("### Data used for these results\n")
    parts.append(
        table(
            ["Item", "Value"],
            [
                ["Source", "Yahoo Finance via yfinance, split and dividend adjusted close"],
                ["Downloaded (UTC)", d["downloaded_utc"]],
                ["Date range", f"{d['first_date']} to {d['last_date']} ({d['n_dates']} trading days)"],
                ["Tickers requested", d["tickers_requested"]],
                ["Tickers returned", f"{d['tickers_returned']} ({d['stocks_returned']} stocks, {d['etfs_returned']} ETFs)"],
                ["Tickers Yahoo returned nothing for", d["tickers_failed"]],
                ["Average 13-week T-bill yield, whole range", pct(d["risk_free_all"], 3)],
            ],
        )
    )
    parts.append("\n### VaR backtest (one-day forecasts, window of %d days)\n" % cfg["window"])
    rows = []
    for r in sorted(summ, key=lambda r: (r["method"], r["confidence"])):
        rows.append(
            [
                r["method"],
                pct(r["confidence"], 0),
                r["n_tickers"],
                f"{r['total_forecasts']:,}",
                pct(r["pooled_breach_rate"]),
                pct(r["expected_rate"]),
                pct(r["share_rejected"], 1),
                r["zero_breach_tickers"],
            ]
        )
    parts.append(table(["Method", "Confidence", "Tickers", "Forecasts", "Pooled breach rate", "Expected", "Share of tickers rejected (Kupiec, 5%)", "Tickers with zero breaches"], rows))
    _m = json.loads((res / "universe_membership_summary.json").read_text())
    _n_all = summ[0]["n_tickers"]
    parts.append(
        f"\nDenominator: the {_n_all} tickers are all tickers with enough history, including {_n_all - _m['backtest']['stocks_tested']} ETFs; "
        f"the {_m['backtest']['stocks_tested']} stocks alone account for {_m['backtest']['forecast_days']:,} of the {summ[0]['total_forecasts']:,} forecasts. "
        "Each forecast is one ticker on one day."
    )

    spy = vt[vt["ticker"] == "SPY"].iloc[0]
    parts.append("\n### Latest one-day VaR and ES for SPY (most recent %d days, ending %s)\n" % (cfg["window"], info["var_latest"]["window_end"]))
    rows = []
    for m in ("historical", "normal"):
        rows.append([m] + [pct(spy[f"{m}_{k}_{c}"]) for c in (95, 99) for k in ("var", "es")])
    parts.append(table(["Method", "VaR 95%", "ES 95%", "VaR 99%", "ES 99%"], rows))
    parts.append(
        "\nVaR and ES are shown as the loss, in percent of position value, for one trading day. "
        f"{info['var_latest']['n_tickers']} tickers had a complete window; {info['var_latest']['excluded_incomplete_window']} did not and are excluded, not filled."
    )

    parts.append(
        "\n### Portfolio experiment: chosen with data to %s, evaluated on %s to %s\n" % (pf["train_end"], pf["test_start"], pf["test_end"])
    )
    spy_t = pf["benchmark_spy_test"]
    rows = []
    for name, e in pf["presets"].items():
        if e["status"] != "constructed":
            rows.append([name, "not constructed", e["n_passing_screen"], "-", "-", "-", "-", "-", "-", "-"])
            continue
        o, i, q = e["out_of_sample_test"], e["in_sample_train"], e["equal_weight_same_holdings_test"]
        rows.append(
            [
                name,
                "constructed",
                e["n_passing_screen"],
                len(e["tickers"]),
                num(i["sharpe"]),
                num(o["sharpe"]),
                pct(o["total_return"], 1),
                pct(o["max_drawdown"], 1),
                num(q["sharpe"]),
                pct(q["total_return"], 1),
            ]
        )
    rows.append(["SPY (benchmark)", "-", "-", "-", "-", num(spy_t["sharpe"]), pct(spy_t["total_return"], 1), pct(spy_t["max_drawdown"], 1), "-", "-"])
    parts.append(
        table(
            ["Preset", "Status", "Stocks passing screen", "Holdings", "Sharpe, training (in-sample)", "Sharpe, test (out-of-sample)", "Total return, test", "Max drawdown, test", "Sharpe, same holdings equal-weighted, test", "Total return, same holdings equal-weighted, test"],
            rows,
        )
    )
    reasons = [f"- {n}: {e['reason']}" for n, e in pf["presets"].items() if e["status"] != "constructed"]
    parts.append(
        "\nRisk-free rate used: %s (training) and %s (test), the average 13-week T-bill yield over each period. %d stocks had a complete training history and were eligible."
        % (pct(pf["risk_free_train"], 3), pct(pf["risk_free_test"], 3), pf["eligible_stocks"])
    )
    if reasons:
        parts.append("\n" + "\n".join(reasons))
    ml = json.loads((res / "ml_summary.json").read_text())
    mi = ml["info"]
    parts.append(
        "\n### Forecasting the %d-day return: out-of-sample evaluation\n" % mi["horizon_trading_days"]
    )
    parts.append(
        "Pooled models trained on %s samples (feature dates %s to %s; every target window ends by %s) and tested on %s samples (feature dates %s to %s). "
        "The latest forecast uses features from %s for %d stocks."
        % (f"{mi['train_samples']:,}", mi["train_first_feature_date"], mi["train_last_feature_date"], mi["train_last_target_end_date"], f"{mi['test_samples']:,}", mi["test_first_feature_date"], mi["test_last_feature_date"], mi["latest_feature_date"], mi["latest_forecasts"])
    )
    b = ml["baselines"]
    rows = []
    for name, m in ml["models"].items():
        ic = m["rank_ic_nonoverlapping"]
        rows.append(
            [
                name.replace("_", " "),
                r2txt(m["oos_r2_vs_training_mean"]),
                pct(m["direction_accuracy"], 1),
                f"{ic['mean']:.3f} (se {ic['std_error']:.3f}, t {ic['t_stat']:.2f}, n {ic['n']})",
                f"{m['rank_ic_all_dates']['mean']:.3f}",
                pct(m["quintile_spread_nonoverlapping"]["mean"], 2),
            ]
        )
    rows.append(["baseline: training-mean forecast", "0.00%", pct(b["training_mean_forecast"]["direction_accuracy"], 1), "-", "-", "-"])
    for f in ("mom_60", "mom_126"):
        ic = b[f"rank_ic_{f}_nonoverlapping"]
        rows.append([f"baseline: {f.replace('_', ' ')} momentum feature alone", "-", "-", f"{ic['mean']:.3f} (se {ic['std_error']:.3f}, t {ic['t_stat']:.2f}, n {ic['n']})", f"{b[f'rank_ic_{f}_all_dates']['mean']:.3f}", "-"])
    parts.append(
        table(
            ["Model", "R squared vs training mean", "Direction accuracy (up vs not up)", "Rank IC, non-overlapping periods", "Rank IC, all dates", "Top-fifth minus bottom-fifth return, non-overlapping"],
            rows,
        )
    )
    parts.append("\nShare of test samples with a positive %d-day return: %s." % (mi["horizon_trading_days"], pct(b["share_of_test_samples_with_positive_return"], 1)))
    mem = json.loads((res / "universe_membership_summary.json").read_text())
    src = mem["source"]["sources"]
    parts.append("\n### Ticker list compared with S&P 500 membership\n")
    parts.append(
        f"Source: Wikipedia's current constituents (revision {src[0]['revision_id']}) and table of index changes (revision {src[1]['revision_id']}), fetched {mem['source']['fetched_utc']}. "
        "This is a volunteer-maintained record, not S&P Dow Jones Indices' own. Membership on a past date is rebuilt by undoing each recorded change from today's list; "
        f"the rebuilt index has between {mem['index_size_rebuilt']['min']} and {mem['index_size_rebuilt']['max']} members on every date of the price window, as a complete table would give.\n"
    )
    cat = mem["categories_stocks"]
    nm = mem["not_member_with_data"]
    cov = mem["coverage_of_index_by_price_data"]
    parts.append(
        table(
            ["Item", "Value"],
            [
                ["Tickers in the list", f"{mem['tickers_in_list']} ({mem['stocks_in_list']} stocks, {mem['etfs_in_list']} ETFs)"],
                ["Stocks that are index members today", f"{mem['current_members_in_list']} of {mem['stocks_in_list']}"],
                ["Stocks that were members on every date of the price window", cat.get("member on every date", 0)],
                ["Stocks that were members on some dates only", cat.get("member on some dates", 0)],
                ["Stocks that were not members on any date, matching by symbol", f"{cat.get('not a member on any date', 0)} ({len(nm)} with price data: {', '.join(nm) or 'none'})"],
                ["Current index members that are not in the list", len(mem["current_members_not_in_list"])],
                ["Index members with price data in this snapshot", f"{pct(cov['first_date'], 1)} on {mem['window_start']}, {pct(cov['last_date'], 1)} on {mem['window_end']}, {pct(cov['mean'], 1)} on average"],
                ["Backtested forecast days on which the stock was an index member", f"{pct(mem['backtest']['share_while_member'], 1)} of {mem['backtest']['forecast_days']:,}"],
                ["Source tables that contradict each other", ", ".join(mem["source_inconsistencies"]["tickers"]) or "none"],
            ],
        )
    )
    rows = []
    for r in mem["pooled_breach_rate_members_only"]:
        rows.append([r["method"], pct(r["confidence"], 0), pct(r["breach_rate_all_days"], 3), pct(r["breach_rate_member_days_only"], 3), pct(r["expected_rate"])])
    parts.append(
        f"\nPooled breach rate over all backtested days, and over only the days when the stock was an index member. Both columns use stocks only "
        f"({mem['backtest']['stocks_tested']} stocks, {mem['backtest']['forecast_days']:,} forecasts, of which {mem['backtest']['forecast_days_while_member']:,} fall on member days), "
        "because ETFs have no index membership; the \"All days\" column therefore differs slightly from the table above, which includes the ETFs:\n"
    )
    parts.append(table(["Method", "Confidence", "All days", "Member days only", "Expected"], rows))
    parts.append(
        "\nMatching is by ticker symbol. A company that changed its symbol cannot be matched, so some of the stocks counted as non-members are renamed members; "
        "the history table itself records, for example, HFC (the earlier symbol of DINO) as a member until 2021-06-04, and describes Equity Residential as renamed. These were not resolved."
    )
    return "\n".join(parts)


def render(readme: str, block: str) -> str:
    a, b = readme.index(START) + len(START), readme.index(END)
    return readme[:a] + "\n\n" + block + "\n\n" + readme[b:]


if __name__ == "__main__":
    path = ROOT / "README.md"
    text = path.read_text()
    new = render(text, build())
    if "--check" in sys.argv:
        sys.exit(0 if new == text else 1)
    path.write_text(new)
    print("README results block updated")
