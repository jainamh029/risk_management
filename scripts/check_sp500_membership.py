"""Compare the ticker list with S&P 500 membership.

Source: Wikipedia, "List of S&P 500 companies" (current constituents) and
"Historical components of the S&P 500" (table of additions and removals with
effective dates). Wikipedia is volunteer-maintained and is not S&P Dow Jones
Indices' own record, so this is a cross-check, not an official verdict.
Text is available under CC BY-SA 4.0; only tickers, dates and reasons are kept.

    python scripts/check_sp500_membership.py --fetch   # download and store the source tables (network)
    python scripts/check_sp500_membership.py           # analyse the stored tables (offline)

Membership on a past date is rebuilt by starting from today's constituents and
undoing every change that took effect after that date. The rebuilt index size
is reported as a consistency check: a complete table keeps it near 500.
"""
import argparse
import io
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA, RES = ROOT / "data", ROOT / "results"
UA = {"User-Agent": "risk_management-research/1.0 (educational; GitHub jainamh029)"}
URL_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_HISTORY = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def norm(t):
    return None if pd.isna(t) else str(t).strip().replace(".", "-")


def page(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode("utf-8")
    rev = re.search(r'"wgRevisionId":(\d+)', html).group(1)
    edited = re.search(r"last edited on ([^<,]+)", html).group(1)
    return html, rev, edited


def fetch():
    h1, rev1, ed1 = page(URL_CURRENT)
    h2, rev2, ed2 = page(URL_HISTORY)
    cur = pd.read_html(io.StringIO(h1))[0]
    out = pd.DataFrame({"ticker": cur["Symbol"].map(norm), "date_added": cur["Date added"]})
    out.to_csv(DATA / "sp500_current.csv", index=False)
    ch = pd.read_html(io.StringIO(h2))[0]
    ch.columns = ["date", "added", "added_name", "removed", "removed_name", "reason", "refs"]
    # every row must be a real change: an unparseable date raises instead of a row being silently dropped
    ch["date"] = pd.to_datetime(ch["date"], format="%B %d, %Y", errors="raise")
    assert ch["added"].notna().any() and ch["removed"].notna().any()
    ch[["date", "added", "removed", "reason"]].assign(added=ch["added"].map(norm), removed=ch["removed"].map(norm)).sort_values("date").to_csv(DATA / "sp500_changes.csv", index=False)
    meta = {
        "sources": [
            {"url": URL_CURRENT, "revision_id": rev1, "last_edited": ed1},
            {"url": URL_HISTORY, "revision_id": rev2, "last_edited": ed2},
        ],
        "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "current_constituents_rows": int(len(out)),
        "changes_rows": int(len(ch)),
        "license": "Wikipedia text is CC BY-SA 4.0; only tickers, dates and reasons are stored",
        "note": "Wikipedia is not the official S&P Dow Jones Indices record",
    }
    (DATA / "sp500_membership_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"stored {len(out)} current constituents and {len(ch)} changes (current page rev {rev1}, history page rev {rev2})")


def membership_matrix(dates: pd.DatetimeIndex, current: set, changes: pd.DataFrame) -> pd.DataFrame:
    """Boolean (date x ticker): was the ticker in the index on that date. Built by undoing later changes."""
    rows = list(changes.sort_values("date", ascending=False).itertuples(index=False))
    state, ptr, sets = set(current), 0, {}
    for d in sorted(dates, reverse=True):
        while ptr < len(rows) and rows[ptr].date > d:
            a, r = rows[ptr].added, rows[ptr].removed
            if isinstance(a, str):
                state.discard(a)
            if isinstance(r, str):
                state.add(r)
            ptr += 1
        sets[d] = frozenset(state)
    every = sorted(set().union(*sets.values()))
    col = {t: i for i, t in enumerate(every)}
    m = np.zeros((len(dates), len(every)), dtype=bool)
    for i, d in enumerate(dates):
        for t in sets[d]:
            m[i, col[t]] = True
    return pd.DataFrame(m, index=dates, columns=every)


def inconsistent_tickers(current: set, changes: pd.DataFrame, start: pd.Timestamp) -> list:
    """Tickers the two source tables contradict each other about: added on or after `start`, not in today's list,
    and never recorded as removed afterwards. The reconstruction cannot place them in the index on any date."""
    out = []
    for t, g in changes[changes["added"].notna()].groupby("added"):
        for d in g["date"][g["date"] >= start]:
            removed_later = ((changes["removed"] == t) & (changes["date"] > d)).any()
            if t not in current and not removed_later:
                out.append(t)
                break
    return sorted(out)


def analyse():
    from riskengine import data

    prices, _, meta = data.load_snapshot()
    returns = data.simple_returns(prices)
    uni = data.load_universe().set_index("ticker")
    cur = pd.read_csv(DATA / "sp500_current.csv")
    ch = pd.read_csv(DATA / "sp500_changes.csv", parse_dates=["date"])
    src = json.loads((DATA / "sp500_membership_meta.json").read_text())
    current = set(cur["ticker"])
    dates = prices.index
    M = membership_matrix(dates, current, ch)

    size = M.sum(axis=1)
    inconsistent = inconsistent_tickers(current, ch, dates[0])
    returned = set(prices.columns)
    rows = []
    for t, r in uni.iterrows():
        days = int(M[t].sum()) if t in M.columns else 0
        if bool(r["is_etf"]):
            cat = "ETF (not applicable)"
        elif days == len(dates):
            cat = "member on every date"
        elif days > 0:
            cat = "member on some dates"
        else:
            cat = "not a member on any date"
        rows.append({"ticker": t, "is_etf": bool(r["is_etf"]), "returned_data": t in returned, "current_member": t in current, "member_days": days, "window_days": len(dates), "category": cat})
    tick = pd.DataFrame(rows)
    tick.to_csv(RES / "universe_membership.csv", index=False)

    # index members on each date that the price data covers
    have = prices.notna()
    cov = []
    for d in dates:
        mem = [t for t in M.columns[M.loc[d].to_numpy()]]
        got = [t for t in mem if t in have.columns and have.at[d, t]]
        cov.append(len(got) / len(mem))
    cov = pd.Series(cov, index=dates)

    # backtest forecast days that fall on dates when the ticker was not an index member
    notna = returns.notna()
    ok = notna & (notna.shift(1).rolling(250).sum() == 250)
    stocks = [t for t in returns.columns if not bool(uni["is_etf"].get(t, False))]
    bt = pd.read_csv(RES / "var_backtest_by_ticker.csv")
    tested = sorted(set(bt["ticker"]) & set(stocks))
    total = member = 0
    for t in tested:
        f = ok[t]
        total += int(f.sum())
        member += int((f & M[t]).sum()) if t in M.columns else 0

    from riskengine import backtest, config

    members_only = []
    for method in ("historical", "normal"):
        for conf in config.CONFS:
            n_all_days = x_all = n_mem = x_mem = 0
            for t in tested:
                r = returns[t].to_numpy(dtype=float)
                f = backtest.rolling_forecasts(r, config.WINDOW, conf, method)
                valid = ~np.isnan(f) & ~np.isnan(r)
                breach = valid & (-r > f)
                mem = M[t].to_numpy() if t in M.columns else np.zeros(len(r), dtype=bool)
                n_all_days += int(valid.sum())
                x_all += int(breach.sum())
                n_mem += int((valid & mem).sum())
                x_mem += int((breach & mem).sum())
            members_only.append({"method": method, "confidence": conf, "breach_rate_all_days": x_all / n_all_days, "breach_rate_member_days_only": x_mem / n_mem, "expected_rate": 1 - conf, "forecast_days_all": n_all_days, "forecast_days_member_only": n_mem})

    n_all = len(uni)
    stocks_df = tick[~tick["is_etf"]]
    summ = {
        "source": src,
        "window_start": str(dates[0].date()),
        "window_end": str(dates[-1].date()),
        "tickers_in_list": int(n_all),
        "etfs_in_list": int(tick["is_etf"].sum()),
        "stocks_in_list": int(len(stocks_df)),
        "current_members_in_list": int(stocks_df["current_member"].sum()),
        "categories_stocks": {k: int(v) for k, v in stocks_df["category"].value_counts().items()},
        "stocks_returned_data": int(stocks_df["returned_data"].sum()),
        "stocks_no_data_by_category": {k: int(v) for k, v in stocks_df[~stocks_df["returned_data"]]["category"].value_counts().items()},
        "current_index_size": int(len(current)),
        "current_members_not_in_list": sorted(current - set(uni.index)),
        "index_size_rebuilt": {"min": int(size.min()), "max": int(size.max()), "first_date": int(size.iloc[0]), "last_date": int(size.iloc[-1])},
        "coverage_of_index_by_price_data": {"first_date": float(cov.iloc[0]), "last_date": float(cov.iloc[-1]), "mean": float(cov.mean()), "min": float(cov.min())},
        "backtest": {"stocks_tested": len(tested), "forecast_days": total, "forecast_days_while_member": member, "share_while_member": member / total},
        "pooled_breach_rate_members_only": members_only,
        "source_inconsistencies": {"tickers": inconsistent, "in_list": sorted(set(inconsistent) & set(uni.index)), "note": "added per the history table, absent from the current list, no removal recorded"},
        "not_member_with_data": sorted(stocks_df[(stocks_df["category"] == "not a member on any date") & stocks_df["returned_data"]]["ticker"]),
        "member_on_some_dates": sorted(stocks_df[stocks_df["category"] == "member on some dates"]["ticker"]),
    }
    (RES / "universe_membership_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k in ("categories_stocks", "backtest", "pooled_breach_rate_members_only", "not_member_with_data", "coverage_of_index_by_price_data", "index_size_rebuilt", "source_inconsistencies")}, indent=1))
    print("current members missing from the list:", len(summ["current_members_not_in_list"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    fetch() if a.fetch else analyse()
