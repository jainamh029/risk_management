"""The charts added to the dashboard (overview, stock explorer, portfolio correlation), checked against independent calculations.

Chart contents are read back from the Plotly specification that the app sends to the browser. Every expected value is computed here
from the frozen snapshot or the stored results with plain numpy loops, not with the dashboard's own code.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.integration.test_dashboard_extra import app, decode, figure, run_app, title  # noqa: F401  (fixture and helpers)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def metrics_csv():
    return pd.read_csv(ROOT / "results" / "metrics.csv")


def plain_returns(prices, t):
    p = prices[t].dropna().to_numpy(dtype=float)
    return p, p[1:] / p[:-1] - 1.0


# ----------------------------------------------------------------------------- overview
def test_overview_growth_chart_equals_an_independent_calculation_for_every_preset_and_spy(app, prices):
    pf = json.loads((ROOT / "results" / "portfolios.json").read_text())
    fig = figure(app, "ov_growth")
    names = [t["name"] for t in fig["data"]]
    built = [f"{n} portfolio" for n, e in pf["presets"].items() if e["status"] == "constructed"]
    assert names == built + ["SPY"]
    r = prices / prices.shift(1) - 1.0
    test = r.loc[pf["test_start"] :]
    test = test[test.index >= pd.Timestamp(pf["test_start"])]
    for trace in fig["data"]:
        if trace["name"] == "SPY":
            daily = test["SPY"].fillna(0.0).to_numpy()
        else:
            e = pf["presets"][trace["name"].split(" ")[0]]
            daily = sum(test[t].fillna(0.0).to_numpy() * w for t, w in e["weights"].items())
        wealth = 1.0
        expected = []
        for d in daily:
            wealth *= 1.0 + d
            expected.append(wealth)
        assert np.allclose(decode(trace["y"]), expected, rtol=1e-12), trace["name"]
    assert title(fig).startswith(f"Growth of 1 since {pf['test_start']}")


@pytest.mark.parametrize("label,conf", [("95%", 0.95), ("99%", 0.99)])
def test_overview_breach_histogram_counts_every_ticker_once_per_method_and_marks_the_expected_rate(app, label, conf):
    app.radio(key="ov_conf").set_value(label)
    app.run()
    assert not app.exception
    fig = figure(app, "ov_breach_hist")
    stored = pd.read_csv(ROOT / "results" / "var_backtest_by_ticker.csv")
    for trace, method in zip(fig["data"], ("historical", "normal")):
        rates = stored[(stored["method"] == method) & (stored["confidence"] == conf)]["breach_rate"].to_numpy()
        assert np.allclose(np.sort(decode(trace["x"])), np.sort(rates), atol=1e-15) and len(decode(trace["x"])) == 458
    line = fig["layout"]["shapes"][0]
    assert line["x0"] == pytest.approx(1 - conf) and f"{1 - conf:.0%}" in fig["layout"]["annotations"][0]["text"]
    assert title(fig) == f"Breach rate of each ticker at {label}"


# ----------------------------------------------------------------------------- stock explorer: the map
def test_the_risk_return_map_plots_exactly_the_stored_statistics(app, metrics_csv):
    fig = figure(app, "se_map")
    stocks, etfs = metrics_csv[~metrics_csv["is_etf"]], metrics_csv[metrics_csv["is_etf"]]
    by_name = {t["name"]: t for t in fig["data"] if not t["name"].startswith("selected")}
    assert set(by_name) == {"stocks", "ETFs"}
    for name, frame in (("stocks", stocks), ("ETFs", etfs)):
        t = by_name[name]
        assert list(t["text"]) == list(frame["ticker"])
        assert np.allclose(decode(t["x"]), frame["annual_vol"].to_numpy(), atol=1e-15)
        assert np.allclose(decode(t["y"]), frame["mean_annual_return"].to_numpy(), atol=1e-15)
        assert t["marker"]["symbol"] == {"stocks": "circle", "ETFs": "diamond"}[name]  # shape as well as colour
    spy = metrics_csv[metrics_csv["ticker"] == "SPY"].iloc[0]
    sel = next(t for t in fig["data"] if t["name"] == "selected: SPY")
    assert decode(sel["x"])[0] == pytest.approx(spy["annual_vol"]) and decode(sel["y"])[0] == pytest.approx(spy["mean_annual_return"])
    assert len(by_name["stocks"]["text"]) + len(by_name["ETFs"]["text"]) == len(metrics_csv) == 458


@pytest.mark.parametrize("group,expected", [("Stocks", {"stocks"}), ("ETFs", {"ETFs"}), ("Both", {"stocks", "ETFs"})])
def test_the_map_group_filter_changes_what_is_plotted(app, group, expected):
    app.radio(key="se_group").set_value(group)
    app.run()
    assert not app.exception
    assert {t["name"] for t in figure(app, "se_map")["data"] if not t["name"].startswith("selected")} == expected


def test_the_selected_ticker_cards_equal_the_stored_row(app, metrics_csv):
    app.selectbox(key="se_ticker").set_value("AAPL")
    app.run()
    assert not app.exception
    row = metrics_csv[metrics_csv["ticker"] == "AAPL"].iloc[0]
    cards = {m.label: m.value for m in app.metric}
    assert cards["Mean annual return"] == f"{100 * row['mean_annual_return']:.1f}%"
    assert cards["Annual volatility"] == f"{100 * row['annual_vol']:.1f}%"
    assert cards["Sharpe ratio"] == f"{row['sharpe']:.2f}"
    assert cards["Maximum drawdown"] == f"{100 * row['max_drawdown']:.1f}%"
    assert cards["Beta against SPY"] == f"{row['beta_vs_spy']:.2f}"
    assert title(figure(app, "se_growth")).startswith("AAPL:")


# ----------------------------------------------------------------------------- stock explorer: the four charts for one ticker
def test_growth_drawdown_and_rolling_volatility_equal_independent_loops(app, prices):
    app.selectbox(key="se_ticker").set_value("MSFT")
    app.run()
    p, r = plain_returns(prices, "MSFT")
    growth = decode(figure(app, "se_growth")["data"][0]["y"])
    assert np.allclose(growth, p / p[0], rtol=1e-12)
    peak, dd = -math.inf, []
    for g in growth:
        peak = max(peak, g)
        dd.append(g / peak - 1.0)
    assert np.allclose(decode(figure(app, "se_drawdown")["data"][0]["y"]), dd, atol=1e-12)
    assert min(dd) == pytest.approx(pd.read_csv(ROOT / "results" / "metrics.csv").set_index("ticker").loc["MSFT", "max_drawdown"], abs=1e-12)  # the chart's deepest point is the stored maximum drawdown
    vol = decode(figure(app, "se_rollvol")["data"][0]["y"])
    window = r[-60:]
    mean = sum(window) / 60
    expected_last = math.sqrt(sum((x - mean) ** 2 for x in window) / 59) * math.sqrt(252)
    assert vol[-1] == pytest.approx(expected_last, rel=1e-12) and len(vol) == len(r) - 59


def test_the_return_histogram_uses_every_return_and_marks_the_independent_historical_var(app, prices):
    app.selectbox(key="se_ticker").set_value("AAPL")
    app.run()
    _, r = plain_returns(prices, "AAPL")
    fig = figure(app, "se_hist")
    x = decode(fig["data"][0]["x"])
    assert len(x) == len(r) and np.allclose(np.sort(x), np.sort(r), atol=1e-15)
    losses = np.sort(-r)
    pos = 0.99 * (len(losses) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    var99 = losses[lo] + (pos - lo) * (losses[hi] - losses[lo])
    line = fig["layout"]["shapes"][0]
    assert line["x0"] == pytest.approx(-var99, abs=1e-12) and f"{var99:.1%}" in fig["layout"]["annotations"][0]["text"]
    mu, sd = r.mean(), r.std(ddof=1)
    xs, ys = decode(fig["data"][1]["x"]), decode(fig["data"][1]["y"])
    assert ys.max() == pytest.approx(1.0 / (sd * math.sqrt(2 * math.pi)), rel=2e-3) and abs(xs[np.argmax(ys)] - mu) < (xs[1] - xs[0]) * 2
    assert "excess kurtosis" in " ".join(c.value for c in app.caption)


@pytest.mark.parametrize("period,years", [("Last 3 years", 3), ("Last 1 year", 1)])
def test_the_period_control_limits_every_chart_to_the_period(app, prices, period, years):
    app.radio(key="se_period").set_value(period)
    app.run()
    assert not app.exception
    last = prices["SPY"].dropna().index[-1]
    cutoff = last - pd.DateOffset(years=years)
    growth = figure(app, "se_growth")["data"][0]
    assert decode(growth["y"])[0] == pytest.approx(1.0)  # rebased at the start of the period
    first = pd.Timestamp(growth["x"][0])
    assert cutoff <= first <= cutoff + pd.Timedelta(days=7)
    assert len(growth["x"]) < len(prices["SPY"].dropna())


def test_a_ticker_with_too_little_history_gives_a_message_not_an_exception(app, prices, metrics_csv):
    counts = prices.notna().sum().sort_values()
    short = counts.index[0]
    app.selectbox(key="se_ticker").set_value(short)
    app.run()
    assert not app.exception, [e.value[:100] for e in app.exception]
    if counts.iloc[0] < 130:
        assert any("too few for the charts" in w.value for w in app.warning)
    if short not in set(metrics_csv["ticker"]):
        assert any("fewer than 250 daily returns" in w.value for w in app.warning)


def test_every_stock_explorer_chart_has_a_data_table_and_no_text_reads_nan(app):
    labels = [e.label for e in app.expander]
    assert "Data table: the statistics behind the map" in labels and "Data table: values behind the four charts" in labels
    assert "Data table: growth of 1 and per-ticker breach rates behind these charts" in labels
    text = " ".join([str(m.value) for m in app.metric] + [c.value for c in app.caption])
    assert "nan" not in text.lower().split() and "None" not in text


# ----------------------------------------------------------------------------- portfolio correlation heatmap
def test_the_portfolio_heatmap_equals_an_independent_correlation_of_the_training_returns(app, prices):
    pf = json.loads((ROOT / "results" / "portfolios.json").read_text())
    app.radio(key="pf_preset").set_value("balanced")
    app.run()
    assert not app.exception
    tickers = pf["presets"]["balanced"]["tickers"]
    heat = figure(app, "pf_stored_corr")["data"][0]
    order = list(heat["x"])
    assert sorted(order) == sorted(tickers) and list(heat["y"]) == order  # the holdings, ordered by weight as in the weights chart
    n = len(order)
    z = decode(heat["z"]).reshape(tuple(int(v) for v in heat["z"]["shape"].split(",")))
    r = (prices / prices.shift(1) - 1.0).loc[: pf["train_end"], order].to_numpy()
    r = r[~np.isnan(r).all(axis=1)]
    assert z.shape == (n, n) and np.allclose(z, np.corrcoef(r.T), atol=1e-12)
    assert np.allclose(np.diag(z), 1.0) and np.allclose(z, z.T)


def test_the_zoom_option_only_changes_the_axis_range_not_the_plotted_data(app, metrics_csv):
    zoomed = figure(app, "se_map")
    x0, x1 = zoomed["layout"]["xaxis"]["range"]
    y0, y1 = zoomed["layout"]["yaxis"]["range"]
    qx, qy = metrics_csv["annual_vol"].quantile([0.01, 0.99]), metrics_csv["mean_annual_return"].quantile([0.01, 0.99])
    assert x0 == pytest.approx(max(0.0, qx.iloc[0] - 0.15 * (qx.iloc[1] - qx.iloc[0]))) and x1 == pytest.approx(qx.iloc[1] + 0.15 * (qx.iloc[1] - qx.iloc[0]))
    assert y0 == pytest.approx(qy.iloc[0] - 0.15 * (qy.iloc[1] - qy.iloc[0])) and y1 == pytest.approx(qy.iloc[1] + 0.15 * (qy.iloc[1] - qy.iloc[0]))
    outside = int(((metrics_csv["annual_vol"] < x0) | (metrics_csv["annual_vol"] > x1) | (metrics_csv["mean_annual_return"] < y0) | (metrics_csv["mean_annual_return"] > y1)).sum())
    assert outside > 0 and any(f"{outside} ticker(s) lie outside the zoomed view" in c.value for c in app.caption)
    app.checkbox(key="se_zoom").set_value(False)
    app.run()
    full = figure(app, "se_map")
    assert "range" not in full["layout"]["xaxis"] and not any("lie outside the zoomed view" in c.value for c in app.caption)
    for a, b in zip(zoomed["data"], full["data"]):
        assert a["name"] == b["name"] and np.array_equal(decode(a["x"]), decode(b["x"])) and np.array_equal(decode(a["y"]), decode(b["y"]))  # every ticker is still plotted
