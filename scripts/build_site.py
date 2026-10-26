"""Build the static report that GitHub Pages serves (docs/index.html).

    python scripts/build_site.py            # write docs/ (the report and the interactive dashboard)
    python scripts/build_site.py --check    # exit 1 if anything in docs/ differs from a fresh build

GitHub Pages serves static files only, so it cannot run a Python server. It publishes two things:

* docs/index.html, a report with the main charts, fixed in time. Every figure comes from results/ and the frozen price
  snapshot, and the page loads nothing from the network (the plotting library is copied next to it).
* docs/app/, the real Streamlit dashboard (dashboard/app.py, unchanged) running inside the visitor's browser with stlite
  (Streamlit on Pyodide, WebAssembly). The page downloads the dashboard code, the riskengine package, results/ and the
  price snapshot from docs/app/files/ and loads stlite and Pyodide from the jsDelivr CDN. Nothing runs on a server.

The build is deterministic, so the checked-in files can be compared with a fresh build.
"""
import argparse
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import config, data, ml, portfolio, var  # noqa: E402

BLUE, LIGHT_BLUE, AMBER, RED, SLATE = "#1E40AF", "#3B82F6", "#D97706", "#DC2626", "#64748B"
OUT = ROOT / "docs"
REPO = "https://github.com/jainamh029/risk_management"


def pct(x, d=2):
    return "undefined" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, fmt=".2f"):
    return "undefined" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def layout(fig, title, yfmt=".1%", ytitle=None, height=400):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=90, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1E293B"),
    )
    fig.update_yaxes(tickformat=yfmt, title=ytitle, gridcolor="#E2E8F0")
    fig.update_xaxes(gridcolor="#E2E8F0")
    return fig


def chart(fig, div_id):
    # explicit ids keep the output identical from one build to the next
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id, config={"responsive": True, "displaylogo": False})


def table(frame, caption):
    head = "".join(f"<th scope='col'>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in frame.itertuples(index=False))
    return f"<div class='scroll'><table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def build() -> str:
    res = ROOT / "results"
    info = json.loads((res / "run_info.json").read_text())
    summary = pd.DataFrame(json.loads((res / "var_backtest_summary.json").read_text()))
    by_ticker = pd.read_csv(res / "var_backtest_by_ticker.csv")
    pf = json.loads((res / "portfolios.json").read_text())
    mlr = json.loads((res / "ml_summary.json").read_text())
    ic = pd.read_csv(res / "ml_rank_ic_by_date.csv", parse_dates=["date"]).set_index("date")
    metrics = pd.read_csv(res / "metrics.csv")
    mem = json.loads((res / "universe_membership_summary.json").read_text())
    prices, _, meta = data.load_snapshot()
    returns = data.simple_returns(prices)

    s = summary.set_index(["method", "confidence"])
    h99, n99 = s.loc[("historical", 0.99)], s.loc[("normal", 0.99)]
    ens = mlr["models"]["ensemble"]
    d = info["data"]

    # ---- charts
    fig1 = go.Figure()
    labels = [f"{m} {int(c * 100)}%" for m, c in zip(summary["method"], summary["confidence"])]
    fig1.add_trace(go.Bar(x=labels, y=summary["pooled_breach_rate"], name="Observed breach rate", marker=dict(color=BLUE, pattern=dict(shape="/")), hovertemplate="%{x}: %{y:.2%}<extra>observed</extra>"))
    fig1.add_trace(go.Scatter(x=labels, y=summary["expected_rate"], mode="markers", name="Expected by a correct model", marker=dict(color="#000000", symbol="line-ew", size=40, line=dict(width=3, color="#000000")), hovertemplate="%{x}: %{y:.2%}<extra>expected</extra>"))
    layout(fig1, "Pooled breach rate against the expected rate", ytitle="share of forecast days with a breach").update_layout(hovermode="closest")

    bt99 = by_ticker[by_ticker["confidence"] == 0.99]
    fig2 = go.Figure()
    for m, color, shape in zip(var.METHODS, (BLUE, AMBER), ("/", "x")):
        fig2.add_trace(go.Histogram(x=bt99[bt99["method"] == m]["breach_rate"], xbins=dict(start=0, size=0.0025), name=m, opacity=0.65, marker=dict(color=color, pattern=dict(shape=shape)), hovertemplate="breach rate %{x:.2%}: %{y} tickers<extra>" + m + "</extra>"))
    fig2.add_vline(x=0.01, line=dict(color="#000000", dash="dash", width=2), annotation_text="expected 1%", annotation_position="top right")
    layout(fig2, "Breach rate of each ticker at 99%", yfmt="d", ytitle="number of tickers").update_layout(barmode="overlay", hovermode="closest")
    fig2.update_xaxes(tickformat=".1%", title="share of forecast days with a breach")

    test = returns.dropna(how="all").loc[config.TEST_START :]
    fig3 = go.Figure()
    dash = {"balanced": ("solid", BLUE), "aggressive": ("dash", AMBER), "conservative": ("dashdot", RED)}
    for name, e in pf["presets"].items():
        if e["status"] == "constructed":
            daily, _ = portfolio.portfolio_returns(test, pd.Series(e["weights"]))
            g = (1 + daily).cumprod()
            fig3.add_trace(go.Scatter(x=g.index, y=g, name=f"{name} portfolio", line=dict(color=dash[name][1], dash=dash[name][0], width=2.5), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + name + "</extra>"))
    spy = (1 + test[config.MARKET].fillna(0)).cumprod()
    fig3.add_trace(go.Scatter(x=spy.index, y=spy, name="SPY", line=dict(color=SLATE, dash="dot", width=2.5), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>SPY</extra>"))
    layout(fig3, f"Growth of 1 since {pf['test_start']} (test period; daily constant weights, no transaction costs)", yfmt=".2f", ytitle="growth of 1")

    fig4 = go.Figure()
    lo_v, hi_v = metrics["annual_vol"].quantile([0.01, 0.99])
    lo_r, hi_r = metrics["mean_annual_return"].quantile([0.01, 0.99])
    for flag, label, symbol in ((False, "stocks", "circle"), (True, "ETFs", "diamond")):
        g = metrics[metrics["is_etf"] == flag]
        fig4.add_trace(
            go.Scatter(
                x=g["annual_vol"], y=g["mean_annual_return"], mode="markers", name=label, text=g["ticker"],
                marker=dict(symbol=symbol, size=9 if flag else 7, color=g["sharpe"], colorscale=[[0, "#B45309"], [0.5, "#E2E8F0"], [1, "#1E40AF"]], cmin=-1.0, cmax=2.0, cmid=0.5, line=dict(width=0.6, color="#475569"), opacity=0.85, showscale=not flag, colorbar=dict(title="Sharpe", thickness=12, len=0.8)),
                customdata=np.stack([g["sharpe"], g["max_drawdown"]], axis=1),
                hovertemplate="<b>%{text}</b><br>return %{y:.1%}, volatility %{x:.1%}<br>Sharpe %{customdata[0]:.2f}, max drawdown %{customdata[1]:.1%}<extra>" + label + "</extra>",
            )
        )
    layout(fig4, "Risk and return of every ticker (whole sample, annualised)", yfmt=".0%", ytitle="mean annual return", height=520).update_layout(hovermode="closest")
    px_, py_ = 0.15 * (hi_v - lo_v), 0.15 * (hi_r - lo_r)
    fig4.update_xaxes(tickformat=".0%", title="annualised volatility", range=[max(0.0, lo_v - px_), hi_v + px_])
    fig4.update_yaxes(range=[lo_r - py_, hi_r + py_])
    outside = int(((metrics["annual_vol"] < max(0.0, lo_v - px_)) | (metrics["annual_vol"] > hi_v + px_) | (metrics["mean_annual_return"] < lo_r - py_) | (metrics["mean_annual_return"] > hi_r + py_)).sum())

    fig5 = go.Figure()
    for m, dash_, color in (("ensemble", "dot", "#0F172A"),):
        fig5.add_trace(go.Scatter(x=ic.index, y=ic[m], name=m, line=dict(color=color, dash=dash_, width=1.6), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + m + "</extra>"))
    indep = ml.nonoverlapping(ic["ensemble"], prices.index)
    fig5.add_trace(go.Scatter(x=indep.index, y=indep, mode="markers", name="ensemble, independent periods", marker=dict(color=RED, size=9, symbol="diamond"), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>independent period</extra>"))
    fig5.add_hline(y=0, line=dict(color="#000000", width=1))
    layout(fig5, "Rank IC by date: forecast against realised 63-day return across stocks", yfmt=".2f", ytitle="rank IC")

    # ---- text and tables
    findings = []
    for m in ("historical", "normal"):
        for c in (0.95, 0.99):
            r = s.loc[(m, c)]
            findings.append(f"{m.capitalize()} VaR at {int(c * 100)}%: breached on {pct(r['pooled_breach_rate'])} of forecast days (expected {pct(r['expected_rate'])}); the Kupiec test rejects {pct(r['share_rejected'], 1)} of tickers.")
    built = [n for n, e in pf["presets"].items() if e["status"] == "constructed"]
    skipped = [n for n, e in pf["presets"].items() if e["status"] != "constructed"]
    findings.append(
        f"Portfolio presets built from data up to {pf['train_end']} and tested from {pf['test_start']}: {len(built)} constructed ({', '.join(built) or 'none'}); "
        f"{len(skipped)} not constructed because too few stocks passed the screen ({', '.join(skipped) or 'none'})."
    )
    r2 = ens["oos_r2_vs_training_mean"]
    r2s = "undefined" if r2 is None else f"{pct(r2)} ({'worse' if r2 < 0 else 'better'} than always predicting the training average)"
    findings.append(
        f"Forecast model (ensemble): out-of-sample R squared {r2s}; direction accuracy {pct(ens['direction_accuracy'], 1)} against {pct(mlr['baselines']['share_of_test_samples_with_positive_return'], 1)} of test samples going up; "
        f"rank IC {num(ens['rank_ic_nonoverlapping']['mean'], '.3f')} with t statistic {num(ens['rank_ic_nonoverlapping']['t_stat'])} over {ens['rank_ic_nonoverlapping']['n']} non-overlapping periods."
    )

    bt_tbl = pd.DataFrame(
        {
            "Method": labels,
            "Tickers": summary["n_tickers"].astype(int),
            "Forecasts": summary["total_forecasts"].map(lambda v: f"{int(v):,}"),
            "Pooled breach rate": summary["pooled_breach_rate"].map(pct),
            "Expected": summary["expected_rate"].map(pct),
            "Share rejected (Kupiec, 5%)": summary["share_rejected"].map(lambda v: pct(v, 1)),
        }
    )
    spy_b = pf["benchmark_spy_test"]
    prow = []
    for n in built:
        o = pf["presets"][n]["out_of_sample_test"]
        prow.append([f"{n} (test)", pct(o["mean_annual_return"], 1), pct(o["annual_vol"], 1), num(o["sharpe"]), pct(o["max_drawdown"], 1), pct(o["total_return"], 1)])
    prow.append(["SPY (test)", pct(spy_b["mean_annual_return"], 1), pct(spy_b["annual_vol"], 1), num(spy_b["sharpe"]), pct(spy_b["max_drawdown"], 1), pct(spy_b["total_return"], 1)])
    pf_tbl = pd.DataFrame(prow, columns=["Portfolio", "Mean annual return", "Annual volatility", "Sharpe ratio", "Maximum drawdown", "Total return"])
    mrows = []
    for name, m in mlr["models"].items():
        ic_ = m["rank_ic_nonoverlapping"]
        mrows.append([name.replace("_", " "), pct(m["oos_r2_vs_training_mean"]), pct(m["direction_accuracy"], 1), num(ic_["mean"], ".3f"), num(ic_["t_stat"]), ic_["n"]])
    ml_tbl = pd.DataFrame(mrows, columns=["Model", "R squared vs training mean", "Direction accuracy", "Rank IC (non-overlapping)", "t statistic", "Periods"])

    cards = [
        ("Tickers backtested", f"{int(h99['n_tickers'])}", ""),
        ("Historical 99% VaR breach rate", pct(h99["pooled_breach_rate"]), f"expected {pct(h99['expected_rate'])}"),
        ("Normal 99% VaR breach rate", pct(n99["pooled_breach_rate"]), f"expected {pct(n99['expected_rate'])}"),
        ("Forecast model R squared (test)", "undefined" if r2 is None else pct(r2), "against the training mean"),
    ]
    card_html = "".join(f"<div class='card'><div class='label'>{html.escape(a)}</div><div class='value'>{html.escape(b)}</div><div class='sub'>{html.escape(c)}</div></div>" for a, b, c in cards)

    limits = [
        f"The ticker list is hand-typed and is not the S&amp;P 500. Compared with Wikipedia's record of index membership, {mem['current_members_in_list']} of its {mem['stocks_in_list']} stocks are current members, {len(mem['current_members_not_in_list'])} current members are missing from it, and Yahoo returned no data for {len(meta['tickers_failed'])} of its tickers. Survivorship bias remains and its size is not measured.",
        "Prices are Yahoo Finance split- and dividend-adjusted closes, used as downloaded; they are not strictly point-in-time.",
        "A 99% VaR from a 250-day window rests on about 2.5 days in the tail, so it is noisy; the per-ticker tests are not independent of each other.",
        "The portfolio and forecasting tests use one train/test split and a short test period. No transaction costs are modelled.",
        "The forecasting model shows no demonstrated skill on the test period. Nothing here recommends buying or selling any security.",
    ]

    css = """
:root{--ink:#0F172A;--muted:#475569;--line:#DBEAFE;--bg:#F8FAFC;--card:#FFFFFF;--primary:#1E40AF;--accent:#D97706}
*{box-sizing:border-box}
html{overflow-x:hidden}
body{margin:0;overflow-x:hidden;background:var(--bg);color:#1E293B;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;line-height:1.55}
a{color:var(--primary)}a:focus-visible,button:focus-visible{outline:3px solid var(--primary);outline-offset:2px}
header,main,footer{max-width:1200px;margin:0 auto;padding:0 16px}
header{padding-top:28px}
h1{margin:0 0 6px;font-size:2rem;letter-spacing:-.02em;color:var(--ink);border-left:6px solid var(--primary);padding-left:12px;line-height:1.15}
h2{margin:36px 0 10px;font-size:1.3rem;color:var(--ink)}
.lead,.muted{color:var(--muted)}
.notice{background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;padding:12px 14px;margin:16px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--primary);border-radius:10px;padding:12px 14px;box-shadow:0 1px 2px rgba(15,23,42,.06)}
.label{color:var(--muted);font-size:.85rem;font-weight:600}.value{color:var(--ink);font-size:1.8rem;font-weight:700;font-variant-numeric:tabular-nums}.sub{color:var(--muted);font-size:.8rem;min-height:1.1em}
.panel{overflow:hidden;background:var(--card);border:1px solid #E2E8F0;border-radius:10px;padding:6px;margin:14px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:14px}
.scroll{overflow-x:auto;margin:10px 0}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid #E2E8F0;font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--muted);font-size:.85rem;padding:6px 2px}
th,td{padding:8px 10px;border-bottom:1px solid #E2E8F0;text-align:left;white-space:nowrap}
th{background:#E9EEF6;color:var(--ink);font-size:.85rem}
ul{padding-left:20px}li{margin:6px 0}
code,pre{background:#E9EEF6;border-radius:6px;font-size:.9rem}code{padding:1px 5px}pre{padding:10px 12px;overflow-x:auto}
footer{padding:28px 16px 40px;color:var(--muted);font-size:.9rem}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk dashboard report</title>
<link rel="icon" href="data:,">
<meta name="description" content="Value at Risk backtests, portfolio experiment and forecasting evaluation on real Yahoo Finance prices.">
<style>{css}</style>
<script src="plotly.min.js"></script>
</head>
<body>
<header>
<h1>Risk dashboard report</h1>
<p class="lead">Real daily prices from Yahoo Finance, {d['first_date']} to {d['last_date']}, downloaded {meta['downloaded_utc']}. {d['tickers_returned']} of {d['tickers_requested']} requested tickers returned data. Educational research tool, not investment advice.</p>
<div class="notice"><strong>This page is a fixed report.</strong> It shows the numbers and main charts as they were when the page was built. <a href="app/"><strong>Open the interactive dashboard</strong></a> to choose any ticker, window and confidence level: it is the real Streamlit dashboard, running in your browser (the first load takes about a minute). You can also run it on your own computer: see the end of this page.</div>
</header>
<main>
<section aria-labelledby="h-over"><h2 id="h-over">Headline numbers</h2>
<div class="cards">{card_html}</div>
<div class="grid2">
<div class="panel">{chart(fig3, "c-growth")}</div>
<div class="panel">{chart(fig2, "c-hist")}</div>
</div></section>
<section aria-labelledby="h-find"><h2 id="h-find">What the real data shows</h2>
<ul>{"".join(f"<li>{html.escape(f)}</li>" for f in findings)}</ul></section>
<section aria-labelledby="h-var"><h2 id="h-var">VaR backtest</h2>
<p class="muted">Each day's one-day VaR forecast uses only the previous 250 returns. A breach is a day on which the loss was bigger than the forecast. The Kupiec test asks whether the breach count is consistent with the stated rate.</p>
<div class="panel">{chart(fig1, "c-bars")}</div>
{table(bt_tbl, "Backtest summary, all tickers and forecast days")}</section>
<section aria-labelledby="h-map"><h2 id="h-map">Every ticker: risk and return</h2>
<div class="panel">{chart(fig4, "c-map")}</div>
<p class="muted">Colour shows the Sharpe ratio and shape shows stocks (circles) or ETFs (diamonds). The view is zoomed to the 1st to 99th percentile of each axis; {outside} ticker(s) lie outside it and are in <code>results/metrics.csv</code>.</p></section>
<section aria-labelledby="h-pf"><h2 id="h-pf">Portfolio experiment</h2>
<p class="muted">Holdings and weights were chosen with data up to {pf['train_end']} only, then held fixed; returns from {pf['test_start']} to {pf['test_end']} assume daily rebalancing to those weights. Screens are never relaxed.</p>
{table(pf_tbl, "Out-of-sample test period")}</section>
<section aria-labelledby="h-ml"><h2 id="h-ml">Forecasting model</h2>
<p class="muted">A random forest and a gradient-boosting model forecast each stock's 63-day return. Only the diamonds below, 63 trading days apart, are independent observations.</p>
{table(ml_tbl, "Out-of-sample evaluation")}
<div class="panel">{chart(fig5, "c-ic")}</div></section>
<section aria-labelledby="h-lim"><h2 id="h-lim">Limits to keep in mind</h2><ul>{"".join(f"<li>{x}</li>" for x in limits)}</ul></section>
<section aria-labelledby="h-run"><h2 id="h-run">Run the interactive dashboard</h2>
<pre>git clone {REPO}.git
cd risk_management
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py       # downloads the real prices (a few minutes)
python scripts/run_pipeline.py
streamlit run dashboard/app.py</pre>
<p>Source code, tests, method and validation record: <a href="{REPO}">{REPO}</a>.</p></section>
</main>
<footer>Educational research tool. Not investment advice. Past data does not predict future results.</footer>
</body>
</html>
"""
    return page


STLITE = "1.12.0"  # the @stlite/browser version the app was tested with; a newer one is not picked up silently
APP_REQUIREMENTS = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "plotly==7.1.0"]  # all but plotly come with Pyodide; plotly is fetched from PyPI when the page loads
APP_FILES = (
    [Path("dashboard/app.py"), Path(".streamlit/config.toml")]
    + [Path("riskengine") / p.name for p in sorted((ROOT / "riskengine").glob("*.py"))]
    + [Path("data") / n for n in ("prices.parquet", "tbill_13w.parquet", "prices_meta.json", "universe.csv")]
    + [Path("results") / n for n in (
        "run_info.json", "var_backtest_summary.json", "var_backtest_by_ticker.csv", "portfolios.json", "ml_summary.json",
        "ml_rank_ic_by_date.csv", "ml_latest_forecasts.csv", "ml_feature_importance.csv", "metrics.csv", "universe_membership_summary.json",
    )]
)


def build_app() -> dict[str, bytes]:
    """The files of docs/app/: the stlite page and a byte-for-byte copy of everything the dashboard reads."""
    out = {f"files/{p.as_posix()}": (ROOT / p).read_bytes() for p in APP_FILES}
    mount = {
        "entrypoint": "dashboard/app.py",
        "requirements": APP_REQUIREMENTS,
        "files": {p.as_posix(): {"url": f"./files/{p.as_posix()}"} for p in APP_FILES},
    }
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Risk dashboard (interactive)</title>
<link rel="icon" href="data:,">
<meta name="description" content="The interactive risk dashboard running in the browser. No server: Python runs in WebAssembly.">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.css">
<style>
body{{margin:0;background:#F8FAFC;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#1E293B}}
#bar{{background:#0F172A;color:#E2E8F0;padding:8px 16px;font-size:.9rem;display:flex;gap:16px;flex-wrap:wrap;align-items:center}}
#bar a{{color:#93C5FD}} #bar a:focus-visible{{outline:3px solid #93C5FD;outline-offset:2px}}
#loading{{max-width:640px;margin:48px auto;padding:0 16px;line-height:1.55}}
#loading h1{{font-size:1.4rem;border-left:6px solid #1E40AF;padding-left:10px;color:#0F172A}}
.muted{{color:#475569}}
</style>
</head>
<body>
<div id="bar"><span>Risk dashboard, running in your browser</span><a href="../">Static report</a><a href="{REPO}">Source code</a></div>
<div id="loading" role="status">
<h1>Loading the dashboard</h1>
<p>Python, pandas and scikit-learn are being downloaded into your browser and the real price data is being read. The first load takes about a minute and uses a few tens of megabytes; later visits are faster because the browser caches them.</p>
<p class="muted">Nothing is sent to a server: everything runs on your computer. Educational research tool, not investment advice.</p>
<noscript>This dashboard needs JavaScript.</noscript>
</div>
<div id="root"></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE}/build/stlite.js";
const config = {json.dumps(mount, indent=2)};
try {{
  mount(config, document.getElementById("root"));
  const hide = () => {{ const el = document.getElementById("loading"); if (el && document.querySelector("#root [data-testid='stApp'], #root .stApp")) el.remove(); }};
  new MutationObserver(hide).observe(document.getElementById("root"), {{ childList: true, subtree: true }});
}} catch (e) {{
  document.getElementById("loading").insertAdjacentHTML("beforeend", "<p><strong>The dashboard could not start:</strong> " + String(e) + "</p>");
}}
</script>
</body>
</html>
"""
    out["index.html"] = page.encode()
    return out


def plotly_js() -> Path:
    return Path(plotly.__file__).parent / "package_data" / "plotly.min.js"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/index.html differs from a fresh build")
    args = ap.parse_args()
    page = build()
    app = build_app()
    target = OUT / "index.html"
    if args.check:
        stale = [] if target.exists() and target.read_text() == page else ["docs/index.html"]
        stale += [f"docs/app/{k}" for k, v in app.items() if not (OUT / "app" / k).exists() or (OUT / "app" / k).read_bytes() != v]
        extra = sorted(str(p.relative_to(OUT)) for p in (OUT / "app" / "files").rglob("*") if p.is_file() and p.relative_to(OUT / "app").as_posix() not in app) if (OUT / "app" / "files").exists() else []
        print("docs/ is up to date" if not stale and not extra else f"docs/ is stale ({', '.join(stale + extra)}): run python scripts/build_site.py")
        return 0 if not stale and not extra else 1
    OUT.mkdir(exist_ok=True)
    target.write_text(page)
    shutil.copyfile(plotly_js(), OUT / "plotly.min.js")
    (OUT / ".nojekyll").write_text("")
    if (OUT / "app").exists():
        shutil.rmtree(OUT / "app")  # the folder is wholly generated; remove files that are no longer part of the app
    for rel, content in app.items():
        path = OUT / "app" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(f"wrote {target} ({len(page):,} characters), plotly.min.js and docs/app/ ({len(app)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
