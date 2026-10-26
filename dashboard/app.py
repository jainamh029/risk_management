"""Interactive dashboard for the real-data risk analysis.

    streamlit run dashboard/app.py

Reads the downloaded snapshot (data/) and the files written by
scripts/run_pipeline.py (results/). The VaR explorer and the optional portfolio
screen recompute live with the same riskengine functions the tests check.
Every number shown comes from real data or from those functions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskengine import backtest, config, data, ml, portfolio, var  # noqa: E402
from riskengine.metrics import TRADING_DAYS  # noqa: E402

# Trading-terminal palette: a deep navy background like a charting screen, green for gains, red for losses.
# Every text and chart colour below has a contrast ratio of at least 4.5:1 against the panel it is drawn on (tests/integration/test_dashboard_theme.py).
BG, PANEL, BORDER, GRID = "#0B1220", "#0F1A2E", "#22304D", "#1E2B47"
INK, MUTED = "#E5E7EB", "#94A3B8"
GREEN, RED, BLUE, LIGHT_BLUE, AMBER, SLATE = "#22C55E", "#F43F5E", "#38BDF8", "#7DD3FC", "#FBBF24", "#94A3B8"
st.set_page_config(page_title="Risk dashboard", layout="wide")

st.markdown(
    """
<style>
  :root {--bg: #0B1220; --panel: #0F1A2E; --panel2: #142038; --border: #22304D; --ink: #E5E7EB; --muted: #94A3B8; --up: #22C55E; --down: #F43F5E; --accent: #38BDF8;}

  /* Screen: deep navy with a faint chart grid and two soft glows, like a trading terminal */
  .stApp, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(1100px 420px at 10% -8%, rgba(56, 189, 248, 0.11), transparent 60%),
      radial-gradient(900px 380px at 100% 0%, rgba(34, 197, 94, 0.08), transparent 55%),
      linear-gradient(rgba(148, 163, 184, 0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px),
      #0B1220;
    background-size: auto, auto, 44px 44px, 44px 44px, auto;
  }
  [data-testid="stHeader"] {background: transparent;}
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
  h1 {font-weight: 750; letter-spacing: -0.02em; color: var(--ink); border-left: 6px solid var(--up); padding-left: 0.75rem; line-height: 1.15;}
  h2, h3 {color: var(--ink); letter-spacing: -0.01em;}
  [data-testid="stCaptionContainer"] {color: var(--muted);}

  /* Price tape across the top */
  .tape {display: flex; overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; margin: 0 0 1.1rem 0; background: linear-gradient(180deg, #111E37, #0C1628); font-variant-numeric: tabular-nums; scrollbar-width: thin;}
  .tape > div {white-space: nowrap; padding: 0.55rem 0.95rem; border-right: 1px solid var(--border); font-size: 0.9rem; flex: 0 0 auto;}
  .tape > div:last-child {border-right: none;}
  .tape .lbl {color: var(--muted); font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; display: flex; align-items: center;}
  .tape .sym {font-weight: 700; color: var(--ink); margin-right: 0.5rem;}
  .tape .px {color: var(--ink); margin-right: 0.45rem;}
  .tape .up {color: var(--up); font-weight: 600;}
  .tape .dn {color: var(--down); font-weight: 600;}
  .tape .flat {color: var(--muted); font-weight: 600;}
  .tape:focus-visible {outline: 3px solid var(--accent); outline-offset: 2px;}

  /* Metric cards */
  [data-testid="stMetric"] {background: var(--panel); border: 1px solid var(--border); border-top: 3px solid var(--accent); border-radius: 10px; padding: 0.85rem 1rem 0.8rem 1rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);}
  [data-testid="stMetricLabel"] {color: var(--muted); font-size: 0.82rem; font-weight: 600;}
  [data-testid="stMetricValue"] {color: var(--ink); font-weight: 700; font-variant-numeric: tabular-nums; font-size: 1.7rem;}
  [data-testid="stMetricValue"] > div {white-space: normal; overflow: visible; text-overflow: clip;}
  [data-testid="stMetricDelta"] {font-size: 0.8rem;}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {gap: 0.25rem; border-bottom: 1px solid var(--border);}
  .stTabs [data-baseweb="tab"] {padding: 0.65rem 1rem; font-weight: 600; color: var(--muted); border-radius: 8px 8px 0 0; transition: background-color 150ms ease, color 150ms ease;}
  .stTabs [data-baseweb="tab"]:hover {background: var(--panel2); color: var(--ink);}
  .stTabs [aria-selected="true"] {color: var(--up);}
  .stTabs [data-baseweb="tab-highlight"] {background-color: var(--up);}

  /* Charts, tables and panels */
  [data-testid="stPlotlyChart"] {background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 0.25rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.30);}
  [data-testid="stDataFrame"] {border: 1px solid var(--border); border-radius: 10px; overflow: hidden;}
  [data-testid="stExpander"] {border: 1px solid var(--border); border-radius: 10px; background: var(--panel);}
  [data-testid="stAlert"] {border-radius: 10px;}
  hr {border-color: var(--border);}
  code {color: var(--accent);}

  /* Keyboard focus and reduced motion */
  button:focus-visible, [role="tab"]:focus-visible, input:focus-visible {outline: 3px solid var(--accent) !important; outline-offset: 2px;}
  @media (prefers-reduced-motion: reduce) {* {animation: none !important; transition: none !important; scroll-behavior: auto !important;}}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading the real price data...")
def load_all():
    prices, rf, meta = data.load_snapshot()
    returns = data.simple_returns(prices)
    uni = data.load_universe().set_index("ticker")
    return prices, rf, meta, returns, uni


REQUIRED_COLUMNS = {
    "summary": {"method", "confidence", "n_tickers", "total_forecasts", "pooled_breach_rate", "expected_rate", "share_rejected", "zero_breach_tickers"},
    "by_ticker": {"ticker", "method", "confidence", "n_forecasts", "breaches", "breach_rate", "expected_rate", "kupiec_lr", "kupiec_p"},
    "ml_ic": {"date", "random_forest", "gradient_boosting", "ensemble"},
    "ml_latest": {"ticker", "predicted_63d_return"},
    "metrics": {"ticker", "annual_vol", "mean_annual_return", "sharpe", "max_drawdown", "current_drawdown", "beta_vs_spy", "is_etf"},
}
REQUIRED_KEYS = {
    "info": set(),
    "portfolios": {"presets", "train_start", "train_end", "test_start", "test_end", "eligible_stocks", "benchmark_spy_test"},
    "ml": {"models", "baselines", "info"},
    "membership": {"current_members_in_list", "stocks_in_list", "current_members_not_in_list"},
}


@st.cache_data(show_spinner=False)
def load_results():
    r = ROOT / "results"
    out = {
        "info": json.loads((r / "run_info.json").read_text()),
        "summary": pd.DataFrame(json.loads((r / "var_backtest_summary.json").read_text())),
        "by_ticker": pd.read_csv(r / "var_backtest_by_ticker.csv"),
        "portfolios": json.loads((r / "portfolios.json").read_text()),
        "ml": json.loads((r / "ml_summary.json").read_text()),
        "ml_ic": pd.read_csv(r / "ml_rank_ic_by_date.csv", parse_dates=["date"]),
        "ml_latest": pd.read_csv(r / "ml_latest_forecasts.csv"),
        "ml_imp": pd.read_csv(r / "ml_feature_importance.csv", index_col=0).iloc[:, 0],
        "metrics": pd.read_csv(r / "metrics.csv"),
        "membership": json.loads((r / "universe_membership_summary.json").read_text()),
    }
    for name, cols in REQUIRED_COLUMNS.items():  # a file with the wrong shape is reported by name, not as a KeyError deep inside a tab
        missing = cols - set(out[name].columns)
        if missing:
            raise ValueError(f"results file for '{name}' is missing columns {sorted(missing)}; regenerate it with python scripts/run_pipeline.py")
    for name, keys in REQUIRED_KEYS.items():
        if not isinstance(out[name], dict) or keys - set(out[name]):
            raise ValueError(f"results file for '{name}' is missing keys {sorted(keys - set(out[name]))}; regenerate it with python scripts/run_pipeline.py" if isinstance(out[name], dict) else f"results file for '{name}' is not a JSON object")
    return out


try:
    prices, rf_series, meta, returns, uni = load_all()
    R = load_results()
except (FileNotFoundError, ValueError) as e:  # ValueError covers malformed JSON and empty or unparseable CSV files
    st.error(f"A required file is missing or unreadable: {e}")
    st.markdown(
        "Create the data and results first, then reload this page:\n\n"
        "```\npython scripts/fetch_data.py\npython scripts/run_pipeline.py\npython scripts/check_sp500_membership.py\n```"
    )
    st.stop()

is_etf = uni["is_etf"].astype(bool)
RF_TRAIN = data.risk_free_annual(rf_series, end=config.TRAIN_END)
RF_TEST = data.risk_free_annual(rf_series, start=config.TEST_START)


def pct(x, d=2):
    return "undefined" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, fmt=".3f"):
    """A number formatted with fmt, or the word 'undefined' for null/NaN."""
    return "undefined" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


def r2txt(x):
    """R squared as a percentage, or the word 'undefined' when the score could not be defined (null in the results file)."""
    return "undefined" if x is None else pct(x)


def line_chart_layout(fig, title, yfmt=".1%", ytitle=None, height=430):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=100, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(color=INK),
    )
    fig.update_yaxes(tickformat=yfmt, title=ytitle, gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def panel_layout(fig, title, yfmt=".1%", ytitle=None, height=360):
    """The layout shared by the smaller charts: left-aligned title, legend above the plot, light grid, unified hover."""
    return line_chart_layout(fig, title, yfmt=yfmt, ytitle=ytitle, height=height)


def test_period_growth(pf):
    """Growth of 1 over the test period for every constructed preset and for SPY, with the same daily-rebalanced constant weights as the portfolio tab."""
    test = returns.dropna(how="all").loc[config.TEST_START :]
    curves = {}
    for name, entry in pf["presets"].items():
        if entry["status"] == "constructed":
            daily, _ = portfolio.portfolio_returns(test, pd.Series(entry["weights"]))
            curves[f"{name} portfolio"] = (1 + daily).cumprod()
    curves["SPY"] = (1 + test[config.MARKET].fillna(0)).cumprod()
    return pd.DataFrame(curves)


def themed(fig):
    """Dark chart surface in the same palette as the page. Applied to every chart just before it is shown."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=INK),
        hoverlabel=dict(bgcolor="#1B2A47", bordercolor=BORDER, font=dict(color=INK)),
        legend=dict(font=dict(color=INK)),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


TAPE_TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "BRK-B", "UNH", "V", "LLY"]


def price_tape(prices):
    """HTML for a strip of last closes and one-day changes of large tickers, from the stored prices. A direction is shown by an arrow and a sign as well as by colour."""
    last = prices.index[-1]
    items = []
    for t in TAPE_TICKERS:
        if t not in prices.columns:
            continue
        sr = prices[t].dropna()
        if len(sr) < 2 or sr.index[-1] != last:  # a ticker that did not trade on the last date is left out, not carried forward
            continue
        chg = sr.iloc[-1] / sr.iloc[-2] - 1.0
        cls, arrow = ("up", "&#9650;") if chg > 0 else (("dn", "&#9660;") if chg < 0 else ("flat", "&#9644;"))
        word = "up" if chg > 0 else ("down" if chg < 0 else "unchanged")
        items.append(f'<div aria-label="{t} {sr.iloc[-1]:,.2f}, {word} {abs(chg):.2%}"><span class="sym">{t}</span><span class="px">{sr.iloc[-1]:,.2f}</span><span class="{cls}">{arrow} {chg:+.2%}</span></div>')
    label = f'<div class="lbl" title="Adjusted close on {last.date()} and its change from the previous trading day">Close {last.date()}</div>'
    return f'<div class="tape" role="region" aria-label="Latest adjusted closing prices" tabindex="0">{label}{"".join(items)}</div>'


# ----------------------------------------------------------------------------- header
st.markdown(price_tape(prices), unsafe_allow_html=True)
st.title("Risk dashboard")
st.caption(
    f"Real daily prices from Yahoo Finance, {meta['first_date']} to {meta['last_date']}, downloaded {meta['downloaded_utc']}. "
    f"{meta['tickers_returned']} of {meta['tickers_requested']} requested tickers returned data. Educational research tool, not investment advice."
)

tab_over, tab_stock, tab_var, tab_bt, tab_pf, tab_ml, tab_help = st.tabs(
    ["Overview", "Stock explorer", "VaR explorer", "Backtest comparison", "Portfolio experiment", "Forecasting model", "Glossary and limits"]
)

# ----------------------------------------------------------------------------- overview
with tab_over:
    summ = R["summary"].set_index(["method", "confidence"])
    h99, n99 = summ.loc[("historical", 0.99)], summ.loc[("normal", 0.99)]
    ens = R["ml"]["models"]["ensemble"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers backtested", int(h99["n_tickers"]), help="Tickers with at least 250 days of history plus 250 forecast days.")
    c2.metric("Historical 99% VaR breach rate", pct(h99["pooled_breach_rate"]), f"expected {pct(h99['expected_rate'])}", delta_color="off", delta_arrow="off", help="Share of all forecast days on which the loss was bigger than the 99% VaR.")
    c3.metric("Normal 99% VaR breach rate", pct(n99["pooled_breach_rate"]), f"expected {pct(n99['expected_rate'])}", delta_color="off", delta_arrow="off")
    c4.metric("Forecast model R squared (test)", r2txt(ens["oos_r2_vs_training_mean"]), help="Against always predicting the training-period average. Below zero means worse than that. Undefined if every test target equals the training average.")

    ov_conf_label = st.radio("Confidence level for the per-ticker breach chart", ["95%", "99%"], index=1, horizontal=True, key="ov_conf")
    ov_conf = 0.95 if ov_conf_label == "95%" else 0.99
    ch1, ch2 = st.columns(2)
    growth = test_period_growth(R["portfolios"])
    figg = go.Figure()
    styles_g = {"SPY": dict(color=SLATE, dash="dot", width=2.5), "balanced portfolio": dict(color=BLUE, dash="solid", width=2.5), "aggressive portfolio": dict(color=AMBER, dash="dash", width=2.5), "conservative portfolio": dict(color=RED, dash="dashdot", width=2.5)}
    for col in growth.columns:
        figg.add_trace(go.Scatter(x=growth.index, y=growth[col], name=col, line=styles_g.get(col, dict(color=LIGHT_BLUE, width=2)), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + col + "</extra>"))
    panel_layout(figg, f"Growth of 1 since {R['portfolios']['test_start']} (test period)", yfmt=".2f", ytitle="growth of 1")
    ch1.plotly_chart(themed(figg), width="stretch", key="ov_growth")
    ch1.caption("Daily constant-weight rebalancing, no transaction costs, one period and one split. Weights were fixed with training data only.")
    bt_all = R["by_ticker"][R["by_ticker"]["confidence"] == ov_conf]
    figh = go.Figure()
    size = 0.0025 if ov_conf == 0.99 else 0.005
    for m, color, shape in zip(var.METHODS, (BLUE, AMBER), ("/", "x")):
        figh.add_trace(go.Histogram(x=bt_all[bt_all["method"] == m]["breach_rate"], xbins=dict(start=0, size=size), name=m, opacity=0.65, marker=dict(color=color, pattern=dict(shape=shape)), hovertemplate="breach rate %{x:.2%}: %{y} tickers<extra>" + m + "</extra>"))
    figh.add_vline(x=1 - ov_conf, line=dict(color=INK, dash="dash", width=2), annotation_text=f"expected {1 - ov_conf:.0%}", annotation_position="top right")
    panel_layout(figh, f"Breach rate of each ticker at {ov_conf_label}", yfmt="d", ytitle="number of tickers")
    figh.update_layout(barmode="overlay", hovermode="closest")
    figh.update_xaxes(tickformat=".1%", title="share of forecast days with a breach")
    ch2.plotly_chart(themed(figh), width="stretch", key="ov_breach_hist")
    with st.expander("Data table: growth of 1 and per-ticker breach rates behind these charts"):
        st.dataframe(growth.round(6), width="stretch")
        st.dataframe(bt_all[["ticker", "method", "n_forecasts", "breaches", "breach_rate"]].sort_values("breach_rate", ascending=False), width="stretch", hide_index=True)

    st.subheader("What the real data shows")
    lines = []
    for m in ("historical", "normal"):
        for c in (0.95, 0.99):
            r = summ.loc[(m, c)]
            lines.append(
                f"- {m.capitalize()} VaR at {int(c * 100)}%: breached on {pct(r['pooled_breach_rate'])} of forecast days (expected {pct(r['expected_rate'])}); "
                f"the Kupiec test rejects {pct(r['share_rejected'], 1)} of tickers (in a simulation with a correct model the test rejected between 0% and 8% of series)."
            )
    pf = R["portfolios"]
    built = [n for n, e in pf["presets"].items() if e["status"] == "constructed"]
    skipped = [n for n, e in pf["presets"].items() if e["status"] != "constructed"]
    lines.append(
        f"- Portfolio presets built from data up to {pf['train_end']} and tested from {pf['test_start']}: {len(built)} constructed ({', '.join(built) or 'none'}); "
        f"{len(skipped)} not constructed because too few stocks passed the screen ({', '.join(skipped) or 'none'})."
    )
    for n in built:
        o = pf["presets"][n]["out_of_sample_test"]
        lines.append(f"  - {n}: test Sharpe {num(o['sharpe'], '.2f')} against {num(pf['benchmark_spy_test']['sharpe'], '.2f')} for SPY over the same period (one period, one split).")
    r2v = ens["oos_r2_vs_training_mean"]
    if r2v is None:
        r2_sentence = "out-of-sample R squared is undefined (every test target equals the training average)"
    else:
        r2_sentence = f"out-of-sample R squared of {pct(r2v)} is {'worse than' if r2v < 0 else 'better than'} always predicting the training average"
    lines.append(
        f"- Forecast model (ensemble): {r2_sentence}; "
        f"direction accuracy {pct(ens['direction_accuracy'], 1)} against {pct(R['ml']['baselines']['share_of_test_samples_with_positive_return'], 1)} of test samples going up; "
        f"rank IC {num(ens['rank_ic_nonoverlapping']['mean'])} with t statistic {num(ens['rank_ic_nonoverlapping']['t_stat'], '.2f')} over {ens['rank_ic_nonoverlapping']['n']} independent periods."
    )
    st.markdown("\n".join(lines))
    mem = R["membership"]
    st.info(
        "Main limitation: the ticker list is hand-typed and is not the S&P 500. Compared with Wikipedia's record of index membership, "
        f"{mem['current_members_in_list']} of its {mem['stocks_in_list']} stocks are current members, {len(mem['current_members_not_in_list'])} current members are missing from it, "
        f"and Yahoo returned no data for {len(meta['tickers_failed'])} of its tickers (mostly names that left the market). "
        f"Counting only days when each stock was a member does not change the backtest conclusions, but survivorship bias remains and its size is not measured."
    )

# ----------------------------------------------------------------------------- stock explorer
with tab_stock:
    st.markdown(
        "Every ticker in the data on one map, and one ticker in detail. The map uses the descriptive statistics in `results/metrics.csv` (whole sample, historical, not forecasts); "
        "the charts below it are computed from the daily prices of the ticker you choose."
    )
    M = R["metrics"].copy()
    a, b, c, d = st.columns([2, 2.4, 2.2, 2.4])
    group = a.radio("Show on the map", ["Stocks", "ETFs", "Both"], index=2, horizontal=True, key="se_group")
    sel = b.selectbox("Ticker to highlight and analyse (type to search)", ["SPY"] + [t for t in prices.columns if t != "SPY"], index=0, key="se_ticker")
    period = c.radio("Period for the charts below", ["All data", "Last 3 years", "Last 1 year"], index=0, horizontal=True, key="se_period")
    zoom = d.checkbox("Zoom the map to the main cluster", value=True, key="se_zoom", help="Sets both axes to the 1st to 99th percentile of the plotted tickers, with a margin, so a few extreme tickers do not squash the rest. Untick to see every ticker.")

    shown = M if group == "Both" else M[M["is_etf"] == (group == "ETFs")]
    fig_map = go.Figure()
    for is_etf_flag, label, symbol in ((False, "stocks", "circle"), (True, "ETFs", "diamond")):
        g = shown[shown["is_etf"] == is_etf_flag]
        if g.empty:
            continue
        fig_map.add_trace(
            go.Scatter(
                x=g["annual_vol"], y=g["mean_annual_return"], mode="markers", name=label, text=g["ticker"],
                marker=dict(symbol=symbol, size=11 if is_etf_flag else 8, color=g["sharpe"], colorscale=[[0, RED], [0.5, "#94A3B8"], [1, GREEN]], cmin=-1.0, cmax=2.0, cmid=0.5,
                            line=dict(width=0.6, color="#0B1220"), opacity=0.92, showscale=not is_etf_flag or shown[~shown["is_etf"]].empty,
                            colorbar=dict(title="Sharpe", thickness=12, len=0.8)),
                customdata=np.stack([g["sharpe"], g["max_drawdown"], g["beta_vs_spy"]], axis=1),
                hovertemplate="<b>%{text}</b><br>return %{y:.1%}, volatility %{x:.1%}<br>Sharpe %{customdata[0]:.2f}, max drawdown %{customdata[1]:.1%}<extra>" + label + "</extra>",
            )
        )
    row = M[M["ticker"] == sel]
    if not row.empty:
        fig_map.add_trace(go.Scatter(x=row["annual_vol"], y=row["mean_annual_return"], mode="markers+text", text=row["ticker"], textposition="top center", name=f"selected: {sel}",
                                     marker=dict(symbol="circle-open", size=18, color=RED, line=dict(width=3, color=RED)), hoverinfo="skip"))
    panel_layout(fig_map, "Risk and return of every ticker (whole sample, annualised)", yfmt=".0%", ytitle="mean annual return", height=520)
    fig_map.update_xaxes(tickformat=".0%", title="annualised volatility")
    fig_map.update_layout(hovermode="closest")
    outside = 0
    if zoom and len(shown) >= 20:
        (x0, x1), (y0, y1) = shown["annual_vol"].quantile([0.01, 0.99]), shown["mean_annual_return"].quantile([0.01, 0.99])
        px_, py_ = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        fig_map.update_xaxes(range=[max(0.0, x0 - px_), x1 + px_])
        fig_map.update_yaxes(range=[y0 - py_, y1 + py_])
        outside = int(((shown["annual_vol"] < max(0.0, x0 - px_)) | (shown["annual_vol"] > x1 + px_) | (shown["mean_annual_return"] < y0 - py_) | (shown["mean_annual_return"] > y1 + py_)).sum())
    st.plotly_chart(themed(fig_map), width="stretch", key="se_map")
    if outside:
        st.caption(f"{outside} ticker(s) lie outside the zoomed view. Untick the zoom option to see them; they are all in the data table below.")
    st.caption("Colour shows the Sharpe ratio and shape shows stocks (circles) or ETFs (diamonds), so neither relies on colour alone. Up and to the left is more return for less risk, in this sample only.")
    with st.expander("Data table: the statistics behind the map"):
        st.dataframe(shown[["ticker", "is_etf", "n_returns", "mean_annual_return", "annual_vol", "sharpe", "max_drawdown", "current_drawdown", "beta_vs_spy"]].sort_values("sharpe", ascending=False), width="stretch", hide_index=True)

    st.subheader(f"{sel} in detail")
    if row.empty:
        st.warning(f"{sel} has fewer than 250 daily returns, so it is not in the statistics table. The charts below still use the prices it has.")
    else:
        r0 = row.iloc[0]
        k = st.columns(6)
        k[0].metric("Mean annual return", pct(r0["mean_annual_return"], 1))
        k[1].metric("Annual volatility", pct(r0["annual_vol"], 1))
        k[2].metric("Sharpe ratio", num(r0["sharpe"], ".2f"))
        k[3].metric("Maximum drawdown", pct(r0["max_drawdown"], 1))
        k[4].metric("Current drawdown", pct(r0["current_drawdown"], 1))
        k[5].metric("Beta against SPY", num(r0["beta_vs_spy"], ".2f"), help="Undefined for SPY itself and when the benchmark is constant.")

    px = prices[sel].dropna()
    if period != "All data":
        px = px.loc[px.index[-1] - pd.DateOffset(years=3 if period == "Last 3 years" else 1) :]
    if len(px) < 130:
        st.warning(f"Only {len(px)} prices are available for {sel} in this period, too few for the charts.")
    else:
        rets = px / px.shift(1) - 1.0
        rets = rets.dropna()
        growth_s = px / px.iloc[0]
        drawdown = growth_s / growth_s.cummax() - 1.0
        roll_vol = (rets.rolling(60, min_periods=60).std() * np.sqrt(TRADING_DAYS)).dropna()
        c1, c2 = st.columns(2)
        f1 = go.Figure(go.Scatter(x=growth_s.index, y=growth_s, name=sel, line=dict(color=BLUE, width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>growth of 1</extra>"))
        panel_layout(f1, f"{sel}: growth of 1 (adjusted close)", yfmt=".2f", ytitle="growth of 1")
        c1.plotly_chart(themed(f1), width="stretch", key="se_growth")
        f2 = go.Figure(go.Scatter(x=drawdown.index, y=drawdown, name="drawdown", fill="tozeroy", line=dict(color=RED, width=1.5), fillcolor="rgba(244, 63, 94, 0.20)", hovertemplate="%{x|%Y-%m-%d}: %{y:.1%}<extra>drawdown</extra>"))
        panel_layout(f2, f"{sel}: fall from the previous peak", yfmt=".0%", ytitle="drawdown")
        c2.plotly_chart(themed(f2), width="stretch", key="se_drawdown")
        c3, c4 = st.columns(2)
        f3 = go.Figure(go.Scatter(x=roll_vol.index, y=roll_vol, name="60-day volatility", line=dict(color=AMBER, width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:.1%}<extra>60-day volatility</extra>"))
        panel_layout(f3, f"{sel}: 60-day rolling volatility (annualised)", yfmt=".0%", ytitle="annualised volatility")
        c3.plotly_chart(themed(f3), width="stretch", key="se_rollvol")
        mu_r, sd_r = float(rets.mean()), float(rets.std(ddof=1))
        v99, _ = var.historical_var_es(rets.to_numpy(), 0.99)
        bins = 60
        f4 = go.Figure(go.Histogram(x=rets, nbinsx=bins, histnorm="probability density", name="daily returns", marker=dict(color=LIGHT_BLUE, line=dict(width=0.3, color=PANEL)), opacity=0.8, hovertemplate="return %{x:.2%}<extra>daily returns</extra>"))
        xs = np.linspace(float(rets.min()), float(rets.max()), 300)
        f4.add_trace(go.Scatter(x=xs, y=np.exp(-0.5 * ((xs - mu_r) / sd_r) ** 2) / (sd_r * np.sqrt(2 * np.pi)), name="normal curve, same mean and standard deviation", line=dict(color=INK, width=2, dash="dash"), hoverinfo="skip"))
        f4.add_vline(x=-v99, line=dict(color=RED, dash="dot", width=2), annotation_text=f"historical 99% VaR {v99:.1%}", annotation_position="top left")
        panel_layout(f4, f"{sel}: distribution of daily returns", yfmt=".0f", ytitle="density")
        f4.update_layout(hovermode="closest")
        f4.update_xaxes(tickformat=".0%", title="daily return")
        c4.plotly_chart(themed(f4), width="stretch", key="se_hist")
        st.caption(
            f"{len(rets):,} daily returns from {rets.index[0].date()} to {rets.index[-1].date()}. Skewness {rets.skew():.2f}, excess kurtosis {rets.kurt():.2f}: "
            "an excess kurtosis above zero means more extreme days than the normal curve allows, which is why a normal VaR can understate tail risk."
        )
        with st.expander("Data table: values behind the four charts"):
            st.dataframe(pd.DataFrame({"growth_of_1": growth_s, "drawdown": drawdown, "rolling_60d_volatility": roll_vol, "daily_return": px / px.shift(1) - 1.0}).round(6), width="stretch")

# ----------------------------------------------------------------------------- VaR explorer
with tab_var:
    st.markdown(
        "Pick a stock and settings. Each day's VaR forecast uses only the previous *window* days of returns. "
        "A **breach** is a day when the loss was bigger than that day's VaR. This tab recomputes everything live."
    )
    stocks_first = ["SPY"] + [t for t in prices.columns if t != "SPY"]
    a, b, c, d = st.columns([2, 2, 1.4, 2])
    ticker = a.selectbox("Ticker (type to search)", stocks_first, index=0, key="var_ticker", help="Any ticker with downloaded data. SPY is an ETF that tracks the S&P 500.")
    methods = b.multiselect("Method", list(var.METHODS), default=list(var.METHODS), key="var_methods", help="Historical: the empirical quantile of past losses. Normal: assumes returns follow a bell curve with the sample mean and standard deviation.")
    conf_label = c.radio("Confidence", ["95%", "99%"], index=1, horizontal=True, key="var_conf", help="99% means the loss should be exceeded on about 1 day in 100.")
    window = d.slider("Window (trading days)", 60, 500, config.WINDOW, step=10, key="var_window", help="How many past days feed each forecast. The research default is 250.")
    position = st.number_input("Position size ($)", min_value=1_000, max_value=100_000_000, value=100_000, step=10_000, key="var_position", help="Used to show today's VaR in dollars.")
    conf = 0.95 if conf_label == "95%" else 0.99

    r = returns[ticker]
    if not methods:
        st.warning("Choose at least one method.")
    elif r.notna().sum() < window + 60:
        st.warning(f"{ticker} has only {int(r.notna().sum())} daily returns, too few for a {window}-day window.")
    else:
        if (1 - conf) * window < 5:
            st.warning(f"With a {window}-day window, a {conf_label} VaR rests on about {(1 - conf) * window:.1f} days in the tail. It will be noisy.")
        fig = go.Figure()
        loss = (-r).dropna()
        fig.add_trace(go.Scatter(x=loss.index, y=loss, mode="lines", name="Daily loss (negative = gain)", line=dict(color=SLATE, width=1), hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>daily loss</extra>"))
        styles = {"historical": dict(color=BLUE, dash="solid", symbol="circle-open"), "normal": dict(color=AMBER, dash="dash", symbol="x")}
        rows, breach_tables = [], []
        for m in methods:
            f = pd.Series(backtest.rolling_forecasts(r.values, window, conf, m), index=r.index)
            ok = f.notna() & r.notna()
            n = int(ok.sum())
            breach = ok & (-r > f)
            x = int(breach.sum())
            lr, pv = backtest.kupiec_pof(n, x, 1 - conf)
            latest, es = var.var_es(r.iloc[-window:].values, conf, m) if r.iloc[-window:].notna().all() else (np.nan, np.nan)
            s = styles[m]
            fig.add_trace(go.Scatter(x=f.index, y=f, mode="lines", name=f"{m} {conf_label} VaR forecast", line=dict(color=s["color"], dash=s["dash"], width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:.2%}<extra>" + m + " VaR</extra>"))
            fig.add_trace(go.Scatter(x=f.index[breach], y=(-r)[breach], mode="markers", name=f"{m} breaches ({x})", marker=dict(color=RED, symbol=s["symbol"], size=9, line=dict(width=2, color=RED)), hovertemplate="%{x|%Y-%m-%d}: loss %{y:.2%}<extra>" + m + " breach</extra>"))
            rows.append({"method": m, "forecast days": n, "breaches": x, "breach rate": x / n, "expected": 1 - conf, "Kupiec statistic": lr, "p-value": pv, "VaR today": latest, "ES today": es})
            bt = pd.DataFrame({"date": f.index[breach], "loss": (-r)[breach].values, "VaR forecast": f[breach].values})
            bt.insert(0, "method", m)
            breach_tables.append(bt)
        line_chart_layout(fig, f"{ticker}: daily loss, {conf_label} VaR forecast and breaches", ytitle="loss as a fraction of position")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=110, b=150), legend=dict(orientation="h", yanchor="top", y=-0.42, x=0))
        fig.update_xaxes(rangeslider_visible=True, rangeselector=dict(x=0, y=1.02, yanchor="bottom", buttons=[dict(count=1, label="1y", step="year", stepmode="backward"), dict(count=2, label="2y", step="year", stepmode="backward"), dict(step="all", label="all")]))
        st.plotly_chart(themed(fig), width="stretch", key="var_chart")
        st.caption("Drag on the chart to zoom, double-click to reset. Breaches are marked by symbol (open circle: historical, cross: normal) as well as colour.")

        tbl = pd.DataFrame(rows)
        for _, row in tbl.iterrows():
            cols = st.columns(5)
            cols[0].metric(f"{row['method']}: VaR today", pct(row["VaR today"]), f"${row['VaR today'] * position:,.0f} on ${position:,}" if not np.isnan(row["VaR today"]) else None, delta_color="off", delta_arrow="off", help="One-day loss that should be exceeded on about (1 - confidence) of days, from the latest window.")
            cols[1].metric("ES today", pct(row["ES today"]), help="Average loss on the days beyond VaR.")
            cols[2].metric("Breaches", f"{int(row['breaches'])} of {int(row['forecast days'])}", help="Days with a loss bigger than that day's VaR.")
            cols[3].metric("Breach rate", pct(row["breach rate"]), f"expected {pct(row['expected'])}", delta_color="off", delta_arrow="off")
            verdict = "rejected at the 5% level" if row["p-value"] < 0.05 else "not rejected at the 5% level"
            cols[4].metric("Kupiec test", f"p = {row['p-value']:.3f}", verdict, delta_color="off", delta_arrow="off", help="Tests whether the breach rate differs from the expected rate by more than chance. Small p-values mean the model's breach rate is off.")
        with st.expander("Data table: results and breach days"):
            st.dataframe(tbl, width="stretch", hide_index=True)
            st.dataframe(pd.concat(breach_tables), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- backtest comparison
with tab_bt:
    st.markdown(
        "Every ticker was backtested the same way (250-day window, one-day forecasts). The bars show how often the VaR was breached across all tickers and days; "
        "the black marks show the rate a correct model would give."
    )
    a, b, c = st.columns([2, 1.2, 2])
    sel_methods = a.multiselect("Methods", list(var.METHODS), default=list(var.METHODS), key="bt_methods")
    sel_conf = b.radio("Confidence", ["95%", "99%", "both"], index=2, horizontal=True, key="bt_conf")
    only_rej = c.checkbox("Only tickers the Kupiec test rejects (p < 0.05)", value=False, key="bt_rejected")
    search = st.text_input("Filter tickers (comma separated, e.g. AAPL, MSFT)", value="", key="bt_search")
    confs = [0.95, 0.99] if sel_conf == "both" else [0.95 if sel_conf == "95%" else 0.99]
    s = R["summary"][R["summary"]["method"].isin(sel_methods) & R["summary"]["confidence"].isin(confs)].copy()
    if s.empty:
        st.warning("Choose at least one method.")
    else:
        s["label"] = s["method"] + " " + (s["confidence"] * 100).astype(int).astype(str) + "%"
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s["label"], y=s["pooled_breach_rate"], name="Observed breach rate", marker=dict(color=BLUE, pattern=dict(shape="/")), hovertemplate="%{x}: %{y:.2%}<extra>observed</extra>"))
        fig.add_trace(go.Scatter(x=s["label"], y=s["expected_rate"], mode="markers", name="Expected by a correct model", marker=dict(color=INK, symbol="line-ew", size=40, line=dict(width=3, color=INK)), hovertemplate="%{x}: %{y:.2%}<extra>expected</extra>"))
        line_chart_layout(fig, "Pooled breach rate against expected", ytitle="share of forecast days with a breach")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(themed(fig), width="stretch", key="bt_bar")
        show = s[["label", "n_tickers", "total_forecasts", "pooled_breach_rate", "expected_rate", "share_rejected", "zero_breach_tickers"]].rename(
            columns={"label": "Method", "n_tickers": "Tickers", "total_forecasts": "Forecasts", "pooled_breach_rate": "Pooled breach rate", "expected_rate": "Expected", "share_rejected": "Share rejected (Kupiec, 5%)", "zero_breach_tickers": "Tickers with zero breaches"}
        )
        st.dataframe(show, width="stretch", hide_index=True, column_config={"Pooled breach rate": st.column_config.NumberColumn(format="percent"), "Expected": st.column_config.NumberColumn(format="percent"), "Share rejected (Kupiec, 5%)": st.column_config.NumberColumn(format="percent")})

        bt = R["by_ticker"]
        bt = bt[bt["method"].isin(sel_methods) & bt["confidence"].isin(confs)]
        fig2 = go.Figure()
        for m, shape in zip(sel_methods, ["/", "x", "."]):
            fig2.add_trace(go.Histogram(x=bt[bt["method"] == m]["kupiec_p"], xbins=dict(start=0, end=1, size=0.05), name=m, opacity=0.6, marker=dict(color=BLUE if m == "historical" else AMBER, pattern=dict(shape=shape))))
        fig2.add_vline(x=0.05, line=dict(color=RED, dash="dash"), annotation_text="p = 0.05")
        fig2.update_layout(barmode="overlay", title="Kupiec p-values across tickers (a correct model gives values spread evenly from 0 to 1)", height=360, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor=PANEL, paper_bgcolor=PANEL)
        fig2.update_xaxes(title="p-value")
        fig2.update_yaxes(title="number of ticker-confidence results")
        st.plotly_chart(themed(fig2), width="stretch", key="bt_hist")

        if only_rej:
            bt = bt[bt["kupiec_p"] < 0.05]
        if search.strip():
            wanted = {t.strip().upper() for t in search.split(",") if t.strip()}
            bt = bt[bt["ticker"].isin(wanted)]
        st.markdown(f"**{len(bt):,} results** (click a column header to sort)")
        st.dataframe(
            bt.sort_values("kupiec_p"),
            width="stretch",
            hide_index=True,
            column_config={"breach_rate": st.column_config.NumberColumn(format="percent"), "expected_rate": st.column_config.NumberColumn(format="percent"), "kupiec_p": st.column_config.NumberColumn("Kupiec p-value", format="%.4f"), "kupiec_lr": st.column_config.NumberColumn("Kupiec statistic", format="%.2f")},
        )
        st.caption("These stored results use the 250-day window. Use the VaR explorer to see any other window for a single ticker. The tables above are the data behind both charts.")


# ----------------------------------------------------------------------------- portfolio experiment
def show_portfolio(name, e, bench, key):
    if e["status"] != "constructed":
        st.warning(f"{name}: not constructed. {e['reason']}")
        return
    w = pd.Series(e["weights"]).sort_values()
    a, b = st.columns([1, 1.6])
    figw = go.Figure(go.Bar(x=w.values, y=w.index, orientation="h", marker=dict(color=BLUE), hovertemplate="%{y}: %{x:.1%}<extra>weight</extra>"))
    figw.update_layout(title="Weights (set with training data only)", height=max(320, 26 * len(w) + 90), margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor=PANEL, paper_bgcolor=PANEL)
    figw.update_xaxes(tickformat=".0%", gridcolor=GRID)
    a.plotly_chart(themed(figw), width="stretch", key=f"{key}_w")
    test = returns.dropna(how="all").loc[config.TEST_START :]
    wt = pd.Series(e["weights"])
    d_p, _ = portfolio.portfolio_returns(test, wt)
    d_e, _ = portfolio.portfolio_returns(test, pd.Series(1 / len(wt), index=wt.index))
    figg = go.Figure()
    figg.add_trace(go.Scatter(x=d_p.index, y=(1 + d_p).cumprod(), name=f"{name} (optimised weights)", line=dict(color=BLUE, width=2.5), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + name + "</extra>"))
    figg.add_trace(go.Scatter(x=d_e.index, y=(1 + d_e).cumprod(), name="same holdings, equal weights", line=dict(color=AMBER, dash="dash", width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>equal weights</extra>"))
    spy = test[config.MARKET].fillna(0)
    figg.add_trace(go.Scatter(x=spy.index, y=(1 + spy).cumprod(), name="SPY", line=dict(color=SLATE, dash="dot", width=2), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>SPY</extra>"))
    line_chart_layout(figg, "Growth of 1 over the test period (no transaction costs)", yfmt=".2f", ytitle="growth of 1", height=max(320, 26 * len(w) + 90))
    b.plotly_chart(themed(figg), width="stretch", key=f"{key}_g")
    i, o, q = e["in_sample_train"], e["out_of_sample_test"], e["equal_weight_same_holdings_test"]
    stats = pd.DataFrame(
        {
            "Training period (in-sample)": [pct(i["mean_annual_return"], 1), pct(i["annual_vol"], 1), num(i['sharpe'], '.2f'), pct(i["max_drawdown"], 1), pct(i["hist_var_99_1d"]), pct(i["total_return"], 1)],
            "Test period (out-of-sample)": [pct(o["mean_annual_return"], 1), pct(o["annual_vol"], 1), num(o['sharpe'], '.2f'), pct(o["max_drawdown"], 1), pct(o["hist_var_99_1d"]), pct(o["total_return"], 1)],
            "Test: same holdings, equal weights": [pct(q["mean_annual_return"], 1), pct(q["annual_vol"], 1), num(q['sharpe'], '.2f'), pct(q["max_drawdown"], 1), pct(q["hist_var_99_1d"]), pct(q["total_return"], 1)],
            "Test: SPY": [pct(bench["mean_annual_return"], 1), pct(bench["annual_vol"], 1), num(bench['sharpe'], '.2f'), pct(bench["max_drawdown"], 1), pct(bench["hist_var_99_1d"]), pct(bench["total_return"], 1)],
        },
        index=["Mean annual return", "Annual volatility", "Sharpe ratio", "Maximum drawdown", "One-day 99% VaR (historical)", "Total return"],
    )
    st.dataframe(stats, width="stretch")
    st.caption("Only the test-period columns are out-of-sample. The training column describes the data used to choose the portfolio.")
    corr = returns.loc[: config.TRAIN_END, w.index].corr()
    show_values = len(corr) <= 12
    figc = go.Figure(
        go.Heatmap(
            z=corr.to_numpy(), x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, zmid=0, colorscale=[[0, AMBER], [0.5, "#1E293B"], [1, BLUE]],
            colorbar=dict(title="correlation", thickness=12), hovertemplate="%{y} and %{x}: %{z:.2f}<extra></extra>",
            **({"text": np.round(corr.to_numpy(), 2), "texttemplate": "%{text}", "textfont": dict(size=11)} if show_values else {}),
        )
    )
    figc.update_layout(title=dict(text="Correlation of the holdings' daily returns in the training period", x=0, xanchor="left", font=dict(size=15)), height=max(380, 34 * len(corr) + 140), margin=dict(l=10, r=10, t=60, b=10), plot_bgcolor=PANEL, paper_bgcolor=PANEL, font=dict(color="#1E293B"))
    figc.update_yaxes(autorange="reversed")
    st.plotly_chart(themed(figc), width="stretch", key=f"{key}_corr")
    st.caption("Correlations close to 1 mean the holdings move together, so the portfolio is less diversified than its number of holdings suggests. Computed on the data used to choose the weights.")
    with st.expander("Data table: weights, growth of 1 and correlations behind the charts"):
        st.dataframe(pd.DataFrame({"weight": w.iloc[::-1]}), width="stretch")
        st.dataframe(corr.round(4), width="stretch")
        growth = pd.DataFrame({f"{name} (optimised weights)": (1 + d_p).cumprod(), "same holdings, equal weights": (1 + d_e).cumprod(), "SPY": (1 + spy).cumprod()})
        st.dataframe(growth, width="stretch")


@st.cache_data(show_spinner="Rebuilding the portfolio with your screen...")
def custom_portfolio(min_sharpe, max_stocks, min_ret_1y, lo, hi):
    preset = dict(min_sharpe=min_sharpe, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=min_ret_1y, max_stocks=int(max_stocks), bounds=(lo, hi))
    return portfolio.build_and_evaluate(returns, prices, is_etf, RF_TRAIN, RF_TEST, presets={"custom": preset})["presets"]["custom"]


with tab_pf:
    pf = R["portfolios"]
    st.markdown(
        f"Holdings and weights were chosen using data from {pf['train_start']} to {pf['train_end']} only, then fixed. Returns from {pf['test_start']} to {pf['test_end']} assume the portfolio is rebalanced to those weights every day (constant weights, not buy-and-hold). "
        f"{pf['eligible_stocks']} stocks with a complete training history were eligible. A preset whose screen leaves fewer than {config.MIN_ASSETS} stocks is not built; the screen is never loosened."
    )
    names = list(pf["presets"])
    first_built = next((i for i, n in enumerate(names) if pf["presets"][n]["status"] == "constructed"), 0)
    preset_name = st.radio("Preset", names, index=first_built, horizontal=True, key="pf_preset", help="Screens and position limits are in riskengine/config.py. A preset that no stock can pass is shown as not constructed.")
    e = pf["presets"][preset_name]
    pr = e["preset"]
    st.caption(
        f"Screen: training Sharpe above {pr['min_sharpe']}"
        + (f", max drawdown better than {pct(pr['min_max_drawdown'], 0)}" if pr.get("min_max_drawdown") is not None else "")
        + (f", one-day 99% VaR below {pct(pr['max_hist_var99_1d'])}" if pr.get("max_hist_var99_1d") is not None else "")
        + (f", trailing 1-year return above {pct(pr['min_return_1y'], 0)}" if pr.get("min_return_1y") is not None else "")
        + f". Up to {pr['max_stocks']} stocks, each between {pct(pr['bounds'][0], 0)} and {pct(pr['bounds'][1], 0)}. {e['n_passing_screen']} stocks passed."
    )
    show_portfolio(preset_name, e, pf["benchmark_spy_test"], "pf_stored")

    with st.expander("Try your own screen"):
        st.warning(
            "Exploratory only. If you change these settings and then read the test-period numbers, you are tuning on the test data. "
            "The results are no longer out-of-sample and will tend to look better than they would in practice."
        )
        a, b, c, d, f = st.columns(5)
        ms = a.slider("Minimum training Sharpe", -0.5, 2.0, 0.1, 0.1, key="pf_sharpe")
        mx = b.slider("Maximum stocks", 4, 25, 15, 1, key="pf_max")
        mr = c.slider("Minimum trailing 1-year return", -0.5, 1.0, 0.05, 0.05, key="pf_ret")
        lo = d.slider("Smallest weight", 0.0, 0.10, 0.02, 0.01, key="pf_lo")
        hi = f.slider("Largest weight", 0.15, 0.50, 0.30, 0.05, key="pf_hi")
        if st.button("Build this portfolio", key="pf_build"):
            st.session_state["pf_custom"] = custom_portfolio(ms, mx, mr, lo, hi)
        if "pf_custom" in st.session_state:
            show_portfolio("custom", st.session_state["pf_custom"], pf["benchmark_spy_test"], "pf_custom_view")

# ----------------------------------------------------------------------------- forecasting model
with tab_ml:
    mi, mb = R["ml"]["info"], R["ml"]["baselines"]
    st.markdown(
        f"A random forest and a gradient-boosting model, trained together on {mi['train_samples']:,} samples (feature dates {mi['train_first_feature_date']} to {mi['train_last_feature_date']}; "
        f"every 63-day target ends by {mi['train_last_target_end_date']}) forecast each stock's {mi['horizon_trading_days']}-day return. "
        f"They were scored on {mi['test_samples']:,} later samples ({mi['test_first_feature_date']} to {mi['test_last_feature_date']}) against baselines."
    )
    ens = R["ml"]["models"]["ensemble"]
    if ens["oos_r2_vs_training_mean"] is None:
        st.info("R squared is undefined on the test period because every test target equals the training average. The other measures below are still shown.")
    elif ens["oos_r2_vs_training_mean"] < 0:
        st.warning("On the test period the ensemble forecasts worse than always predicting the training average (R squared below zero).")
    else:
        st.success("On the test period the ensemble forecasts better than always predicting the training average. See the rank IC statistics before drawing conclusions.")
    rows = []
    for name, m in R["ml"]["models"].items():
        ic = m["rank_ic_nonoverlapping"]
        rows.append({"Model": name.replace("_", " "), "R squared vs training mean": r2txt(m["oos_r2_vs_training_mean"]), "Direction accuracy": m["direction_accuracy"], "Rank IC (independent periods)": ic["mean"], "t statistic": ic["t_stat"], "Independent periods": ic["n"], "Rank IC (all dates)": m["rank_ic_all_dates"]["mean"]})
    rows.append({"Model": "baseline: training mean", "R squared vs training mean": "0.00%", "Direction accuracy": mb["training_mean_forecast"]["direction_accuracy"]})
    for f_ in ("mom_60", "mom_126"):
        ic = mb[f"rank_ic_{f_}_nonoverlapping"]
        rows.append({"Model": f"baseline: {f_.replace('_', ' ')} alone", "Rank IC (independent periods)": ic["mean"], "t statistic": ic["t_stat"], "Independent periods": ic["n"], "Rank IC (all dates)": mb[f"rank_ic_{f_}_all_dates"]["mean"]})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={"Direction accuracy": st.column_config.NumberColumn(format="percent"), "Rank IC (independent periods)": st.column_config.NumberColumn(format="%.3f"), "Rank IC (all dates)": st.column_config.NumberColumn(format="%.3f"), "t statistic": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"{pct(mb['share_of_test_samples_with_positive_return'], 1)} of test samples went up, so a model that always says up scores that on direction accuracy.")

    chosen = st.multiselect("Models in the chart", ["random_forest", "gradient_boosting", "ensemble"], default=["ensemble"], key="ml_models")
    figi = go.Figure()
    dash = {"random_forest": "solid", "gradient_boosting": "dash", "ensemble": "dot"}
    colr = {"random_forest": BLUE, "gradient_boosting": AMBER, "ensemble": INK}
    ic_df = R["ml_ic"].set_index("date")
    for m in chosen:
        figi.add_trace(go.Scatter(x=ic_df.index, y=ic_df[m], name=m.replace("_", " "), line=dict(color=colr[m], dash=dash[m], width=1.6), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>" + m + "</extra>"))
    if "ensemble" in chosen:
        indep = ml.nonoverlapping(ic_df["ensemble"], prices.index)
        figi.add_trace(go.Scatter(x=indep.index, y=indep, mode="markers", name="ensemble, independent periods", marker=dict(color=RED, size=9, symbol="diamond"), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>independent period</extra>"))
    figi.add_hline(y=0, line=dict(color=INK, width=1))
    line_chart_layout(figi, "Rank IC by date: correlation between forecast and realised 63-day return across stocks", yfmt=".2f", ytitle="rank IC")
    st.plotly_chart(themed(figi), width="stretch", key="ml_ic_chart")
    st.caption("Above zero means stocks forecast to do better did better. Neighbouring dates share most of their 63-day window, so only the diamonds are independent.")
    with st.expander("Data table: rank IC by date and feature importance behind the charts"):
        st.dataframe(ic_df[chosen] if chosen else ic_df, width="stretch")
        st.dataframe(R["ml_imp"].sort_values(ascending=False).to_frame("random_forest_importance"), width="stretch")

    imp = R["ml_imp"].sort_values()
    figf = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker=dict(color=BLUE)))
    figf.update_layout(title="Random-forest feature importance (final model)", height=360, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor=PANEL, paper_bgcolor=PANEL)
    figf.update_xaxes(gridcolor=GRID)
    st.plotly_chart(themed(figf), width="stretch", key="ml_imp_chart")

    st.subheader(f"Latest forecasts (features as of {mi['latest_feature_date']})")
    st.warning("These are model outputs with the out-of-sample accuracy shown above. They are not recommendations.")
    a, b = st.columns([2, 1])
    q = a.text_input("Filter tickers (comma separated)", value="", key="ml_search")
    topn = b.slider("Rows to show", 10, 200, 25, 5, key="ml_topn")
    lf = R["ml_latest"].copy()
    if q.strip():
        wanted = {t.strip().upper() for t in q.split(",") if t.strip()}
        lf = lf[lf["ticker"].isin(wanted)]
    order = st.radio("Order", ["highest forecast first", "lowest forecast first"], horizontal=True, key="ml_order")
    lf = lf.sort_values("predicted_63d_return", ascending=(order == "lowest forecast first")).head(topn)
    st.dataframe(lf, width="stretch", hide_index=True, column_config={"predicted_63d_return": st.column_config.NumberColumn("Forecast 63-day return", format="percent")})
    st.download_button("Download all latest forecasts (CSV)", R["ml_latest"].to_csv(index=False), "ml_latest_forecasts.csv", "text/csv", key="ml_download")

# ----------------------------------------------------------------------------- glossary
with tab_help:
    st.subheader("Plain-language glossary")
    terms = {
        "Value at Risk (VaR)": "A loss level for one day. A 99% VaR of 2% means that on about 1 day in 100 you would expect to lose more than 2% of the position.",
        "Expected Shortfall (ES)": "The average loss on the days when the loss is bigger than VaR. It says how bad the bad days are, not just where they start.",
        "Breach": "A day on which the actual loss was bigger than that day's VaR forecast.",
        "Backtest": "Checking past forecasts against what happened. Each day's forecast uses only earlier data.",
        "Kupiec test and p-value": "Asks whether the number of breaches is consistent with the rate the model claims. A small p-value (below 0.05) means the observed breach rate is unlikely if the model were right.",
        "Out-of-sample": "Data that was not used to build or choose the model. Only out-of-sample results say anything about how a method might do on new data.",
        "Sharpe ratio": "Average return above the risk-free rate, divided by volatility. Higher means more return per unit of risk.",
        "Maximum drawdown": "The biggest fall from a peak to a later low.",
        "R squared against the training mean": "Compares a forecast with always predicting the training-period average return. Below zero means the forecast is worse than that.",
        "Rank IC": "On one date, how well the ranking of stocks by forecast matches the ranking by what they actually returned (1 is perfect, 0 is no relationship).",
        "Survivorship bias": "If stocks that left the market are missing from the data, results look better than they were in real time.",
        "Data snooping": "Trying many settings and reporting the best-looking result. The portfolio tab warns you when you do this.",
    }
    for k, v in terms.items():
        with st.expander(k):
            st.write(v)
    st.subheader("Limits to keep in mind")
    st.markdown(
        "- The ticker list is hand-typed and is not the S&P 500 (see the Overview and the README comparison with Wikipedia's membership record); names that left the market are missing.\n"
        "- Prices are Yahoo Finance adjusted closes, used as downloaded.\n"
        "- A 99% VaR from a 250-day window rests on about 2.5 days in the tail, so it is noisy.\n"
        "- The per-ticker tests are not independent of each other.\n"
        "- The portfolio and forecasting tests use one train/test split and a short test period.\n"
        "- No transaction costs are modelled."
    )

st.divider()
st.caption("Educational research tool. Not investment advice. Past data does not predict future results. Source code and tests: see the repository README.")
