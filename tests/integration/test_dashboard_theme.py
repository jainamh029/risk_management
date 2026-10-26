"""The trading-terminal theme: a dark navy surface, green for gains and red for losses.

Checks that the colours stay readable (WCAG contrast of at least 4.5:1 for text and for coloured chart marks), that Streamlit's own theme matches
the palette, that every chart is drawn on the dark surface, and that the price tape equals an independent calculation from the stored prices."""
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from tests.integration.test_dashboard_extra import app, decode, figure  # noqa: F401  (fixture and helpers)

ROOT = Path(__file__).resolve().parents[2]
SRC = (ROOT / "dashboard" / "app.py").read_text()


def palette():
    ns = {}
    block = re.search(r"BG, PANEL, BORDER, GRID = .*?\n(?:.*\n){0,2}?GREEN, RED, BLUE, LIGHT_BLUE, AMBER, SLATE = [^\n]+", SRC).group(0)
    exec(block, ns)
    return {k: v for k, v in ns.items() if k.isupper()}


def luminance(hex_colour):
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    lin = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a, b):
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_the_palette_is_the_documented_one_and_every_text_and_mark_colour_is_readable_on_the_panel():
    p = palette()
    assert (p["BG"], p["PANEL"]) == ("#0B1220", "#0F1A2E")
    for name in ("INK", "MUTED", "GREEN", "RED", "BLUE", "LIGHT_BLUE", "AMBER", "SLATE"):
        assert contrast(p[name], p["PANEL"]) >= 4.5, (name, contrast(p[name], p["PANEL"]))
        assert contrast(p[name], p["BG"]) >= 4.5, name
    assert contrast(p["INK"], p["BG"]) >= 12  # body text is far above the minimum
    assert contrast(p["GREEN"], p["RED"]) > 1.0 and p["GREEN"] != p["RED"]


def test_streamlit_is_configured_with_the_same_dark_palette():
    theme = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())["theme"]
    p = palette()
    assert theme["base"] == "dark" and theme["backgroundColor"].upper() == p["BG"] and theme["secondaryBackgroundColor"].upper() == p["PANEL"]
    assert theme["textColor"].upper() == p["INK"] and theme["primaryColor"].upper() == p["GREEN"]
    assert contrast(theme["textColor"], theme["backgroundColor"]) >= 12 and contrast(theme["primaryColor"], theme["backgroundColor"]) >= 4.5


def test_no_light_surface_colours_are_left_in_the_dashboard_source():
    light = re.findall(r"#(?:FFFFFF|F8FAFC|E2E8F0|DBEAFE|E9EEF6|F1F5F9)\b", SRC, re.I)
    assert light == []
    assert "prefers-reduced-motion" in SRC and "focus-visible" in SRC  # accessibility rules survived the restyle


def collect_figures(at):
    return {e.key: __import__("json").loads(e.proto.spec) for e in at.get("plotly_chart")}


def test_every_chart_is_drawn_on_the_dark_surface(app):
    p = palette()
    figs = collect_figures(app)
    assert len(figs) >= 7
    for key, spec in figs.items():
        assert spec["layout"]["paper_bgcolor"] == p["PANEL"] and spec["layout"]["plot_bgcolor"] == p["PANEL"], key
        assert spec["layout"]["font"]["color"] == p["INK"], key


def test_gains_are_green_and_losses_are_red_in_the_charts(app):
    p = palette()
    f = figure(app, "se_map")
    stocks = next(t for t in f["data"] if t["name"] == "stocks")
    scale = dict((v, c) for v, c in stocks["marker"]["colorscale"])
    assert scale[0] == p["RED"] and scale[1] == p["GREEN"]  # a negative Sharpe ratio is red, a positive one is green
    dd = figure(app, "se_drawdown")["data"][0]
    assert dd["line"]["color"] == p["RED"] and dd["fillcolor"].startswith("rgba(244, 63, 94")  # drawdown (a loss) is red
    var = figure(app, "var_chart")
    assert any(t.get("marker", {}).get("color") == p["RED"] for t in var["data"])  # breaches are red marks


def test_the_price_tape_equals_an_independent_calculation_from_the_stored_prices(app, prices):
    tape = next(m.value for m in app.markdown if 'class="tape"' in m.value)
    last = prices.index[-1]
    assert f"Close {last.date()}" in tape
    rows = re.findall(r'<span class="sym">([^<]+)</span><span class="px">([^<]+)</span><span class="(up|dn|flat)">(?:&#\d+;) ([+\-][\d.]+%)</span>', tape)
    assert len(rows) >= 8
    for sym, px, cls, chg in rows:
        s = prices[sym].dropna()
        assert s.index[-1] == last
        assert px == f"{s.iloc[-1]:,.2f}"
        change = s.iloc[-1] / s.iloc[-2] - 1.0
        assert chg == f"{change:+.2%}" and cls == ("up" if change > 0 else ("dn" if change < 0 else "flat"))
        assert (f"&#9650; {chg}" in tape) if change > 0 else (f"&#9660; {chg}" in tape or change == 0)  # direction is also an arrow and a sign, not colour alone
    assert 'role="region"' in tape and 'aria-label="Latest adjusted closing prices"' in tape and "tabindex" in tape


def test_a_ticker_that_did_not_trade_on_the_last_date_is_left_out_of_the_tape_not_carried_forward():
    import pandas as pd

    ns = {"prices": None}
    start = SRC.index("TAPE_TICKERS =")
    end = SRC.index("# ----------------------------------------------------------------------------- header")
    exec("import pandas as pd\n" + SRC[start:end], ns)
    idx = pd.bdate_range("2024-01-02", periods=4)
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 102.0, 103.0], "AAPL": [10.0, 11.0, np.nan, np.nan], "MSFT": [50.0, 50.0, 50.0, 49.0]}, index=idx)
    html = ns["price_tape"](prices)
    assert "SPY" in html and "MSFT" in html and "AAPL" not in html
    assert "&#9660; -2.00%" in html and "&#9650; +0.98%" in html
    flat = ns["price_tape"](prices.assign(MSFT=[50.0, 50.0, 50.0, 50.0]))
    assert "&#9644; +0.00%" in flat and 'class="flat"' in flat


def test_the_web_report_and_the_loading_page_use_the_same_palette():
    site = (ROOT / "scripts" / "build_site.py").read_text()
    p = palette()
    for name in ("BG", "PANEL", "BORDER", "INK", "MUTED", "GREEN", "RED", "BLUE", "AMBER"):
        assert f'"{p[name]}"' in site, name
    assert not re.findall(r"#(?:F8FAFC|FFFFFF|E2E8F0|DBEAFE|E9EEF6|FFF7ED)\b", site, re.I)
