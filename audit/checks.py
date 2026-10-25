"""Stage 1 audit checks. Run from the repository root:

    python3 -I audit/checks.py

Reads the ORIGINAL notebook (read-only) and executes selected original code on
SYNTHETIC data to test specific behaviours. Nothing here is a market result and
nothing here downloads data.
"""
import collections
import contextlib
import io
import json
import math
import re
import textwrap

import numpy as np
import pandas as pd
from scipy.stats import norm

NB = "Risk_Management_and_Var_es_engine_for_stocks.ipynb"
nb = json.load(open(NB))
src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}

print("== 1. syntax check of each code cell")
for i, s in src.items():
    try:
        compile(s, f"cell{i}", "exec")
        print(f"cell {i}: compiles")
    except SyntaxError as e:
        print(f"cell {i}: SYNTAX ERROR {e}")

print("\n== 2. universe list parsed from cell 1")
m = re.search(r"self\.tickers = \[(.*?)\n        \]", src[1], re.S)
body = m.group(1)
tickers = re.findall(r"'([A-Z\.\-]+)'", body)
print("entries in literal:", len(tickers), "| unique:", len(set(tickers)))
print("duplicated entries:", [k for k, v in collections.Counter(tickers).items() if v > 1])
print("tickers containing '-':", sorted(x for x in set(tickers) if "-" in x))
cur, counts = None, {}
for line in body.splitlines():
    mm = re.search(r"# ([A-Za-z ]+) \((\d+) (?:stocks|ETFs)\)", line)
    if mm:
        cur = (mm.group(1), int(mm.group(2)))
        counts[cur] = 0
        continue
    if cur:
        counts[cur] += len(re.findall(r"'([A-Z\.\-]+)'", line))
for (n, claimed), actual in counts.items():
    print(f"  {n}: comment says {claimed}, list has {actual}")
print("sum of README sector-table lower bounds + 10 ETFs:", 50 + 63 + 61 + 72 + 56 + 37 + 23 + 24 + 26 + 31 + 31 + 10)

print("\n== 3. VaRESEngine (original cell 3 code) on synthetic iid normal returns, mu=0.0005, sigma=0.02")
ns = {"np": np, "pd": pd, "norm": norm}
exec(src[3], ns)
rng = np.random.default_rng(0)
idx = pd.bdate_range("2019-01-02", periods=1800)
ret = pd.DataFrame({"SYN": rng.normal(0.0005, 0.02, 1800)}, index=idx)
eng = ns["VaRESEngine"](ret)
with contextlib.redirect_stdout(io.StringIO()):
    df = eng.calculate_var_es(window=250)
r = ret["SYN"].tail(250)
mu, sg = r.mean(), r.std()
s252 = np.sqrt(252)
for conf in (0.95, 0.99):
    z, a, c = norm.ppf(conf), 1 - conf, int(conf * 100)
    print(f"conf {c}: code Param_VaR/sqrt(252) = {df[f'Param_VaR_{c}'].iloc[0] / s252:+.5f} | closed-form 1-day VaR = {-(mu - z * sg):+.5f}")
    print(f"         code Param_ES /sqrt(252) = {df[f'Param_ES_{c}'].iloc[0] / s252:+.5f} | closed-form 1-day ES  = {-mu + sg * norm.pdf(z) / a:+.5f}")
    print(f"         code MC_VaR   /sqrt(252) = {df[f'MC_VaR_{c}'].iloc[0] / s252:+.5f} (draws from Normal(mu, sigma))")
print("code Hist_VaR_99 (sqrt(252)-scaled) for this series:", round(df["Hist_VaR_99"].iloc[0], 4), "(1.0 = 100% of position)")

print("\n== 4. Kupiec boundary case: original nested kupiec_test with zero breaches")
mm = re.search(r"( {12}def kupiec_test\(.*?return lr\n)", src[3], re.S)
kns = {"np": np}
exec(textwrap.dedent(mm.group(1)), kns)
k = kns["kupiec_test"]
for p, n in ((0.01, 1500), (0.05, 1500)):
    print(f"kupiec_test(0, {p}, {n}) = {k(0, p, n)} | closed-form LR for zero breaches = {-2 * n * math.log(1 - p):.4f}")


def lr_pof(n, x, p):
    ll0 = (n - x) * math.log(1 - p) + (x * math.log(p) if x > 0 else 0)
    ll1 = ((n - x) * math.log(1 - x / n) if x < n else 0) + (x * math.log(x / n) if x > 0 else 0)
    return -2 * (ll0 - ll1)


print("\n== 5. Kupiec formula for x>0 against independent log-likelihood form (saved notebook rows)")
print("n=1525, x=23, p=0.01 -> independent LR", round(lr_pof(1525, 23, 0.01), 6), "| saved output 3.441928")
print("n=1525, x=24, p=0.01 -> independent LR", round(lr_pof(1525, 24, 0.01), 6), "| saved output 4.317578")
print("underflow probe: (0.95)**(n-x) * 0.05**x for n=10000, x=500 ->", (0.95) ** 9500 * 0.05 ** 500, "(0.0 means the product underflows; the code takes log of this product)")

print("\n== 6. ffill().bfill() with leading NaNs, then the 30%-missing filter")
p = pd.DataFrame({"OLD": np.linspace(100, 120, 10), "NEW": [np.nan] * 6 + [50, 51, 52, 53]})
f = p.ffill().bfill()
print("NEW after ffill/bfill:", f["NEW"].tolist())
miss = f.isna().sum() / len(f)
print("missing share after fill:", miss.tolist(), "-> columns kept by 'missing_pct < 0.3':", list(f.columns[miss < 0.3]))
print("returns of NEW:", f["NEW"].pct_change().dropna().round(4).tolist())

print("\n== 7. optimizer")
print("occurrences of max_var in cell 9 (1 means only the signature):", len(re.findall(r"\bmax_var\b", src[9])))
print("bounds feasibility: weights summing to 1 need n*lower <= 1 <= n*upper; (0.05, 0.25) -> 4 <= n <= 20; (0.02, 0.30) -> 4 <= n <= 50")

print("\n== 8. requirements.txt vs third-party imports")
req = [l.strip() for l in open("requirements.txt") if l.strip()]
imps = set(re.findall(r"^(?:import|from)\s+([A-Za-z0-9_]+)", "\n".join(src.values()), re.M))
pkgmap = {"sklearn": "scikit-learn", "yfinance": "yfinance", "plotly": "plotly", "scipy": "scipy", "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib", "seaborn": "seaborn"}
third = sorted(pkgmap[i] for i in imps if i in pkgmap)
print("third-party imports:", third)
print("requirements.txt   :", req)
print("missing            :", sorted(set(third) - set(req)))

print("\n== 9. ML index arithmetic (saved run had 1776 prices and 1775 returns)")
L, lookback = 1775, 60
last_i = L - 63 - 1
print(f"last feature row i={last_i}; its target uses prices[{last_i + 63}] of prices 0..{L}; the row used for 'latest' prediction ends {L - last_i} return-days before the last date")
n_samples = L - 63 - lookback
split = int(0.8 * n_samples)
print(f"samples={n_samples}, train={split}, test={n_samples - split}, approx non-overlapping 63-day windows in test={(n_samples - split) // 63}")

print("\n== 10. pandas frequency alias used in cell 5 (resample('M')) on the installed pandas")
_s = pd.Series(np.random.default_rng(0).normal(size=100), index=pd.bdate_range("2020-01-01", periods=100))
for _alias in ("M", "ME"):
    try:
        print(f"pandas {pd.__version__}: resample('{_alias}') ->", len(_s.resample(_alias).sum()), "periods")
    except Exception as _e:
        print(f"pandas {pd.__version__}: resample('{_alias}') raises {type(_e).__name__}")
print("cell 5 wraps this call inside try/except Exception, which prints a truncated message and skips the ticker")
