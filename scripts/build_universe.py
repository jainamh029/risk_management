"""Build data/universe.csv from the ticker list in the original notebook (git commit 888104c).

The list is the original author's hand-typed list. It is not verified against
any index membership. Order is preserved, the first occurrence of a duplicate
is kept, tickers keep their Yahoo Finance form (BRK-B, not BRK.B).

    python scripts/build_universe.py
"""
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "888104c"
NB_NAME = "Risk_Management_and_Var_es_engine_for_stocks.ipynb"
ETFS = {"SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "IVV", "EFA", "EEM", "AGG"}

nb = json.loads(subprocess.run(["git", "show", f"{COMMIT}:{NB_NAME}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
src = "".join(nb["cells"][1]["source"])
body = re.search(r"self\.tickers = \[(.*?)\n        \]", src, re.S).group(1)

rows, seen, group = [], set(), None
for line in body.splitlines():
    m = re.search(r"# ([A-Za-z ]+) \(\d+ (?:stocks|ETFs)\)", line)
    if m:
        group = m.group(1).strip()
        continue
    for t in re.findall(r"'([A-Z\.\-]+)'", line):
        if t in seen:
            continue
        seen.add(t)
        rows.append((t, group, int(t in ETFS)))

out = ROOT / "data" / "universe.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ticker", "group_in_original_list", "is_etf"])
    w.writerows(rows)
print(f"wrote {out} with {len(rows)} unique tickers, {sum(r[2] for r in rows)} ETFs")
