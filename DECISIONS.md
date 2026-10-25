# Decisions

Stage 1 listed these as proposals. After the Stage 1 review the owner asked for everything judged wrong to be fixed, with one ground rule: no data may be created, assumed, simulated or filled in; everything uses real data. The decisions below record what was applied and any change from the Stage 1 proposal.

| ID | Decision | Status | Reason | Change from the proposal |
|---|---|---|---|---|
| D1 | Remove the Production Ready, Plotly and Python 3.7+ badges, "not a class project", and every comparison with professional or institutional practice | applied | No evidence supports them | none |
| D2 | Remove the ML module (per-stock forecasts, signals, intervals) and the optimizer's use of predictions | applied | Stale inference row, overlapping targets across the split, no baseline, interval not a prediction interval. Repairing it properly is new research | none |
| D3 | Remove annualisation and dollar VaR; one-day VaR and ES only | applied | Reported VaR exceeded 100%; validated and displayed quantities differed | the optional 10-day figure was not added |
| D4 | Remove "Monte Carlo" VaR | applied | It sampled the normal distribution already used by the parametric method | none |
| D5 | Delete Cornish-Fisher, Risk Parity, Min Vol, efficient frontier, heatmaps and attribution from the README | applied | They did not exist in the code | none; none was implemented |
| D6 | Optimizer: remove unused parameters, never relax screens, label in-sample figures, report out-of-sample results | applied | Two of three saved portfolios ran without their screen; "expected" figures were in-sample | Added a single train/test split (training to 2023-12-29, test from 2024-01-02) so that out-of-sample figures exist. This replaces simply deleting the "expected" numbers and is the only structural addition |
| D7 | Describe the universe as a hand-maintained list; keep ETFs out of portfolio eligibility and flag them in the metrics file; fix dash tickers; record failures with reasons | applied | The list was not shown to be the S&P 500; ETFs were ranked as stocks; BRK-B and BF-B were dropped by a symbol conversion | none |
| D8 | Remove hard-coded advice and promotional output | applied | Printed regardless of results | none |
| D9 | No dashboard | deferred | Build only after the core is accepted | none |
| D10 | Package structure instead of a single notebook | changed | Stage 1 proposed fixing logic inside the notebook. Tests on real data need importable functions, and a notebook cannot be tested in isolation. The notebook is now a thin walkthrough with no logic | Reversal of the Stage 1 proposal |
| D11 | README tables are generated from results files | applied | The README total (468) mixed requested and fetched counts | stronger than proposed: a test fails if the README block differs from the results |
| D12 | Price data is not committed; `data/prices_meta.json` and the code that regenerates the snapshot are | applied | Size and vendor licensing; provenance is preserved by the metadata | none |
| D13 | No contact section in the new README | applied | LinkedIn slug and email in the original README could not be verified | Owner can add a contact section after confirming |
| D14 | Keep `AUDIT.md`, `CLAIMS.md`, `PROGRESS.md`, `DECISIONS.md` in the repository root | applied | Evidence behind the changes | `audit/checks.py` was removed because it used synthetic data (kept in commit `47cf740`) |
| D15 | No synthetic data anywhere, including tests | applied (new rule) | Owner's ground rule | Tests use the real snapshot; expected values come from independent calculations (closed forms, numerical integration, hand-written loops, the binomial log-pmf). The few scalar-argument tests of the Kupiec function (counts such as n=1500, x=0) are function arguments, not market data. Two tests use `monkeypatch` to change screen thresholds (parameters, not data) |
| D16 | Risk-free rate from the real 13-week T-bill yield (`^IRX`) averaged over each window, replacing the hard-coded 4% | applied | A fixed 4% was an assumption presented as data | none |
| D17 | Remove `LT_Score`, RSI and moving-average fields | applied | The score had unjustified weights mixing units; the technical indicators had no role in any retained analysis | removed rather than repaired |
| D18 | Do not add the Christoffersen independence test | applied | New method, outside the repair scope | The original "clustering score" was removed instead; the README states that only the Kupiec test is implemented |
| D19 | Analysis settings (window 250 days, confidence 95% and 99%, training end 2023-12-31, screen thresholds from the original notebook divided by sqrt(252) to express them as one-day figures) are in `riskengine/config.py` | applied | One place for every parameter | The settings were written before any result was computed on the downloaded data; they were not changed afterwards. This is a statement about the working session and cannot be verified from the repository alone |

## Open items for the owner

1. Keep the ML module removed, or commission a properly validated version as a separate feature?
2. Add a contact section to the README (D13)?
3. Which optional features in `PROGRESS.md` section 4, if any, to build next?
4. Remove the stray user-level Jupyter kernel created during the session (`PROGRESS.md` section 7).
