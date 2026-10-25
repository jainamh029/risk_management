# Decisions (Stage 1)

All entries are proposals for the repair stage. Nothing has been removed or changed in the repository. Each needs the owner's approval before it is applied.

| ID | Proposal | Reason | Alternative | Audit / claim refs |
|---|---|---|---|---|
| D1 | Remove the "Production Ready" and "Plotly Interactive" badges, the Python 3.7+ badge, the line "It is not a class project", and every comparison with professional or institutional practice | No evidence in the repository supports any of them; Plotly is imported and never used | Keep only a Python-version badge after testing a stated version | A1-A4, A7, A67 |
| D2 | Remove the ML module (per-stock random forest and gradient boosting, signals, intervals) from the core, along with the optimizer's use of predictions | The "latest" prediction uses features from 64 days before the last date; overlapping targets leak across the split; no baseline; the interval is not a prediction interval; fixing it properly is new work, not a repair | Keep it, repair it (true latest row, embargo, baseline, no interval or BUY/SELL labels) and present results even if the model has no skill | B8, B9, A5, A11, A39-A41 |
| D3 | Remove annualisation of VaR/ES by sqrt(252) and the dollar VaR derived from it. Report 1-day VaR and ES, and state the horizon in every column name | Reported VaR exceeds 100% of position value; the validated quantity (1-day) differs from the displayed one | Keep a clearly labeled multi-day figure scaled by sqrt(h) for a stated h such as 10 | B2, A25 |
| D4 | Delete the "Monte Carlo" VaR | It samples the same normal distribution as the parametric method and is not independent | Replace with a method that uses a different distribution (new feature) | B7, B1c |
| D5 | Delete Cornish-Fisher, Risk Parity, Minimum Volatility, efficient frontier, correlation heatmaps, attribution and similar items from the README in the core repair | They do not exist in the code. Implementing any of them is a new feature | Implement later as optional features (`PROGRESS.md` section 6) | A24, A43, A47-A51, A60, A63 |
| D6 | Optimizer: remove `max_var`; stop silently relaxing the screens (fail with a clear message, or report that the screen was relaxed in the output); rename or describe the three "strategies" as three screens over one max-Sharpe objective; label portfolio figures as in-sample | Two of three portfolios in the saved run ran without their screen; "Expected Return" is an in-sample statistic | Add an out-of-sample walk-forward test (new feature) | B10, B9c |
| D7 | Describe the universe as "a hand-maintained list of 483 unique tickers including 10 ETFs"; keep ETFs out of stock rankings (or label them separately); fix the dash handling for `BRK-B` and `BF-B`; list tickers that fail with reasons | The list is not shown to be the S&P 500; ETFs are ranked as stocks; two active tickers were dropped by a symbol conversion | Replace the list with point-in-time index membership (new feature, needs a data source) | B4, B15, A6, A18, A65 |
| D8 | Remove the hard-coded "INVESTMENT RECOMMENDATIONS", "ready for investment decisions", "TOP INVESTMENT PICKS", "Perfect for LinkedIn" text; keep a neutral summary of computed values | Printed regardless of results; contradicts the disclaimer | Replace with output derived from computed results only | B14, B6c-B8c, B12c |
| D9 | Defer the dashboard | An earlier planning prompt included one. Adding it before the numbers it would display are correct would present unverified figures more attractively | Build it after core repairs, reading from generated results files | out of scope for Stage 1 |
| D10 | Core repair stage keeps the single-notebook structure | The smallest correct change is to fix the logic where it is. Extracting a package is a refactor best done once behaviour is correct and tested | Extract a package first, then fix | n/a |
| D11 | Use the pre-existing README sector counts only as "requested" counts, and report fetched counts from a generated file | The README total (468) is the number fetched, while the rows sum to 484 | Remove the table | A64, A65 |
| D12 | Do not commit downloaded price data in the core repair; commit only a small metadata file (download date, tickers requested and returned, tickers dropped and why) and the code to regenerate | Size and licensing of vendor data; reproducibility comes from metadata plus a pinned retrieval script | Commit a compressed snapshot if the owner accepts the size | B13 |
| D13 | Contact details: leave unchanged until the owner confirms the LinkedIn slug and email | Cannot be verified from the repository | Use the owner's current details once supplied | A69, S10 |
| D14 | Audit script and notes live in `audit/` and `*.md` files at the repository root, and are committed on the local branch only | They are the evidence behind the findings; they change no source file | Keep them untracked outside the repository | n/a |

## Open questions for the owner

1. Remove the ML module (D2) or keep and repair it?
2. Is a stated 10-day horizon figure wanted (D3), or 1-day only?
3. Should ETFs stay in the universe as a separate group or be dropped?
4. Confirm the contact details to publish (D13).
