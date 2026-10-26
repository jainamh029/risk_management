# Progress

Branch `fix/rigor-pass`. Nothing has been pushed; no pull request, remote change or message was made.

Owner's ground rules: no data is created, assumed, simulated or filled in; every analysis and every test uses the real snapshot downloaded from Yahoo Finance. In the latest request the owner also asked that the ML module stay, that every formula, method and piece of logic be checked, that the whole project be tested with the result reported, that the dashboard be interactive, and that no synthetic data remain.

## 1. Status

| Request | Status |
|---|---|
| Stage 1 audit | done (`47cf740`) |
| Core repairs (VaR, backtest, metrics, portfolio, data layer) | done (`34b4357`, `fd2fce9`, `104a584`) |
| ML module stays | done: rebuilt with an honest evaluation (section 4) |
| Check every formula, method and piece of logic | done: `FORMULAS.md`; two defects found and fixed (section 3) |
| Run the tests on the whole project and report | done (section 2) |
| Interactive dashboard | done: `dashboard/app.py`, six tabs, tested headless and in a browser |
| No synthetic data in the project | done: scan in section 5; original notebook removed from the working tree |
| Test the dashboard | done: headless tests plus a full browser pass (section 2) |
| Test the backtest | done: independent re-implementation matches all 1,832 results (section 2) |
| Check the ticker list against S&P 500 membership | done against Wikipedia's record (section 9) |
| Push to GitHub once everything passes | see section 10 |

## 2. Test results on the whole project (final)

| Check | Result |
|---|---|
| `pytest tests` in the working environment | 106 passed in 108 s (backtest 13, dashboard 13, data 9, membership 9, metrics 9, ML 17, portfolio 18, README and documents 9, VaR 9) |
| Same suite in a brand-new virtual environment built only from `requirements.txt` | see section 10 |
| `scripts/run_pipeline.py` rerun in the clean environment, then `diff -rq results` against the repository | every file byte-identical (after rounding the forecast files to 10 decimals; before the rounding, `ml_latest_forecasts.csv` differed by 1.1e-16 from multithreaded floating-point summation, with the same ranking) |
| `scripts/make_readme_tables.py --check` | README results block matches the results files |
| `notebooks/walkthrough.ipynb` executed with nbclient | 6 code cells, 0 errors, 4 figures |
| Dashboard, headless (Streamlit AppTest) | 13 tests: loads, overview numbers equal the results files, VaR explorer breach counts equal an independent loop, backtest filters, portfolio presets and custom screen, forecast table filters, every control option, data tables, missing-data message, no unsupported text |
| Dashboard, live browser | opened at `localhost:8501` and used twice. First pass: Overview, VaR explorer (changed confidence to 95% and saw SPY update to 86 breaches of 1,713), Portfolio experiment and Forecasting model. Second pass: Overview with the new membership text; Backtest comparison (bars with hatching and expected marks, ticker filter returned exactly the 12 results expected for 3 tickers, p-value histogram with a pile-up near zero, sortable table); Glossary; keyboard navigation (arrow keys moved through all six tabs, focused tab had a solid 3px blue outline); phone width 375px (no horizontal overflow, chart 343px wide). Layout problems found and fixed: chart title overlapping legend and range buttons; misleading upward arrow on "expected" values; default portfolio preset was the one that cannot be built. The server was stopped afterwards |
| Independent backtest verification | `scripts/verify_backtest_independent.py` (plain loops and `statistics.NormalDist`, no package code): recomputed every forecast, breach count, Kupiec statistic and p-value for 1,832 ticker x method x confidence results and the summary table; all match (31 s) |
| Defect reintroduction: 21 original or likely defects put back one at a time into a scratch copy | 21 of 21 made the suite fail. Misses found along the way and fixed in the tests: screen relaxation (two boundary tests added); drawdown without starting capital (test changed to use real series where the definitions differ); a mutant that changed nothing was discarded |

## 3. Formula and logic review

Every formula is listed in `FORMULAS.md` with the independent test that checks it. Defects found in this pass:

| Defect | Fix | Guard |
|---|---|---|
| Maximum drawdown in `portfolio.performance` ignored the starting capital as a peak | start the wealth series at 1.0 | `test_max_drawdown_counts_the_starting_capital_as_a_peak` |
| ML direction accuracy treated an exactly flat return differently from the baseline | both use up versus not up | `test_baseline_and_scores_match_independent_formulas` |

Other logic reviewed and found correct: return calculation, VaR and ES definitions, rolling forecast alignment, Kupiec statistic and p-value, breach definition, descriptive statistics, screen and optimiser constraints, constant-weight portfolio returns, feature windows, target alignment, purge boundary, rank IC and its independent-period inference. Judgement calls that are not errors but are conventions are listed at the end of `FORMULAS.md`.

## 4. ML module: what the real data says

Pooled random forest and gradient boosting (equal blend), 11 price-based features, 63-day forward return, 94,371 training samples (feature dates 2019-07-10 to 2023-09-25, all targets ending by 2023-12-22), 287,999 test samples (2024-01-02 to 2026-07-27). On the test period the models do not beat the training-mean forecast: out-of-sample R squared is -0.9% (random forest), -6.0% (gradient boosting) and -3.0% (ensemble); direction accuracy is about 53.6% against 57.5% of samples that went up; the rank IC of 0.044 is not distinguishable from zero (t = 0.60 on 11 independent periods). Neither the 60-day nor the 126-day momentum feature alone shows a significant rank IC on independent periods either. This is reported as a negative result, in the README and in the dashboard.

## 5. Synthetic-data scan

Searched every tracked and working file outside `.venv` for random-number generation, generators and the words synthetic, simulated, fake, dummy and mock.
- Package, scripts, tests, dashboard and notebook: no random-number generation anywhere. The only matches are the guard test's own ban list and comments that say data is not simulated. `test_no_data_is_generated_or_simulated_in_code_or_in_the_unit_and_integration_tests` enforces this.
- The original notebook contained `np.random.seed(42)` and `np.random.normal(mu, sigma, 10000)` (its "Monte Carlo" VaR). It was removed from the working tree and remains in git history (`git show 888104c:Risk_Management_and_Var_es_engine_for_stocks.ipynb`).
- The Stage 1 audit documents quoted numbers from synthetic demonstrations. Those numbers were replaced by the same quantities measured on the real snapshot (for example: on the last 250 real SPY returns the original VaR expression gives -0.01922 at 99% against the correct +0.01815; 90 of 458 tickers had a sqrt(252)-scaled 99% VaR above 100%; 1,196 exact-zero returns would have been created for VLTO by back-filling).
- Remaining mentions of the word "synthetic" in the documents are statements that Stage 1 used it and that it was removed.
- Every price and rate in `data/` and `results/` comes from Yahoo Finance via yfinance (`data/prices_meta.json` records the download).
- The only fixed numeric constants in tests that are not computed from the snapshot are the two Kupiec statistics saved in the original notebook's output from a real run (3.441928 and 4.317578) and analysis settings in `riskengine/config.py`.

## 6. Commands that mattered, and what went wrong

| Event | Detail |
|---|---|
| Notebook kernel registered with `--user` | Created `~/Library/Jupyter/kernels/python3` (outside the repository). The permission system declined my deletion and I did not work around it. It is still there; removal command in section 8 |
| Two shell calls blocked by a permission-service error | Retried unchanged; both succeeded afterwards |
| A bad mutant (`jac=None`) reported "not caught" | Discarded as meaningless and replaced by real optimiser mutants |
| A test of mine used a wrong tolerance for the historical quantile | Corrected the test, not the code |
| A test indexed a row before VLTO was listed | Corrected the test |
| Streamlit and Python caches made a "missing data" test see the real data | Test now clears the cache and uses a loader that raises the real exception |

## 7. Not done or not verified

- Membership was compared with Wikipedia, a volunteer-maintained record, not S&P's own. Renamed symbols cannot be matched by symbol, so some stocks counted as non-members are renamed members; this was not resolved.
- Dashboard reduced-motion behaviour is set in CSS but was not tested with the operating-system setting.
- No independence test for VaR breaches; no multi-day VaR; no minimum-variance, risk-parity or walk-forward allocation; no Ledoit-Wolf covariance. All listed as optional features in earlier notes and not requested.
- Commit author is the laptop default, not a GitHub address.

## 8. Needs owner action

```bash
ls -la ~/Library/Jupyter/kernels/python3     # should hold only kernel.json and three logo files
rm -r ~/Library/Jupyter/kernels              # removes the directory created during this session
```

## 9. Ticker list versus S&P 500 membership

Source: Wikipedia's current constituents (revision 1381923657) and its table of index changes (revision 1379179365, 412 changes from 1976 to 2026-10-06), fetched 2026-10-25 and stored in `data/`. Wikipedia is not S&P's own record. Results (also in the README):
- 483 tickers: 473 stocks and 10 ETFs. 412 of the 473 stocks are current members; 91 current members are missing from the list.
- Over the price window (2019-01-02 to 2026-10-23): 357 stocks were members on every date, 105 on some dates, 11 on none when matched by symbol (5 of them have price data: AGL, DINO, EQR, FNF, PARA). The history table records DINO's earlier symbol HFC as a member until 2021-06-04 and describes Equity Residential as renamed, so some of these are renamed members; not resolved.
- Price data covers 75% of the index's members on the first date and 82% on the last.
- 95.0% of backtested forecast days fell on days when the stock was an index member. Restricting the backtest to member days changes the pooled breach rates by less than 0.02 percentage points (historical 99%: 1.588% all days, 1.583% member days), so the VaR conclusions do not depend on this.
- The rebuilt index has 503 to 508 members on every date, consistent with a complete table.
- The source tables contradict each other for one ticker (SATS: added 2026-03-23, absent from today's list, no removal recorded). It is not in the list.
- A parsing bug of mine (dropping the first table row, the 2026-10-06 WBD and TWLO change) was found by checking WBD and fixed; a test now guards it.
- Conclusion: the list is not the S&P 500. It is a hand-typed list that overlaps it heavily.

## 11. Acceptance-test stage (validation before merging into main)

Full details are in `VALIDATION_REPORT.md`. Summary: 212 tests (196 passed, 16 failed, 0 skipped, 0 blocked) in the working environment and again in a clean virtual environment; an independent recomputation of all 1,832 backtest results matches; with the price snapshot removed, 103 tests report BLOCKED and the run exits with status 10. The 16 failures come from two input-validation gaps (`var.py` does not validate confidence, finiteness or sample size; `ml.score` raises `ZeroDivisionError` for a zero denominator). No model or analysis code was changed in this stage. The test directories were restructured into `tests/unit`, `tests/integration` and `tests/statistical`. The counts in section 2 above describe the earlier 106-test suite and are kept as history.

## 12. Repair stage (bounded repair of the validation failures)

Baseline `970faf1` reproduced the reported 212 tests (196 passed, 16 failed). After the repair the complete suite is 350 tests, 350 passed (working environment and a clean virtual environment), with the real-data results unchanged except for one added field (`oos_r2_status`) in `results/ml_summary.json`. See `VALIDATION_REPORT.md` Part 2. Repaired: `var.py` input validation and `ml.score`'s undefined R squared (NaN in code, null in JSON, "undefined" in README and dashboard). Remaining: `backtest.rolling_forecasts` does not validate its confidence argument; RuntimeWarnings from `np.corrcoef` on constant targets. Nothing was pushed.

## 13. Final bounded repair (rolling-forecast confidence, undefined rank IC)

Baseline `4ccb8bb`. Both defects were reproduced, repaired and covered by 101 new tests. The complete suite is 451 tests, 451 passed, 0 warnings, in the working environment and in a fresh virtual environment; the independent verifier matches all 1,832 results; all 12 result files and the README are byte-identical to the pre-edit copies. See `VALIDATION_REPORT.md` Part 3. Left as decisions for the owner: whether infinite returns inside a rolling history should make windows unavailable, and whether the rank-IC status should be stored in the results files. Nothing was pushed.

## 14. Module-by-module acceptance audit

Baseline `154a5a0` (451 tests). Every module and entry point was compared with an independent reference and probed with valid, invalid and boundary input; the result is in `MODULE_TEST_MATRIX.md` (per module) and `ALL_MODULES_VALIDATION.md` (commands, counts, coverage, defects, conclusions). Ten defects were found and repaired with regression tests (input validation in the data, backtest, configuration, scoring and optimiser code; beta against a constant benchmark; a risk-free download error path; the membership script's handling of damaged cached tables; three dashboard failure modes; and last-bit nondeterminism of the forest's predictions). The complete suite is now 679 tests, 679 passed in the working environment and in a fresh virtual environment; with the snapshot removed it reports 151 BLOCKED, exits with status 10 and makes no network attempt. Statement and branch coverage of the measured files is 98% (the uncovered lines are the network fetch of the membership tables and five dashboard lines that only run in temporary copies of the app). All result files, the generated README block and the notebook are byte-identical to the copies taken before the audit. Nothing was pushed.

## 15. The two numerical edge cases left by the audit

Baseline `8512e60` (679 tests). An infinite value anywhere in a rolling history now raises `ValueError` (NaN keeps its policy: affected windows have no forecast; D42 closed). The Sharpe ratio is undefined at zero volatility: NaN in reported figures, and the optimiser refuses such a portfolio so the preset is "not constructed" with the reason instead of returning a successful-looking allocation (D48; a numerical floor of 1e-12 on annualised volatility). 40 regression tests; the complete suite is 719 tests, 719 passed in the working and a fresh environment; every result file, the README block and the notebook are byte-identical to their pre-change copies. Details: `ALL_MODULES_VALIDATION.md` section 12. Nothing was pushed.

## 16. Incremental data update

Added `scripts/update_data.py` and `data.update_snapshot` (D49): the snapshot is brought up to date by downloading only the days after the last stored date, with a one-week overlap compared with the stored prices; tickers whose stored prices no longer match Yahoo's (dividend, split, restatement) are downloaded in full again and replaced; new and previously failed tickers are downloaded in full; if nothing is new no file is touched; files are written atomically; invalid data is refused before anything is written. 19 new tests against a stand-in provider (two at script level). The real Yahoo Finance update has not been run, so the checked-in snapshot, results, README tables and web report are unchanged. To update for real: `python scripts/update_data.py --rebuild`.

## 17. The interactive dashboard on GitHub Pages

`scripts/build_site.py` now also writes `docs/app/`: a page that starts the unchanged `dashboard/app.py` in the browser with stlite 1.12.0 (Streamlit 1.65.0 on Pyodide, Python 3.14), plus byte copies of the code, `riskengine/`, `results/` and the price snapshot. Checked by hand in the browser pane on a local server: all seven tabs, 15 charts, no exceptions; the VaR explorer recomputed on a control change with numbers equal to the native run; a custom portfolio built (scipy in WebAssembly); the forecast table downloaded. Known quirks: the first load takes about a minute; a harmless console message about a download-button fetch and a missing `toml` module appear. D50.

## 18. Daily automatic update

`.github/workflows/daily-update.yml` (D51): at 23:00 UTC on weekdays it restores the price snapshot from the `pages` branch, runs `scripts/update_data.py`, and when new days arrived regenerates the results and README tables, runs the verifier and the whole test suite, commits to `main`, and publishes the web pages on the `pages` branch. Holidays and weekends add no rows and change nothing; a missed run is caught up by the next. Real-data check of the incremental method (2026-10-26): a copy of the snapshot cut back to 2026-10-16 and updated to 2026-10-23 equals the full download for 461 of 462 tickers (every price identical); the exception, PARA, was reassigned by Yahoo to another company, which led to the 90% coverage rule (D52). Tests that hard-coded numbers of the current snapshot now follow the data. Pages source moved to the `pages` branch; `docs/` left main.

## 19. Trading-terminal theme

Dark navy theme for the dashboard, the web report and the loading page (D54): faint chart-grid background, a strip of last closes and one-day changes of 14 large tickers, green for gains and red for losses, dark chart panels through one `themed()` function. Contrast of every text and chart colour is tested (at least 4.5:1). Checked by eye in the browser pane on the overview, stock explorer and VaR explorer tabs and on a phone-width view; two details were fixed that way (map markers too dark near a Sharpe ratio of zero; a tape label that hid the tickers on a phone).

