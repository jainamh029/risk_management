# All-modules validation

A module-by-module acceptance audit of the repository: does every module behave as intended, do the modules work
together, and does the documentation say only what the code does. Validation first: no new feature, no tuning, no
methodology change. The per-module detail (intended behaviour, independent reference, tests, status) is in
`MODULE_TEST_MATRIX.md`; this file records what was run, what it returned, what was found and what is not verified.

Nothing in this document was estimated or remembered: every count below is copied from a run log or junit file
produced in this session, and the dashboard figures from the live browser smoke were compared with the stored results.

## 1. Commit and environment

| Item | Value |
|---|---|
| Base commit | `154a5a02234616badb6ac48706c5e4e33796a413` on branch `fix/rigor-pass` (this audit's changes are uncommitted until the commit that adds this file) |
| Operating system | Darwin 25.2.0, arm64 |
| Python | 3.12.2 (conda-forge build) |
| Working environment (`.venv`) | numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, yfinance 1.7.0, streamlit 1.66.1, plotly 7.1.0, lxml 6.1.3, pytest 9.1.1, nbconvert 7.17.2, nbclient 0.11.0, ipykernel 7.4.0 |
| Fresh environment | a new virtual environment created from the same interpreter, `pip install -r requirements.txt` (all pinned versions), exit 0; the suite ran there on a copy of the working tree |
| Dependency added | `pytest-cov==7.1.0` (brings `coverage==7.17.0`, also pinned) in `requirements.txt`, used only to measure coverage; `.coveragerc` added; `.coverage` files added to `.gitignore` |

### Frozen snapshot (no data was downloaded)

SHA-256, identical before and after the audit (checked with `shasum -a 256 -c` against a list taken before any edit):

```
a9a0c76423e6a28b9bb0de37b6e5a4d3e6be961e367a56f7bf26784004c7922e  data/prices.parquet
8deddbe70e309bb3243da8e753e9251c3a95bdd76db5716be2f13c4235f45aba  data/tbill_13w.parquet
000efc5756df561895e97a1d999da1e7db518e78b90814ac3bd7a93caa8b8687  data/prices_meta.json
d84ee026702c1668c9558c9069a3522897bb7580392e66cc3767958a6c95e025  data/universe.csv
ca471e1dcf27d689e2722511fbc6024475e18a4d3e0ed37f56d78abf912f5f43  data/sp500_current.csv
d8b6f3ba26964787351a5419844c1fc4421036ae65d2dbffa417ed2d31cdeba1  data/sp500_changes.csv
7149c332661a2a3e0afa4be895751a89ae39e3c089d689fd7d40df33e4a77aad  data/sp500_membership_meta.json
```

The snapshot covers 2019-01-02 to 2026-10-23 (1,964 rows, 462 of 483 requested tickers), downloaded 2026-10-25T05:57:48Z.

## 2. Module verdicts

| Module | Verdict | Evidence (section of `MODULE_TEST_MATRIX.md`) |
|---|---|---|
| `riskengine/data.py` | PASS | matrix section 1 (includes the mocked provider with retry delays mocked and sockets blocked) |
| `riskengine/var.py` | PASS | matrix section 2 |
| `riskengine/backtest.py` | PASS | matrix section 3 (includes the independent loop reimplementation of all 1,832 results) |
| `riskengine/metrics.py` | PASS | matrix section 4 |
| `riskengine/portfolio.py` | PASS | matrix section 5 |
| `riskengine/ml.py` | PASS (software); the empirical result is negative, see section 9 of this file | matrix section 6 |
| `riskengine/config.py`, `riskengine/serialize.py` | PASS | matrix section 7 |
| `scripts/build_universe.py`, `fetch_data.py`, `run_pipeline.py`, `make_readme_tables.py`, `build_notebook.py`, `verify_backtest_independent.py`, `check_sp500_membership.py` | PASS; the live download and `check_sp500_membership.py --fetch` are BLOCKED (network) | matrix section 8 |
| `dashboard/app.py` | PASS | matrix section 9 (headless tests of every tab, damaged files, undefined values; one live browser smoke) |
| `notebooks/walkthrough.ipynb` | PASS | matrix section 10 |
| README and documents | PASS | matrix section 10 |
| Test harness (BLOCKED reporting) | PASS | matrix section 10 and the missing-snapshot run (command 5) |

No module is FAIL. The audit found and repaired ten defects (section 5); all other behaviour was already correct.

## 3. Commands executed and their outcomes

All commands were run from `~/Desktop/risk_management_work` unless stated. Logs and junit files were kept in the
session's scratch area; counts are copied from them.

| # | Command | Result |
|---|---|---|
| 1 | Baseline, before any edit: `.venv/bin/python -m pytest -q` | exit 0, **451 passed**, 235 s (`baseline_junit.xml`) |
| 2 | Baseline coverage of the package only (scripts and dashboard were not measured then): `pytest --cov=riskengine --cov-branch` | 85.3% statements, 79.5% branches in total; `data.py` 39.9% (see section 6) |
| 3 | Final, working environment: `COVERAGE_PROCESS_START=$PWD/.coveragerc COVERAGE_FILE=$PWD/.coverage .venv/bin/python -m pytest -p no:cacheprovider -o junit_family=xunit1 --cov=riskengine --cov=dashboard --cov=scripts --cov-branch --cov-report=term-missing --cov-report=json:… --junitxml=… -q` (after `python -m coverage run scripts/verify_backtest_independent.py` to measure the verifier, which the tests run as a subprocess) | exit 0, **679 passed, 0 failed, 0 errors, 0 skipped, 0 blocked**, 360 s; junit: tests=679 failures=0 errors=0 skipped=0 |
| 3b | Same suite, working environment, run once more after the last edit to the documents (no coverage): `.venv/bin/python -m pytest -q -p no:cacheprovider` | exit 0, **679 passed**, 276 s |
| 4 | Final, fresh virtual environment, on a copy of the working tree (`cd <copy> && <fresh venv>/bin/python -m pytest -p no:cacheprovider -o junit_family=xunit1 --junitxml=… -q`) | exit 0, **679 passed**, 290 s |
| 5 | Missing snapshot, a copy without `data/prices.parquet`, with a plugin that blocks and logs every socket connection and DNS lookup: `PYTHONPATH=<harness> python -m pytest -p nonet -q` | exit **10**, 528 passed, **151 BLOCKED**, 0 failed; **0 network attempts recorded**; the snapshot was not recreated |
| 6 | `python -m compileall -q riskengine scripts dashboard tests` | exit 0 |
| 7 | Cached pipeline in the fresh environment, in the copy: `python scripts/run_pipeline.py` | exit 0, 82 s; every result file byte-identical to the copies taken before the audit |
| 8 | `python scripts/verify_backtest_independent.py` (working and fresh environments) | exit 0: "compared 1832 ticker x method x confidence results against an independent implementation. ALL MATCH" |
| 9 | `python scripts/make_readme_tables.py --check` (working and fresh) | exit 0 |
| 10 | Notebook: `python -m jupyter nbconvert --to notebook --execute --output-dir <scratch> notebooks/walkthrough.ipynb` | exit 0; 8 outputs, 0 error outputs, 4 figures; the checked-in notebook was not modified. The same notebook is also executed by `test_the_walkthrough_notebook_executes_top_to_bottom_without_errors_using_this_interpreter`. |
| 11 | Dashboard automated tests (part of 3 and 4): `tests/integration/test_dashboard.py`, `test_dashboard_extra.py`, `test_undefined_r2_outputs.py` | 13 + 28 + 3 test cases, all passed |
| 12 | Live browser smoke (section 8 below) | see below |

### Failures and flakes observed during the audit (none hidden)

| When | What | Cause | Resolution |
|---|---|---|---|
| First full run after the new script tests were added | `test_the_walkthrough_notebook_executes_top_to_bottom…` failed in the full suite (passed alone): 0 figures instead of 4 | importing Streamlit in the pytest process sets `MPLBACKEND=Agg`, which the notebook kernel inherited and which disables inline figures | test isolation fix: the test removes `MPLBACKEND` for the kernel. Not a software defect. |
| First full run | `test_no_data_is_generated_or_simulated…` failed | my new `test_config.py` contained the forbidden strings it was scanning for | removed the duplicate scan (the README-claims test already covers the whole tree) |
| Second coverage run | `test_smoke_pipeline_rerun_with_unchanged_inputs_reproduces_every_file_byte_for_byte` failed once: `ml_summary.json` differed in the last digit of `oos_r2_vs_training_mean` (0.9033930335059022 against …023) | a **real defect**: see D-10 below | repaired; the test has not failed since (it ran in three further full runs) |

## 4. Counts

| | Passed | Failed | Skipped (BLOCKED) | Errors |
|---|---|---|---|---|
| Baseline (start of audit) | 451 | 0 | 0 | 0 |
| Final, working environment | 679 | 0 | 0 | 0 |
| Final, fresh environment | 679 | 0 | 0 | 0 |
| Final, snapshot missing | 528 | 0 | 151 | 0 |

228 test cases were added (from 451 to 679). No existing test was weakened, skipped, marked xfail or deleted; two
assertions inside new tests were corrected for my own mistakes (section 3), and `test_every_test_named_in_the_documents_exists`
now also checks `MODULE_TEST_MATRIX.md` and this file.

## 5. New defects found and repaired

Each was found by probing the behaviour against the documented contract, repaired minimally, and covered by a
regression test. The valid-data results are unchanged (section 7). The probe outputs that reproduced each defect before
the repair were printed in the session; they are not stored as files. For D-10 the regression test was additionally run
with the repair removed and failed, and with the repair in place it passes.

| ID | Module | Defect (reproduced before repair) | Repair | Regression test |
|---|---|---|---|---|
| D-1 | `data.simple_returns` | Accepted duplicate or unsorted timestamps and zero, negative or infinite prices, silently producing returns | New `validate_prices` (unique, increasing index; observed prices finite and strictly positive) called by `simple_returns` | `test_unsorted_or_duplicate_timestamps_are_rejected_not_silently_paired`, `test_observed_prices_must_be_finite_and_positive` |
| D-2 | `backtest` | Windows of 0, 1, 2.5, `True` or a string were accepted or failed with an unhelpful error; `kupiec_pof` accepted NaN, fractional and boolean counts and booleans for p; `summarise` of an empty table raised `KeyError` | `validate_window` (integer ≥ 2, no booleans), `_count`, an empty summary with the usual columns | `test_invalid_windows_raise_value_error_everywhere`, `test_kupiec_rejects_invalid_counts`, `test_kupiec_rejects_invalid_probabilities`, `test_a_series_with_no_possible_forecast_returns_no_rows_and_an_empty_summary` |
| D-3 | `metrics.beta` | Against a constant benchmark it returned a number (19.2) instead of undefined | returns NaN when the aligned benchmark values are all equal | `test_beta_uses_only_dates_present_in_both_series_and_is_undefined_for_constant_or_disjoint_benchmarks` |
| D-4 | `ml.score` | Silently dropped NaN predictions and targets, accepted `-inf` and compared by position when the indexes differed | requires identical indexes and finite predictions, targets and baseline | `test_nonfinite_predictions_or_targets_are_rejected_not_dropped`, `test_a_nonfinite_baseline_is_rejected`, `test_misaligned_prediction_and_target_indexes_are_rejected_with_an_explanation` |
| D-5 | `config` | No validation: an overlapping train/test period, a non-integer window or impossible preset bounds were accepted | `config.validate()`, called by `portfolio.fit`, `ml.run` and `scripts/run_pipeline.py` | `test_invalid_settings_fail_with_a_message_that_names_the_setting`, `test_invalid_preset_bounds_and_counts_fail_clearly`, `test_a_test_period_that_overlaps_training_is_refused_by_every_consumer` |
| D-6 | `portfolio.max_sharpe_weights` | NaN weights from the solver were returned (the following sum check does not fail on NaN) | raises `RuntimeError("optimizer returned non-finite weights")` first | `test_weights_that_are_not_finite_or_violate_the_constraints_are_refused` |
| D-7 | `data.fetch_risk_free` | `KeyError: 'Close'` on a response without that column; a one-row response crashed in `.squeeze()` | explicit `ValueError`s (no data, no Close column, no observations) and a safe column selection | `test_risk_free_download_success_and_failure_modes`, `test_risk_free_response_with_one_close_column_per_symbol_is_accepted` |
| D-8 | `scripts/check_sp500_membership.py` | Missing or malformed cached tables produced raw pandas errors | `load_sources()` with a message naming the file and the command to recreate it; no results are written | `test_missing_or_malformed_cached_membership_sources_are_reported_honestly` |
| D-9 | `dashboard/app.py` | A missing `universe_membership_summary.json` raised a raw `FileNotFoundError`; a valid-JSON file with the wrong shape raised `KeyError` deep inside a tab; a null rank IC or Sharpe raised `TypeError`; undefined percentages read "n/a" | membership summary loaded with the other results inside the guarded block; column and key checks naming the file; `num()` helper; undefined reads "undefined" everywhere | `test_every_missing_results_file_gives_an_actionable_message_not_a_traceback`, `test_malformed_or_wrongly_shaped_results_files_are_reported_by_name`, `test_an_undefined_rank_ic_and_sharpe_are_displayed_as_undefined_not_as_a_crash` |
| D-10 | `ml.fit_models` | The forest predicted with `n_jobs=-1`; per-tree outputs are summed in thread-completion order, so a forecast changed in its last bit from call to call (measured: 300 distinct outputs in 300 identical calls on a hand-built matrix) and a rerun of the pipeline could change the last digit of a stored R squared, contradicting the byte-for-byte reproducibility claim | the forest is fitted in parallel (exact: each tree has its own seed) and then set to `n_jobs=1` for prediction | `test_forecasts_are_bit_for_bit_reproducible_because_prediction_is_single_threaded` |

Decisions are recorded in `DECISIONS.md` (D43 to D47).

## 6. Statement and branch coverage by module

Measured in run 3 above with pytest-cov (`--cov-branch`); subprocess runs of the scripts are included through
`coverage.process_startup()`, enabled only when `COVERAGE_PROCESS_START` is set.

| Module | Statements | Missed | Branches | Partial | Statement and branch cover | Before the audit |
|---|---|---|---|---|---|---|
| `riskengine/backtest.py` | 87 | 0 | 36 | 0 | 100% | 97.8% |
| `riskengine/config.py` | 50 | 0 | 26 | 0 | 100% | 100% (15 statements) |
| `riskengine/data.py` | 133 | 0 | 48 | 0 | 100% | **39.9%** (48 of 111 statements; the download and snapshot code was untested) |
| `riskengine/metrics.py` | 40 | 0 | 8 | 0 | 100% | 95.5% |
| `riskengine/ml.py` | 131 | 0 | 26 | 0 | 100% | 98.6% |
| `riskengine/portfolio.py` | 104 | 0 | 30 | 0 | 100% | 92.2% |
| `riskengine/serialize.py` | 19 | 0 | 10 | 0 | 100% | 100% |
| `riskengine/var.py` | 79 | 0 | 34 | 0 | 100% | 98.2% |
| `dashboard/app.py` | 298 | 5 | 56 | 5 | 97% | not measured |
| `scripts/build_notebook.py` | 11 | 0 | 0 | 0 | 100% | not measured |
| `scripts/build_universe.py` | 29 | 0 | 8 | 0 | 100% | not measured |
| `scripts/check_sp500_membership.py` | 154 | 18 | 48 | 0 | 91% | not measured |
| `scripts/fetch_data.py` | 20 | 0 | 0 | 0 | 100% | not measured |
| `scripts/make_readme_tables.py` | 97 | 0 | 20 | 1 | 99% | not measured |
| `scripts/run_pipeline.py` | 53 | 0 | 4 | 0 | 100% | not measured |
| `scripts/verify_backtest_independent.py` | 59 | 2 | 24 | 1 | 96% (see below) | not measured |
| **All measured files** | 1,423 | 28 | 402 | 10 | **98%** | 85.3% (package only) |

Inspection of every uncovered statement or branch:

| Where | Why it is uncovered | Disposition |
|---|---|---|
| `scripts/check_sp500_membership.py` lines 37, 41-71 (`norm`, `page`, `fetch`) | the `--fetch` path downloads Wikipedia pages; running it needs the network | BLOCKED by the no-download rule; the parsing it feeds is tested from the stored tables |
| `dashboard/app.py` lines 80, 83, 176, 401, 405 | exercised only inside temporary copies of the app (damaged-file tests and altered result values), whose file path the coverage tool does not attribute to `dashboard/app.py` | covered by behaviour: `test_malformed_or_wrongly_shaped_results_files_are_reported_by_name`, `test_an_undefined_r_squared_is_displayed_as_undefined_everywhere`, `test_a_positive_r_squared_shows_the_success_message_with_the_caveat_and_the_matching_overview_sentence` assert the text each line produces; not counted in the 97% |
| `scripts/verify_backtest_independent.py` lines 59-60 (missing-row branch) | run in the temporary copy used by `test_the_verifier_exits_nonzero_when_a_row_is_missing_or_a_kupiec_p_value_is_wrong`; the tool reports that copy as a separate file (3 statements missed there: different lines) | the union of the two runs covers every line; the table shows the repository copy only |
| `scripts/make_readme_tables.py` branch 132->134 | the branch where the `reasons` list is non-empty (a preset that was not constructed) is reached for the checked-in results but the opposite arm is not | immaterial: formatting only |

Coverage measures that code ran, not that it was right. It does not count the arms of conditional expressions as branches,
and 100% on a module says nothing about inputs that were not tried. Correctness evidence is the independent references
listed in `MODULE_TEST_MATRIX.md`.

## 7. Differences between the checked-in artifacts and regenerated ones

| Artifact | Comparison | Result |
|---|---|---|
| The 10 pipeline result files and the 2 membership result files in `results/` | `diff -rq` against the copy taken before the audit; the pipeline regenerated in a temporary location (working and fresh environments) and compared byte for byte (`test_F31_the_pipeline_regenerates_every_result_in_a_temporary_directory_without_touching_the_checked_in_results`) | identical |
| Generated results block of `README.md` | `make_readme_tables.py --check` | identical to the results |
| `README.md`, remaining text | edited on purpose: the coverage procedure in the Reproduce section and one row in the Layout table | differs from the baseline copy in those two places only |
| `notebooks/walkthrough.ipynb` | byte comparison with the pre-audit copy; regeneration compared cell by cell | identical (the regenerated notebook has the same cells and no stored outputs) |
| Snapshot and cached membership sources | SHA-256 | identical |

The one numerical hazard found (D-10) never changed a checked-in value: every regeneration of the real results (the
byte-comparison test and the explicit pipeline runs) matched the stored files. It was observed once, as a one-unit
difference in the last digit of an unrounded R squared written by the pipeline on the hand-built 41-ticker fixture. Whether the
real results could ever have shown it was not established; the repair removes the possibility.

## 8. Dashboard: focused live browser smoke

Run once, against the real snapshot and results, in the built-in browser, server started with
`streamlit run dashboard/app.py --server.headless true --server.port 8765` and stopped afterwards.

| Check | Observed |
|---|---|
| Overview | 458 tickers; historical 99% breach rate 1.59% against 1.00% expected; normal 99% 1.95%; forecast model R squared -2.99% (all equal to the stored results) |
| VaR explorer, SPY, 99%, both methods, 250 days | historical VaR today 1.75% ($1,746 on $100,000), ES 2.13%; normal VaR today 1.81%, ES 2.09%; breaches 30 of 1,713 and 46 of 1,713; the stored `var_latest.csv` row for SPY gives 0.017463, 0.021342, 0.018148 and 0.020869; the chart rendered |
| Portfolio experiment | the "Exploratory only." warning is displayed above the custom-screen controls; the forecast tab shows the warning that the ensemble does worse than the training average |
| Browser console | no errors |

The browser pane was narrow, so tab clicks were done by coordinates. No layout, contrast or responsive-design audit was
performed; those are not verified.

## 9. Three separate conclusions

### Software correctness: **confirmed for everything tested, with the limits listed below**

Every module was compared with an independent reference (hand calculation, plain-loop reimplementation, `scipy`,
`statistics.NormalDist`, closed form) on valid, invalid and boundary input and through its integration points. The full
suite is 679 passing tests in two environments, the independent verifier reproduces all 1,832 per-ticker results, and the
pipeline regenerates every result file byte for byte. Ten defects were found and repaired (section 5). The tests cannot
prove the absence of defects in paths they do not reach: the live download, the membership `--fetch`, browser rendering,
and conditional-expression arms are not covered.

### Leakage controls: **confirmed for the controls tested**

Features at t use only data up to t; the 63-day targets of all training samples end by 2023-12-31 and none reaches the
test period; changing every price after the cutoff changes neither the selected holdings, the optimised weights nor any
fitted model; the risk-free rate used to choose weights is the training average; forecasts for day t ignore day t and later.
These are tested on hand-built fixtures and on the real snapshot (perturbation tests). Not controlled: the prices are
split- and dividend-adjusted by Yahoo as of the download date, so the snapshot is not strictly point-in-time
(`test_A5b_the_snapshot_is_split_and_dividend_adjusted_so_it_is_not_strictly_point_in_time`); the ticker list is a
hand-typed list of mostly current names, so survivorship bias exists and its size is not measured; the dashboard's custom
portfolio screen lets a user tune on the test period and says so.

### Empirical performance: **reported, not asserted; no skill is claimed**

Separate from correctness, on this snapshot and this one train/test split: historical 99% VaR was breached on 1.59% of
forecast days (1.00% expected) and the Kupiec test rejects 69.2% of tickers; the normal 99% VaR 1.95% and 91.5%; at 95%
the historical method 5.69% and the normal 4.64% (5.00% expected). The forecasting ensemble's R squared against the
training mean is -2.99% (worse than predicting the training average), direction accuracy 53.6% against 57.5% of test
samples that went up, rank IC 0.044 with t = 0.60 over 11 non-overlapping periods. Two of three portfolios were built
(test Sharpe 1.14 and 1.32 against 0.96 for SPY over one period, no transaction costs); one preset was not constructed
because no stock passed its screen. These are findings from a single sample with survivorship bias, not evidence that
any method works.

## 10. Unverified external assumptions

- Yahoo Finance adjusted closes are correct and adjusted consistently; the snapshot's content was checked for internal
  consistency (positive, sorted, unique, gaps recorded), not against another vendor.
- Wikipedia's constituents and change tables are accurate and complete; they are not S&P's own record (one internal
  contradiction, SATS, is documented).
- The 13-week T-bill yield `^IRX` is a suitable risk-free proxy.
- `scipy`, `scikit-learn`, `pandas` and `numpy` compute what they document; they are used as references and are not
  re-derived. The sklearn tree algorithm is trusted.
- Streamlit and Plotly render what the app specifies; the specification was checked, the pixels were not (apart from one smoke).
- Results were produced on one machine (arm64 macOS); other platforms may differ in the last digits of floating-point values.
- The infinite-return policy inside a rolling history (`DECISIONS.md` D42) is an open owner decision.
- The portfolio objective returns 0 when a portfolio's volatility is exactly 0 (`neg_sharpe` in `portfolio.py`); this
  needs constant returns and was not reached with any data or fixture here, and no test covers it.

## 11. Residual items for the owner

1. The stray user-level Jupyter kernel directory created in an earlier phase (`PROGRESS.md` section 7) still needs removal by the owner.
2. Three earlier local commits (`970faf1`, `4ccb8bb`, `154a5a0`) plus this audit's commit are unpushed; nothing was pushed, no remote was changed and no PR was opened.
3. Decide whether to change the rolling-history infinite-value policy (D42).
