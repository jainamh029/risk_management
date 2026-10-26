# Validation report (independent acceptance-test stage)

Scope: validate the implementation on branch `fix/rigor-pass` before merging into `main`. No model, analysis or feature code was changed in this stage (`git diff HEAD -- riskengine/` is empty). Nothing was pushed, no pull request was opened, no remote was changed.

> **Update, final bounded repair (Part 3).** The two items left open at the end of Part 2 are now repaired: `backtest.rolling_forecasts` validates its confidence argument with the shared validator, and `ml.rank_ic_by_date` reports constant or insufficient inputs as undefined without warnings. The complete suite is now **451 tests, 451 passed, 0 failed, 0 skipped, 0 blocked, 0 warnings**, in the working environment and in a fresh virtual environment; all 12 result files and the README are byte-identical to their pre-edit copies.
>
> **Update, repair stage.** Part 2 at the end of this document records the bounded repair of the 16 failures reported below. After the repair the complete suite has **350 tests: 350 passed, 0 failed, 0 skipped, 0 blocked**, in the working environment and in a clean virtual environment; the stored real-data results are unchanged (except for one added status field); the independent backtest verifier still matches all 1,832 results. Sections 1 to 11 below are the original acceptance-run record and are kept unchanged as the audit trail of the 212-test run (196 passed, 16 failed).

## 1. Verdict in one paragraph

The calculations are correct and the leakage controls hold: every formula checked against an independent reference matches, an independent plain-loop recomputation reproduces all 1,832 cached backtest results, and a simulation with a known truth shows the forecasting pipeline finds nothing in noise and finds a planted signal. **Sixteen tests fail, and all sixteen come from two input-handling gaps, not from wrong numbers:** `var.py` does not validate its inputs (15 failing cases) and `ml.score` raises an unhandled `ZeroDivisionError` when the R squared denominator is zero (1 case). Neither affects the stored results, because the real price data is finite and the pipeline masks missing windows before calling these functions, but the functions can silently return NaN, infinity or a plausible-looking number when called directly with bad input. The smallest repair is about 15 lines (section 5). I did not make it, because this stage is validation, not development. Empirically, the VaR models are miscalibrated at 99% and the forecasting model shows no demonstrated skill (section 9).

## 2. What was tested

| Item | Value |
|---|---|
| Code under test | branch `fix/rigor-pass`, `HEAD` `3b8ef2b` (pushed) plus uncommitted test, README-wording and documentation changes made in this stage |
| Python | 3.12.2 |
| Operating system | macOS 26.2, arm64 |
| Packages | numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1, yfinance 1.7.0, streamlit 1.66.1, plotly 7.1.0, lxml 6.1.3, pytest 9.1.1 (exact pins in `requirements.txt`) |
| Data | the frozen snapshot `data/prices.parquet` (downloaded 2026-10-25T05:57:48Z; 462 of 483 tickers; 2019-01-02 to 2026-10-23). No data was downloaded in this stage. |
| Wall time of the full suite | about 230 seconds |

## 3. Commands executed

```
pytest tests/unit                                   (fast, no data)
pytest tests/statistical -o junit_family=xunit1     (observed values read from the junit properties)
pytest                                              (all three directories) -- run three times: working environment (twice) and a clean virtual environment
python3 -m venv <tmp>/venv && pip install -r requirements.txt      (clean environment)
python scripts/run_pipeline.py                      (clean environment, frozen snapshot; fetch_data.py deliberately NOT run)
diff -rq results <checked-in results>               (byte comparison)
python scripts/verify_backtest_independent.py       (clean environment and via test F33)
python scripts/check_sp500_membership.py            (offline analysis, clean environment)
python scripts/make_readme_tables.py --check
jupyter notebook execution with nbclient            (clean environment)
pytest tests/unit tests/integration with data/prices.parquet removed from a scratch copy   (item 35)
```

## 4. Counts

Final full run (working environment): **212 tests: 196 passed, 16 failed, 0 skipped, 0 blocked.** The clean virtual environment gave the same: 196 passed, 16 failed.

| Directory | Passed | Failed | Skipped | Blocked | Needs |
|---|---|---|---|---|---|
| `tests/unit` | 67 | 16 | 0 | 0 | nothing: hand-constructed deterministic fixtures, no market data, no network |
| `tests/integration` | 121 | 0 | 0 | 0 | the cached real-price snapshot and stored results |
| `tests/statistical` | 8 | 0 | 0 | 0 | nothing: simulated series (explicitly synthetic, marker `statistical`) |

Item 35 (snapshot missing), scratch copy without `data/prices.parquet`: 85 passed, 16 failed (the same 16), **103 BLOCKED**, run exit status 10. Nothing was downloaded and the snapshot was not recreated. Before the fix described in section 7 four tests read the snapshot directly and failed instead of reporting BLOCKED; they now request the fixture.

## 5. Failures (16), each reproducible, and the smallest repair

Reproduce any of them with `pytest tests/unit/test_var_es.py tests/unit/test_ml_unit.py`. The specification used is the acceptance criterion in your request: invalid input must not produce a silently manufactured number.

### F-1. `var.var_es` does not validate its inputs (15 failing cases in `tests/unit/test_var_es.py`)

Observed behaviour, with `r = [-0.01, 0.0, 0.01, -0.02, 0.015]`:

| Call | Observed | Problem |
|---|---|---|
| `var_es(r, 0.0, "normal")` | `(-inf, 0.001)` | invalid confidence accepted, infinity returned |
| `var_es(r, 1.0, "normal")` | `(inf, nan)` | same |
| `var_es(r, -0.1 / 1.5 / nan, "normal")` | `(nan, nan)` | silent NaN |
| `var_es(r, 0.0 / 1.0, "historical")` | `(-0.015, 0.001)` / `(0.02, 0.02)` | confidence 0 and 1 accepted (the quantile function allows them) |
| returns containing `+inf`, `historical` | `(0.018, 0.02)` | **a plausible-looking number from a non-finite input** |
| returns containing `-inf`, `historical` | `(nan, nan)` | silent NaN |
| returns containing `+inf` or `-inf`, `normal` | `(nan, nan)` | silent NaN |
| empty returns, `historical` | `IndexError` | unclear failure |
| empty returns, `normal` | `(nan, nan)` plus runtime warnings | silent NaN |
| one return, `historical` | `(0.01, 0.01)` | a one-observation "VaR" |
| one return, `normal` | `(nan, nan)` | silent NaN |

What already behaves correctly: a `NaN` return raises `ValueError` for both methods; a historical confidence of -0.1, 1.5 or NaN raises `ValueError` (from NumPy); an unknown method raises `ValueError`.

Smallest repair: one guard at the top of `historical_var_es` and `normal_var_es`: raise `ValueError` unless `0 < conf < 1`, `len(returns) >= 2` and every return is finite. About 8 lines plus a docstring sentence. Impact on results: none, because the pipeline never passes a non-finite value or an invalid confidence (checked by test A3 and by the stored results).

### F-2. `ml.score` raises an unhandled `ZeroDivisionError` when the R squared denominator is zero (1 failing test, `test_D22b`)

Reproduce: a test set whose targets all equal the training mean makes `sum((y - mean(y_train))^2)` zero; `ml.score(...)` raises `ZeroDivisionError: float division by zero`. The specification asked for explicit handling (NaN or a documented error). Smallest repair: if the denominator is zero, report `nan` for `oos_r2_vs_training_mean` (or raise `ValueError` with an explanatory message). About 3 lines. Impact on results: none (the real denominator is far from zero).

## 6. Independence source for the expected values

| Area | Independent source |
|---|---|
| Normal VaR and ES | `scipy.stats.norm.ppf` and `norm.pdf` closed forms (B6, B7); the defining equation `P(R <= -VaR) = 1 - c` through `norm.cdf`; numerical integration with `scipy.integrate.quad` (existing integration test) |
| Historical VaR and ES | hand-calculated examples from the specification (VaR 0.02 / ES 0.03 at 0.75; 0.036 / 0.04 at 0.95); a hand-written interpolation over sorted real losses |
| Kupiec | direct log likelihoods written in the test; `scipy.stats.binom.logpmf`; `scipy.stats.chi2.sf`; the exact boundary values from the specification (LR = 0 at x = np; `-200 ln 0.95`; `-200 ln 0.05`); a value calculated by hand (4.1308 for n=100, x=10, p=0.05) |
| Breach counts and pooled rates | a plain-loop reimplementation inside the test (C16) that imports nothing from the package; a second one in `scripts/verify_backtest_independent.py`; exact i.i.d. theory in the statistical test (historical breach probability `(W - c(W-1))/(W+1)`; normal-method breach probability from the Student t distribution) |
| Features and target | direct arithmetic on constructed and real prices, `scipy.stats.skew/kurtosis`; the target compared bit-for-bit with `P[t+63]/P[t]-1` |
| R squared, direction accuracy, baseline accuracy | formulas recomputed in the test with `numpy` sums; Spearman IC against `scipy.stats.spearmanr` |
| Portfolio performance | hand loops for wealth, drawdown (starting capital as a peak), Sharpe; the buy-and-hold versus daily-rebalanced example from the specification |
| README tables | independently formatted values recomputed from the result files (F32) |

Limits of that independence: the verifier script and the tests were written by the same author as the package and share its conventions (strict `>` breach, previous 250 returns, linear interpolation, sample standard deviation). They are independent code and independent arithmetic, not an independent specification. The inputs they read are the same snapshot.

## 7. Item-by-item status

| Item | Result |
|---|---|
| A1 returns of [100, 110, 99] | pass |
| A2 [missing, 100, missing, 110] gives all-missing returns, nothing filled | pass |
| A3 timestamps unique, sorted, valid (weekdays, no timezone); observed prices finite and positive | pass |
| A4 requested, returned, failed, equity and ETF counts reconcile with universe, metadata, run_info and membership files (no hard-coded counts) | pass |
| A5 later prices cannot change returns or features as of t | pass (constructed and real data). Recorded limitation: prices are split and dividend adjusted by the vendor retroactively, so the snapshot is not strictly point-in-time (asserted by `test_A5b`) |
| B6 normal VaR and ES for [-0.01, 0, 0.01] | pass (both confidences) |
| B7 shift by +0.001 lowers VaR and ES by exactly 0.001; no clipping | pass (shows a negative loss VaR for a large shift) |
| B8 historical hand example | pass |
| B9 positive scaling | pass (3 scalars, both methods, both confidences) |
| B10 ES >= VaR; confidence ordering | pass (constant returns give VaR = ES = -constant) |
| B11 invalid confidence, insufficient data, missing and non-finite inputs | **FAIL: F-1** (15 cases); missing (`NaN`) values and unknown methods behave correctly |
| C12 Kupiec boundary and ordinary cases | pass |
| C13 loss equal to VaR is not a breach | pass (strictly greater counts; a loss 1e-7 above counts) |
| C14 forecast for t ignores the return at t and later; breach flag can change | pass (both methods, four dates) |
| C15 missing return gives no forecast; exact dates and counts around the gap; recovery | pass (both methods) |
| C16 all cached per-ticker counts and pooled rates recomputed independently; aggregate versus member-only tables reconciled by equity versus ETF | pass. Finding: the README did not state that the two breach-rate tables have different denominators (458 tickers including 10 ETFs; 448 stocks); the README now states it |
| D17 target equals P[t+63]/P[t]-1 exactly | pass (bit-for-bit, constructed and real prices) |
| D18 every training label finishes by the cutoff; none reaches the test period | pass |
| D19 later prices change neither features nor fitted models nor the training baseline | pass |
| D20 latest inference uses the latest eligible date; per-ticker exclusions reconcile | pass (independent rule: a ticker is eligible iff its last 127 prices exist; real and constructed data) |
| D21 no learned preprocessing exists; ensemble weights equal in evaluation and inference | pass (source contains no scaler or pipeline; the one blend function serves both paths) |
| D22 R squared equals the documented formula | pass. Zero-denominator handling: **FAIL: F-2** |
| D23 direction accuracy, baseline accuracy and positive share recomputed separately | pass (the fixture has a negative training mean, so baseline accuracy is 1 minus the positive share, not equal to it) |
| D24 rank IC is cross-sectional by date; non-overlapping dates at least 63 sessions apart; n, standard error, statistic | pass (constructed ranks, `scipy.stats.spearmanr`, real stored results) |
| D25 deterministic refit reproduces predictions | pass (within 1e-12; the stored real results also equal a fresh run) |
| E26 perturbing every test-period price leaves holdings and weights unchanged | pass (constructed and real data) |
| E27 weight sums, bounds and screens; infeasible stays unconstructed; no relaxation | pass |
| E28 buy-and-hold or daily constant weight | **The implementation is daily constant-weight rebalancing.** With the specified example it gives final wealth 1.125, not the buy-and-hold 1.00. The README already says "constant weights, rebalanced daily", so no README correction was needed; a test asserts the wording |
| E29 total return, drawdown, Sharpe and the same-holdings equal-weight benchmark recomputed; matching dates and risk-free convention | pass |
| E30 risk-free rate used to choose weights uses training dates only | pass (the test-period rate changes only evaluation figures; the pipeline passes `rf_train` to selection) |
| F31 results regenerated in a temporary directory without touching the checked-in results | pass (10 files byte-identical; checked-in digests unchanged) |
| F32 README tables reconcile with result files, including rounding, labels, units, denominators and date ranges | pass |
| F33 existing verifier inspected and run | pass. Its imports are json, math, sys, pathlib, statistics, numpy, pandas, scipy only; no `riskengine`; no dynamic import or eval. See the independence limits in section 6 |
| F34 clean venv, documented pipeline, tests, notebook against the frozen snapshot | pass: pipeline exit 0 and results byte-identical to the checked-in ones; verifier all match; README block matches; notebook 6 code cells, 0 errors, 4 figures; full suite 196 passed and the same 16 failed |
| F35 missing snapshot reported as BLOCKED, not a pass, nothing downloaded | pass after one fix (section 8) |

## 8. Corrections made during this stage, including my own mistakes

| What | Detail |
|---|---|
| Test architecture | The 106 existing tests moved to `tests/integration`; new `tests/unit` (83 tests) and `tests/statistical` (8 tests) created. A snapshot-missing run now reports BLOCKED (skip with a BLOCKED reason, run exit status 10) instead of aborting the session. |
| My arithmetic error | A hand-calculated sanity value in a Kupiec test was wrong (I wrote 3.5256; the correct hand value is 4.1308). Corrected the test, not the code. |
| My over-strict assertions | One test forbade the ensemble weight from appearing in `run()`'s metadata; another compared a recorded weight after changing it. Both were test errors and were corrected. |
| **Statistical bands revised after a failure (disclosed)** | The first version of experiment S1 asserted that the Kupiec rejection rate under a correct model lies in bands such as [0.02, 0.15] at 95% confidence, and S2 required a power above 0.95. Result: S1 failed (normal 0.8%, historical 0.0% rejected) and S2 failed for the normal method (93.2%). Diagnosis: the bands rested on a wrong expectation. Rolling windows share data, so breach counts are under-dispersed (standard deviation 3.6 and 5.6 against 7.6 for independent trials), which makes the Kupiec test **conservative** for rolling forecasts. The pooled breach rates themselves match exact i.i.d. theory within about one standard error. The criteria were then replaced by (a) agreement with the analytical breach probability, (b) rejection rate no higher than 0.10, (c) power above 0.90. These replacements were written after seeing the failure and are labelled as such in the test file. |
| README: the Kupiec sentence | The README (and the dashboard overview) said about 5% of tickers would be rejected by chance if the model were right. The experiment shows that is wrong for rolling-window forecasts. Both now say that a correct i.i.d. simulation was rejected 0% to 8% of the time, depending on method and level. |
| README: denominators | Added a generated sentence stating that the headline table counts 458 tickers including 10 ETFs and that the member-day table uses the 448 stocks only. Numbers unchanged. |
| README: test description and data statement | The README now says that the unit and statistical tests use hand-constructed and simulated fixtures that never enter results. Layout and run instructions updated. |
| Four tests that read the snapshot without the fixture | They failed instead of reporting BLOCKED when the snapshot was missing; they now request the fixture. |

## 9. Verdicts

### Calculational correctness: **confirmed, with two input-validation gaps**

Every formula matched its independent reference. The backtest was recomputed independently for all 458 tickers, two methods and two confidence levels (1,832 results) with exact agreement on forecast counts, breach counts, Kupiec statistics and p-values, and the pooled and member-only tables reconcile once ETFs and stocks are separated. A simulation with a known truth reproduces the exact i.i.d. breach probabilities for both methods. The only defects found are F-1 and F-2, which do not affect any stored result.

### Leakage control: **confirmed**

Features at date t are unchanged when later prices change; targets are exactly `P[t+63]/P[t]-1`; every training label ends by the cutoff and none reaches the test period; fitted models and the training baseline are unchanged when test-period prices change; portfolio holdings and weights are unchanged when every test-period price is perturbed; the weight-selection risk-free rate uses training dates only. The ML pipeline finds nothing in a simulated pure-noise panel (out-of-sample R squared -0.4%, rank IC t statistic -0.04 over 7 independent periods) and finds a planted cross-sectional signal (rank IC 0.27, t = 7.0), so neither a leak nor a broken alignment is visible in either direction. Caveats: prices are retroactively adjusted by the vendor (not strictly point-in-time); the test period has been inspected repeatedly during development and in this report, so it is not blind. No model setting was changed in this stage (`git diff HEAD -- riskengine/` is empty), and `riskengine/config.py` has not been modified since the commit that added the forecasting settings; I cannot prove from the repository alone that those settings were written before their results were computed, so that remains my statement about the working session. The test period is certainly not blind: it has been inspected repeatedly during development and in this report.

### Empirical performance (separate from correctness; no benchmark had to be beaten to pass)

Observed on the frozen real snapshot:

- **VaR (one-day, 250-day window, 458 tickers, 779,137 forecasts).** Pooled breach rates and Kupiec rejections: historical 95%: 5.69% breaches against 5.00% nominal (the i.i.d. expectation for this window is 5.36%), 17.9% of tickers rejected; historical 99%: 1.59% against 1.00% (i.i.d. expectation 1.39%), 69.2% rejected; normal 95%: 4.64% (i.i.d. expectation 5.10%), 15.1% rejected; normal 99%: 1.95% (i.i.d. expectation 1.05%), 91.5% rejected. Because the simulation shows a correct model is rejected 0% to 8% of the time, rejection shares of 69% and 92% at 99% are strong evidence of miscalibration (fat tails and volatility clustering). The normal method's 99% VaR is the worst. Restricting to days when a stock was an index member does not change this (differences under 0.02 percentage points).
- **Forecasting model (63-day return; 94,371 training and 287,999 test samples).** Out-of-sample R squared against the training mean: -0.90% (random forest), -6.03% (gradient boosting), -2.99% (ensemble); direction accuracy 53.7%, 53.3%, 53.6% against a baseline of 57.5% (the positive share of test targets is also 57.5%, because the training mean is positive here); rank IC 0.044 with standard error 0.074 and t = 0.60 on only 11 non-overlapping periods; neither 60-day nor 126-day momentum alone shows a significant IC. No demonstrated skill, and the 11-period sample is too small to detect a modest one.
- **Portfolio experiment (selected on data to 2023-12-29; tested 2024-01-02 to 2026-10-23).** Conservative: not constructed (0 stocks passed). Balanced: test Sharpe 1.14, total return 72.5%, against 0.76 and 46.0% for the same holdings equally weighted and 0.96 and 65.1% for SPY. Aggressive: 1.32 and 127.0% against 1.08 and 106.9% for equal weights. One split, one period, three presets tried, no cost model, a rising market, and a selection by past Sharpe: this is not evidence of skill.

## 10. Remaining unverified claims

- Strict point-in-time validity of the prices (vendor adjusts retroactively; cannot be tested without a second dated snapshot).
- The accuracy of Yahoo Finance data itself; only structural checks (positive finite prices, sorted unique timestamps, counts) were made.
- Membership truth: the comparison with the S&P 500 relies on Wikipedia's tables, which contradict each other for one ticker (SATS) and cannot match renamed symbols.
- Dashboard reduced-motion behaviour (CSS rule not tested with the operating-system setting) and any browser other than the in-app one.
- Long-horizon stability of the results on other time periods or other universes; the forecasting and portfolio results rest on a single split.
- The `fetch_data.py` download step was not exercised in this stage (frozen snapshot used by design); its output is covered only through the snapshot's metadata checks.
- Coverage: line coverage of the package was not measured.

## 11. Recommendation

Do not weaken the 16 failing tests. Either (a) apply the roughly 15-line repair in section 5 in a separate commit and rerun the suite, or (b) merge with the 16 failures documented as known input-validation gaps. There is no failure that affects a stored number, a README figure or a leakage control. The decision and the repair are yours to authorise; I stopped here.

---

# Part 2. Repair stage

Scope: a bounded repair of the confirmed failures, then the complete validation and reproducibility checks again. No financial method, model setting, feature, evaluation period, allocation rule or universe was changed. Nothing was pushed, merged or opened as a pull request; no Git configuration was changed (the commit author is set per commit with `-c` flags only).

## 12. Baseline

| Item | Value |
|---|---|
| Branch / commit | `fix/rigor-pass`, `970faf1` (verified, not assumed); working tree clean; `origin/fix/rigor-pass` is at `3b8ef2b`, so `970faf1` was local only |
| Git identity configured | none (read-only check) |
| Python / OS | 3.12.2, macOS 26.2 arm64 |
| Frozen snapshot, SHA-256 | `data/prices.parquet` `a9a0c76423e6a28b9bb0de37b6e5a4d3e6be961e367a56f7bf26784004c7922e`; `data/tbill_13w.parquet` `8deddbe70e309bb3243da8e753e9251c3a95bdd76db5716be2f13c4235f45aba`; `data/prices_meta.json` `000efc5756df561895e97a1d999da1e7db518e78b90814ac3bd7a93caa8b8687`; `data/universe.csv` `d84ee026702c1668c9558c9069a3522897bb7580392e66cc3767958a6c95e025`. Recomputed after all repair work and after the pipeline runs: unchanged. No market data was downloaded. |
| Result artifacts | copies of `results/` and `README.md` taken before any edit, compared byte for byte afterwards |

Baseline run, before any edit: `.venv/bin/python -m pytest -q`, exit status 1, **212 tests: 196 passed, 16 failed, 0 skipped, 0 blocked, 0 errors**, 235 s. The 16 failing tests were exactly the reported ones: `test_D22b_a_zero_denominator_is_handled_explicitly`, `test_B11a_invalid_confidence_is_rejected` (7 parameter sets), `test_B11b_missing_or_nonfinite_returns_are_rejected` (4), `test_B11c_insufficient_data_is_rejected` (4). The reported baseline was therefore reproduced exactly.

## 13. Defects, root causes and repairs

### F-1. `var.py` accepted invalid input

Root cause: `historical_var_es` and `normal_var_es` only checked for NaN. The confidence went straight to `np.quantile` or `scipy.stats.norm.ppf`, and finiteness, dimensionality and sample size were never checked, so `conf=1` gave `inf`/`nan`, a `+inf` return gave a plausible `(0.018, 0.02)` from the historical method, and empty or one-observation input was accepted or failed obscurely.

Repair (`riskengine/var.py`): two shared validators, `validate_confidence` and `validate_returns`, called at the top of both calculation functions and therefore by `var_es` and `latest_var_table`; `var_es` also requires the method to be exactly `historical` or `normal`. Contract: confidence must be a real scalar strictly between 0 and 1, finite, and not a boolean; returns must be a real numeric one-dimensional sequence of at least two finite values (list, tuple, NumPy array or pandas Series); complex values, booleans, strings, objects, scalars, multidimensional inputs, NaN and both infinities raise `ValueError` with an explanatory message. Nothing is dropped, filled, clipped or flattened; constant finite returns remain valid; the quantile, ES and normal formulas were not touched. Incomplete windows remain unavailable forecasts in `backtest.rolling_forecasts` and are excluded by `latest_var_table`, as documented.

Internal callers checked: `portfolio.performance`, `latest_var_table`, and the dashboard's VaR explorer all pass finite data of at least two observations (verified by the unchanged real-data results and the dashboard tests); none needed changing. The `var.py` docstring previously said no annualisation is applied "anywhere in this package"; it is now scoped to VaR and ES.

| Input | Before | After |
|---|---|---|
| `var_es(r, 1.0, "normal")` | `(inf, nan)` | `ValueError: confidence must be strictly between 0 and 1 ...` |
| `var_es([..., +inf, ...], 0.95, "historical")` | `(0.018, 0.02)` | `ValueError: returns must be finite: found 0 NaN, 1 +inf and 0 -inf ...` |
| `var_es([], 0.95, "normal")` | `(nan, nan)` plus warnings | `ValueError: at least 2 returns are required, got 0` |
| `var_es(r, True, "normal")` | accepted as 1 | `ValueError: ... not a boolean` |

### F-2. `ml.score` raised `ZeroDivisionError` for a zero denominator

Root cause: `1.0 - sse_model / sse_base` with Python floats and no handling of `sse_base == 0.0`.

Repair (`riskengine/ml.py`): the reference sum of squares is computed exactly as before, using the training mean (not scikit-learn's test-mean reference); if it equals exactly 0.0 the score is `NaN` and a new field `oos_r2_status` records "undefined: every test target equals the training mean, so the reference sum of squares is zero"; otherwise the status is "defined". No tolerance is used, so small positive denominators are calculated (tested down to 1e-300 per observation). All other scores of the same evaluation are returned unchanged, and the existing return structure is preserved plus the one status key. Propagation: a new `riskengine/serialize.py` (the previous `clean()` moved out of `scripts/run_pipeline.py` so it can be tested; same logic, plus `allow_nan=False`) writes the undefined value as JSON `null` and can never emit a nonstandard `NaN` literal; `scripts/make_readme_tables.py` and `dashboard/app.py` render it as the word "undefined" (Overview metric and finding, Forecasting tab banner and table) instead of an exception, a zero or a percentage; the pipeline's console line prints "undefined".

### Test-side corrections in this stage (not software repairs)

| What | Detail |
|---|---|
| `test_E28b_readme_describes_the_implemented_convention` | It forbade the string "buy-and-hold" anywhere in the README. The README now says "not buy-and-hold" deliberately. The test now requires every mention to be negated and the daily-rebalancing sentence to be present, and forbids the ambiguous phrase "held fixed over the test period". |
| Expectations I got wrong while writing the new tests | The recovery index after a gap (day 14, not 13); the message produced by a nullable pandas NA (it becomes NaN and is rejected as non-finite, which is correct); the tolerance for a p-value near LR = 0 (the chi-square survival function has infinite slope there, so a 4.5e-13 rounding difference in LR moves the p-value by 1.3e-7). In each case the code was right and the test was corrected, with the reason written in the test. |
| `test_F31` (regeneration into a temporary directory) | It correctly failed once, because the checked-in `ml_summary.json` lacked the new status field. It passes after the results were regenerated from the frozen snapshot (section 15). |

## 14. Regression tests added (138 new tests; every existing test kept, none weakened, skipped or marked xfail)

| File | What it covers |
|---|---|
| `tests/unit/test_var_validation.py` | Both methods and both direct entry points. Invalid confidence: 0, 1, below 0, above 1, NaN, plus and minus infinity, booleans (Python and NumPy), list, tuple, array, 0-d array, dict, string, None, complex. Invalid returns: empty, one observation, scalars, 2-D list and array and DataFrame, NaN, plus and minus infinity, strings, numeric strings, mixed, None inside, booleans, series of strings, nullable NA, complex list and complex array with zero imaginary part, ragged and dict. Unknown methods. List, tuple, float64 and float32 arrays, Series with and without a date index, integer dtype. Constant returns. Unchanged reference results. Negative VaR not clipped. Missing windows still unavailable forecasts and still excluded by `latest_var_table`. |
| `tests/unit/test_ml_score.py` | Zero denominator with perfect and with imperfect predictions; constant targets that differ from the training mean; small positive denominators (1e-12 and 1e-300); the training-mean reference versus a test-mean reference; the return structure; other metrics intact; JSON null and strict parsing; valid numbers unchanged by serialisation. |
| `tests/integration/test_undefined_r2_outputs.py` | README table generation and dashboard rendering with a null R squared (in a temporary copy of the results), and the unchanged rendering of the real results. |
| `tests/statistical/test_known_distribution.py` | Experiment A (section 17). |

Independent expected values: `scipy.stats.norm.ppf` and `norm.pdf` closed forms for the normal method; hand arithmetic for the historical method on `[-0.01, 0, 0.01]` (VaR `(2c - 1) * 0.01`, ES 0.01) and on `[-0.04, -0.02, 0, 0.01, 0.03]`; constant returns `[0.01, 0.01, 0.01]` give exactly -0.01 within 1e-15; R squared worked by hand: training mean 1, targets `[2, 2]`, predictions `[1, 1]` gives `1 - 2/2 = 0`; training mean 0, targets `[1, 3]`, predictions `[2, 2]` gives `1 - 2/10 = 0.8` (a test-mean reference would give 0); targets 2.0 against training mean 1.0 with predictions 1.5 gives 0.75; targets 1e-6 above the training mean with predictions halfway gives 0.75; the likelihood ratios come from the binomial likelihoods written in the test.

## 15. Verification commands and outcomes

| Step | Command | Exit | Outcome |
|---|---|---|---|
| A. Syntax | `python -m compileall -q riskengine tests scripts dashboard` | 0 | no errors |
| B. Focused tests | `pytest tests/unit/test_var_validation.py tests/unit/test_var_es.py`, then `tests/unit/test_ml_score.py tests/unit/test_ml_unit.py`, then `tests/integration/test_undefined_r2_outputs.py`, then `tests/statistical` | 0 | 163, 22, 3 and 12 passed (before the full suite) |
| C1. First full run after the repairs | `pytest -q` | 1 | 350 tests: 348 passed, 2 failed (`test_F31` for the stale `ml_summary.json`; `test_E28b` for the intentionally negated README wording). Both explained in section 13. |
| D. Frozen-snapshot pipeline | `python scripts/run_pipeline.py` (no `fetch_data.py`), `python scripts/check_sp500_membership.py` (offline), `python scripts/verify_backtest_independent.py`, `python scripts/make_readme_tables.py` | 0, 0, 0, 0 | verifier: all 1,832 results match; see section 16 for the comparison |
| C2. Final full run, working environment | `pytest -q` | 0 | **350 passed, 0 failed, 0 skipped, 0 blocked, 0 errors** (238 s) |
| E. Clean environment | new venv, `pip install -r requirements.txt` (now including `nbconvert==7.17.2`), then with that venv's interpreter: `pytest -q`, `run_pipeline.py`, `check_sp500_membership.py`, `verify_backtest_independent.py`, `make_readme_tables.py --check` | 0 for each | **350 passed**; pipeline results byte-identical to the working environment; verifier all match; README block matches; the copied snapshot's checksum equals the original |
| F. Notebook | `jupyter nbconvert --to notebook --execute --output-dir <temporary directory> notebooks/walkthrough.ipynb` in the clean environment | 0 | 6 code cells executed, 0 errors, 4 figures, nothing on stderr; the original notebook's hash is unchanged |
| G. Dashboard | the automated dashboard tests (13 existing and 2 new, inside the 350) plus a live browser check | | see below |
| H. Missing snapshot | scratch copy of the clean environment without `data/prices.parquet`; `PYTHONPATH=<harness> python -m pytest -p nonet -q`, where `nonet` blocks and logs every socket connect, DNS lookup and `create_connection` | 10 | 245 passed, **105 BLOCKED**, 0 failed; **0 network attempts recorded**; the snapshot was not recreated; the user's real snapshot was never touched |

Dashboard, performed: the automated tests (valid defaults agree with the results files; filter, preset, forecast-table and control-option tests; missing-data message; the new undefined-R-squared test and the unchanged-real-results test). Live browser check (real, in the in-app browser): served the dashboard from a temporary copy whose three R squared values were set to `null`; the Overview metric and the findings sentence read "undefined", the Forecasting tab shows an information banner explaining why, the three model rows show "undefined" while direction accuracy, rank IC and t statistics are still shown, and the baseline row shows 0.00%. A screenshot of the Forecasting tab was taken. Not repeated in this stage: keyboard navigation, phone width, and the VaR explorer in a browser (these were checked in the earlier stage and the explorer is covered by the automated tests). The dashboard's "None" text in not-applicable table cells of the baseline rows is pre-existing cosmetic behaviour, unrelated to this repair.

## 16. Reproducibility: did valid-data results change?

No. From the same frozen snapshot, the regenerated artifacts compared with the copies taken before any edit:

| Artifact | Result |
|---|---|
| `metrics.csv`, `ml_feature_importance.csv`, `ml_latest_forecasts.csv`, `ml_rank_ic_by_date.csv`, `portfolios.json`, `run_info.json`, `universe_membership.csv`, `universe_membership_summary.json`, `var_backtest_by_ticker.csv`, `var_backtest_summary.json`, `var_latest.csv` | byte-identical (11 of 12) |
| `ml_summary.json` | differs by exactly three added lines, `"oos_r2_status": "defined"`, one per model. After removing that key the file is identical to the baseline, including every number. This is an actual schema addition, not a numerical change or a timestamp. |
| README results block | unchanged (the README diff is wording only: VaR/ES annualisation scope, portfolio rebalancing wording, the undefined-R-squared sentence) |
| Clean-environment regeneration | byte-identical to the working-environment results |

## 17. Statistical-test integrity

Two experiments are now kept apart and labelled as such in the test files.

**Experiment A, independent observations from a known distribution** (`tests/statistical/test_known_distribution.py`, new; seed 20261101 fixed in advance). The true VaR is known, so each day is an independent Bernoulli trial. The exact probability that the Kupiec test rejects at 5% was computed by enumerating the binomial distribution with a separately written likelihood ratio. Observed rejection rate against exact size: 95% confidence, 5.10% against 4.67% (+0.91 standard errors); 99% confidence, 3.85% against 4.06% (-0.48 standard errors). The test also checks that `kupiec_pof` equals the reference for every possible breach count (0 to 1200) for both confidence levels. A shows that the Kupiec function behaves correctly under its own assumptions. **It does not show anything about experiment B.**

**Experiment B, rolling-window estimated forecasts** (`tests/statistical/test_statistical.py`, S1 and S2, thresholds and seeds unchanged). Here the VaR is estimated from the previous 250 returns, neighbouring days share data, and the breach counts are not independent trials. The original result and the audit trail are preserved verbatim in that file and in section 8 of this report: the first criteria (rejection rate within bands such as [0.02, 0.15]) failed, with observed rejection rates of 0.8% (normal, 95%) and 0.0% (historical, 95%) and a power of 93.2% for the normal method against a threshold of 0.95. The replacement criteria were written after seeing that failure. The justification: (a) agreement of the pooled breach rate with the exact i.i.d. breach probability, derived analytically (this one is a mathematically grounded criterion); (b) a rejection rate of at most 0.10, and (c) a power above 0.90, which are **empirical guards, not derived bounds**, chosen after observing 0% to 8% and 93% to 96%. This was a test-design correction (the first expectation was mathematically invalid for overlapping windows), not a software repair; no production code changed because of it. The observed dispersion (variance below that of independent trials) is an empirical finding, not a proven theorem, here. The simulation tests S3 and S4 (pure noise, planted signal) remain as pipeline sanity checks; they say nothing about whether any forecastable signal exists in real markets.

## 18. Remaining failures, blockers, warnings and unverified claims

*(Status after Part 3: the rolling-forecast confidence gap and the correlation warnings listed below are resolved; the rest stands.)*

- **Failures: none.** 350 of 350 pass in both environments.
- **Blocked:** none in the complete runs. The deliberate missing-snapshot run reports 105 BLOCKED and exits with status 10.
- **Warnings (10, one cause):** `RuntimeWarning: invalid value encountered in divide` from `np.corrcoef` inside `ml.rank_ic_by_date` when the test targets are constant on a date, as in the undefined-R-squared test scenarios. The rank IC is correctly NaN there; the warning is noise. Proposed (not made): skip zero-variance dates explicitly.
- **Remaining gap, same class as F-1, in a layer I was told to leave alone:** `backtest.rolling_forecasts` does not validate its confidence argument. Observed: `conf=1.0` with the normal method returns `inf` forecasts, `conf=0.0` returns `-inf`, `conf=1.5` returns `nan`; the historical method raises only for confidence outside [0, 1]. `kupiec_pof` does validate. The pipeline always passes the fixed `config.CONFS`, so no stored result is affected. Proposed repair: call `var.validate_confidence(conf)` at the top of `rolling_forecasts` (two lines plus tests).
- **Unverified:** the vendor's retroactive price adjustment (not strictly point-in-time); the accuracy of Yahoo Finance data; Wikipedia as the membership source (SATS contradiction, renamed symbols); reduced-motion behaviour of the dashboard; the `fetch_data.py` download step (the frozen snapshot was used by design); line coverage; any browser other than the in-app one.

## 19. Confirmations

- **No tuning and no change of evaluation:** `riskengine/config.py`, `riskengine/portfolio.py`, `riskengine/backtest.py`, `riskengine/metrics.py`, `riskengine/data.py`, the universe, the training and test dates, the screens and the model settings are unchanged in this stage (`git diff 970faf1 -- <those files>` is empty). Production files changed: `riskengine/var.py` (validation), `riskengine/ml.py` (the score), `riskengine/serialize.py` (new, moved from the pipeline script), `scripts/run_pipeline.py` (uses the serialiser, prints "undefined"), `scripts/make_readme_tables.py` and `dashboard/app.py` (render an undefined score). No poorly performing method, ticker or period was excluded and no simulated data replaced an observation.
- **Verdicts.** Software correctness checks: passed (350 of 350 in both environments, the independent verifier matches all 1,832 results, the notebook runs without error, the dashboard checks pass). Leakage controls: passed (the earlier leakage tests and the simulation checks are unchanged and still pass). What the unchanged market-performance results show: the VaR models are miscalibrated at 99% (historical 1.59% breaches against 1% expected, 69.2% of tickers rejected; normal 1.95% and 91.5%), with the Kupiec test known to be conservative for rolling forecasts; the forecasting model shows no demonstrated skill (out-of-sample R squared between -0.9% and -6.0% against the training mean, direction accuracy about 53.6% against a 57.5% baseline, rank IC 0.044 with t = 0.60 on 11 independent periods); the portfolio results beat SPY's Sharpe ratio in one period on one split and are not evidence of skill. Passing tests shows the software computes what it claims, not that any strategy has investment skill.

---

# Part 3. Final bounded input-validation repair

Scope: two named items only. No financial method, model setting, feature, evaluation period, universe or allocation rule changed. Nothing was pushed, merged or opened as a pull request; no Git configuration or existing commit was changed (author set per commit with `-c` flags).

## 20. Baseline

| Item | Value |
|---|---|
| Branch / commit | `fix/rigor-pass`, `4ccb8bb`; working tree clean; `origin/fix/rigor-pass` still at `3b8ef2b` |
| Frozen snapshot, SHA-256 | unchanged and re-verified before the repair, after it, and in the fresh copy: `prices.parquet` `a9a0c764...7922e`, `tbill_13w.parquet` `8deddbe7...45aba`, `prices_meta.json` `000efc57...8687`, `universe.csv` `d84ee026...5e025` (full values in section 12). Nothing was fetched. |
| Result artifacts | copies of `results/` and `README.md` saved before any edit |
| Baseline suite | not re-run in this stage; the last complete run of `4ccb8bb` was 350 passed (Part 2). Both defects were instead reproduced directly before any edit (below). |

## 21. Defects, root causes and repairs

### G-1. `backtest.rolling_forecasts` did not validate its confidence argument

Reproduced before the edit (window 10, a 30-day deterministic series): the historical method accepted `conf=1.0` (it returned the maximum loss), `0.0` (the minimum), the boolean `True` (as 1) and the list `[0.95]`, and raised only for `1.5` and `nan` (from NumPy's quantile check); the normal method returned `inf` for `1.0` and `True`, `-inf` for `0.0`, `nan` for `1.5` and for `nan`, and accepted the list `[0.95]`; a string raised an obscure `UFuncTypeError` in both. Root cause: the confidence went straight to `np.quantile` or `norm.ppf`; only `kupiec_pof` validated its own argument.

Repair (`riskengine/backtest.py`): `rolling_forecasts` calls the shared `var.validate_confidence` first (the same function object the public VaR functions use, so the policy lives in one place: a real scalar strictly between 0 and 1; booleans, strings, bytes, non-scalars, `None`, complex values, NaN and infinities raise `ValueError` with an explanatory message) and validates the method before any window is built, so nothing is calculated for invalid settings, including for histories shorter than the window. `backtest_series` and `backtest_universe` inherit the check because they call it. The history is deliberately **not** checked for finiteness: a window with a missing return still produces an unavailable forecast and the rest of the history is still forecast.

### G-2. Constant inputs to the rank IC produced `RuntimeWarning` and an accidental NaN

Reproduced: constant targets on a date raised `RuntimeWarning: invalid value encountered in divide` (shown as an error under `-W error`). Root cause: `np.corrcoef` was called on constant rank vectors.

Repair (`riskengine/ml.py`, `rank_ic_by_date`): before any correlation, pairs with missing or infinite values are removed, and a date is **undefined** if it then has fewer than 10 paired observations, or the predictions are all equal, or the targets are all equal. An undefined date is omitted from the returned series (the treatment dates with too few stocks always had), with the reason recorded in `result.attrs["undefined_dates"]` (for example "constant targets across 12 stocks"); no value is substituted, no warning is emitted, and no warning filter is changed globally. Defined dates are computed exactly as before. The existing output schema has no status field for the rank IC, so none was added: the statistics already represent "undefined" as `n = 0` and null statistics (checked to serialise as JSON null), and the reason is carried on the series. A schema change would alter every `ml_summary.json` for no numerical gain; if a status field in the results files is wanted, that is a separate decision.

| Case | Before | After |
|---|---|---|
| constant targets on a date | `RuntimeWarning`, NaN correlation silently kept | date undefined, omitted, reason recorded, no warning |
| constant predictions on a date | same | same as above |
| fewer than 10 finite pairs (including infinities) | infinities were ranked as values | undefined with the count in the reason |
| ordinary non-constant dates, including tied but non-constant data | computed | computed identically |

## 22. Regression tests added (101 new tests; every existing test kept, none weakened, skipped or marked xfail)

| File | What it covers |
|---|---|
| `tests/unit/test_rolling_confidence.py` | Valid confidences (0.5 to 0.999, `np.float64`, `Fraction`) for both methods against an independent loop; breach counts against an independent loop; confidence 0 and 1, below 0, above 1, NaN, plus and minus infinity, Python and NumPy booleans, strings, bytes, lists, tuples, arrays, 0-d array, dict, `None` and complex, for both methods; the shared-helper identity and identical messages; validation happens before `sliding_window_view` is reached (patched to fail); short histories still validate; unknown methods; inheritance by `backtest_series` and `backtest_universe`; missing observations make exactly the affected windows unavailable and match the independent loop; an invalid confidence is rejected even when the history has gaps. |
| `tests/unit/test_rank_ic_undefined.py` | Constant predictions, constant targets, both, all dates constant, fewer than 10 paired finite observations (missing and infinite values), tied but non-constant data, perfect and reversed rankings; each under `warnings.simplefilter("error")` and with `catch_warnings(record=True)` to show nothing is emitted and no global filter is changed; `ml.score` with constant targets (undefined statistics, no warning, JSON null); `ml.score` with ordinary data against `scipy.stats.spearmanr`. |

Independent expected values: loops in the test files (sorted losses with linear interpolation; sample mean and `statistics.NormalDist` quantile); `scipy.stats.spearmanr` for every defined correlation; hand-known +1 and -1 for perfectly agreeing and reversed rankings.

## 23. Verification and outcomes

| Check | Command | Exit | Outcome |
|---|---|---|---|
| Focused regression tests | `pytest tests/unit/test_rolling_confidence.py tests/unit/test_rank_ic_undefined.py` | 0 | 101 passed |
| Affected existing tests | `pytest tests/unit/test_ml_unit.py tests/unit/test_ml_score.py tests/unit/test_kupiec_backtest.py tests/integration/test_ml.py tests/integration/test_backtest.py` | 0 | 64 passed |
| Complete suite, working environment | `pytest -q` | 0 | **451 passed, 0 failed, 0 skipped, 0 blocked, 0 errors, 0 warnings** (235 s). The 10 warnings of the previous run (all from the correlation of constant inputs) are gone. |
| Complete suite, fresh venv | new venv from `requirements.txt`, then that interpreter: `pytest -q` | 0 | **451 passed, 0 warnings** (231 s) |
| Cached-snapshot pipeline | `run_pipeline.py`, `check_sp500_membership.py` (offline), `make_readme_tables.py` | 0 | no fetch |
| Independent backtest verifier | `verify_backtest_independent.py`, in the working and the fresh environment | 0 | all 1,832 results match |
| README / result consistency | `make_readme_tables.py --check`; the F32 test (inside the 451) | 0 | README block matches the results |
| Fresh environment pipeline | same scripts with the fresh venv | 0 | results byte-identical to the working environment |
| Notebook (not required this round; run anyway) | `jupyter nbconvert --execute` in the fresh environment, output to a temporary directory | 0 | 6 code cells, 0 errors |

## 24. Did valid-data results change?

No. All **12 of 12** result files and `README.md` are byte-identical to the copies saved before the first edit, in the working environment, and the fresh environment's results are byte-identical to the working environment's. There is no artifact difference to explain: no schema was changed in this stage. The stored forecasts, breach counts, Kupiec statistics, rank ICs, portfolio results and ML results are unchanged, which also confirms that no defined, non-constant correlation changed.

## 25. Completed checks, and what remains unverified

Completed in this stage (each by a command that was run, above): both defects reproduced, repaired and covered by tests; full suite twice; independent verifier twice; byte comparison of every result file; README consistency; notebook.

Remaining and not repaired:
- **Infinite values inside a rolling history.** Missing returns (NaN) correctly make windows unavailable. An infinite return in the history is not treated as missing: the historical method then returns an ordinary-looking number for windows containing it and the normal method returns NaN (observed on a 30-day constructed series). I left this alone because the instruction was to preserve the missing-window policy and the data layer guarantees finite returns (test A3). Whether non-finite values should also make a window unavailable is a policy decision for you.
- **Other arguments of the backtest functions** (for example `window`) are not validated.
- **External-data assumptions that no test can settle:** the vendor's retroactive price adjustment (not strictly point-in-time); the accuracy of Yahoo Finance prices; Wikipedia as the membership source (the SATS contradiction, renamed symbols); and the `fetch_data.py` download step, which was not run (frozen snapshot).
- **Not repeated in this stage:** the live browser checks (earlier stages) and dashboard reduced-motion behaviour.
- **Empirical results are unchanged and are not evidence of skill:** VaR is miscalibrated at 99% (historical 1.59% breaches against 1% expected; normal 1.95%); the forecasting model shows no demonstrated skill (R squared against the training mean between -0.9% and -6.0%, rank IC 0.044, t = 0.60 on 11 independent periods); the portfolio results are one period on one split.

## 26. Later audit

`ALL_MODULES_VALIDATION.md` and `MODULE_TEST_MATRIX.md` record a module-by-module audit made after this report. Where they differ from sections 25 and earlier, they supersede them: the backtest `window` argument and the other arguments listed in section 25 as unvalidated are now validated (D43), and the final suite is 679 tests. The infinite-value policy in section 25 is unchanged and remains an open owner decision.

## 27. Infinite values and zero volatility (later change)

Section 25 listed an infinite return inside a rolling history as an open item; it is closed. `backtest.rolling_forecasts` now raises `ValueError` for infinity, and the NaN policy is unchanged. The zero-volatility Sharpe ratio is also now undefined (NaN) and the optimiser refuses such portfolios. See `DECISIONS.md` D42 and D48 and `ALL_MODULES_VALIDATION.md` section 12; the final suite is 719 tests.

