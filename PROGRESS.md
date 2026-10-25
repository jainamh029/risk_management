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

## 2. Test results on the whole project (final)

| Check | Result |
|---|---|
| `pytest tests` in the working environment | 97 passed in 96 s (backtest 13, dashboard 13, data 9, metrics 9, ML 17, portfolio 18, README and documents 9, VaR 9) |
| Same suite in a brand-new virtual environment built only from `requirements.txt` | 97 passed (final run, below) |
| `scripts/run_pipeline.py` rerun in the clean environment, then `diff -rq results` against the repository | every file byte-identical (after rounding the forecast files to 10 decimals; before the rounding, `ml_latest_forecasts.csv` differed by 1.1e-16 from multithreaded floating-point summation, with the same ranking) |
| `scripts/make_readme_tables.py --check` | README results block matches the results files |
| `notebooks/walkthrough.ipynb` executed with nbclient | 6 code cells, 0 errors, 4 figures |
| Dashboard, headless (Streamlit AppTest) | 13 tests: loads, overview numbers equal the results files, VaR explorer breach counts equal an independent loop, backtest filters, portfolio presets and custom screen, forecast table filters, every control option, data tables, missing-data message, no unsupported text |
| Dashboard, live browser | opened at `localhost:8501`; Overview, VaR explorer (changed confidence to 95% and saw SPY update to 86 breaches of 1,713), Portfolio experiment and Forecasting model tabs viewed. Layout problems found and fixed: chart title overlapping legend and range buttons; misleading upward arrow on "expected" values; default portfolio preset was the one that cannot be built. The Backtest comparison and Glossary tabs were exercised by the headless tests but not looked at in the browser. The server was stopped afterwards |
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
- Package, scripts, tests, dashboard and notebook: no random-number generation anywhere. The only matches are the guard test's own ban list and comments that say data is not simulated. `test_no_data_is_generated_or_simulated_anywhere` enforces this.
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

- The ticker list was never checked against real S&P 500 membership; the README says so.
- Dashboard keyboard navigation and reduced-motion behaviour were set in CSS but not tested with a keyboard or the operating-system setting.
- The Backtest comparison and Glossary tabs were not viewed in a browser.
- No independence test for VaR breaches; no multi-day VaR; no minimum-variance, risk-parity or walk-forward allocation; no Ledoit-Wolf covariance. All listed as optional features in earlier notes and not requested.
- Commit author is the laptop default, not a GitHub address.

## 8. Needs owner action

```bash
ls -la ~/Library/Jupyter/kernels/python3     # should hold only kernel.json and three logo files
rm -r ~/Library/Jupyter/kernels              # removes the directory created during this session
```
