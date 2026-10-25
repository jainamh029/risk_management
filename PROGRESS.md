# Progress

Branch `fix/rigor-pass`. Nothing has been pushed; no pull request, remote change or message was made.

Ground rule from the owner for this stage: no data is created, assumed, simulated or filled in. Every analysis and every test uses the real snapshot downloaded from Yahoo Finance.

## 1. Status

| Stage | Status |
|---|---|
| Stage 1: audit (`AUDIT.md`, `CLAIMS.md`, `DECISIONS.md`) | done, committed as `47cf740` |
| Core repairs (section 3) | done and tested, not yet committed (see section 6) |
| Optional new features (section 4) | not started |

## 2. Commands actually run and their outcomes

| # | Command | Outcome |
|---|---|---|
| 1 | `python3 -m venv .venv` and `pip install` of numpy, pandas, scipy, yfinance, pyarrow, matplotlib, pytest, nbformat, nbclient, ipykernel | Installed numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, yfinance 1.7.0, pyarrow 26.0.0, matplotlib 3.11.2, pytest 9.1.1, nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0, jupyter_client 8.10.0 (Python 3.12) |
| 2 | Download test: `BRK-B`, `BRK.B`, `BF-B`, `SPY`, `VLTO`, `^IRX` | `BRK.B` returned nothing; `BRK-B` returned 1,964 rows; `VLTO` began 2023-10-04 (767 rows); confirmed the dot-conversion bug and the late-listing case on real data |
| 3 | `python scripts/build_universe.py` | `data/universe.csv`: 483 unique tickers, 10 ETFs (duplicate `CVS` dropped, order kept) |
| 4 | `python scripts/fetch_data.py` | 462 of 483 tickers returned (452 stocks, 10 ETFs), 21 returned nothing; 2019-01-02 to 2026-10-23, 1,964 trading days; no internal gaps; 16 tickers listed after the start; 4 tickers stopped before the end |
| 5 | Retried the 21 failures individually (10 of them, including BK, CTRA, K, MMC) | All still returned nothing |
| 6 | Looked at the series of the 4 early-stopping names | AVB, EQR, EA, WBD each have exactly one valid price; recorded in `data/prices_meta.json`, excluded from analyses that need history |
| 7 | `python scripts/run_pipeline.py` (several times during development; final run after the last code change) | Writes `results/`. Final run output: 458 tickers backtested; pooled breach rates and rejection shares as in the README; conservative preset not constructed (0 stocks passed), balanced and aggressive constructed |
| 8 | `pytest tests` | 69 passed (final). Earlier: one test failed because of my own wrong tolerance in `test_historical_var_is_the_empirical_quantile` (the exceedance share can exceed the target by 1/n under linear interpolation); the test was corrected, not the code. Later failures in `test_readme_claims.py` (bare file names in the README layout table, a regex that contained emoji characters) were fixed in the test and README |
| 9 | Defect reintroduction ("mutation") run: original defects put back one at a time into a scratch copy, full suite run each time | 11 distinct reintroduced defects, all made the suite fail: wrong VaR sign, sqrt(252) scaling, Kupiec zero-breach returns 0, `bfill`, forecast window including the forecast day, downside deviation as std of negatives, fit using post-training data, screen relaxation, minimum assets lowered to 1, dot tickers, selection order reversed. The first run had one miss (screen relaxation), because a zero-passer screen was rejected by the optimiser anyway; two boundary tests using real training Sharpe ratios were added and the mutant is now caught |
| 10 | Two optimizer defects: weights replaced by equal weights; Sharpe minimised instead of maximised | Both made the suite fail (`test_optimiser_beats_equal_weight_on_the_training_sharpe` / `test_weights_are_a_local_optimum_of_the_training_sharpe` and others). My first attempt used a mutant that changed nothing (`jac=None`), so its "not caught" line was discarded; the shell was then blocked twice by a tool-side permission service error and the corrected run was done afterwards. Total: 13 reintroduced defects, 13 caught |
| 11 | Executed `notebooks/walkthrough.ipynb` with nbclient against a kernel registered inside `.venv` | 6 code cells, no errors, 4 figures; figures inspected by eye |
| 12 | Clean-environment gate: new venv built only from `requirements.txt`, copy of the repository, `pytest` | 69 passed in that environment |
| 13 | In that copy: `scripts/run_pipeline.py`, then `diff -rq results` against the repository's `results/`; `scripts/make_readme_tables.py --check` | Results files identical after the clean rerun; README results block matches the results |
| 14 | In that copy: moved `data/prices.parquet` away and ran `pytest tests/test_data.py` | Pytest stopped with "No price snapshot at ... Create it with: python scripts/fetch_data.py" |

## 3. Core repairs (all done)

| Repair | Audit IDs | Where |
|---|---|---|
| Pinned, complete requirements; `resample('M')` removed | B12 | `requirements.txt` |
| Data layer: dashes kept, no filling, failures and gaps recorded, explicit end date | B3, B4, B13 | `riskengine/data.py`, `scripts/fetch_data.py` |
| One-day VaR/ES, sign fixed, annualisation removed, Monte Carlo removed | B1, B2, B7 | `riskengine/var.py` |
| Kupiec rewritten; both methods backtested; rejection shares reported | B5, B6 | `riskengine/backtest.py` |
| Metrics corrected; risk-free rate from real T-bill yields | B11 | `riskengine/metrics.py` |
| Portfolio: train/test split, screens never relaxed, out-of-sample reporting | B10 | `riskengine/portfolio.py`, `riskengine/config.py` |
| ML module removed | B8, B9 | decision D2 |
| Advice text and promotional output removed; README rewritten from results | B14, A1-A70 | `README.md`, `scripts/make_readme_tables.py` |
| Universe described truthfully; ETFs flagged and kept out of portfolios | B15 | `data/universe.csv`, README |
| Tests on real data, 69 in total | all | `tests/` |
| Original notebook and README moved to `legacy/` | n/a | `legacy/` |

## 4. Optional new features (not started, need approval)

- Christoffersen independence and conditional-coverage tests; Cornish-Fisher and filtered historical VaR; 10-day horizon.
- Minimum-variance and risk-parity allocation; Ledoit-Wolf covariance; rolling walk-forward allocation with costs.
- Point-in-time index membership data.
- A properly validated ML forecast.
- A dashboard.
- Continuous integration.

## 5. Not done, blocked or not applicable

| Item | Category | Detail |
|---|---|---|
| Check of the ticker list against real S&P 500 membership | not done | No authoritative source consulted. The README says the list is unverified |
| Line-by-line audit of the plotting in the legacy cell 14 | not done | Legacy code, superseded |
| Independence test for VaR breaches | not applicable | Not added; the original "clustering score" was removed because it was not a test |
| Owner confirmation of contact details | pending | The new README has no contact section; add one after confirmation |
| Stray user-level Jupyter kernel | needs owner action | See section 7 |

## 6. Commit status

Not committed yet at the time of writing this file; the commit hashes are in the stage report.

## 7. Side effect outside the repository that needs attention

While registering a notebook kernel I ran `ipykernel install --user`, which created `~/Library/Jupyter/kernels/python3/` (kernel.json plus three logo files, all created during this session; the `kernels` directory did not exist before). Anaconda's Jupyter now lists a "python3" kernel that points to this repository's `.venv`. My attempt to delete the directory was declined by the permission system and I did not work around it. To remove it:

```bash
rm -r ~/Library/Jupyter/kernels
```

Check first with `ls -la ~/Library/Jupyter/kernels/python3`: the directory should contain only `kernel.json`, `logo-32x32.png`, `logo-64x64.png`, `logo-svg.svg`. The repository's own check kernel is registered inside `.venv/share/jupyter` and is not affected.
