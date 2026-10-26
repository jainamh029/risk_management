# risk_management

Value at Risk (VaR) and Expected Shortfall (ES) estimation, out-of-sample VaR backtesting, descriptive risk statistics, a train/test portfolio experiment, and an out-of-sample test of a machine-learning forecast of the 63-day return, with an interactive dashboard. Every result comes from real daily prices downloaded from Yahoo Finance for a hand-maintained list of 483 tickers. No data is simulated, generated or filled in. The test suite also uses small hand-constructed fixtures (`tests/unit`) and simulated series (`tests/statistical`); these exist only inside tests and are never written to `data/` or `results/`.

## Results

<!-- RESULTS:START -->

### Data used for these results

| Item | Value |
|---|---|
| Source | Yahoo Finance via yfinance, split and dividend adjusted close |
| Downloaded (UTC) | 2026-10-25T05:57:48Z |
| Date range | 2019-01-02 to 2026-10-23 (1964 trading days) |
| Tickers requested | 483 |
| Tickers returned | 462 (452 stocks, 10 ETFs) |
| Tickers Yahoo returned nothing for | 21 |
| Average 13-week T-bill yield, whole range | 2.742% |

### VaR backtest (one-day forecasts, window of 250 days)

| Method | Confidence | Tickers | Forecasts | Pooled breach rate | Expected | Share of tickers rejected (Kupiec, 5%) | Tickers with zero breaches |
|---|---|---|---|---|---|---|---|
| historical | 95% | 458 | 779,137 | 5.69% | 5.00% | 17.9% | 0 |
| historical | 99% | 458 | 779,137 | 1.59% | 1.00% | 69.2% | 0 |
| normal | 95% | 458 | 779,137 | 4.64% | 5.00% | 15.1% | 0 |
| normal | 99% | 458 | 779,137 | 1.95% | 1.00% | 91.5% | 0 |

Denominator: the 458 tickers are all tickers with enough history, including 10 ETFs; the 448 stocks alone account for 762,007 of the 779,137 forecasts. Each forecast is one ticker on one day.

### Latest one-day VaR and ES for SPY (most recent 250 days, ending 2026-10-23)

| Method | VaR 95% | ES 95% | VaR 99% | ES 99% |
|---|---|---|---|---|
| historical | 1.36% | 1.67% | 1.75% | 2.13% |
| normal | 1.27% | 1.60% | 1.81% | 2.09% |

VaR and ES are shown as the loss, in percent of position value, for one trading day. 458 tickers had a complete window; 4 did not and are excluded, not filled.

### Portfolio experiment: chosen with data to 2023-12-29, evaluated on 2024-01-02 to 2026-10-23

| Preset | Status | Stocks passing screen | Holdings | Sharpe, training (in-sample) | Sharpe, test (out-of-sample) | Total return, test | Max drawdown, test | Sharpe, same holdings equal-weighted, test | Total return, same holdings equal-weighted, test |
|---|---|---|---|---|---|---|---|---|---|
| conservative | not constructed | 0 | - | - | - | - | - | - | - |
| balanced | constructed | 6 | 6 | 0.92 | 1.14 | 72.5% | -16.5% | 0.76 | 46.0% |
| aggressive | constructed | 258 | 15 | 1.78 | 1.32 | 127.0% | -17.7% | 1.08 | 106.9% |
| SPY (benchmark) | - | - | - | - | 0.96 | 65.1% | -18.8% | - | - |

Risk-free rate used: 1.884% (training) and 4.270% (test), the average 13-week T-bill yield over each period. 436 stocks had a complete training history and were eligible.

- conservative: 0 stocks passed the screen; at least 4 are required. The screen was not relaxed.

### Forecasting the 63-day return: out-of-sample evaluation

Pooled models trained on 94,371 samples (feature dates 2019-07-10 to 2023-09-25; every target window ends by 2023-12-22) and tested on 287,999 samples (feature dates 2024-01-02 to 2026-07-27). The latest forecast uses features from 2026-10-23 for 448 stocks.
| Model | R squared vs training mean | Direction accuracy (up vs not up) | Rank IC, non-overlapping periods | Rank IC, all dates | Top-fifth minus bottom-fifth return, non-overlapping |
|---|---|---|---|---|---|
| random forest | -0.90% | 53.7% | 0.044 (se 0.071, t 0.62, n 11) | 0.028 | 4.72% |
| gradient boosting | -6.03% | 53.3% | 0.044 (se 0.076, t 0.58, n 11) | 0.028 | 4.57% |
| ensemble | -2.99% | 53.6% | 0.044 (se 0.074, t 0.60, n 11) | 0.028 | 5.27% |
| baseline: training-mean forecast | 0.00% | 57.5% | - | - | - |
| baseline: mom 60 momentum feature alone | - | - | -0.069 (se 0.057, t -1.20, n 11) | -0.032 | - |
| baseline: mom 126 momentum feature alone | - | - | 0.033 (se 0.048, t 0.68, n 11) | 0.013 | - |

Share of test samples with a positive 63-day return: 57.5%.

### Ticker list compared with S&P 500 membership

Source: Wikipedia's current constituents (revision 1381923657) and table of index changes (revision 1379179365), fetched 2026-10-25T17:14:40Z. This is a volunteer-maintained record, not S&P Dow Jones Indices' own. Membership on a past date is rebuilt by undoing each recorded change from today's list; the rebuilt index has between 503 and 508 members on every date of the price window, as a complete table would give.

| Item | Value |
|---|---|
| Tickers in the list | 483 (473 stocks, 10 ETFs) |
| Stocks that are index members today | 412 of 473 |
| Stocks that were members on every date of the price window | 357 |
| Stocks that were members on some dates only | 105 |
| Stocks that were not members on any date, matching by symbol | 11 (5 with price data: AGL, DINO, EQR, FNF, PARA) |
| Current index members that are not in the list | 91 |
| Index members with price data in this snapshot | 74.9% on 2019-01-02, 81.9% on 2026-10-23, 82.9% on average |
| Backtested forecast days on which the stock was an index member | 95.0% of 762,007 |
| Source tables that contradict each other | SATS |

Pooled breach rate over all backtested days, and over only the days when the stock was an index member. Both columns use stocks only (448 stocks, 762,007 forecasts, of which 723,873 fall on member days), because ETFs have no index membership; the "All days" column therefore differs slightly from the table above, which includes the ETFs:

| Method | Confidence | All days | Member days only | Expected |
|---|---|---|---|---|
| historical | 95% | 5.696% | 5.685% | 5.00% |
| historical | 99% | 1.588% | 1.583% | 1.00% |
| normal | 95% | 4.626% | 4.636% | 5.00% |
| normal | 99% | 1.938% | 1.934% | 1.00% |

Matching is by ticker symbol. A company that changed its symbol cannot be matched, so some of the stocks counted as non-members are renamed members; the history table itself records, for example, HFC (the earlier symbol of DINO) as a member until 2021-06-04, and describes Equity Residential as renamed. These were not resolved.

<!-- RESULTS:END -->

All numbers above are written by `scripts/make_readme_tables.py` from the files in `results/`, which `scripts/run_pipeline.py` writes from the downloaded data. A test fails if this block and the results files disagree.

### How to read the results

- **Backtest.** A VaR model is calibrated if the share of days with a loss larger than the VaR is close to one minus the confidence level. The Kupiec test asks whether the observed number of such days is consistent with that rate. It is run for each ticker separately. Rolling-window forecasts share data from day to day, so their breach counts vary less than independent trials would, and the test is conservative: in a simulation with correct i.i.d. data (`tests/statistical`) it rejected between 0% and 8% of series, depending on method and confidence level. A share of rejections far above that range therefore means the model's breach rate differs from nominal for more tickers than chance explains. Tickers move together, so the share is not a formal test.
- **Forecasting model.** Pooled over all eligible stocks. Eleven features, each using prices up to and including the forecast day only: 5, 20, 60 and 126-day price returns, 20 and 60-day volatility, 60-day skewness and excess kurtosis, distance from the 20 and 50-day average price, and the drawdown from the 60-day high. The target is the simple return from the close on day t to the close 63 trading days later. Training uses every fifth day, and only days whose whole target window ends on or before the training end date; the test period starts after that date. The forecast is an equal blend of a random forest and gradient boosting, the same blend that is evaluated. Settings are in `riskengine/config.py` and were not tuned on the test period. Forecasts are not used by the portfolio experiment. No confidence intervals and no buy or sell labels are produced.

**Portfolio experiment.** Holdings and target weights are chosen using only data up to the training end date and then fixed. The test-period return assumes the portfolio is rebalanced back to those target weights every day (constant weights, not buy-and-hold, so positions do not drift between days). The "in-sample" Sharpe ratio is measured on the data used to choose the portfolio and says nothing about the future. The test-period figures are realised results for one period and one split, for three presets, with no correction for having tried more than one. They are not evidence of future performance. The equal-weighted column uses the same holdings without the optimiser's weights.
- **Forecasting.** R squared is measured against the training-mean forecast: a value below zero means the model forecasts worse than always predicting the training average. If every test target equals the training mean exactly, the score is undefined and is shown as "undefined" (null in the results files); the other measures are still reported. Direction accuracy is compared with the share of test samples that went up, because a model that always says "up" scores that much. Rank IC is the cross-sectional Spearman correlation between forecast and realised return on one date; neighbouring dates share most of their 63-day window, so the standard error and t statistic use only dates 63 trading days apart.

## Dashboard

`dashboard/app.py` reads the downloaded data and the files in `results/`. Tabs: overview, a VaR explorer (choose ticker, confidence, window and method; the forecast, breaches and Kupiec test are recomputed live), the backtest comparison with filters and a sortable table, the portfolio experiment (with optional screen sliders), the forecasting evaluation with a filterable table of the latest forecasts, and a glossary. Charts support hover and zoom, series are distinguished by line style and marker as well as colour, and every chart has a data table. Changing portfolio thresholds and then reading the test-period numbers is tuning on the test data, and the dashboard says so next to the controls.

## Method

**Data.** Split and dividend adjusted closing prices from Yahoo Finance through `yfinance`. Daily simple returns are `P_t / P_(t-1) - 1`. A missing price gives a missing return. Missing values are never filled, so a stock that started trading later has no returns before its first price. A ticker is included in an analysis only if it has the history that analysis needs; otherwise it is excluded from that analysis and nothing is filled in. The risk-free rate is the average yield of the 13-week Treasury bill series (`^IRX`, from Yahoo Finance) over the analysis window.

**VaR and ES.** One trading day, reported as a positive loss fraction of position value. No annualisation and no square-root-of-time scaling is applied to VaR or ES. The descriptive statistics (returns, volatility, Sharpe and related ratios) are annualised with 252 trading days.
- Historical: VaR is the `conf` quantile of daily losses (linear interpolation); ES is the mean of losses at or above that VaR.
- Normal: with sample mean `mu` and sample standard deviation `sigma` (ddof 1), `VaR = -(mu - z*sigma)` and `ES = -mu + sigma*pdf(z)/(1-conf)`, where `z` is the normal quantile of `conf`.

**Backtest.** For each day the forecast uses only the 250 returns before that day. A breach is a day with loss greater than the forecast. A window containing a missing return gives no forecast for that day; an infinite return anywhere in the history is a data error and raises an error rather than being treated as missing. The Kupiec likelihood-ratio statistic for `x` breaches in `n` forecasts against rate `p` is `-2 * [ (n-x)ln(1-p) + x ln p - (n-x)ln(1-x/n) - x ln(x/n) ]`, computed in log space with `x = 0` and `x = n` handled exactly, compared with a chi-square distribution with one degree of freedom.

**Descriptive statistics** (`results/metrics.csv`): mean annualised return (mean daily return times 252), CAGR, annualised volatility, downside deviation (root mean square shortfall below the daily risk-free rate over all days, annualised), Sharpe, Sortino, Calmar, maximum and current drawdown, skewness, excess kurtosis, beta against SPY, and 1/3/6/12-month price returns (missing when the history is too short). The Sharpe ratio is undefined (missing) when the volatility is zero, whatever the excess return; in the portfolio optimiser a portfolio with zero volatility is refused and the preset is reported as not constructed, with the reason.

**Portfolio experiment.** Eligible stocks are non-ETF tickers with a return on every day of the training window. Three presets screen on training-window statistics (Sharpe, maximum drawdown, one-day historical 99% VaR, trailing one-year return), keep the best by training Sharpe, and choose long-only weights that maximise the training Sharpe ratio with position bounds, using the sample mean and covariance. A preset with too few stocks passing its screen is reported as not constructed; the screen is never relaxed. Test-period results use constant weights, rebalanced daily (not buy-and-hold), with no transaction costs. The settings are in `riskengine/config.py`.

## Limitations

- The ticker list is hand-typed and is not the S&P 500. The comparison table above (Wikipedia's record of membership, not S&P's own) shows how it differs: some current members are missing, some listed stocks were members for only part of the window, and a few matched by symbol are not members at all. It also mixes names from different years, and Yahoo Finance returns no data for some of them (listed in `data/prices_meta.json`), so stocks that left the market are missing. The backtest conclusions do not change when only member days are counted, but the portfolio and forecasting results use whichever stocks were in the list, and the size of the resulting bias is not measured.
- Adjusted prices from Yahoo Finance are not point-in-time and are used as downloaded. The only checks are those in `data/prices_meta.json` and the tests.
- A 99% VaR from a 250-day window rests on about 2.5 days in the tail, so it is a noisy estimate.
- The per-ticker Kupiec tests are not independent of each other, and the share rejected is not a formal test.
- Only one-day VaR horizons, and only the historical and normal methods, are implemented and tested.
- The forecasting test covers one train/test split and a test period of about 2.7 years, which contains only a handful of non-overlapping 63-day periods, so the rank IC inference rests on few independent observations. The features are price-based only.
- The portfolio experiment uses one split, sample covariance, no transaction costs, and chooses stocks by past performance.
- The risk-free rate is a proxy: the average 13-week T-bill yield over the window.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/run_pipeline.py
python scripts/verify_backtest_independent.py
python scripts/check_sp500_membership.py
python scripts/make_readme_tables.py
pytest -q
```

`pytest -q` runs all three test directories. `pytest tests/unit` needs no data. `pytest -m statistical` runs only the simulation experiments. Statement and branch coverage of the package, the scripts and the dashboard is measured with `pytest --cov=riskengine --cov=dashboard --cov=scripts --cov-branch --cov-report=term-missing`; the scripts that the tests run as subprocesses are included only when `COVERAGE_PROCESS_START` points to `.coveragerc` (the procedure and the measured figures are in `ALL_MODULES_VALIDATION.md`).

`scripts/verify_backtest_independent.py` recomputes every backtest result with a separate plain-loop implementation and exits with an error if any number differs. `python scripts/check_sp500_membership.py --fetch` refreshes the stored Wikipedia tables. `scripts/fetch_data.py` takes several minutes and writes `data/prices.parquet` (git-ignored) and `data/prices_meta.json`. Downloading again later gives a different end date and different results; run the pipeline and the README script again afterwards. The walkthrough notebook is `notebooks/walkthrough.ipynb`; it can be executed without modifying it with `jupyter nbconvert --to notebook --execute --output-dir /tmp/notebook_check notebooks/walkthrough.ipynb`. The dashboard starts with `streamlit run dashboard/app.py` after the pipeline has run.

## Layout

| Path | Contents |
|---|---|
| `riskengine/` | `riskengine/data.py`, `riskengine/var.py`, `riskengine/backtest.py`, `riskengine/metrics.py`, `riskengine/portfolio.py`, `riskengine/ml.py`, `riskengine/config.py` |
| `scripts/` | `scripts/build_universe.py`, `scripts/fetch_data.py`, `scripts/run_pipeline.py`, `scripts/make_readme_tables.py`, `scripts/build_notebook.py`, `scripts/verify_backtest_independent.py`, `scripts/check_sp500_membership.py` |
| `dashboard/` | `dashboard/app.py` |
| `tests/` | `tests/unit` (hand-constructed fixtures, no market data, no network), `tests/integration` (the cached real-price snapshot and the stored results), `tests/statistical` (simulation experiments, marker `statistical`). If the snapshot is missing, every test that needs it is reported as BLOCKED and the run exits with status 10; nothing is downloaded |
| `results/` | files written by the pipeline |
| `data/` | `data/universe.csv`, `data/prices_meta.json`, `data/sp500_current.csv`, `data/sp500_changes.csv`, `data/sp500_membership_meta.json`; the price file `data/prices.parquet` is git-ignored |
| `legacy/` | the original README, kept for the audit trail; the original notebook is in git history (commit `888104c`) and is superseded by the code above |
| `MODULE_TEST_MATRIX.md`, `ALL_MODULES_VALIDATION.md`, `VALIDATION_REPORT.md` | the per-module test matrix, the module-by-module acceptance audit with its commands and counts, and the earlier acceptance report |
| `AUDIT.md`, `CLAIMS.md`, `FORMULAS.md`, `PROGRESS.md`, `DECISIONS.md` | the review of the original notebook, the register of every formula and the test that checks it, and the record of what was changed |

## Disclaimer

Educational research code. It is not investment advice, and nothing here recommends buying or selling any security. The latest forecasts in `results/ml_latest_forecasts.csv` are model outputs whose out-of-sample accuracy is reported above; they are not recommendations.
