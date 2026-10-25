# risk_management

Value at Risk (VaR) and Expected Shortfall (ES) estimation, out-of-sample VaR backtesting, descriptive risk statistics, and a train/test portfolio experiment. Everything runs on real daily prices downloaded from Yahoo Finance for a hand-maintained list of 483 tickers. No data is simulated, generated or filled in.

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

<!-- RESULTS:END -->

All numbers above are written by `scripts/make_readme_tables.py` from the files in `results/`, which `scripts/run_pipeline.py` writes from the downloaded data. A test fails if this block and the results files disagree.

### How to read the results

- **Backtest.** A VaR model is calibrated if the share of days with a loss larger than the VaR is close to one minus the confidence level. The Kupiec test asks whether the observed number of such days is consistent with that rate. It is run for each ticker separately. If a model were correct, about 5% of tickers would be rejected by chance at the 5% level; tickers move together, so that figure is approximate. A share of rejections far above 5% means the model's breach rate differs from nominal for more tickers than chance explains.
- **Portfolio experiment.** Holdings and weights are chosen using only data up to the training end date, then held fixed over the test period. The "in-sample" Sharpe ratio is measured on the data used to choose the portfolio and says nothing about the future. The test-period figures are realised results for one period and one split, for three presets, with no correction for having tried more than one. They are not evidence of future performance. The equal-weighted column uses the same holdings without the optimiser's weights.

## Method

**Data.** Split and dividend adjusted closing prices from Yahoo Finance through `yfinance`. Daily simple returns are `P_t / P_(t-1) - 1`. A missing price gives a missing return. Missing values are never filled, so a stock that started trading later has no returns before its first price. A ticker is included in an analysis only if it has the history that analysis needs; otherwise it is excluded from that analysis and nothing is filled in. The risk-free rate is the average yield of the 13-week Treasury bill series (`^IRX`, from Yahoo Finance) over the analysis window.

**VaR and ES.** One trading day, reported as a positive loss fraction of position value. No annualisation and no square-root-of-time scaling is applied anywhere.
- Historical: VaR is the `conf` quantile of daily losses (linear interpolation); ES is the mean of losses at or above that VaR.
- Normal: with sample mean `mu` and sample standard deviation `sigma` (ddof 1), `VaR = -(mu - z*sigma)` and `ES = -mu + sigma*pdf(z)/(1-conf)`, where `z` is the normal quantile of `conf`.

**Backtest.** For each day the forecast uses only the 250 returns before that day. A breach is a day with loss greater than the forecast. A window containing a missing return gives no forecast for that day. The Kupiec likelihood-ratio statistic for `x` breaches in `n` forecasts against rate `p` is `-2 * [ (n-x)ln(1-p) + x ln p - (n-x)ln(1-x/n) - x ln(x/n) ]`, computed in log space with `x = 0` and `x = n` handled exactly, compared with a chi-square distribution with one degree of freedom.

**Descriptive statistics** (`results/metrics.csv`): mean annualised return (mean daily return times 252), CAGR, annualised volatility, downside deviation (root mean square shortfall below the daily risk-free rate over all days, annualised), Sharpe, Sortino, Calmar, maximum and current drawdown, skewness, excess kurtosis, beta against SPY, and 1/3/6/12-month price returns (missing when the history is too short).

**Portfolio experiment.** Eligible stocks are non-ETF tickers with a return on every day of the training window. Three presets screen on training-window statistics (Sharpe, maximum drawdown, one-day historical 99% VaR, trailing one-year return), keep the best by training Sharpe, and choose long-only weights that maximise the training Sharpe ratio with position bounds, using the sample mean and covariance. A preset with too few stocks passing its screen is reported as not constructed; the screen is never relaxed. Test-period results use constant weights, rebalanced daily, with no transaction costs. The settings are in `riskengine/config.py`.

## Limitations

- The ticker list is hand-typed and was never checked against any index membership. It mixes names from different years, and Yahoo Finance returns no data for some of them (listed in `data/prices_meta.json`). Stocks that left the market are therefore missing from the data. The size and direction of the resulting bias are not measured.
- Adjusted prices from Yahoo Finance are not point-in-time and are used as downloaded. The only checks are those in `data/prices_meta.json` and the tests.
- A 99% VaR from a 250-day window rests on about 2.5 days in the tail, so it is a noisy estimate.
- The per-ticker Kupiec tests are not independent of each other, and the share rejected is not a formal test.
- Only one-day horizons, and only the historical and normal methods, are implemented and tested.
- The portfolio experiment uses one split, sample covariance, no transaction costs, and chooses stocks by past performance.
- The risk-free rate is a proxy: the average 13-week T-bill yield over the window.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/run_pipeline.py
python scripts/make_readme_tables.py
pytest -q
```

`scripts/fetch_data.py` takes several minutes and writes `data/prices.parquet` (git-ignored) and `data/prices_meta.json`. Downloading again later gives a different end date and different results; run the pipeline and the README script again afterwards. The walkthrough notebook is `notebooks/walkthrough.ipynb`.

## Layout

| Path | Contents |
|---|---|
| `riskengine/` | `riskengine/data.py`, `riskengine/var.py`, `riskengine/backtest.py`, `riskengine/metrics.py`, `riskengine/portfolio.py`, `riskengine/config.py` |
| `scripts/` | `scripts/build_universe.py`, `scripts/fetch_data.py`, `scripts/run_pipeline.py`, `scripts/make_readme_tables.py`, `scripts/build_notebook.py` |
| `tests/` | tests on the real downloaded data; they fail with a clear message if the snapshot is missing |
| `results/` | files written by the pipeline |
| `data/` | `data/universe.csv`, `data/prices_meta.json`; the price file `data/prices.parquet` is git-ignored |
| `legacy/` | the original notebook and README, kept for the audit trail and superseded by the code above |
| `AUDIT.md`, `CLAIMS.md`, `PROGRESS.md`, `DECISIONS.md` | the review of the original notebook and the record of what was changed |

## Disclaimer

Educational research code. It is not investment advice, and nothing here recommends buying or selling any security.
