# Claims register (Stage 1)

Every factual claim made by `README.md` at commit `888104c`, plus the claims the notebook prints about itself. Nothing here has been changed in the repository.

Status values:
- **verified**: confirmed by reading the code and/or by executing it on synthetic data (reproduce with `python3 -I audit/checks.py`). A claim is never marked verified because it appears in a comment or in saved notebook output.
- **unsupported**: contradicted by the code, absent from the code, or not backed by any evidence in the repository.
- **requires execution**: cannot be confirmed without running the pipeline (network, time, a pinned environment) or without information from the repository owner.

Cell references: 0-based notebook cell index, 1-based line within the cell. README line numbers are lines of `README.md`. Audit IDs (B1, B2, ...) refer to `AUDIT.md`.

## A. README claims

| ID | Exact claim | Code reference | Evidence | Status |
|---|---|---|---|---|
| A1 | Badge: "Status: Production Ready" | none | No tests, no CI, no pinned dependencies, missing requirements (B12), fails on pandas 3.0.3 (B12), several bugs (B1-B10) | unsupported |
| A2 | Badge: "Python 3.7+" | none | Never tested on 3.7; installed stack here is Python 3.12 with current libraries | unsupported |
| A3 | Badge: "Plotly Interactive" | cell 1 lines 24-25 | Plotly is imported; no `go.` or `make_subplots(` call exists in any cell; all figures are matplotlib | unsupported |
| A4 | "A full-stack quantitative investment system replicating core infrastructure used in professional asset management" | whole notebook | No evidence of what professional asset-management infrastructure contains; single notebook, no tests | unsupported |
| A5 | "ML-based alpha generation across 468+ securities" | cell 7 | No out-of-sample evidence of alpha; stale inference row (B8); negative R2 on many models in saved output | unsupported |
| A6 | "It is not a class project. It is a working framework applied to real market data covering the full S&P 500 universe" | cell 1 lines 45-124 | Universe is a hand-typed 483-ticker list incl. 10 ETFs and merged/renamed names (B15). "Working" depends on the pandas version (B12) | unsupported |
| A7 | "built with the same mathematical foundations ... that institutional investors use in production" | none | Comparative claim with no evidence | unsupported |
| A8 | "Multi-method VaR/ES engine at 95% & 99% confidence" | cell 3 lines 31-77 | Historical, normal and a normal-sampling "Monte Carlo" at 95% and 99% exist and run on synthetic data. The methods are not independent (B7) and the parametric VaR has the wrong sign (B1) | unsupported |
| A9 | "47+ risk-adjusted metrics ranked across 468+ securities" | cell 5 lines 107-145 | 28 keys including `Ticker` (27 metrics); several are not risk-adjusted. 468 fetched includes 10 ETFs | unsupported |
| A10 | "Mean-variance optimization: Max Sharpe, Min Vol, Risk Parity" | cell 9 | Only Sharpe maximisation exists. "Min vol" and "risk parity" do not appear in the notebook | unsupported |
| A11 | "Ensemble ML models forecasting 63-day forward returns" | cell 7 | Random forest and gradient boosting with a 63-day target exist. The reported "latest" forecast uses a row from 64 days before the last date (B8) | unsupported |
| A12 | Architecture: "SP500DataFetcher -> 468+ stocks, 5-year history" | cell 1 | Saved run: 2019-01-02 to 2026-01-26 (about 7 years); 468 fetched including ETFs | unsupported |
| A13 | Architecture: "VaRESEngine -> Risk at position level" | cell 3 | Class exists and computes per-stock VaR/ES. No portfolio-level engine; portfolio VaR is computed inside the optimizer (cell 9 lines 133-135) | verified |
| A14 | Architecture: "ComprehensiveRiskAnalyzer -> 47+ metrics per stock" | cell 5 | See A9 | unsupported |
| A15 | Architecture: "AdvancedMLPredictor -> 63-day return forecasts" | cell 7 | See A11 | unsupported |
| A16 | Architecture: "PortfolioOptimizer -> Optimal weight allocation" | cell 9 line 1 | Class is named `AdvancedPortfolioOptimizer`. Weights are in-sample Sharpe maximisers, not shown to be optimal out of sample (B10) | unsupported |
| A17 | Architecture: "AdvancedAnalysis -> Visualizations & reports" | cell 11 line 1 | No class named `AdvancedAnalysis`; a function `generate_comprehensive_report` exists (name search) | unsupported |
| A18 | "Fetches 468+ S&P 500 stocks and major ETFs (SPY, QQQ, DIA, IWM, VTI, VOO, IVV, EFA, EEM, AGG) via yfinance" (line 66) | cell 1 lines 45-124, 136-350 | The 10 ETFs are in the list. The universe is 483 unique tickers; 468 were fetched in the saved run, including ETFs. S&P 500 membership unverified | unsupported |
| A19 | "Covers all 11 GICS sectors" (line 67) | cell 1 lines 46-120 | The list has 11 sector comment groups whose counts match the listed tickers (executed). GICS classification of each ticker was not checked | requires execution |
| A20 | "Robust batch fetching (20 stocks/batch) with automatic error handling and fallback mechanisms" (line 68) | cell 1 lines 177-286 | Batch size 20 and individual-retry fallbacks exist. Bare `except:` hides causes; BRK-B and BF-B were dropped in the saved run (B4) | unsupported (the word "robust"); batch size verified |
| A21 | "Outputs: clean price series, return series, and volume data aligned across all tickers" (line 69) | cell 1 lines 305-317 | Prices, returns and volume are produced. Prices are back-filled with future values (B3), so "clean" is not supported | unsupported |
| A22 | VaR table: Historical Simulation, "Uses actual return distribution" (line 79) | cell 3 lines 46-49 | Implemented; `np.percentile` of losses and mean of losses beyond it | verified |
| A23 | VaR table: Parametric (Normal) | cell 3 lines 34-44 | VaR sign error (B1); ES matches the closed form (executed) | unsupported (VaR); ES verified |
| A24 | VaR table: Cornish-Fisher "Adjusts for skewness and excess kurtosis" (line 81) | none | The string "Cornish" does not occur in the notebook | unsupported |
| A25 | "Outputs at 95% and 99% confidence: VaR, ES / CVaR" (lines 83-85) | cell 3 lines 31-77 | Both levels exist. Returned values are scaled by sqrt(252) and labeled annual, not 1-day (B2) | unsupported |
| A26 | Analyzer table: returns "Daily, annual, and period returns (1M / 3M / 6M / 1Y)" (line 95) | cell 5 lines 28-35 | Implemented as mean daily, mean times 252, and price ratios over 21, 63, 126 and 252 prices | verified |
| A27 | Analyzer table: volatility "Annual volatility, downside deviation, downside volatility" (line 96) | cell 5 lines 38-43 | Annual volatility exists. One downside figure exists (standard deviation of negative returns), not the standard downside deviation (B11); there is no separate "downside deviation" field | unsupported |
| A28 | Analyzer table: "Sharpe, Sortino, Calmar ratios" (line 97) | cell 5 lines 47-49 | All three exist. Sortino uses the non-standard downside figure; risk-free rate fixed at 4% (B11) | requires execution (definitions differ from standard; values unvalidated) |
| A29 | Analyzer table: drawdown "Maximum drawdown, current drawdown, drawdown duration" (line 98) | cell 5 lines 157-169 | Max and current drawdown exist. The word "duration" does not occur in the cell | unsupported (duration); max and current verified by reading |
| A30 | Analyzer table: "Beta vs S&P 500, rolling correlation to benchmark" (line 99) | cell 5 lines 71-74 | Beta against the SPY ETF exists. No rolling correlation anywhere | unsupported (rolling correlation); beta verified by reading |
| A31 | Analyzer table: "RSI, SMA 20/50/200, Price-to-SMA deviation ratios" (line 100) | cell 5 lines 60-68, 137-139 | RSI and three SMAs computed; stored ratios cover SMA 20 and SMA 200 only | verified (partially: the 50-day ratio is not stored) |
| A32 | "Investment Score: composite long-term quality metric synthesizing risk, return, and momentum signals" (line 102) | cell 5 lines 99-105 | `LT_Score` exists with unjustified weights. Inputs are Sharpe, Sortino, drawdown, positive months, and 1-year return over volatility; no momentum signal as such | unsupported |
| A33 | "Feature Engineering (50+ features)" (line 110) | cell 7 lines 31-66; feature list at cell 7 around line 150 | 12 features: 12 `features.append(...)` calls (by reading) and a 12-name `feature_names` list (by reading) | unsupported |
| A34 | Features: "Statistical Moments: mean returns, volatility, skewness, kurtosis" (line 113) | cell 7 lines 35-38 | Implemented | verified |
| A35 | Features: "Momentum: 1M, 3M, 6M, 12M momentum signals" (line 114) | cell 7 lines 41-42 | Only 20-day and 60-day momentum | unsupported |
| A36 | Features: "Moving Averages: SMA ratios, EMA crossovers" (line 115) | cell 7 lines 45-48 | SMA ratios (20, 50) exist; no EMA | unsupported (EMA); SMA ratios verified |
| A37 | Features: "Market Correlation: rolling beta, correlation to SPY" (line 116) | cell 7 | Absent from the feature list | unsupported |
| A38 | Features: "Relative Strength: cross-sectional momentum ranks" (line 117) | cell 7 | Models are per stock; no cross-sectional ranks | unsupported |
| A39 | Models: "Random Forest Regressor: Robust to overfitting"; "Gradient Boosting: strong out-of-sample performance" (lines 124-125) | cell 7 lines 103-141 | Both models exist. No evidence for the performance statements; saved output shows negative R2 for many models | unsupported (statements); models verified |
| A40 | "Time-series cross-validation: future data is never seen during training (no leakage)" (line 128) | cell 7 line 106; cell 1 line 15 | One chronological 80/20 split. `TimeSeriesSplit` imported, never called. 63-day target overlap crosses the boundary (B8) | unsupported |
| A41 | "SelectKBest feature selection to eliminate noise" (line 129) | cell 1 line 17 | Imported, never used | unsupported |
| A42 | "Evaluation metrics: MSE, MAE, R2" (line 130) | cell 7 lines 144-147 | Computed (plus direction accuracy) | verified |
| A43 | "Implements mean-variance optimization via scipy to construct three distinct portfolios: Maximum Sharpe, Minimum Volatility, Risk Parity" (lines 136-142) | cell 9 | `scipy.optimize.minimize` SLSQP with a Sharpe objective is used for three strategies that differ by screen only (B10). No min-vol or risk-parity | unsupported |
| A44 | Constraints: "Long-only (no short selling)" (line 145) | cell 9 lines 112-115 | Lower bounds of 0.05 and 0.02 | verified |
| A45 | Constraints: "Individual position limits to prevent over-concentration" (line 146) | cell 9 lines 112-115 | Upper bounds 0.25 and 0.30 | verified |
| A46 | Constraints: "Leverage constraints consistent with typical fund mandates" (line 147) | cell 9 lines 100-109 | Only sum-to-one and bounds exist; "typical fund mandates" is not defined | unsupported |
| A47 | Visualisation: "Efficient frontier with Max Sharpe, Min Vol, and Risk Parity portfolio overlays" (line 155) | cell 11, cell 14 | Absent ("frontier" does not occur) | unsupported |
| A48 | Visualisation: "VaR/ES return distribution histograms" (line 156) | cell 14 lines 327-346 | A returns histogram with VaR levels exists for one selected stock in cell 14 | requires execution (cell 14 not audited line by line; single stock only) |
| A49 | Visualisation: "Sector-level correlation heatmaps" (line 157) | none | Absent (no heatmap call) | unsupported |
| A50 | Visualisation: "Drawdown time series per security" (line 158) | none | Absent | unsupported |
| A51 | Visualisation: "Performance attribution by sector and factor" (line 159) | none | Absent | unsupported |
| A52 | Visualisation: "ML feature importance rankings" (line 160) | cell 7 lines 149-156 | Importances are computed and stored in the model dict; no cell plots or prints them | unsupported |
| A53 | Quickstart: `pip install numpy pandas yfinance scipy scikit-learn matplotlib seaborn plotly` (line 169) | `requirements.txt` | The README line covers all imports; `requirements.txt` omits `yfinance`, `scikit-learn`, `plotly`. Installability on current versions is untested | requires execution |
| A54 | Quickstart: `fetcher.fetch_data(start_date='2019-01-01', end_date='2024-01-01')` and `fetcher.returns` (lines 186-190) | cell 1 lines 136, 317 | Signature and attribute exist. `main()` uses no end date (cell 13 line 13) and the saved run ended 2026-01-26 | verified (interface); mismatch with the notebook's own run noted |
| A55 | Quickstart: `VaRESEngine(returns, portfolio_value=100_000)` and `calculate_var_es(confidence_levels=[0.95, 0.99], window=250)` (lines 194-199) | cell 3 lines 4, 10 | Signatures match; executed on synthetic data | verified |
| A56 | Quickstart: `ComprehensiveRiskAnalyzer(returns, prices, var_es_results).calculate_all_metrics()` (lines 203-204) | cell 5 lines 4, 9 | Signatures match | verified (interface only). Under pandas 3.0.3 the `resample('M')` alias raises (executed) inside a `try/except` in this method, so it is expected to return no metrics (B12); the method itself was not run |
| A57 | Quickstart: `AdvancedMLPredictor(...).predict_returns(selected_tickers=..., train_size=0.7)` (lines 209-213) | cell 7 | No `predict_returns` method; the methods are `train_models` and `generate_predictions` | unsupported |
| A58 | Quickstart: `PortfolioOptimizer(returns, metrics)`, `optimize_max_sharpe(selected_tickers=...)`, `optimize_min_volatility(...)` (lines 218-221) | cell 9 | None exist. The class is `AdvancedPortfolioOptimizer` with `optimize_portfolio(strategy=...)` | unsupported |
| A59 | Quickstart: `AdvancedAnalysis(...).create_comprehensive_report()` (lines 225-226) | cell 11 | Neither exists | unsupported |
| A60 | Methodology: "Cornish-Fisher ... the most appropriate method for individual stock-level risk measurement" (line 234) | none | Method absent; the appropriateness claim is not tested | unsupported |
| A61 | Methodology: Sortino "producing a more accurate picture of risk-adjusted return quality" (line 237) | cell 5 lines 42-48 | Opinion; Sortino here uses a non-standard downside figure (B11) | unsupported |
| A62 | Methodology: "Why time-series cross-validation for ML?" (lines 239-240) | none | Describes a method the code does not use | unsupported |
| A63 | Methodology: "Why Risk Parity?" (lines 242-243) | none | Method absent | unsupported |
| A64 | Data coverage table: per-sector counts (lines 249-262) | cell 1 section comments and list | Executed count per section equals the comment count (50, 23, 63, 61, 56, 37, 24, 72, 26, 31, 31, 10). These are counts requested, not fetched | verified (as requested counts) |
| A65 | Data coverage table: "Total ~468" (line 263) | cell 1 | Table rows sum to 484 (lower bounds with "+" plus 10 ETFs). 468 is the number fetched in the saved run | unsupported |
| A66 | Repository structure: notebook Parts 1-6 and README only (lines 269-280) | repository | Parts exist roughly as listed (class/function names differ, A17); `requirements.txt` and an extra backtest-plot cell (cell 14) are not listed | unsupported (listing incomplete) |
| A67 | Tech stack: plotly "Interactive dashboards and charts" (line 294) | cell 1 | Imported, not used (A3) | unsupported |
| A68 | Disclaimer: "built for educational and research purposes only. It is not financial advice" (line 300) | notebook prints | Present. The notebook's printed text contradicts it in places ("INVESTMENT RECOMMENDATIONS", "ready for investment decisions"; B14) | verified (present); consistency unsupported |
| A69 | Contact: LinkedIn `linkedin.com/in/s-jainam`, email `shah.jainamh@northeastern.edu` (lines 308-310) | none | Cannot be checked from the repository | requires execution (owner confirmation, S10) |
| A70 | Footer: "Built with Python - February 2026 - Boston, MA" (line 315) | git history | Commits dated 2026-02-18 to 2026-02-24; the notebook's saved data ends 2026-01-26 | verified (date range); location not checkable |

## B. Claims the notebook makes about itself (printed text)

| ID | Claim | Code reference | Evidence | Status |
|---|---|---|---|---|
| B1c | "VaR-ES Risk Engine (Parametric, Historical, Monte Carlo)" | cell 13 line 92 | The three labels exist; Monte Carlo is normal sampling and Parametric VaR has the wrong sign (B1, B7) | unsupported |
| B2c | "ML Predictions with Confidence Intervals" | cell 13 line 94 | Intervals are tree dispersion, not prediction intervals (B9) | unsupported |
| B3c | "Multi-Strategy Portfolio Optimization" | cell 13 line 95 | Three screens over one objective; two silently relaxed in the saved run (B10) | unsupported |
| B4c | "Backtesting with Kupiec Test" | cell 3 lines 158-175 | Implemented; correct for breaches > 0 (executed against independent formula), wrong for zero breaches (B5); no independence test | verified (partially) |
| B5c | "Best VaR Models (most accurate)" | cell 3 lines 243-247 | First five "Good" rows in iteration order | unsupported |
| B6c | "INVESTMENT RECOMMENDATIONS ... Use Conservative portfolio for capital preservation ... Set stop-losses based on VaR thresholds ... Rebalance quarterly" | cell 13 lines 99-112 | Printed unconditionally; the "conservative" portfolio was built with the screen relaxed (B10); no analysis supports the other statements | unsupported |
| B7c | "The complete analysis is ready for investment decisions!" | cell 13 line 160 | No evidence | unsupported |
| B8c | "TOP INVESTMENT PICKS", "PORTFOLIO RECOMMENDATIONS" | cell 11 lines 106, 138 | Rankings by in-sample Sharpe; recommendations not supported | unsupported |
| B9c | "Expected Return" for each portfolio | cell 11 line 143 | In-sample statistic of the optimised portfolio (B10) | unsupported |
| B10c | "Annual VaR (%)" axis label | cell 11 line 64 | Scaled 1-day figure, exceeds 100% for some stocks (B2) | unsupported |
| B11c | "Strong model calibration" / "Excellent breach rate accuracy: within 1% of expected" / conclusions on reliance for risk decisions | cell 14 lines 551-572 | Conditional text keyed to pass rates; the "within 1%" criterion is loose; saved pass rates are 37% overall | requires execution (cell 14 logic reviewed only at the conclusion block) |
| B12c | "Perfect for LinkedIn, GitHub, and portfolio presentations!" | cell 14 line 581 | Promotional text | unsupported |

## C. Numbers recorded in saved notebook output (not re-run)

These are listed so they are not mistaken for verified results. All come from cell 13's saved output (execution count 7) and none has been reproduced in this audit.

| Saved figure | Status |
|---|---|
| 483 tickers requested, 468 fetched, 15 failed (list in AUDIT B15) | requires execution |
| Date range 2019-01-02 to 2026-01-26, 1776 trading days | requires execution |
| 95% VaR "valid" 428/468 (91.5%); 99% "valid" 182/468 (38.9%); "Overall good" 173/468 (37.0%) | requires execution |
| Average breach rate 5.58% at 95% and 1.60% at 99% | requires execution |
| ML "High accuracy" lines (for example SPY 72.8% with R2 -0.047) | requires execution |
| Optimized portfolios: expected return 44.2%, 44.8%, 42.0%; Sharpe 1.61, 1.68, 1.69 (in-sample) | requires execution; in-sample by construction (B10) |

## D. Claims in the rewritten README (current)

Sections A to C above describe the original repository (now in `legacy/`) and are kept as the Stage 1 record. This section covers the new `README.md`. Status was assigned after the tests passed on the real snapshot; "verified" means a test on real data checks it against an independent calculation.

| Claim (paraphrased from the new README) | Evidence | Status |
|---|---|---|
| Runs on real Yahoo Finance daily prices for a hand-maintained list of 483 tickers | `tests/test_data.py::test_universe_is_the_original_list_deduplicated`, `test_ticker_count_in_readme_matches_the_universe`, `test_every_ticker_comes_from_the_universe` | verified |
| No data is simulated, generated or filled in | `test_no_data_is_generated_or_simulated_anywhere`, `test_no_filled_values_in_downloaded_series`, `test_late_listed_tickers_are_not_back_filled` | verified |
| Results tables (data, backtest, SPY VaR/ES, portfolios) | `test_results_block_equals_what_the_results_files_produce`; stored results equal a fresh run (`test_stored_*_match_a_fresh_run`) | verified |
| Daily simple returns `P_t / P_(t-1) - 1`; missing price gives missing return | `test_returns_match_an_independent_calculation` | verified |
| Risk-free rate is the average `^IRX` yield over the window | `test_risk_free_is_the_real_tbill_series` | verified |
| Historical VaR and ES definitions | `test_historical_var_is_the_empirical_quantile` (hand interpolation) | verified |
| Normal VaR and ES formulas | `test_normal_var_satisfies_its_defining_equation`, `test_normal_es_matches_numerical_integration`, `test_the_original_sign_error_is_real_on_real_data` | verified |
| No annualisation or sqrt-time scaling | `test_one_day_var_is_a_plausible_fraction_not_a_scaled_number` | verified |
| Forecast uses only the 250 returns before the day; window with a missing return gives no forecast | `test_forecasts_do_not_use_the_day_they_forecast`, `test_late_listed_ticker_has_no_forecasts_before_a_full_window`, `test_backtest_breach_counts_match_an_independent_loop` | verified |
| Kupiec statistic, log-space computation, `x = 0` and `x = n` handled exactly, chi-square(1) | `test_kupiec_*` | verified |
| Descriptive statistics list and definitions | `tests/test_metrics.py` (downside deviation, drawdowns, beta, Sharpe, Sortino, Calmar, CAGR, skew, excess kurtosis, period returns, missing 1-year returns) | verified |
| A ticker is included in an analysis only with the history it needs | `test_incomplete_windows_are_excluded_not_filled`, `test_short_histories_get_nan_not_a_substitute`, `test_universe_backtest_and_summary_are_consistent` | verified |
| Eligible stocks: non-ETF with a return on every training day | `test_only_stocks_with_a_complete_training_history_are_eligible` | verified |
| Screens on training-window statistics; best by training Sharpe kept | `test_selected_stocks_meet_the_screen_by_independent_calculation`, `test_selected_stocks_are_the_best_training_sharpe_among_those_passing` | verified |
| Long-only weights maximising the training Sharpe within position bounds, from the sample mean and covariance | `test_constructed_portfolios_respect_every_constraint`, `test_optimiser_beats_equal_weight_on_the_training_sharpe`, `test_weights_are_a_local_optimum_of_the_training_sharpe` | verified (local optimality only; global optimality not claimed) |
| A preset with too few passing stocks is reported as not constructed; the screen is never relaxed | `test_screens_are_never_relaxed`, `test_a_screen_with_too_few_passers_is_refused_not_relaxed`, `test_a_screen_with_exactly_the_minimum_passers_is_built_from_those_stocks` | verified |
| Holdings and weights use only data up to the training end | `test_fit_ignores_everything_after_the_training_end`, `test_train_and_test_windows_do_not_overlap` | verified |
| Test results use constant weights rebalanced daily, no transaction costs; equal-weighted column uses the same holdings | `test_out_of_sample_numbers_match_an_independent_recalculation`, `test_equal_weight_comparison_uses_the_same_holdings` | verified |
| Interpretation paragraphs (what the backtest and portfolio numbers do and do not show) | Statements of method and caution; contain no numbers | not testable (methodological statements) |
| Limitations list | Items about Yahoo data quality, survivorship, noisy 99% VaR (250 x 1% = 2.5 days), dependence between tickers, single split | partly verified: the 21 missing tickers and the 4 one-price tickers are in `data/prices_meta.json` and tested; the direction and size of survivorship bias is stated as not measured |
| Reproduce commands work | Steps executed during development: `scripts/fetch_data.py`, `scripts/run_pipeline.py`, `scripts/make_readme_tables.py`, `pytest`; clean-environment install from `requirements.txt` | verified on the author's machine (macOS, Python 3.12) |
| `requirements.txt` pins what was tested | `test_requirements_pin_the_versions_that_ran_the_tests` | verified |
| Every file path named in the README exists | `test_every_path_named_in_the_readme_exists` | verified |
| Tests fail with a clear message when the snapshot is missing | Observed in the clean-environment copy (`PROGRESS.md` row 14) | verified once, not an automated test |
| No emoji, no forbidden phrases in README and code | `test_no_forbidden_claims_in_readme_or_code`, `test_no_emoji_anywhere_in_documentation_or_code` | verified |
