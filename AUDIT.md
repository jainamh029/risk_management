# Audit: risk_management

This document was written in Stage 1 (audit only) about the original notebook, which now lives in `legacy/`. The section directly below records how each finding was resolved in the repair stage that followed. Sections 1 to 6 are the original audit text, kept unchanged apart from the notes marked "Repair note".

## Repair status of each finding

All repairs use real downloaded data only. The Stage 1 script `audit/checks.py` demonstrated several defects on synthetic data; the owner's rule is that no data may be created or simulated, so that script was removed (it remains in git history, commit `47cf740`) and each finding is now guarded by tests on the real snapshot (`tests/`). The tests were checked by reintroducing 11 of the original defects one at a time into a scratch copy; all 11 made the suite fail (listed in `PROGRESS.md`).

| ID | Resolution | Guarded by |
|---|---|---|
| B1 normal VaR sign | `riskengine/var.py` uses `-(mu - z*sigma)` | `test_normal_var_satisfies_its_defining_equation`, `test_the_original_sign_error_is_real_on_real_data` (real SPY returns) |
| B2 sqrt(252) scaling and "annual" labels | Removed everywhere. One-day VaR and ES only; the dollar columns are gone | `test_one_day_var_is_a_plausible_fraction_not_a_scaled_number` |
| B3 `bfill`, dead missing-data filter | No filling anywhere; returns are NaN where a price is missing; the filter is deleted. Real data: VLTO has no price before 2023-10-04 | `test_late_listed_tickers_are_not_back_filled`, `test_returns_match_an_independent_calculation`, `test_no_filled_values_in_downloaded_series` |
| B4 ticker handling | Dashes kept (re-checked against Yahoo during repair: `BRK.B` returns nothing, `BRK-B` returns 1,964 rows). Failures are recorded with their reason in `data/prices_meta.json`. Order is the order of `data/universe.csv` | `test_dash_tickers_are_present_in_the_downloaded_data`, `test_universe_is_the_original_list_deduplicated`, `test_snapshot_metadata_matches_the_files` |
| B5 Kupiec | Rewritten in log space, zero and full breaches computed exactly, p-values returned. The clustering score was removed. No independence test was added (see `DECISIONS.md`) | `test_kupiec_*` (reference from the binomial log-pmf, original saved values 3.441928 and 4.317578 reproduced, zero-breach closed form, large-sample stability) |
| B6 backtest scope and labels | Historical and normal VaR are both backtested; the "best models" list is gone; results are reported as breach rates against the nominal rate and the share of tickers rejected | `test_backtest_breach_counts_match_an_independent_loop` (4 real tickers, both methods, both levels), `test_forecasts_do_not_use_the_day_they_forecast`, `test_universe_backtest_and_summary_are_consistent` |
| B7 "Monte Carlo" and Cornish-Fisher | Monte Carlo removed. Cornish-Fisher is not implemented and is not claimed | `test_no_forbidden_claims_in_readme_or_code` |
| B8, B9 ML module | Removed (decision D2). No ML claims remain | `test_no_forbidden_claims_in_readme_or_code`, `test_every_import_in_the_package_is_in_requirements` |
| B10 optimizer | Unused parameters removed. Screens are never relaxed: a preset with fewer than 4 passing stocks is reported "not constructed". Holdings and weights use training data only; results are reported out-of-sample against SPY and the same holdings equal-weighted | `test_fit_ignores_everything_after_the_training_end`, `test_screens_are_never_relaxed`, `test_a_screen_with_too_few_passers_is_refused_not_relaxed`, `test_a_screen_with_exactly_the_minimum_passers_is_built_from_those_stocks`, `test_selected_stocks_meet_the_screen_by_independent_calculation`, `test_out_of_sample_numbers_match_an_independent_recalculation` |
| B11 metric definitions | Downside deviation is the root mean square shortfall over all days. The risk-free rate comes from real 13-week T-bill yields (`^IRX`), not a fixed 4%. `LT_Score`, RSI and moving-average fields were removed. Missing 1-year returns stay missing | `test_downside_deviation_*`, `test_compute_metrics_row_matches_hand_calculation`, `test_short_histories_get_nan_not_a_substitute` |
| B12 environment | `requirements.txt` lists every dependency with the exact tested versions; `resample('M')` is no longer used; Plotly is not used | `test_requirements_pin_the_versions_that_ran_the_tests`, `test_every_import_in_the_package_is_in_requirements`, clean-environment run (see `PROGRESS.md`) |
| B13 reproducibility | Explicit end date, snapshot metadata with download time, pinned versions, settings fixed in `riskengine/config.py` | `test_snapshot_metadata_matches_the_files`, `test_stored_*_match_a_fresh_run` |
| B14 hard-coded advice | Removed. The pipeline prints only numbers it just computed | `test_no_forbidden_claims_in_readme_or_code` |
| B15 universe | The README describes it as a hand-maintained list of 483 tickers. ETFs are flagged in `results/metrics.csv` and are excluded from portfolio eligibility. Membership of the real S&P 500 was not verified and the README says so | `test_ticker_count_in_readme_matches_the_universe`, `test_only_stocks_with_a_complete_training_history_are_eligible` |

New facts learned from the real download during the repair (not in the original audit):
- Yahoo Finance returned data for 462 of 483 tickers on 2026-10-25, with 21 returning nothing even when retried one at a time (list in `data/prices_meta.json`). The saved original run had 15 failures.
- Four tickers (AVB, EQR, EA, WBD) have exactly one valid price in the snapshot. They are excluded from every analysis that needs history, not corrected.
- Pooled breach rates for historical VaR on this snapshot are close to those in the original saved output (5.69% and 1.59% against 5.58% and 1.60%). The share of tickers rejected differs (see the README); the two runs differ in end date, ticker set and data cleaning, and the difference was not reconciled.

## Original audit text (Stage 1)

Audited commit: `888104c` on `main` (local branch `fix/rigor-pass`, no source file changed).
Files audited: `Risk_Management_and_Var_es_engine_for_stocks.ipynb` (15 cells, 8 code cells), `README.md`, `requirements.txt`. There are no tests and no CI configuration.

## 1. How to read this document

- Cell references use the 0-based notebook cell index. "Line" is the 1-based line within that cell's source. Example: "cell 3, line 39".
- Evidence tags:
  - **[Executed]**: I ran the original code or an independent calculation. In Stage 1 some of these runs used synthetic data. Under the owner's rule that no data may be created or simulated, that script was removed and every such statement below was re-established on the real downloaded snapshot; the numbers quoted in this document come from real data. The checks now live in `tests/`.
  - **[Read]**: established by reading the code. Not executed.
  - **[Saved output]**: taken from the outputs saved in the notebook (cell 13, execution count 7). I did not re-run the pipeline, so these numbers are not independently reproduced.
  - **[Unverified]**: not established.
- The pipeline was not re-run end to end. It needs network access to Yahoo Finance and several minutes, and it is not in scope for an audit-only stage.
- Cell 14 (581 lines of plotting and a printed conclusion) was audited by outline and by reading its final conclusion block only. Its plotting code was not audited line by line.

## 2. Corrections to my earlier review

The earlier chat review was based on a partial read. Verified against the full notebook:

| Earlier statement | Status after full audit |
|---|---|
| The ticker list is "today's constituents applied to 2019-2024 history" | Not supported. The list is hand-typed, 483 unique tickers including 10 ETFs and 13 names that failed to download as delisted or renamed (section 3, B15). It is neither the current index nor a point-in-time index. Its membership versus the real S&P 500 is unverified. |
| README claims 12 vs 50+ features, 28 vs 47+ metrics | Confirmed. 12 ML features; 28 keys in the metrics dict, 27 excluding `Ticker`. |
| Parametric VaR sign error | Confirmed by execution on real SPY returns (B1). |
| Kupiec test returns 0 with zero breaches | Confirmed by execution (B5). Whether it changed any saved result is unknown. |
| Single 80/20 split, `TimeSeriesSplit` unused | Confirmed. |
| Optimizer is in-sample | Confirmed, and the saved output shows two of three "strategies" silently relaxed their screens (B10). |
| The ML module "likely shows R2 near zero" | Partly supported: many models flagged "High accuracy" in the saved output have negative R2 (examples in B8). The averages were not captured by my extraction, so no overall figure is claimed here. |
| Stale "latest" prediction, interval not a prediction interval, `max_var` unused, ETFs ranked as stocks, `resample('M')` incompatibility | New findings, not in the earlier review (B8, B9, B10, B12, B15). |

## 3. Confirmed issues

Severity: **High** = changes reported numbers or invalidates a stated claim. **Medium** = wrong or misleading but contained. **Low** = hygiene.

### B1. Normal VaR has the wrong sign (High) [Executed]
- Location: cell 3, line 39 (`param_var_daily = -(mu + z_score * sigma)`); same expression, unused, at lines 131-132.
- Closed form for the loss quantile is `-(mu - z*sigma)`. On the last 250 real SPY returns (ending 2026-10-23) the original expression gives -0.01375 at 95% and -0.01922 at 99%, against the closed form +0.01267 and +0.01815.
- The ES formula at line 40 matches the closed form (checked in `tests/test_var.py` by numerical integration on real returns).
- Consequence: `Param_VaR_95` and `Param_VaR_99` are negative for ordinary inputs. Downstream code reads only the `Hist_*` columns (cell 5, lines 84-87), so the optimizer and report numbers are not affected, but the parametric column is wrong in the returned DataFrame and the README presents parametric VaR as a feature.

### B2. One-day VaR/ES multiplied by sqrt(252) and labeled annual (High) [Executed, Saved output]
- Locations: cell 3, lines 43-44, 52-53, 63-64 and the dollar columns at lines 76-77; cell 9, lines 134-135 (portfolio VaR); cell 11, line 64 (axis label "Annual VaR (%)"); thresholds on these scaled values in cell 9 (`VaR_99 < 0.35`, `< 0.45`).
- The windows are 250 days of daily returns. Scaling a 1-day quantile by sqrt(252) is a 252-day horizon approximation that assumes iid returns, and for the parametric formulas it scales the mean by sqrt(T) when the mean scales with T.
- Consequence: reported "VaR" exceeds 100% of position value. Saved output shows NVDA "VaR 99%: 120.4%", PWR 88.7%, KLAC 110.0%, and the long-term score table lists MU at 1.66 (166%). A long-only position cannot lose more than 100%. On the real snapshot, 99% historical VaR multiplied by sqrt(252) exceeds 100% of position value for 90 of 458 tickers (PARA 407%, AGL 305%) and is 27.7% for SPY.
- The backtest (B6) validates the 1-day figure, while every report prints the scaled figure. What is validated is not what is shown.
- `VaR_95_Dollar` and `VaR_99_Dollar` (lines 76-77) multiply the scaled historical figure by `portfolio_value`.

### B3. Missing-data filling uses future prices; the missing-data filter is dead code (High) [Executed, Read]
- Location: cell 1, lines 305-313 (`close_prices.ffill().bfill()`, then a missing-share filter, then `dropna(thresh=...)`).
- `bfill()` fills leading NaNs with the first valid price. On the real snapshot VLTO has no price before 2023-10-04: back-filling creates 1,197 rows at the first price, 1,196 exact-zero returns, and a daily return standard deviation of 0.894% against 1.435% for the real data.
- The "remove columns with more than 30% missing data" filter at lines 308-310 is computed after filling, so the missing share is always 0 and nothing is ever removed. The `dropna(thresh=...)` at line 313 is a no-op for the same reason. Volume is also `ffill().bfill().fillna(0)` (line 316).
- Consequence: assets that began trading after 2019-01-02 get a flat, zero-volatility history, which understates volatility, VaR and drawdown and inflates Sharpe for those names, and feeds the ML and optimizer stages.
- Likely affected in the saved run: names in the universe whose shares began trading after the data start, for example CARR, OTIS, CEG, GEHC, VLTO, CTVA, DOW, ABNB, WBD, FOX, FOXA. None of these appears in the saved failed-ticker list. Listing dates are from general knowledge and were not verified online, so which tickers were actually back-filled in the saved run is **unverified**.

### B4. Ticker handling drops two active tickers; failures are swallowed; ordering is nondeterministic (Medium) [Saved output, Read]
- Location: cell 1, line 159 (`ticker.replace('-', '.')`, with the comment "Yahoo Finance uses dots"); bare `except:` at lines 173, 218, 256, 280; line 149 (`list(set(...))`).
- Saved output shows `BRK.B` and `BF.B` failing ("possibly delisted; no timezone found"). Both are listed as failures, `['BF-B', 'BRK-B', ...]`. The dot conversion is the likely cause. I did not re-run the download, so the cause is inferred from the saved output and the code.
- `list(set(...))` makes the column and batch order depend on string hashing, so ordering differs between runs. It also makes the "first five" lists in the report order-dependent (B6).
- Bare `except:` clauses record a failure without the reason. `warnings.filterwarnings('ignore')` (cell 1, line 11) hides warnings globally.
- `CVS` appears twice in the literal list (removed by the dedupe at line 128).

### B5. Kupiec test: zero-breach case, no p-value, fragile arithmetic, no independence test (Medium) [Executed, Read]
- Location: cell 3, lines 158-175 and 177-183.
- With zero breaches the function returns 0 (lines 160-161). The correct likelihood ratio is -2*n*ln(1-p): 34.43 at 99% and 175.73 at 95% for SPY's real n=1713 forecasts. Such a model (too conservative) is rejected by the real test but is labeled "Valid" here, and `Model_Quality` becomes "Good" if the other level passes.
- For breaches > 0 the formula matches an independent log-likelihood calculation: n=1525, x=23, p=0.01 gives 3.441928 and x=24 gives 4.317578, both identical to the saved rows. So the core statistic is correct for the common case.
- The code takes the log of a product of powers `(1-p)**(n-x) * p**x`. At the pooled size of the real 99% historical backtest (n=779,137, x=12,384) that product is exactly 0.0 in float64. At the saved run's per-ticker n=1525 it is fine but fragile.
- Validity uses a hard-coded 3.84 cutoff. There are no p-values.
- `clustering_score` (lines 177-183) is the std/mean of calendar days between 95% breaches. It is not an independence test and is not labeled as one in the code, but the comment says "(independence test)".
- Whether any stock in the saved run had zero breaches is **unverified** (the backtest DataFrame was not saved). At the saved 1.60% average 99% breach rate it is unlikely.

### B6. Backtest scope and labels (Medium) [Read, Saved output]
- Only historical-simulation VaR is backtested (cell 3, lines 123-126, 135-142). The parametric VaR is computed at lines 131-132 and never used. The README describes three methods; none but the historical one is validated.
- No look-ahead in the rolling windows: each day's VaR uses `iloc[i-lookback:i]`, which excludes the day being tested. [Read]
- "Best VaR Models (most accurate)" (cell 3, lines 243-247) prints `head(5)` of the rows labeled "Good", which is the first five in (nondeterministic) iteration order, not the five most accurate.
- 468 per-stock tests are summarized as a count of "valid" models. Under a correct model about 5% would reject by chance. Saved output: 95% VaR "valid" 428/468 (91.5%, i.e. 8.5% rejected), 99% "valid" 182/468 (38.9%), "Overall good" 173/468 (37.0%), average breach rates 5.58% at 95% and 1.60% at 99%. These are recorded in the notebook only; they have not been reproduced.
- Cell 14's final conclusion text is conditional on those pass rates (lines 551-572), but it also prints "Excellent breach rate accuracy: within 1% of expected" because the 95% average (5.58%) is within 1.0 percentage point of 5%. That is a loose criterion (20% of the target rate) and sits beside a 37% overall pass rate.

### B7. "Monte Carlo" VaR is the normal model again; Cornish-Fisher is absent (Medium) [Executed, Read]
- Cell 3, lines 55-60 draw 10,000 samples from `Normal(mu, sigma)`. Executed: MC VaR 0.03647 and 0.05067 versus parametric closed form 0.03625 and 0.05080 (differences are sampling noise). It is the parametric method with noise, not an independent third method.
- `np.random.seed(42)` is re-set inside the per-ticker, per-confidence loop (line 56).
- The string "Cornish" appears nowhere in the notebook.

### B8. ML: stale inference row, overlapping targets, no baseline, evaluation differs from use (High) [Read, Executed arithmetic, Saved output]
- **Stale "latest" prediction.** `engineer_features` builds rows for `i` in `range(lookback, len(returns) - 63)` (cell 7, line 26) because each row needs a realized 63-day target. `generate_predictions` takes the last row of that matrix (cell 7, lines 228 and 234). In the saved run (1775 returns) the last row ends 64 return-days before the last date, and its target window is already realized. It is then reported as `Predicted_3M_Return`, converted to BUY/SELL signals (lines 284-294), and used as a screening filter in the optimizer (cell 9, lines 44-52). The model is not predicting the future from current data.
- **Overlapping targets, no embargo.** Targets are 63-day returns on consecutive days (line 71). The split is one chronological 80/20 cut (line 106). In the saved run the 1652 samples give 1321 train and 331 test rows; the last 63 train rows have target windows that overlap the test period, and the test set contains roughly 5 non-overlapping 63-day windows. Per-stock R2 and accuracy therefore rest on very little independent data.
- **No baseline.** "Direction accuracy" (cell 7, line 147, `np.sign(pred) == np.sign(y_test)`) has no comparison to always predicting the sign most common in the sample. The report picks the "Top 5 Most Accurate Models" by test accuracy. The saved output lists many "High accuracy" models with negative R2, for example SPY 72.8% with R2 -0.047, DAL 76.7% with -0.037, SLB 60.1% with -1.514, SO 60.7% with -0.297.
- **Evaluated model is not the deployed model.** Evaluation uses a 50/50 average of random forest and gradient boosting (cell 7, line 140). The prediction uses 0.6/0.4 (line 245).
- Scaling is correct: `StandardScaler` is fit on the training rows only (lines 111-113). [Read]
- Feature windows are offset by one day: the return window ends at price `i` while the price window (RSI, moving averages) ends at price `i-1`. This is an inconsistency, not leakage.
- `SelectKBest` and `TimeSeriesSplit` are imported (cell 1, lines 15, 17) and never used. Comment "Volume features (if available)" (line 50) precedes two return-based features.
- `generate_predictions` ends with `except Exception: continue` (lines 313-314), which hides failures.

### B9. "Confidence intervals" are not prediction intervals (High) [Read]
- Cell 7, lines 248-250 and 264-265: the interval is `prediction ± 1.96 * std(individual tree predictions of the random forest)`, centred on a 0.6/0.4 blend that includes the gradient boosting model, which contributes nothing to the width.
- The standard deviation across trees measures disagreement between trees, not the spread of realized outcomes around the prediction. It ignores residual noise. No calibration or coverage check exists.
- The interval is used to assign "STRONG BUY" (`confidence_lower > 0`, line 284) and to count "high confidence" predictions. Coverage was not measured here (**unverified**), but the construction cannot be interpreted as a 95% interval.

### B10. Portfolio optimizer: unused parameters, silent mandate relaxation, in-sample "expected" figures (High) [Read, Saved output]
- `max_var` is accepted and never used (cell 9, line 10; one occurrence in the cell). `target_return` is used only as half its value, as a floor for the non-conservative strategies (line 108, default floor 6%).
- Silent relaxation: when no stock passes the screen, the code prints "No stocks meet {strategy} criteria, relaxing constraints..." and replaces the screen with `Sharpe > 0` over the whole universe (lines 54-57). Saved output shows this message for both "conservative" and "balanced". Those two portfolios therefore carry no drawdown or VaR screen, yet are reported and described as "Use Conservative portfolio for capital preservation" (cell 13, lines 99-103). After relaxation the predicted-return column is absent, so the screen differs from the aggressive case (lines 44-52 run before the relaxation).
- All three "strategies" maximise the same in-sample Sharpe objective (lines 94-98); they differ only in the screen, the number of stocks and bounds. The README's Min Volatility and Risk Parity portfolios do not exist.
- In-sample estimation: the stock screen uses full-sample Sharpe, drawdown and 1-year return; weights maximise Sharpe on the same sample means (cell 9, lines 90-91, `mean_returns = selected_returns.mean() * 252`). The returned `expected_return`, `expected_volatility` and `sharpe_ratio` (lines 128-130, 146-155) are in-sample statistics of the optimised portfolio, yet are printed as "Expected Return" (cell 11, line 143). Saved output: 44.2%, 44.8%, 42.0% expected return and Sharpe 1.61, 1.68, 1.69 for the three portfolios. No out-of-sample test exists.
- Portfolio VaR (lines 133-135) is a full-sample historical percentile scaled by sqrt(252) (B2).
- The composite `Score` (lines 63-70) mixes quantities in different units with unjustified weights; `LT_Score` in cell 5 (lines 99-105) has the same property.
- Constraints present: weights sum to 1 and per-asset bounds (0.05-0.25 conservative, 0.02-0.30 otherwise), so long-only with position limits. With the stock counts used (10, 12, 15) the bounds are feasible; they would be infeasible for fewer than 4 assets. No leverage constraint exists beyond the sum-to-one constraint.
- Chained assignment on a filtered slice (`filtered['Predicted_Return'] = ...`, lines 46 and 52) is hidden by the global warning filter. Behaviour depends on the pandas version (**unverified**).

### B11. Risk metric definitions (Medium) [Read]
- `downside_vol` (cell 5, lines 42-43) is the standard deviation of the negative returns only, annualised. The standard downside deviation is the root mean square of shortfalls below a target over all observations. The Sortino ratio built on it (line 48) is therefore non-standard, and the README calls it "downside deviation".
- The risk-free rate is a hard-coded 4% (line 46, repeated in cell 9, lines 97 and 130). Annual return is the arithmetic mean daily return times 252 (line 29), while `Return_1Y` is price-based (line 35), with a fallback to the arithmetic figure when fewer than 252 prices exist.
- Monthly consistency uses summed simple returns (cell 5, line 93).
- RSI is the simple-moving-average variant (lines 171-178), with an additive 0.0001 in the denominator that is not scaled to price units.
- Beta against SPY is correctly aligned by index (line 73-74).

### B12. Environment compatibility and requirements (High) [Executed]
- `requirements.txt` lists numpy, pandas, matplotlib, seaborn, scipy. The notebook also imports `yfinance`, `scikit-learn` and `plotly`; all three are missing (checked by script). Nothing is version-pinned.
- Cell 5, line 93 uses `resample('M')`. On the installed pandas 3.0.3 this raises `ValueError: Invalid frequency: M` (the replacement alias is `'ME'`). The call sits inside `try/except Exception` (cell 5, lines 20 and 147), which prints a truncated message and skips the ticker. By reading, every ticker would be skipped and `calculate_all_metrics` would return an empty DataFrame. I executed the alias call, not the full function.
- The saved run (data through 2026-01-26) evidently used an older pandas.
- `plotly` is imported (cell 1, lines 24-25) but `go.` and `make_subplots(` are never called; all figures use matplotlib.
- The notebook is valid nbformat 4.0 (validated).
- The default shell Python here prints NumPy 1.x/2.x binary-compatibility warnings from a pre-installed `pyarrow`; the audit checks still ran. Use a fresh virtual environment for repairs.

### B13. Reproducibility (Medium) [Read, Saved output]
- `main()` calls `fetch_data(start_date='2019-01-01')` with no end date (cell 13, line 13), so the end date is "now" (cell 1, lines 138-139). Saved output shows 2019-01-02 to 2026-01-26, 1776 prices. The README Quickstart shows `end_date='2024-01-01'`.
- No data snapshot, no seeds for the universe ordering, global warning suppression, and no pinned versions.

### B14. Hard-coded advice, conclusions and promotional text printed regardless of results (Medium) [Read]
- Cell 13, lines 99-112 print "INVESTMENT RECOMMENDATIONS" (use the conservative portfolio for capital preservation, follow ML signals, set stop-losses on VaR thresholds, rebalance quarterly or at >5% drift) every run. Line 160 prints "ready for investment decisions!". Lines 92-97 print a checklist claiming "ML Predictions with Confidence Intervals" and "Multi-Strategy Portfolio Optimization".
- Cell 11 prints "TOP INVESTMENT PICKS" and "PORTFOLIO RECOMMENDATIONS" (lines 106, 138).
- Cell 14, line 568-569 says the models "can be relied upon for risk management decisions" when more than 75% pass, and line 581 prints "Perfect for LinkedIn, GitHub, and portfolio presentations!".
- None of these statements is derived from an analysis in the repository.

### B15. The universe is not demonstrably the S&P 500, and mixes ETFs with stocks (High) [Executed, Saved output]
- The literal in cell 1 (lines 45-124) has 484 entries and 483 unique tickers (`CVS` is duplicated), including 10 ETFs (`SPY, QQQ, DIA, IWM, VTI, VOO, IVV, EFA, EEM, AGG`). Section comments match the section counts (50, 23, 63, 61, 56, 37, 24, 72, 26, 31, 31, 10), summing to 484.
- Saved output: 483 requested, 468 fetched, 15 failed: `BF-B, BRK-B, PARA, MRO, ABC, JNPR, CTLT, PKI, HES, ANSS, DFS, WBA, RE, PXD, DISH`. Thirteen of these are names that merged, were acquired or were renamed (so the list mixes vintages); two (`BRK-B`, `BF-B`) are the B4 symbol issue.
- The fetched 468 include the ETFs, so at most 458 are stocks. Calling this "468 S&P 500 stocks" is inaccurate on both counts.
- ETFs are ranked with stocks: the saved "lowest risk" table lists `AGG` and `EFA`, and `SPY` is trained as a stock in the ML stage.
- Membership of the list against the real index is **unverified**; no authoritative source was consulted.
- Consequence for survivorship: the list is neither current constituents nor point-in-time membership, and some removed names fail to download and are silently absent. The direction and size of the bias are unmeasured.

## 4. Suspected issues that remain unverified

| ID | Suspicion | What would verify it |
|---|---|---|
| S1 | Zero-breach stocks inflated the saved "valid" counts | Re-run the backtest and count `Breaches_95 == 0` or `Breaches_99 == 0` rows |
| S2 | Which tickers were back-filled in the saved run | Re-download and count leading NaNs per ticker before `bfill` |
| S3 | `BRK-B` and `BF-B` fail only because of the dot conversion | Re-run those two downloads with the dash form |
| S4 | The ML interval's actual coverage | Out-of-sample coverage test (needs a repaired pipeline) |
| S5 | Whether ML has any out-of-sample skill beyond the sign base rate | Baseline comparison under purged splits |
| S6 | Chained-assignment behaviour in cell 9 across pandas versions | Run under pinned versions |
| S7 | Whether the `Calmar` and `Beta` fields behave at edge values | Checked on real data in the repair stage (`test_compute_metrics_row_matches_hand_calculation`, `test_beta_matches_least_squares_slope`) |
| S8 | Cell 14 plotting code beyond the conclusion block | Line-by-line review in a later stage |
| S9 | Whether yfinance adjusted closes are stable between downloads | Two downloads on different days compared |
| S10 | Contact details in the README (LinkedIn slug, institutional email) | Confirmation by the repository owner |

## 5. README claims unsupported by the implementation

Full per-claim table: `CLAIMS.md`. Summary of unsupported items:

- "Production Ready" badge; "not a class project"; "replicating core infrastructure used in professional asset management"; "same mathematical foundations ... institutional investors use in production"; "alpha generation".
- Methods: Cornish-Fisher VaR, Risk Parity, Minimum Volatility portfolios (absent), "three distinct methodologies" (the third is the normal model sampled), time-series cross-validation, `SelectKBest` feature selection.
- Counts: 47+ metrics (27), 50+ features (12), 468+ S&P 500 stocks (468 fetched including ETFs, from a 483-ticker list), 5-year history (about 7 years in the saved run), sector table total (rows sum to 484, stated total 468).
- Features: efficient frontier, correlation heatmaps, drawdown time series per security, performance attribution, ML feature-importance plots, rolling correlation to benchmark, drawdown duration, EMA crossovers, 1M/3M/6M/12M momentum features, rolling beta features, cross-sectional momentum ranks, leverage constraints, Plotly interactivity.
- Interfaces in the Quickstart that do not exist: `AdvancedMLPredictor.predict_returns`, `PortfolioOptimizer`, `optimize_max_sharpe`, `optimize_min_volatility`, `AdvancedAnalysis`, `create_comprehensive_report`. The code has `AdvancedPortfolioOptimizer.optimize_portfolio` and the function `generate_comprehensive_report`.
- `pip install` line omits nothing the notebook needs but does not match `requirements.txt`, which omits three packages.
- The Python 3.7+ badge is untested and implausible with current dependency versions (unverified; the installed stack is Python 3.12).

## 6. Prioritised repair plan

The smallest stage that makes the repository truthful and the numbers defensible. New features are listed separately in `PROGRESS.md` and need approval.

1. **Environment and data (B12, B13, B3, B4, B15).** Add the missing requirements with version bounds, fix `resample('M')`, preserve dashes in tickers, remove `bfill`, delete the dead missing-data filter and replace it with an explicit documented policy, make ticker order deterministic, log failures with reasons, require an explicit end date, and rename the universe in code and text to what it is.
2. **VaR/ES correctness (B1, B2, B7).** Fix the parametric sign, report 1-day VaR/ES, state the horizon on every column, delete the sqrt(252) annual and dollar columns (or relabel as a stated approximation), remove or rename the "Monte Carlo" method.
3. **Backtest correctness (B5, B6).** Compute Kupiec in log space with zero-breach handling and a p-value, drop the clustering score or relabel it, fix the "best models" selection, report rejection rates against the nominal rate.
4. **ML module (B8, B9).** Recommended: remove it from the core, with its signals and the optimizer's dependence on predictions. If kept: use the true latest feature row, an embargo, a stated baseline, and drop the interval and signal labels.
5. **Optimizer (B10).** Remove unused parameters, stop silent relaxation (fail or report explicitly), relabel in-sample figures, and make the README describe the max-Sharpe-with-screens that exists.
6. **Metrics (B11).** Correct or relabel downside deviation and Sortino; document the fixed risk-free rate.
7. **Text and README (B14, section 5).** Delete hard-coded advice and promotional output; rewrite the README to contain only claims in `CLAIMS.md` marked verified; fix the Quickstart interfaces.
8. **Tests.** Unit tests with independent expected values for each repaired function (closed-form VaR/ES, Kupiec zero-breach and a hand-calculated case, no-lookahead, bounds, fill policy).
9. **Housekeeping.** Confirm contact details (S10).

Dependencies: 1 before everything else (nothing else can be run reliably until the environment works); 4 before 5 if the ML module is kept.
