# Formula and method register

Every formula and method in `riskengine/`, what it is, where it lives, and the test that checks it against an independent calculation on the real downloaded data. "Independent" means the test computes the expected value by a different route than the package (a defining equation, numerical integration, the binomial log-pmf, `scipy`, or a hand-written loop), never by calling the package function it is checking.

Conventions used throughout: daily simple returns; 252 trading days per year; sample standard deviation with `ddof = 1`; losses are positive numbers; VaR and ES are one-day figures as a fraction of position value.

## Data (`riskengine/data.py`)

| Item | Definition | Test |
|---|---|---|
| Daily simple return | `r_t = P_t / P_(t-1) - 1`; NaN if either price is missing; nothing is filled | `test_returns_match_an_independent_calculation` (`np.diff` on SPY, AAPL, VLTO) |
| Prices | Yahoo Finance adjusted close (splits and dividends), so returns are total returns | metadata and snapshot tests in `tests/test_data.py` |
| Average risk-free rate | mean of the 13-week T-bill yield (`^IRX`, percent per year) over the window, divided by 100 | `test_risk_free_is_the_real_tbill_series` |

## VaR and ES (`riskengine/var.py`)

| Item | Definition | Test |
|---|---|---|
| Historical VaR | `quantile(losses, conf)` with linear interpolation between order statistics | `test_historical_var_is_the_empirical_quantile` (hand interpolation on sorted real losses) |
| Historical ES | mean of losses at or above the historical VaR | same test |
| Normal VaR | `-(mu - z*sigma)`, `z = Phi^-1(conf)`; equivalently the loss `L` with `P(R <= -L) = 1 - conf` under Normal(mu, sigma) | `test_normal_var_satisfies_its_defining_equation` (checks the defining equation, not the closed form) |
| Normal ES | `-mu + sigma * phi(z) / (1 - conf)` | `test_normal_es_matches_numerical_integration` (tail integral with `scipy.integrate.quad`) |
| Ordering and scale | ES >= VaR; VaR(99) >= VaR(95); ES(99) >= ES(95); all values are below 50% | `test_var_and_es_orderings_hold_for_every_ticker`, `test_one_day_var_is_a_plausible_fraction_not_a_scaled_number` |

## Backtest (`riskengine/backtest.py`)

| Item | Definition | Test |
|---|---|---|
| Forecast for day t | VaR from the `window` returns before day t; no forecast if the window has a missing return | `test_forecasts_do_not_use_the_day_they_forecast`, `test_late_listed_ticker_has_no_forecasts_before_a_full_window` |
| Breach | `loss_t > forecast_t` (strict) | `test_backtest_breach_counts_match_an_independent_loop` (4 real tickers, both methods, both levels, forecasts and counts from a loop that shares no code) |
| Kupiec likelihood ratio | `LR = -2 [ (n-x) ln(1-p) + x ln p - (n-x) ln(1-x/n) - x ln(x/n) ]`, computed in log space; terms with `x = 0` or `x = n` taken as 0 exactly | `test_kupiec_matches_binomial_reference_on_real_backtest_counts` (reference from `scipy.stats.binom.logpmf`), `test_kupiec_boundaries_use_the_real_forecast_count`, `test_kupiec_is_stable_at_the_pooled_real_sample_size`, `test_kupiec_reproduces_the_numbers_saved_in_the_original_notebook` |
| p-value | `chi2.sf(LR, df=1)` | same tests |
| Pooled breach rate | total breaches / total forecasts | `test_universe_backtest_and_summary_are_consistent` |
| Share rejected | fraction of tickers with p < 0.05 | same test |

## Descriptive statistics (`riskengine/metrics.py`)

| Item | Definition | Test |
|---|---|---|
| Mean annual return | `252 * mean(r)` | `test_compute_metrics_row_matches_hand_calculation` |
| Volatility | `sqrt(252) * std(r)` | same |
| CAGR | `(P_last / P_first)^(1/years) - 1`, years = calendar days / 365.25 | same |
| Downside deviation | `sqrt(mean(min(r - mar, 0)^2)) * sqrt(252)` over all days; `mar` = daily risk-free rate | `test_downside_deviation_matches_hand_formula`, `test_downside_deviation_is_not_the_std_of_negative_returns` |
| Sharpe | `(mean annual return - rf) / volatility` | `test_compute_metrics_row_matches_hand_calculation` |
| Sortino | `(mean annual return - rf) / downside deviation` | same |
| Calmar | `CAGR / abs(max drawdown)` | same |
| Max and current drawdown | worst and latest `P_t / max(P_s, s <= t) - 1` | `test_drawdowns_match_an_independent_loop` |
| Skewness, excess kurtosis | sample (bias-adjusted) estimators | `scipy.stats.skew/kurtosis(bias=False)` in `test_compute_metrics_row_matches_hand_calculation` |
| Beta | `cov(r_i, r_SPY) / var(r_SPY)` over common dates | `test_beta_matches_least_squares_slope` (`np.polyfit`) |
| 1/3/6/12-month return | `P_last / P_(last-k) - 1`, k = 21, 63, 126, 252; NaN if the history is shorter | `test_compute_metrics_row_matches_hand_calculation`, `test_short_histories_get_nan_not_a_substitute` |

## Portfolio (`riskengine/portfolio.py`)

| Item | Definition | Test |
|---|---|---|
| Eligibility | non-ETF stocks with a return on every training day | `test_only_stocks_with_a_complete_training_history_are_eligible` |
| Screen | thresholds on training Sharpe, max drawdown, one-day historical 99% VaR over the last 250 training days, trailing 1-year return | `test_selected_stocks_meet_the_screen_by_independent_calculation` |
| Selection | best training Sharpe among passers, up to the preset's maximum | `test_selected_stocks_are_the_best_training_sharpe_among_those_passing` |
| Weights | maximise `(w.mu - rf) / sqrt(w' S w)` subject to `sum(w) = 1` and position bounds, `mu` and `S` annualised sample mean and covariance of training returns | `test_constructed_portfolios_respect_every_constraint`, `test_optimiser_beats_equal_weight_on_the_training_sharpe`, `test_weights_are_a_local_optimum_of_the_training_sharpe` |
| No relaxation | fewer than 4 passers: not constructed | `test_screens_are_never_relaxed`, `test_a_screen_with_too_few_passers_is_refused_not_relaxed`, `test_a_screen_with_exactly_the_minimum_passers_is_built_from_those_stocks` |
| No look-ahead | holdings and weights are unchanged if all data after the training end is removed | `test_fit_ignores_everything_after_the_training_end` |
| Portfolio return | `sum_i w_i r_i,t`, constant weights each day; a missing return contributes 0 (held as cash) and is counted | `test_out_of_sample_numbers_match_an_independent_recalculation` |
| Performance figures | mean annual return, volatility, Sharpe, total return, max drawdown (the starting capital counts as a peak), one-day historical VaR and ES | `test_out_of_sample_numbers_match_an_independent_recalculation`, `test_max_drawdown_counts_the_starting_capital_as_a_peak`, `test_performance_var_and_es_match_the_var_module_formulas`, `test_spy_benchmark_matches_a_direct_calculation` |
| Equal-weight comparison | same holdings, weights 1/n | `test_equal_weight_comparison_uses_the_same_holdings` |

## Forecasting model (`riskengine/ml.py`)

| Item | Definition | Test |
|---|---|---|
| Features at day t | 5, 20, 60, 126-day price returns; 20 and 60-day volatility, 60-day skewness and excess kurtosis of daily returns; distance of the price from its 20 and 50-day mean; drawdown from the 60-day high. All use prices up to and including day t. Missing if any price in the window is missing | `test_features_match_a_hand_calculation` (AAPL, XOM, JPM at four dates), `test_a_feature_is_missing_unless_its_whole_window_has_prices` |
| No look-ahead | a feature at day t is unchanged when all later prices are removed | `test_features_use_only_data_up_to_the_day` |
| Target | `P_(t+63) / P_t - 1`; missing for the last 63 days | `test_target_is_the_return_to_63_trading_days_later` |
| Training sample | every 5th trading day whose whole target window ends on or before the training end date | `test_training_set_is_purged_and_test_set_starts_after_it` |
| Test sample | feature dates on or after the test start with a realised target | same test |
| Baseline forecast | mean target of the training sample | `test_baseline_and_scores_match_independent_formulas` |
| Out-of-sample R squared | `1 - SSE(model) / SSE(training-mean forecast)` on the test sample | same test |
| Direction accuracy | share of test samples where `(forecast > 0) == (return > 0)` | same test |
| Rank IC | Spearman correlation across stocks between forecast and realised return, per date | `test_rank_ic_matches_scipy_spearman` |
| Independent periods | dates exactly 63 trading days apart; mean, standard error `sd / sqrt(n)`, t statistic `mean / se` | `test_non_overlapping_dates_are_a_full_horizon_apart` |
| Ensemble | `w * random forest + (1 - w) * gradient boosting`, the same blend scored and deployed | `test_the_evaluated_ensemble_is_the_blend_that_is_deployed` |
| Latest forecast | features at the last date; final model trained on samples whose target window has ended | `test_latest_forecast_uses_features_from_the_last_date` |
| No intervals or signals | none produced | `test_no_intervals_or_trading_signals_are_produced` |

## Membership cross-check (`scripts/check_sp500_membership.py`)

| Item | Definition | Test |
|---|---|---|
| Membership on date d | start from today's constituents; for every recorded change effective after d, remove the ticker that was added and restore the ticker that was removed | `test_every_dated_change_flips_membership_on_its_effective_date` (for every change in the window, checked against the day before and the first trading day on or after the effective date), `test_last_day_membership_is_exactly_the_current_list`, `test_known_membership_facts_from_the_stored_tables` |
| Table completeness | the rebuilt index size stays between 500 and 510 on every date | `test_rebuilt_index_size_stays_near_500_on_every_date` |
| No dropped rows | every row of the changes table is kept; the latest change (2026-10-06) is present | `test_no_change_row_was_dropped_while_parsing` |
| Contradictions in the source | tickers added since the window start, absent from today's list, never recorded as removed | `test_source_inconsistencies_are_reported_not_hidden` |
| Stored results | equal a fresh analysis of the stored tables | `test_stored_membership_results_equal_a_fresh_analysis`, `test_summary_counts_are_consistent` |

## Independent backtest verification (`scripts/verify_backtest_independent.py`)

Recomputes, with plain loops and no package code, every forecast, breach count, Kupiec statistic (from the binomial log-pmf) and p-value for all 1,832 ticker, method and confidence combinations, and the summary table. Last run: all match.

## Input contract and undefined scores (repair stage)

| Item | Rule | Test |
|---|---|---|
| VaR/ES confidence | a real scalar, finite, strictly between 0 and 1; booleans rejected; otherwise `ValueError` | `test_invalid_confidence_raises_value_error_with_an_explanation` (19 invalid values, both methods), `test_the_direct_entry_points_validate_too` |
| VaR/ES returns | real numeric, one-dimensional, at least 2 observations, all finite; complex, boolean, string, object, scalar, multidimensional, NaN and infinite inputs raise `ValueError`; nothing is dropped, filled, clipped or flattened | `test_invalid_returns_raise_value_error_with_an_explanation` (24 inputs, both methods), `test_the_returns_are_not_modified_dropped_clipped_or_filled` |
| VaR/ES method | exactly `historical` or `normal` | `test_unknown_methods_raise_value_error` |
| Constant finite returns | valid; VaR = ES = minus the constant | `test_constant_finite_returns_are_valid_and_give_minus_the_constant` |
| Missing windows | unavailable forecasts in the backtest layer, exclusions in `latest_var_table`; never an error and never a filled value | `test_a_missing_return_in_a_window_remains_an_unavailable_forecast_not_an_error`, `test_latest_var_table_excludes_an_incomplete_window_instead_of_raising_or_filling` |
| Out-of-sample R squared | `1 - sum((y - prediction)^2) / sum((y - mean(y_train))^2)`; if the denominator is exactly 0.0 the score is NaN with `oos_r2_status` "undefined: ..." (no tolerance, no substituted value) | `test_the_independent_nonzero_denominator_example_gives_exactly_zero`, `test_the_score_uses_the_training_mean_not_the_test_mean_as_its_reference`, `test_zero_denominator_is_undefined_and_does_not_crash`, `test_a_small_positive_denominator_is_calculated_not_classified_as_zero` |
| Rolling-forecast confidence | the same rule as the public VaR functions (shared helper `var.validate_confidence`); checked before any window is built; the history may contain missing returns | `test_valid_confidence_levels_give_the_independent_reference_forecasts`, `test_invalid_confidence_raises_an_explanatory_value_error`, `test_the_policy_is_the_shared_one_not_a_copy`, `test_validation_happens_before_any_calculation_starts`, `test_missing_observations_make_only_the_affected_windows_unavailable`, `test_the_higher_level_backtest_functions_inherit_the_validation` |
| Rank IC undefined cases | a date is undefined (omitted, reason in `attrs["undefined_dates"]`, no warning, no substituted value) if it has fewer than 10 paired finite observations or constant predictions or constant targets; defined dates are unchanged | `test_a_constant_prediction_date_is_undefined_and_the_others_are_unchanged`, `test_a_constant_target_date_is_undefined`, `test_fewer_than_ten_paired_finite_observations_is_undefined_for_missing_and_infinite_values`, `test_no_warning_is_emitted_for_any_constant_case_and_nothing_is_suppressed_globally`, `test_score_with_constant_targets_returns_undefined_statistics_without_warnings` |
| Serialisation | an undefined score is JSON `null`; a nonstandard `NaN` or `Infinity` literal is never written | `test_an_undefined_score_serialises_to_json_null_and_never_to_a_nonstandard_literal`, `test_the_serialiser_leaves_valid_numbers_unchanged_and_refuses_to_write_nan` |
| Rendering | the README table and the dashboard show "undefined" | `test_readme_tables_show_undefined_instead_of_a_number_or_an_exception`, `test_dashboard_renders_undefined_without_an_exception_or_a_misleading_number` |

## Statistical experiments

| Item | Rule | Test |
|---|---|---|
| Experiment A: independent observations, known distribution | the Kupiec rejection rate equals the exact binomial-enumeration size within 4 standard errors; the statistic equals the reference for every possible breach count | `test_A1_kupiec_statistic_and_pvalue_equal_the_reference_for_every_possible_breach_count`, `test_A2_with_independent_trials_the_rejection_rate_matches_the_exact_theoretical_size` |
| Experiment B: rolling estimated forecasts | pooled breach rate equals the exact i.i.d. breach probability (historical: `(W - c(W-1))/(W+1)`; normal: Student t); rejection rate is an empirical finding, not implied by A | `test_S1_breach_rate_matches_the_exact_theory_and_kupiec_does_not_over_reject`, `test_S2_kupiec_has_power_against_a_volatility_shock` |

## Approximations and conventions that are not errors

- Mean annual return is the arithmetic mean times 252; CAGR is shown beside it.
- The risk-free rate is the average of a 13-week bill yield, a proxy.
- Historical VaR from 250 days at 99% uses interpolation between the second and third largest losses.
- The rank-IC t statistic rests on about 11 independent periods.

## Defects found and fixed in this review pass

| Defect | Effect | Fix | Guard |
|---|---|---|---|
| `performance()` did not count the starting capital as a drawdown peak | A first-day loss, or a loss from the start, was missed in the maximum drawdown of any portfolio or benchmark series | start the wealth series at 1.0 | `test_max_drawdown_counts_the_starting_capital_as_a_peak` (uses real series where the two definitions differ) |
| Direction accuracy used `sign(forecast) == sign(return)`, which scored an exactly flat return as a miss while the baseline share used `return > 0` | one sample in 288,000 differed; definitions were inconsistent | both use "up versus not up" | `test_baseline_and_scores_match_independent_formulas` |
