"""Analysis settings, fixed BEFORE any result was computed on the downloaded data.

Changing a value here changes the results; the values used for a given result
file are written into that file.
"""
import math

WINDOW = 250  # trading days of history behind each VaR forecast (unchanged from the original notebook)
CONFS = (0.95, 0.99)
TRAIN_END = "2023-12-31"  # portfolios are chosen using data up to and including this date
TEST_START = "2024-01-01"  # and evaluated on data from this date on (the original README used 2019 to 2024)
MARKET = "SPY"
MIN_ASSETS = 4  # weights summing to 1 are infeasible under the position bounds below this count

# The original notebook applied its VaR thresholds to one-day VaR multiplied by sqrt(252).
# Dividing the original thresholds by sqrt(252) gives the equivalent one-day thresholds.
_S = math.sqrt(252)
PRESETS = {
    "conservative": dict(min_sharpe=0.5, min_max_drawdown=-0.20, max_hist_var99_1d=0.35 / _S, min_return_1y=None, max_stocks=10, bounds=(0.05, 0.25)),
    "balanced": dict(min_sharpe=0.3, min_max_drawdown=-0.30, max_hist_var99_1d=0.45 / _S, min_return_1y=None, max_stocks=12, bounds=(0.02, 0.30)),
    "aggressive": dict(min_sharpe=0.1, min_max_drawdown=None, max_hist_var99_1d=None, min_return_1y=0.05, max_stocks=15, bounds=(0.02, 0.30)),
}

# Machine-learning module. These values were written down before the model was fitted and are not tuned
# on the test period. The seed fixes the models' internal randomness; it does not create any data.
ML_HORIZON = 63  # forecast the simple return from the close on day t to the close 63 trading days later
ML_STEP = 5  # use every 5th trading day as a training sample (neighbouring days overlap almost completely)
ML_SEED = 0
ML_RF_WEIGHT = 0.5  # ensemble = weight * random forest + (1 - weight) * gradient boosting, used for evaluation and for the latest prediction
ML_RF_PARAMS = dict(n_estimators=200, max_depth=8, min_samples_leaf=100, max_features="sqrt", n_jobs=-1)
ML_GB_PARAMS = dict(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.7)
