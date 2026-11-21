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


def validate() -> None:
    """Raise ValueError, naming the setting, if the current settings are invalid or inconsistent.

    Checks ranges and the relations between settings (for example that the test period starts after the training
    end, which prevents the evaluation from overlapping the data used to choose portfolios or fit models).
    It is called at the start of the pipeline, of portfolio construction and of the forecasting run.
    """
    import numbers

    import pandas as pd

    def is_int(v, minimum):
        return isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= minimum

    if not is_int(WINDOW, 2):
        raise ValueError(f"WINDOW must be an integer of at least 2, got {WINDOW!r}")
    if not CONFS or any(isinstance(c, bool) or not isinstance(c, numbers.Real) or not 0.0 < c < 1.0 for c in CONFS):
        raise ValueError(f"CONFS must be a non-empty sequence of numbers strictly between 0 and 1, got {CONFS!r}")
    try:
        train_end, test_start = pd.Timestamp(TRAIN_END), pd.Timestamp(TEST_START)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"TRAIN_END and TEST_START must be dates: {exc}") from exc
    if not train_end < test_start:
        raise ValueError(f"TEST_START ({TEST_START}) must be after TRAIN_END ({TRAIN_END}); otherwise the evaluation period overlaps the training data")
    if not isinstance(MARKET, str) or not MARKET:
        raise ValueError(f"MARKET must be a ticker symbol, got {MARKET!r}")
    if not is_int(MIN_ASSETS, 1):
        raise ValueError(f"MIN_ASSETS must be an integer of at least 1, got {MIN_ASSETS!r}")
    for name, p in PRESETS.items():
        lo, hi = p["bounds"]
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"PRESETS[{name!r}]: bounds must satisfy 0 <= low <= high <= 1, got {p['bounds']!r}")
        if not is_int(p["max_stocks"], MIN_ASSETS):
            raise ValueError(f"PRESETS[{name!r}]: max_stocks must be an integer of at least MIN_ASSETS ({MIN_ASSETS}), got {p['max_stocks']!r}")
        if not any(n * lo <= 1.0 <= n * hi for n in range(MIN_ASSETS, p["max_stocks"] + 1)):
            raise ValueError(f"PRESETS[{name!r}]: weights cannot sum to 1 for any holding count from {MIN_ASSETS} to {p['max_stocks']} under bounds {p['bounds']!r}")
    if not is_int(ML_HORIZON, 1):
        raise ValueError(f"ML_HORIZON must be a positive integer, got {ML_HORIZON!r}")
    if not is_int(ML_STEP, 1):
        raise ValueError(f"ML_STEP must be a positive integer, got {ML_STEP!r}")
    if isinstance(ML_RF_WEIGHT, bool) or not isinstance(ML_RF_WEIGHT, numbers.Real) or not 0.0 <= ML_RF_WEIGHT <= 1.0:
        raise ValueError(f"ML_RF_WEIGHT must be a number between 0 and 1, got {ML_RF_WEIGHT!r}")
    if not isinstance(ML_RF_PARAMS, dict) or not isinstance(ML_GB_PARAMS, dict):
        raise ValueError("ML_RF_PARAMS and ML_GB_PARAMS must be dictionaries of model parameters")
