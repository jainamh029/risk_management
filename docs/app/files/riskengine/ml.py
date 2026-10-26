"""Forecasting the 63-day forward return, evaluated honestly.

Design choices that exist because of defects found in the original notebook:
- One pooled model over all eligible stocks (not one model per stock), so it
  has far more independent history to learn from.
- Every feature at day t uses prices up to and including day t only.
- The target for day t is the simple return from the close of day t to the
  close HORIZON trading days later. Those windows overlap for neighbouring
  days, so training samples are thinned (ML_STEP) and the training set is
  purged: a training sample is used only if its whole target window ends on or
  before the training end date. The test set starts after that date, so no
  training target window overlaps a test feature date.
- Models are scored against baselines they must beat: the training-mean
  forecast (out-of-sample R squared), the share of up moves (direction), and a
  plain momentum feature (rank IC).
- Inference on the rank IC uses non-overlapping dates only.
- The latest forecast uses features from the last available date.
- No confidence intervals and no buy/sell labels are produced.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from . import config
from .data import simple_returns

FEATURES = [
    "ret_5", "mom_20", "mom_60", "mom_126", "vol_20", "vol_60",
    "skew_60", "kurt_60", "dist_ma20", "dist_ma50", "drawdown_60",
]


def build_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide (date x ticker) feature frames. A feature is NaN if any price or return in its window is missing."""
    ret = simple_returns(prices)
    f = {
        "ret_5": prices / prices.shift(5) - 1.0,
        "mom_20": prices / prices.shift(20) - 1.0,
        "mom_60": prices / prices.shift(60) - 1.0,
        "mom_126": prices / prices.shift(126) - 1.0,
        "vol_20": ret.rolling(20, min_periods=20).std(),
        "vol_60": ret.rolling(60, min_periods=60).std(),
        "skew_60": ret.rolling(60, min_periods=60).skew(),
        "kurt_60": ret.rolling(60, min_periods=60).kurt(),
        "dist_ma20": prices / prices.rolling(20, min_periods=20).mean() - 1.0,
        "dist_ma50": prices / prices.rolling(50, min_periods=50).mean() - 1.0,
        "drawdown_60": prices / prices.rolling(60, min_periods=60).max() - 1.0,
    }
    return {k: f[k] for k in FEATURES}


def forward_return(prices: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    h = horizon or config.ML_HORIZON
    return prices.shift(-h) / prices - 1.0


def build_panel(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Long table indexed by (date, ticker): all features plus the target `y` (NaN for the last HORIZON days)."""
    px = prices[tickers]
    feats = build_features(px)
    idx = pd.MultiIndex.from_product([px.index, tickers], names=["date", "ticker"])
    data = {k: v[tickers].to_numpy().ravel() for k, v in feats.items()}
    data["y"] = forward_return(px)[tickers].to_numpy().ravel()
    return pd.DataFrame(data, index=idx)


def split_samples(panel: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Purged train / test split by trading-day position."""
    h = config.ML_HORIZON
    pos = pd.Series(np.arange(len(dates)), index=dates)
    train_end_pos = int(pos[: config.TRAIN_END].iloc[-1])
    last_pos = len(dates) - 1
    p = pos.reindex(panel.index.get_level_values("date")).to_numpy()
    complete = panel[FEATURES].notna().all(axis=1).to_numpy()
    has_y = panel["y"].notna().to_numpy()
    thinned = (p % config.ML_STEP) == 0
    train = complete & has_y & thinned & (p + h <= train_end_pos)
    test = complete & has_y & (dates[p] >= pd.Timestamp(config.TEST_START))
    return {"train": panel[train], "test": panel[test], "train_end_pos": train_end_pos, "last_pos": last_pos}


def fit_models(train: pd.DataFrame):
    X, y = train[FEATURES].to_numpy(), train["y"].to_numpy()
    rf = RandomForestRegressor(random_state=config.ML_SEED, **config.ML_RF_PARAMS).fit(X, y)
    gb = GradientBoostingRegressor(random_state=config.ML_SEED, **config.ML_GB_PARAMS).fit(X, y)
    # Fitting in parallel is exact (every tree has its own pre-drawn seed). Predicting in parallel is not: the per-tree
    # outputs are summed in whatever order the threads finish, so the last bit of a forecast changes from call to call.
    # Predictions are therefore made single-threaded, which makes every forecast, and every statistic derived from it, reproducible.
    rf.set_params(n_jobs=1)
    return rf, gb


def ensemble_predict(rf, gb, X: np.ndarray) -> np.ndarray:
    w = config.ML_RF_WEIGHT
    return w * rf.predict(X) + (1.0 - w) * gb.predict(X)


def rank_ic_by_date(pred: pd.Series, y: pd.Series) -> pd.Series:
    """Spearman correlation between forecast and realised return across stocks, for each date.

    A date's correlation is undefined, and the date is omitted from the result, when after removing missing and
    infinite values it has fewer than 10 paired observations, or when the predictions (or the targets) are all equal.
    The check is made before any correlation is computed, so no division-by-zero warning is produced and no
    value is invented. The reason for each omitted date is recorded in `result.attrs["undefined_dates"]`.
    Valid, non-constant dates are computed exactly as before.
    """
    df = pd.DataFrame({"p": pred, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    out, undefined = {}, {}
    for d, g in df.groupby(level="date"):
        n = len(g)
        if n < 10:
            undefined[d] = f"fewer than 10 paired finite observations ({n})"
            continue
        reasons = []
        if g["p"].nunique() < 2:
            reasons.append("constant predictions")
        if g["y"].nunique() < 2:
            reasons.append("constant targets")
        if reasons:
            undefined[d] = " and ".join(reasons) + f" across {n} stocks"
            continue
        pr, yr = g["p"].rank().to_numpy(), g["y"].rank().to_numpy()
        out[d] = float(np.corrcoef(pr, yr)[0, 1])
    s = pd.Series(out, name="rank_ic", dtype=float)
    s.attrs["undefined_dates"] = undefined
    return s


def quintile_spread_by_date(pred: pd.Series, y: pd.Series) -> pd.Series:
    """Mean realised return of the top fifth by forecast minus the bottom fifth, for each date."""
    df = pd.DataFrame({"p": pred, "y": y}).dropna()
    out = {}
    for d, g in df.groupby(level="date"):
        if len(g) < 50:
            continue
        r = g["p"].rank(method="first")
        k = len(g) // 5
        out[d] = float(g["y"][r > len(g) - k].mean() - g["y"][r <= k].mean())
    return pd.Series(out, name="quintile_spread")


def nonoverlapping(series: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Every HORIZON-th trading date of the series, so the forecast windows do not overlap."""
    pos = pd.Series(np.arange(len(dates)), index=dates).reindex(series.index)
    first = pos.min()
    return series[((pos - first) % config.ML_HORIZON) == 0]


def _stats(s: pd.Series) -> dict:
    n = int(len(s))
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "std": sd, "std_error": se, "t_stat": mean / se if se and se > 0 else float("nan")}


def score(pred: pd.Series, test: pd.DataFrame, baseline_mean: float, dates: pd.DatetimeIndex) -> dict:
    y = test["y"]
    if not pred.index.equals(test.index):
        raise ValueError("predictions and test targets must have identical indexes (same dates and tickers in the same order)")
    if not (np.isfinite(pred.to_numpy(dtype=float)).all() and np.isfinite(y.to_numpy(dtype=float)).all() and math.isfinite(baseline_mean)):
        raise ValueError("scoring inputs must be finite: predictions, targets and the baseline contain NaN or infinite values")
    sse_model = float(((y - pred) ** 2).sum())
    sse_base = float(((y - baseline_mean) ** 2).sum())  # reference: the training-mean forecast (not the test mean)
    if sse_base == 0.0:
        # every test target equals the training mean exactly, so the score is undefined; no finite value is substituted
        r2, r2_status = float("nan"), "undefined: every test target equals the training mean, so the reference sum of squares is zero"
    else:
        r2, r2_status = 1.0 - sse_model / sse_base, "defined"
    ic = rank_ic_by_date(pred, y)
    qs = quintile_spread_by_date(pred, y)
    return {
        "n_test_samples": int(len(y)),
        "oos_r2_vs_training_mean": r2,
        "oos_r2_status": r2_status,
        "direction_accuracy": float(((pred > 0) == (y > 0)).mean()),  # up versus not up
        "rank_ic_all_dates": _stats(ic),
        "rank_ic_nonoverlapping": _stats(nonoverlapping(ic, dates)),
        "quintile_spread_all_dates": _stats(qs),
        "quintile_spread_nonoverlapping": _stats(nonoverlapping(qs, dates)),
    }, ic


def run(prices: pd.DataFrame, is_etf: pd.Series) -> dict:
    config.validate()
    stocks = [t for t in prices.columns if not bool(is_etf.get(t, False))]
    panel = build_panel(prices, stocks)
    dates = prices.index
    sp = split_samples(panel, dates)
    train, test = sp["train"], sp["test"]
    if train.empty or test.empty:
        raise ValueError("empty train or test sample")

    baseline_mean = float(train["y"].mean())
    rf, gb = fit_models(train)
    Xte = test[FEATURES].to_numpy()
    preds = {
        "random_forest": pd.Series(rf.predict(Xte), index=test.index),
        "gradient_boosting": pd.Series(gb.predict(Xte), index=test.index),
        "ensemble": pd.Series(ensemble_predict(rf, gb, Xte), index=test.index),
    }
    y_test = test["y"]
    models, ics = {}, {}
    for name, p in preds.items():
        models[name], ics[name] = score(p, test, baseline_mean, dates)

    # baselines
    base = {
        "training_mean_forecast": {
            "value": baseline_mean,
            "direction_accuracy": float(((baseline_mean > 0) == (y_test > 0)).mean()),
        },
        "share_of_test_samples_with_positive_return": float((y_test > 0).mean()),
    }
    for f in ("mom_60", "mom_126"):
        ic_f = rank_ic_by_date(test[f], y_test)
        base[f"rank_ic_{f}_all_dates"] = _stats(ic_f)
        base[f"rank_ic_{f}_nonoverlapping"] = _stats(nonoverlapping(ic_f, dates))

    # final model for the latest forecast: every sample whose target window has ended by the last date
    h = config.ML_HORIZON
    p_all = pd.Series(np.arange(len(dates)), index=dates).reindex(panel.index.get_level_values("date")).to_numpy()
    comp = panel[FEATURES].notna().all(axis=1).to_numpy()
    final_train = panel[comp & panel["y"].notna().to_numpy() & ((p_all % config.ML_STEP) == 0) & (p_all + h <= sp["last_pos"])]
    rf2, gb2 = fit_models(final_train)
    last_date = dates[-1]
    latest = panel.xs(last_date, level="date")[FEATURES].dropna()
    latest_pred = pd.DataFrame(
        {
            "ticker": latest.index,
            "feature_date": last_date,
            "predicted_63d_return": ensemble_predict(rf2, gb2, latest.to_numpy()),
        }
    ).sort_values("predicted_63d_return", ascending=False)
    importance = pd.Series(rf2.feature_importances_, index=FEATURES, name="random_forest_importance").sort_values(ascending=False)

    info = {
        "horizon_trading_days": h,
        "features": FEATURES,
        "n_eligible_stocks": len(stocks),
        "train_samples": int(len(train)),
        "train_first_feature_date": str(train.index.get_level_values("date").min().date()),
        "train_last_feature_date": str(train.index.get_level_values("date").max().date()),
        "train_last_target_end_date": str(dates[int(pd.Series(np.arange(len(dates)), index=dates).reindex([train.index.get_level_values("date").max()]).iloc[0]) + h].date()),
        "test_samples": int(len(test)),
        "test_first_feature_date": str(test.index.get_level_values("date").min().date()),
        "test_last_feature_date": str(test.index.get_level_values("date").max().date()),
        "final_model_train_samples": int(len(final_train)),
        "final_model_last_feature_date": str(final_train.index.get_level_values("date").max().date()),
        "latest_feature_date": str(last_date.date()),
        "latest_forecasts": int(len(latest_pred)),
        "rf_params": config.ML_RF_PARAMS,
        "gb_params": config.ML_GB_PARAMS,
        "rf_weight": config.ML_RF_WEIGHT,
        "seed": config.ML_SEED,
    }
    return {
        "info": info,
        "models": models,
        "baselines": base,
        "ic_by_date": pd.DataFrame(ics).rename_axis("date"),
        "latest": latest_pred,
        "importance": importance,
        "_objects": {"panel": panel, "split": sp, "preds": preds, "baseline_mean": baseline_mean, "rf": rf, "gb": gb},
    }
