"""ml.score refuses inputs that would otherwise be silently dropped or mis-aligned."""
import numpy as np
import pandas as pd
import pytest

from riskengine import ml

DATES = pd.bdate_range("2024-01-01", periods=60)
IDX = pd.MultiIndex.from_product([DATES[:3], [f"T{i}" for i in range(12)]], names=["date", "ticker"])


def parts():
    y = pd.Series(np.linspace(-0.1, 0.2, 36), index=IDX)
    p = pd.Series(np.linspace(-0.05, 0.1, 36), index=IDX)
    return p, pd.DataFrame({"y": y})


def test_valid_inputs_are_scored_and_mae_mse_are_not_part_of_the_documented_output():
    p, t = parts()
    out, _ = ml.score(p, t, 0.01, DATES)
    assert out["oos_r2_vs_training_mean"] == pytest.approx(1 - ((t["y"] - p) ** 2).sum() / ((t["y"] - 0.01) ** 2).sum(), rel=1e-12)
    assert not [k for k in out if "mae" in k.lower() or "mse" in k.lower()]  # not implemented, so not claimed


@pytest.mark.parametrize("where,bad", [("pred", np.nan), ("pred", np.inf), ("pred", -np.inf), ("target", np.nan), ("target", np.inf)])
def test_nonfinite_predictions_or_targets_are_rejected_not_dropped(where, bad):
    p, t = parts()
    if where == "pred":
        p.iloc[4] = bad
    else:
        t.iloc[4, 0] = bad
    with pytest.raises(ValueError, match="must be finite"):
        ml.score(p, t, 0.01, DATES)


@pytest.mark.parametrize("baseline", [np.nan, np.inf, -np.inf])
def test_a_nonfinite_baseline_is_rejected(baseline):
    p, t = parts()
    with pytest.raises(ValueError, match="must be finite"):
        ml.score(p, t, baseline, DATES)


def test_misaligned_prediction_and_target_indexes_are_rejected_with_an_explanation():
    p, t = parts()
    with pytest.raises(ValueError, match="identical indexes"):
        ml.score(p.iloc[:30], t, 0.01, DATES)
    with pytest.raises(ValueError, match="identical indexes"):
        ml.score(p.iloc[::-1], t, 0.01, DATES)  # same labels, different order


def test_forecasts_are_bit_for_bit_reproducible_because_prediction_is_single_threaded():
    """Regression: with n_jobs=-1 at prediction time the per-tree outputs were summed in thread-completion order, so the last
    bit of the forest's forecast changed from call to call (observed: 300 distinct outputs in 300 identical calls), and a rerun
    of the pipeline could change the last digit of a stored R squared. Fitting stays parallel; the trees are identical either way."""
    import hashlib

    import numpy as np
    import pandas as pd

    from riskengine import ml

    i = np.arange(2000, dtype=float)  # a smooth, hand-built feature matrix: no random numbers
    X = np.column_stack([np.sin(i * (0.013 + 0.007 * k)) if k % 2 == 0 else np.cos(i * (0.011 + 0.005 * k)) for k in range(len(ml.FEATURES))])
    frame = pd.DataFrame(X, columns=ml.FEATURES)
    frame["y"] = 0.5 * X[:, 0] - 0.3 * X[:, 1] * X[:, 2] + 0.1 * np.sin(i * 0.5)
    rf, gb = ml.fit_models(frame)
    assert rf.n_jobs == 1
    hashes = {hashlib.sha256(ml.ensemble_predict(rf, gb, X).tobytes()).hexdigest() for _ in range(25)}
    assert len(hashes) == 1
