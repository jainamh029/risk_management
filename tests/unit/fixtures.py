"""Hand-constructed deterministic fixtures for unit tests.

These are test fixtures only. They contain no random numbers, are never written into
results/ or data/, and are never presented as observations.
"""
import numpy as np
import pandas as pd


def smooth_prices(n_tickers: int, n_days: int, start: str = "2015-01-01", early_drift: float = -0.0012, late_drift: float = 0.0015, switch: int = 400) -> pd.DataFrame:
    """Deterministic price paths: a piecewise drift plus ticker-specific sine waves. Daily log return is a
    closed-form function of (ticker, day), so the same call always gives the same prices."""
    dates = pd.bdate_range(start, periods=n_days)
    t = np.arange(n_days)
    cols = {}
    for k in range(n_tickers):
        drift = np.where(t < switch, early_drift, late_drift) * (1.0 + 0.25 * np.sin(k))
        wave = 0.012 * np.sin(0.05 * (k + 1) * t + 0.7 * k) + 0.006 * np.sin(0.31 * (k + 2) * t + k)
        log_r = drift + wave
        log_r[0] = 0.0
        cols[f"S{k:02d}"] = 100.0 * np.exp(np.cumsum(log_r))
    return pd.DataFrame(cols, index=dates)


def wavy_returns(n: int, scale: float = 0.01, phase: float = 0.0) -> np.ndarray:
    """A deterministic return series that is neither constant nor trending."""
    t = np.arange(n)
    return scale * (np.sin(0.9 * t + phase) + 0.5 * np.sin(2.3 * t + 1.0) - 0.1)
