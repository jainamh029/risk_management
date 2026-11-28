"""JSON-safe serialisation of results.

NaN and infinity are not valid JSON, so they are written as null. dumps() refuses to emit
a nonstandard NaN or Infinity literal under any circumstances (allow_nan=False).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def clean(o):
    """Recursively convert to JSON-safe values: NaN/inf -> None, numpy scalars -> python, Timestamps -> dates."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def dumps(obj, **kwargs) -> str:
    return json.dumps(clean(obj), allow_nan=False, **kwargs)
