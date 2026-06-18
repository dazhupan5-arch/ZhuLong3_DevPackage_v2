"""Strict JSON 序列化：禁止 NaN/Inf，避免 C# System.Text.Json 解析失败。"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    if pd is not None:
        if value is pd.NA:
            return None
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                return None
            return value.isoformat()
        if isinstance(value, pd.Timedelta):
            if pd.isna(value):
                return None
            return value.isoformat()
        if isinstance(value, pd.Series):
            return json_safe(value.tolist())
        if isinstance(value, pd.DataFrame):
            return json_safe(value.to_dict(orient="list"))
    if np is not None:
        if isinstance(value, np.ndarray):
            return json_safe(value.tolist())
        if isinstance(value, np.generic):
            return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    return str(value)


def dumps_strict(obj: Any, **kwargs: Any) -> str:
    safe = json_safe(obj)
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("allow_nan", False)
    text = json.dumps(safe, **kwargs)
    if "NaN" in text or "Infinity" in text or "-Infinity" in text:
        raise ValueError("json_safe leak: non-finite token in serialized output")
    return text
