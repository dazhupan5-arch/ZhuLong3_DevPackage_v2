"""特征 golden 回归：固定 CSV → 特征矩阵 hash 不变。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "golden_m5.csv"
META = ROOT / "tests" / "fixtures" / "golden_feature_hash.json"


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.astype(np.float64).tobytes()).hexdigest()


def compute_python_features() -> np.ndarray:
    from zhulong.feature_engine import compute_m5_features, compute_mtf_trend_features

    df = pd.read_csv(FIXTURE, parse_dates=["time"]).set_index("time").sort_index()
    m5 = compute_m5_features(df).dropna()
    mtf = compute_mtf_trend_features(df).loc[m5.index]
    fused = np.concatenate([m5.values, mtf.values], axis=1)
    return fused.astype(np.float32)


def test_golden_feature_hash_stable():
    assert FIXTURE.is_file(), "缺少 golden_m5.csv"
    feats = compute_python_features()
    h = _hash_array(feats)

    if not META.is_file():
        META.write_text(json.dumps({"hash": h, "rows": len(feats), "cols": feats.shape[1]}), encoding="utf-8")
        assert len(feats) > 0
        return

    expected = json.loads(META.read_text(encoding="utf-8"))
    assert h == expected["hash"], f"特征 hash 变化: {h} != {expected['hash']}"
    assert feats.shape[1] == 28  # 22 + 6 MTF


def test_feature_schema_exists():
    schema = ROOT / "zhulong" / "feature_schema.json"
    data = json.loads(schema.read_text(encoding="utf-8"))
    assert data["model_feature_dim"] == 30
    assert len(data["base_columns"]) == 22
    assert len(data["mtf_columns"]) == 6
