"""Horizon V16 位置感知方向标签：在 1h 方向标签上施加结构位置门控。"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np

from zhulong.agent.kn2_location_labels import (
    LocationLabelConfig,
    REGIME_CODES,
    build_entry_masks,
    compute_pos_in_range,
    regime_code_array,
)


def apply_location_gate_to_horizon_labels(
    struct: np.ndarray,
    labels_legacy: np.ndarray,
    close: np.ndarray,
    cfg: LocationLabelConfig | None = None,
) -> dict[str, np.ndarray]:
    """
    对 Horizon 三分类方向标签（-1/0/1）施加结构位置门控。

    - legacy long 且非 long_candidate → flat
    - legacy short 且非 short_candidate → flat
    """
    cfg = cfg or LocationLabelConfig()
    n = min(len(struct), len(labels_legacy), len(close))
    struct = np.asarray(struct[:n], dtype=np.float32)
    legacy = np.asarray(labels_legacy[:n], dtype=np.int8)
    close = np.asarray(close[:n], dtype=np.float64)

    pos = compute_pos_in_range(close.astype(np.float32))
    regime = regime_code_array(struct)
    long_cand, short_cand, _ = build_entry_masks(struct, pos, regime, cfg)

    labels = np.zeros(n, dtype=np.int8)
    legacy_long = legacy == 1
    legacy_short = legacy == -1
    gated_long = legacy_long & long_cand
    gated_short = legacy_short & short_cand
    labels[gated_long] = 1
    labels[gated_short] = -1

    gate_mask = np.zeros(n, dtype=np.int8)
    gate_mask[legacy_long & ~long_cand] = 1
    gate_mask[legacy_short & ~short_cand] = 2

    return {
        "labels": labels,
        "labels_legacy": legacy,
        "long_candidate": long_cand.astype(np.uint8),
        "short_candidate": short_cand.astype(np.uint8),
        "pos_in_range": pos,
        "regime_code": regime,
        "gate_mask": gate_mask,
    }


def summarize_horizon_location_labels(
    result: dict[str, np.ndarray],
    cfg: LocationLabelConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or LocationLabelConfig()
    legacy = result["labels_legacy"]
    labels = result["labels"]
    n = len(labels)
    inv_regime = {v: k for k, v in REGIME_CODES.items()}

    def _counts(arr: np.ndarray) -> dict[str, int]:
        return {
            "short": int((arr == -1).sum()),
            "flat": int((arr == 0).sum()),
            "long": int((arr == 1).sum()),
        }

    legacy_c = _counts(legacy)
    gated_c = _counts(labels)
    report: dict[str, Any] = {
        "total_bars": n,
        "config": asdict(cfg),
        "legacy_counts": legacy_c,
        "gated_counts": gated_c,
        "legacy_pct": {k: round(v / max(n, 1) * 100, 3) for k, v in legacy_c.items()},
        "gated_pct": {k: round(v / max(n, 1) * 100, 3) for k, v in gated_c.items()},
        "filtered_long": int(((legacy == 1) & (labels != 1)).sum()),
        "filtered_short": int(((legacy == -1) & (labels != -1)).sum()),
        "long_candidate_pct": round(float(result["long_candidate"].mean()) * 100, 3),
        "short_candidate_pct": round(float(result["short_candidate"].mean()) * 100, 3),
    }

    regime = result.get("regime_code")
    if regime is not None:
        by_regime: dict[str, Any] = {}
        for code, name in inv_regime.items():
            m = regime == code
            if not m.any():
                continue
            by_regime[name] = {
                "bars": int(m.sum()),
                "legacy_long": int((legacy[m] == 1).sum()),
                "gated_long": int((labels[m] == 1).sum()),
                "legacy_short": int((legacy[m] == -1).sum()),
                "gated_short": int((labels[m] == -1).sum()),
            }
        report["by_regime"] = by_regime

    gm = result.get("gate_mask")
    if gm is not None:
        report["gate_reasons"] = {
            "blocked_long": int((gm == 1).sum()),
            "blocked_short": int((gm == 2).sum()),
        }

    return report


def resolve_horizon_training_labels(
    data: dict[str, Any],
    *,
    label_mode: str = "auto",
) -> tuple[np.ndarray, str]:
    """
    解析 Horizon 训练用 signed labels (-1/0/1)。

    label_mode:
      - auto: NPZ 含 loc_horizon_version → location，否则 legacy
      - location: 使用门控后 labels（或 labels + loc_* 现场门控）
      - legacy: labels_legacy 若存在，否则 labels
    """
    mode = (label_mode or "auto").strip().lower()
    has_loc = "loc_horizon_version" in data or "labels_legacy" in data
    if mode == "auto":
        mode = "location" if has_loc else "legacy"

    if mode == "legacy":
        if "labels_legacy" in data:
            return np.asarray(data["labels_legacy"], dtype=np.int8), "legacy_forward_return"
        return np.asarray(data["labels"], dtype=np.int8), "legacy_forward_return"

    if mode != "location":
        raise ValueError(f"未知 label_mode: {label_mode}")

    if "loc_horizon_version" in data:
        ver = data["loc_horizon_version"]
        ver_s = str(ver[0] if hasattr(ver, "__len__") and len(ver) else ver)
        return np.asarray(data["labels"], dtype=np.int8), ver_s

    if "labels_legacy" not in data:
        raise KeyError("location 模式需要 labels_legacy 或 loc_horizon_version；请先 prepare_horizon_v16_location_labels.py")

    struct = np.asarray(data["struct"], dtype=np.float32)
    legacy = np.asarray(data["labels_legacy"], dtype=np.int8)
    close = np.asarray(data["close"], dtype=np.float64)
    cfg = LocationLabelConfig()
    if "loc_config_json" in data:
        raw = data["loc_config_json"]
        txt = raw[0] if hasattr(raw, "__len__") else raw
        try:
            cfg = LocationLabelConfig(**json.loads(str(txt)))
        except Exception:
            pass
    gated = apply_location_gate_to_horizon_labels(struct, legacy, close, cfg=cfg)
    return gated["labels"], "location_v1_runtime"


def load_horizon_v16_location_npz(path: str | Any) -> dict[str, Any]:
    """加载 NPZ 为 dict（allow_pickle）。"""
    import numpy as np

    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}
