#!/usr/bin/env python3
"""KN2 V16 验收：模型加载 + 2025 验证集动作分类指标。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
except Exception:
    pass

import torch  # noqa: F401 — 须在 numpy/pandas 之前

import numpy as np
import pandas as pd

from zhulong.agent.kn2_location_labels import load_kn2_v16_labels
from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state
from zhulong.agent.training_utils import load_npz, signed_to_class, temporal_train_val_masks, VAL_YEAR_DEFAULT
from scripts.v16_acceptance_metrics import (
    apply_classification_thresholds,
    apply_train_test_f1_gates,
    classification_report,
    load_f1_floor,
)

# 验收门槛（与 config/v16_acceptance.json 对齐，KN2 动作为 hold/long/short）
MIN_VAL_ACC = 0.50
MARKET_DIM = 65


def _load_acceptance(root: Path) -> dict:
    p = root / "config" / "v16_acceptance.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8-sig"))
    return {
        "min_val_macro_f1": 0.50,
        "min_class_precision": 0.80,
        "min_class_recall": 0.80,
        "strict_classes": ["long", "short"],
    }


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> dict:
    return classification_report(y_true, y_pred, class_names=("hold", "long", "short"))


def _eval_val_sequences(
    kn2: KN2Inference,
    market_feat: np.ndarray,
    pos_states: np.ndarray,
    y_true: np.ndarray,
    seq_starts: list[int],
    seq_len: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    truths = []
    kn2.reset_hidden()
    for start in seq_starts:
        end = min(start + seq_len, len(market_feat))
        for t in range(start, end):
            if t >= len(y_true):
                break
            dec = kn2.predict(market_feat[t], pos_states[t])
            preds.append(int(dec["action"]))
            truths.append(int(y_true[t]))
    return np.asarray(truths, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/kn2_trader_v16.pth")
    parser.add_argument("--npz", default="data/clean/kn2_training_v16_location.npz")
    parser.add_argument(
        "--label-mode",
        choices=("auto", "location", "legacy"),
        default="auto",
    )
    parser.add_argument("--max-val-bars", type=int, default=20000, help="0=all val bars")
    args = parser.parse_args()
    acc_cfg = _load_acceptance(_ROOT)

    model_path = _ROOT / args.model
    meta_path = model_path.with_suffix(".meta.json")
    npz_path = _ROOT / args.npz

    report: dict = {"architecture": "kn2_v16", "passed": False, "failures": []}

    if not model_path.is_file():
        report["failures"].append(f"missing_model:{model_path}")
        print(json.dumps(report, indent=2))
        return 1
    if not meta_path.is_file():
        report["failures"].append(f"missing_meta:{meta_path}")
        print(json.dumps(report, indent=2))
        return 1

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    report["meta"] = meta
    if int(meta.get("market_dim", 0)) != MARKET_DIM:
        report["failures"].append(f"market_dim_expected_{MARKET_DIM}_got_{meta.get('market_dim')}")
    if meta.get("architecture") not in ("kn2_v16", None) and meta.get("market_dim") == MARKET_DIM:
        pass  # architecture field optional on older meta

    kn2 = KN2Inference(model_path)
    report["model_ready"] = kn2.is_ready
    report["inferred_market_dim"] = kn2.market_dim
    if not kn2.is_ready:
        report["failures"].append("model_not_ready")
        print(json.dumps(report, indent=2))
        return 1
    if kn2.market_dim != MARKET_DIM:
        report["failures"].append(f"inference_market_dim_{kn2.market_dim}")

    # smoke predict
    mf0 = np.zeros(MARKET_DIM, dtype=np.float32)
    dec0 = kn2.predict(mf0, encode_position_state())
    report["smoke"] = {
        "action": dec0.get("action_name"),
        "confidence": float(dec0.get("confidence", 0)),
        "should_trade": bool(dec0.get("should_trade")),
    }

    if not npz_path.is_file():
        report["failures"].append(f"missing_npz:{npz_path}")
        print(json.dumps(report, indent=2))
        return 1

    data = load_npz(npz_path)
    market_feat = np.asarray(data["market_feat"], dtype=np.float32)
    n = len(market_feat)
    times = pd.to_datetime(data["time"]) if "time" in data else pd.date_range("2020-01-01", periods=n, freq="5min")
    df = pd.DataFrame(
        {
            "open": data.get("open", np.zeros(n)),
            "high": data.get("high", np.zeros(n)),
            "low": data.get("low", np.zeros(n)),
            "close": data.get("close", np.zeros(n)),
            "volume": data.get("volume", np.zeros(n)),
        },
        index=times[:n],
    )
    _, val_mask = temporal_train_val_masks(
        times[:n],
        val_year=int(acc_cfg.get("val_year", VAL_YEAR_DEFAULT)),
    )
    val_mask = np.asarray(val_mask, dtype=bool)
    train_mask, _ = temporal_train_val_masks(
        times[:n],
        val_year=int(acc_cfg.get("val_year", VAL_YEAR_DEFAULT)),
    )
    train_mask = np.asarray(train_mask, dtype=bool)
    if int(val_mask.sum()) < 1000:
        report["failures"].append(
            f"val_year_{acc_cfg.get('val_year', VAL_YEAR_DEFAULT)}_sample_too_small_{int(val_mask.sum())}"
        )
        print(json.dumps(report, indent=2))
        return 2

    print(f"Loading labels for validation ({val_mask.sum():,} bars, mode={args.label_mode})...", flush=True)
    t0 = time.perf_counter()
    labels, label_version = load_kn2_v16_labels(
        data,
        df.reset_index(drop=True),
        market_feat,
        label_mode=args.label_mode,
    )
    report["label_version"] = label_version
    print(
        f"labels done in {time.perf_counter() - t0:.1f}s | "
        f"should_trade={labels['should_trade'].mean() * 100:.1f}%",
        flush=True,
    )

    pos_states = np.tile(encode_position_state(), (n, 1)).astype(np.float32)
    seq_len = 64

    def _eval_on_mask(mask: np.ndarray, max_bars: int) -> tuple[dict, np.ndarray, np.ndarray]:
        idx = np.where(mask)[0]
        if max_bars > 0 and len(idx) > max_bars:
            step = max(1, len(idx) // max_bars)
            idx = idx[::step][:max_bars]
        val_set = set(idx.tolist())
        seq_starts = [
            s for s in range(0, n - seq_len, seq_len // 2)
            if any(i in val_set for i in range(s, min(s + seq_len, n)))
        ]
        if max_bars > 0:
            seq_starts = seq_starts[: max(1, max_bars // seq_len)]
        y_true_local, y_pred_local = _eval_val_sequences(
            kn2, market_feat, pos_states, labels["action"], seq_starts, seq_len=seq_len
        )
        local_mask = (y_true_local < 3) & (y_pred_local < 3)
        y_t = y_true_local[local_mask]
        y_p = y_pred_local[local_mask]
        return _class_metrics(y_t, y_p, n_classes=3), y_t, y_p

    print(f"Evaluating train split ({train_mask.sum():,} bars)...", flush=True)
    train_metrics, _, _ = _eval_on_mask(train_mask, args.max_val_bars)
    report["train_eval"] = train_metrics

    print(f"Evaluating test split ({val_mask.sum():,} bars)...", flush=True)
    t1 = time.perf_counter()
    test_metrics, y_true, y_pred = _eval_on_mask(val_mask, args.max_val_bars)
    metrics = test_metrics
    report["test_eval"] = {
        "bars": int(len(y_true)),
        "elapsed_sec": round(time.perf_counter() - t1, 1),
        **test_metrics,
    }
    report["val_eval"] = report["test_eval"]

    acc = metrics["accuracy"]
    if acc <= MIN_VAL_ACC:
        report["failures"].append(f"test_accuracy_below_{MIN_VAL_ACC}")
    apply_classification_thresholds(metrics, acc_cfg, report["failures"], prefix="test")
    apply_train_test_f1_gates(train_metrics, test_metrics, acc_cfg, report["failures"], prefix="kn2")

    min_f1 = load_f1_floor(acc_cfg)
    report["thresholds"] = {
        "acceptance_contract_version": acc_cfg.get("acceptance_contract_version"),
        "min_test_accuracy": MIN_VAL_ACC,
        "min_train_macro_f1": acc_cfg.get("min_train_macro_f1", min_f1),
        "min_test_macro_f1": acc_cfg.get("min_test_macro_f1", min_f1),
        "max_train_test_f1_gap": acc_cfg.get("max_train_test_f1_gap", 0.10),
        "min_class_precision": acc_cfg.get("min_class_precision", 0.80),
        "min_class_recall": acc_cfg.get("min_class_recall", 0.80),
        "strict_classes": acc_cfg.get("strict_classes", ["long", "short"]),
        "require_no_data_leak": acc_cfg.get("require_no_data_leak", True),
        "require_no_future_function": acc_cfg.get("require_no_future_function", True),
    }
    report["passed"] = len(report["failures"]) == 0

    out_dir = _ROOT / "data" / "training" / "reports" / "kn2_v16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "acceptance_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== KN2 V16 Acceptance ===")
    print(f"train macro_f1: {train_metrics.get('macro_f1', 0):.4f} (need > {min_f1})")
    print(f"test macro_f1: {metrics.get('macro_f1', 0):.4f} (need > {min_f1})")
    print(f"train-test gap: {float(train_metrics.get('macro_f1', 0)) - float(metrics.get('macro_f1', 0)):.4f}")
    print(f"test accuracy: {acc:.2%} (need > {MIN_VAL_ACC:.0%})")
    for cls in ("hold", "long", "short"):
        pc = metrics["per_class"][cls]
        print(
            f"  {cls}: prec={pc['precision']:.2%} recall={pc['recall']:.2%} "
            f"f1={pc.get('f1', 0):.2%} support={pc['support']}"
        )
    print(f"val_loss(meta): {meta.get('val_loss')}")
    print(f"PASSED: {report['passed']}")
    if report["failures"]:
        print("Failures:", ", ".join(report["failures"]))
    print(f"Report: {out_path}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
