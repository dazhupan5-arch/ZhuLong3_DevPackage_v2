"""V16 分类验收指标（macro F1 + long/short 精确率/召回率）。"""
from __future__ import annotations

from typing import Any

import numpy as np


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: tuple[str, ...] = ("short", "flat", "long"),
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(class_names)
    per_class: dict[str, dict[str, float | int]] = {}
    supports = []
    precisions = []
    recalls = []
    f1s = []
    for c, name in enumerate(class_names):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        pred_n = int((y_pred == c).sum())
        true_n = int((y_true == c).sum())
        prec = tp / pred_n if pred_n else 0.0
        rec = tp / true_n if true_n else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": true_n,
            "pred_n": pred_n,
        }
        supports.append(true_n)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    # macro（仅含 support>0 的类）
    active = [i for i, s in enumerate(supports) if s > 0]
    macro_f1 = float(np.mean([f1s[i] for i in active])) if active else 0.0
    macro_prec = float(np.mean([precisions[i] for i in active])) if active else 0.0
    macro_rec = float(np.mean([recalls[i] for i in active])) if active else 0.0
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": per_class,
        "n_samples": int(len(y_true)),
    }


def apply_classification_thresholds(
    metrics: dict[str, Any],
    acc: dict[str, Any],
    failures: list[str],
    *,
    prefix: str = "",
) -> None:
    """F1>0.5；long/short 精确率、召回率均 >=80%（可配置）。"""
    pfx = f"{prefix}_" if prefix else ""
    min_f1 = float(acc.get("min_val_macro_f1", 0.50))
    min_prec = float(acc.get("min_class_precision", 0.80))
    min_rec = float(acc.get("min_class_recall", 0.80))
    strict = acc.get("strict_classes") or ["long", "short"]

    macro_f1 = float(metrics.get("macro_f1", 0))
    if macro_f1 <= min_f1:
        failures.append(f"{pfx}macro_f1_{macro_f1:.4f}_lte_{min_f1}")

    per = metrics.get("per_class") or {}
    for cls in strict:
        pc = per.get(cls) or {}
        prec = float(pc.get("precision", 0))
        rec = float(pc.get("recall", 0))
        if prec < min_prec:
            failures.append(f"{pfx}{cls}_precision_{prec:.4f}_lt_{min_prec}")
        if rec < min_rec:
            failures.append(f"{pfx}{cls}_recall_{rec:.4f}_lt_{min_rec}")
