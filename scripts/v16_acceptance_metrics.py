"""V16 分类验收指标（macro F1 + long/short 精确率/召回率 + 训练/测试双门槛）。"""
from __future__ import annotations

from typing import Any

import numpy as np


def load_f1_floor(acc: dict[str, Any]) -> float:
    """macro F1 下限（训练集与测试集均须严格大于该值）。"""
    for key in ("min_macro_f1", "min_test_macro_f1", "min_train_macro_f1", "min_val_macro_f1"):
        if key in acc:
            return float(acc[key])
    return 0.50


def load_win_rate_floor(acc: dict[str, Any]) -> float:
    return float(acc.get("min_win_rate", acc.get("min_oos_win_rate", 0.60)))


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
    """测试集 long/short 精确率、召回率门槛（默认 >=80%）。"""
    pfx = f"{prefix}_" if prefix else ""
    min_prec = float(acc.get("min_class_precision", 0.80))
    min_rec = float(acc.get("min_class_recall", 0.80))
    strict = acc.get("strict_classes") or ["long", "short"]

    per = metrics.get("per_class") or {}
    for cls in strict:
        pc = per.get(cls) or {}
        prec = float(pc.get("precision", 0))
        rec = float(pc.get("recall", 0))
        if prec < min_prec:
            failures.append(f"{pfx}{cls}_precision_{prec:.4f}_lt_{min_prec}")
        if rec < min_rec:
            failures.append(f"{pfx}{cls}_recall_{rec:.4f}_lt_{min_rec}")


def apply_train_test_f1_gates(
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    acc: dict[str, Any],
    failures: list[str],
    *,
    prefix: str = "",
) -> None:
    """训练集与测试集 macro F1 均须 > min_macro_f1；禁止 train 高 test 低（gap 超限）。"""
    pfx = f"{prefix}_" if prefix else ""
    min_train = float(acc.get("min_train_macro_f1", acc.get("min_macro_f1", 0.50)))
    min_test = float(acc.get("min_test_macro_f1", acc.get("min_macro_f1", 0.50)))
    max_gap = float(acc.get("max_train_test_f1_gap", 0.10))

    train_f1 = float(train_metrics.get("macro_f1", 0))
    test_f1 = float(test_metrics.get("macro_f1", 0))

    if train_f1 <= min_train:
        failures.append(f"{pfx}train_macro_f1_{train_f1:.4f}_lte_{min_train}")
    if test_f1 <= min_test:
        failures.append(f"{pfx}test_macro_f1_{test_f1:.4f}_lte_{min_test}")

    gap = train_f1 - test_f1
    if gap > max_gap:
        failures.append(
            f"{pfx}train_test_f1_gap_{gap:.4f}_gt_{max_gap}_train_high_test_low"
        )


def check_win_rate(
    win_rate: float,
    acc: dict[str, Any],
    failures: list[str],
    *,
    label: str,
    override_min: float | None = None,
) -> None:
    """胜率须严格大于 min_win_rate（默认 >60%）。"""
    floor = float(override_min if override_min is not None else load_win_rate_floor(acc))
    if float(win_rate) <= floor:
        failures.append(f"{label}_win_rate_{float(win_rate):.4f}_lte_{floor}")
