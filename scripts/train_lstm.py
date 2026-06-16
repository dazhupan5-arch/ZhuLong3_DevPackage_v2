#!/usr/bin/env python3
"""训练 v7 LSTM 二分类模型。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from zhulong.training.lstm.model import CLASS_WEIGHT, build_lstm_model, default_callbacks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/training/lstm/XAUUSD")
    parser.add_argument("--output", default="models/XAUUSD/lstm/lstm_model.keras")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lstm-units", default="64,32")
    parser.add_argument("--quick", action="store_true", help="仅 5 轮快速验证")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)
    root = _ROOT
    data_dir = root / args.data_dir
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = np.load(data_dir / "train.npz")
    val = np.load(data_dir / "val.npz")
    X_tr, y_tr = train["X"], train["y"]
    X_va, y_va = val["X"], val["y"]
    logger.info("train=%s val=%s pos_train=%.2f%%", len(y_tr), len(y_va), 100 * y_tr.mean())

    units = tuple(int(x) for x in args.lstm_units.split(","))
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    seq_len = meta["seq_len"]
    n_feat = len(meta["features"])

    model = build_lstm_model(seq_len, n_feat, lstm_units=units)  # type: ignore[arg-type]
    epochs = 5 if args.quick else args.epochs

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=epochs,
        batch_size=args.batch_size,
        class_weight=CLASS_WEIGHT,
        callbacks=default_callbacks(str(out_path)),
        verbose=1,
    )

    model.save(out_path)
    proba = model.predict(X_va, batch_size=512, verbose=0).ravel()
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "val_auc": float(roc_auc_score(y_va, proba)),
        "val_precision": float(precision_score(y_va, pred, zero_division=0)),
        "val_recall": float(recall_score(y_va, pred, zero_division=0)),
        "epochs_run": len(history.history.get("loss", [])),
        "lstm_units": list(units),
    }
    logger.info("val metrics @0.5: %s", metrics)

    report_dir = root / "data" / "training" / "reports" / "lstm" / args.symbol
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "train_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}, indent=2),
        encoding="utf-8",
    )
    (report_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history.get("loss", []), label="train")
    ax[0].plot(history.history.get("val_loss", []), label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.history.get("auc", []), label="train")
    ax[1].plot(history.history.get("val_auc", []), label="val")
    ax[1].set_title("AUC")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(report_dir / "train_curve.png", dpi=120)
    plt.close(fig)
    print(f"model -> {out_path}")
    print(f"curve -> {report_dir / 'train_curve.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
