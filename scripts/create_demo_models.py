#!/usr/bin/env python3
"""生成演示用最小模型包，供 validate_models 与 smoke 推理通过。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zhulong.inference_engine import TransformerEncoder  # noqa: E402

FEATURE_DIM = 30
SEQ_LEN = 60
HOURLY = 10
MACRO = 8
EMB = 32
FUSED = EMB + HOURLY + MACRO

DEFAULT_SYMBOLS = ("XAUUSD", "USOIL")


def write_symbol(symbol: str, models_root: Path) -> None:
    d = models_root / symbol
    d.mkdir(parents=True, exist_ok=True)

    encoder = TransformerEncoder(feature_dim=FEATURE_DIM)
    torch.save(encoder.state_dict(), d / "transformer_encoder.pth")

    rng = np.random.default_rng(42)
    flat = rng.normal(size=(200, FEATURE_DIM)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(flat)
    joblib.dump(scaler, d / "scaler.pkl")

    x = rng.normal(size=(120, FUSED)).astype(np.float32)
    y_cls = rng.integers(0, 3, size=120)
    y_reg = rng.normal(0, 0.1, size=120).astype(np.float32)

    clf = xgb.XGBClassifier(
        n_estimators=8, max_depth=3, objective="multi:softprob", num_class=3, verbosity=0
    )
    clf.fit(x, y_cls)
    clf.save_model(str(d / "xgb_classifier.json"))

    reg = xgb.XGBRegressor(n_estimators=8, max_depth=3, verbosity=0)
    reg.fit(x, y_reg)
    reg.save_model(str(d / "xgb_regressor.json"))

    manifest = {
        "symbol": symbol,
        "kind": "demo",
        "feature_dim": FEATURE_DIM,
        "seq_len": SEQ_LEN,
        "note": "原油占位，正式模型待 deploy" if symbol == "USOIL" else "演示占位模型，非实盘策略",
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"OK {d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write demo model placeholders")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Symbols to write (default: XAUUSD USOIL)",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="models root (default: repo models/)",
    )
    args = parser.parse_args()
    models_root = Path(args.output_dir) if args.output_dir else ROOT / "models"
    for sym in args.symbols:
        write_symbol(sym.upper(), models_root)
    print(f"演示模型已写入 {models_root}")


if __name__ == "__main__":
    main()
