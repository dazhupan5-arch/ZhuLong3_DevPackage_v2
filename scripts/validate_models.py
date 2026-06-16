#!/usr/bin/env python3
"""校验 models/ 下各品种四件套是否可被 InferenceEngine 加载。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INFERENCE = ROOT / "ZhuLong.PythonEngine" / "inference.py"
spec = importlib.util.spec_from_file_location("zhulong_inference", INFERENCE)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

SYMBOLS = ("XAUUSD", "USOIL")
REQUIRED = (
    "transformer_encoder.pth",
    "scaler.pkl",
    "xgb_classifier.json",
    "xgb_regressor.json",
    "manifest.json",
)


def main() -> int:
    missing_files: list[str] = []
    for sym in SYMBOLS:
        d = ROOT / "models" / sym
        for name in REQUIRED:
            p = d / name
            if not p.is_file() or p.stat().st_size == 0:
                missing_files.append(str(p))

    if missing_files:
        print("缺少模型文件:")
        for p in missing_files:
            print(" ", p)
        return 1

    result = mod.validate_models(list(SYMBOLS))
    if not result.get("ok"):
        print("InferenceEngine 校验失败, missing:", result.get("missing"))
        return 1

    print("OK validate_models:", ", ".join(SYMBOLS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
