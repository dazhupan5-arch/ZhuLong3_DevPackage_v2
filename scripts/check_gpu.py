#!/usr/bin/env python3
"""检查本机 GPU / CUDA 是否可用于烛龙训练。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.utils.device import print_gpu_status

if __name__ == "__main__":
    raise SystemExit(print_gpu_status())
