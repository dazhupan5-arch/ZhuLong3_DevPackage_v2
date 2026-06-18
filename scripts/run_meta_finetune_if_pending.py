#!/usr/bin/env python3
"""若 meta_learning/finetune_pending.json 存在则运行 weekly_finetune。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.utils.paths import resolve_writable_data_path


def main() -> int:
    pending = resolve_writable_data_path("meta_learning/finetune_pending.json")
    if not pending.is_file():
        print("无 pending finetune")
        return 0
    blob = json.loads(pending.read_text(encoding="utf-8"))
    sym = str(blob.get("symbol", "XAUUSD"))
    steps = int(blob.get("timesteps", 5000))
    script = _ROOT / "scripts" / "weekly_finetune.py"
    rc = subprocess.call([sys.executable, str(script), "--symbol", sym, "--timesteps", str(steps)], cwd=str(_ROOT))
    if rc == 0:
        pending.unlink(missing_ok=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
