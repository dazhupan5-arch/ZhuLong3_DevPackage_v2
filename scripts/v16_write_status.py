#!/usr/bin/env python3
"""V16 管线状态快照（供 watchdog / 人工查看）。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _read_progress() -> dict:
    p = _ROOT / "data" / "training" / "v16" / "XAUUSD" / "struct_progress.json"
    if not p.is_file():
        return {"done": 0, "total": 0, "pct": 0.0}
    d = json.loads(p.read_text(encoding="utf-8-sig"))
    total = int(d.get("total", 0))
    done = int(d.get("done", 0))
    pct = round(100.0 * done / total, 2) if total else 0.0
    return {"done": done, "total": total, "pct": pct, "meta_key": d.get("meta_key")}


def _npz_rows() -> int:
    p = _ROOT / "data" / "training_horizon_v16.npz"
    if not p.is_file():
        return 0
    try:
        import numpy as np

        return int(len(np.load(p, allow_pickle=True)["struct"]))
    except Exception:
        return 0


def _horizon_meta() -> dict:
    p = _ROOT / "models" / "horizon_v16.meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def build_status(*, pipeline_running: bool, note: str = "") -> dict:
    prog = _read_progress()
    npz_rows = _npz_rows()
    meta = _horizon_meta()
    if npz_rows >= 700000:
        stage = "prep_done"
    elif prog["total"] and prog["done"] >= prog["total"]:
        stage = "prep_writing_npz"
    elif prog["done"] > 0:
        stage = "struct_features"
    else:
        stage = "idle"

    if (_ROOT / "data" / "training" / "reports" / "v16" / "acceptance_report.json").is_file():
        try:
            rep = json.loads(
                (_ROOT / "data" / "training" / "reports" / "v16" / "acceptance_report.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            if rep.get("passed"):
                stage = "acceptance_passed"
            elif stage == "idle":
                stage = "acceptance_failed"
        except Exception:
            pass

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "struct_done": prog["done"],
        "struct_total": prog["total"],
        "struct_pct": prog["pct"],
        "npz_rows": npz_rows,
        "pipeline_running": pipeline_running,
        "horizon_val_acc": meta.get("val_accuracy"),
        "horizon_passed": meta.get("passed"),
        "note": note,
    }


def main() -> int:
    running = "--running" in sys.argv
    note = ""
    for i, a in enumerate(sys.argv):
        if a == "--note" and i + 1 < len(sys.argv):
            note = sys.argv[i + 1]
    status = build_status(pipeline_running=running, note=note)
    out = _ROOT / "logs" / "v16_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
