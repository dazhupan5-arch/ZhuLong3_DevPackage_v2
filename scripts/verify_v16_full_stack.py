#!/usr/bin/env python3
"""V16 full-stack verification: Horizon passed + gates + agent_validate + KN2 LIVE."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from v16_cli_runner import APPDATA, run_cli  # noqa: E402

CFG = APPDATA / "config_agent.json"


def _chk(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=== V16 Full Stack Verification ===\n")
    ok_all = True

    meta_path = _ROOT / "models" / "horizon_v16.meta.json"
    if not meta_path.is_file():
        meta_path = APPDATA / "models" / "horizon_v16.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok_all &= _chk("horizon meta passed", bool(meta.get("passed")), f"trial={meta.get('trial')} f1={meta.get('macro_f1')}")
    else:
        ok_all &= _chk("horizon meta", False, "missing")

    if not CFG.is_file():
        ok_all &= _chk("config_agent.json", False, str(CFG))
        return 1
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    arch = (cfg.get("architecture") or {}).get("version", "")
    eg = cfg.get("execution_gates") or {}
    kn2 = cfg.get("kn2") or {}
    ok_all &= _chk("architecture=v16", arch == "v16", arch)
    ok_all &= _chk("structure_location_gate", bool(eg.get("structure_location_gate")))
    ok_all &= _chk("horizon_lock=false", not bool(eg.get("horizon_lock_direction")))
    ok_all &= _chk("kn2.enabled (LIVE)", kn2.get("enabled") is True)
    ok_all &= _chk("kn2.shadow_mode=false", kn2.get("shadow_mode") is False)

    from zhulong.agent.kn2_location_labels import replay_bar_diagnosis
    import numpy as np

    bad = np.zeros(30, dtype=np.float32)
    bad[0], bad[3], bad[4], bad[5], bad[6] = 0.05, 1.2, 0.25, 0.2, 0.5
    ok_all &= _chk("replay block chase long", replay_bar_diagnosis(bad, 0.72, "ranging")["verdict"] == "would_block_long")

    good = np.zeros(30, dtype=np.float32)
    good[0], good[3], good[4], good[5], good[6] = 0.08, 0.35, 1.8, 0.55, 0.3
    ok_all &= _chk("replay allow bottom long", replay_bar_diagnosis(good, 0.28, "ranging").get("long_candidate") is True)

    for rel in ("models/horizon_v16.onnx", "models/horizon_v16_scaler.pkl", "models/kn2_trader_v16.pth"):
        p = APPDATA / rel
        if not p.is_file():
            p = _ROOT / rel
        ok_all &= _chk(rel, p.is_file(), str(p))

    v = run_cli({"cmd": "agent_validate", "config_path": str(CFG)})
    ok_all &= _chk("agent_validate", bool(v.get("ok")), str(v.get("error", v.get("architecture", ""))))
    if v.get("ok"):
        print(f"       kn2_ready={v.get('kn2_ready')} kn2_live={v.get('kn2_live')}")

    print()
    if ok_all:
        print("=== OVERALL: PASS ===")
        return 0
    print("=== OVERALL: FAIL ===")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
