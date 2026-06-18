#!/usr/bin/env python3
"""V16 训练无泄露契约审计（数据切分 / 模型 meta / NPZ 标记）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.training_utils import (
    PIPELINE_CONTRACT_VERSION,
    TRAIN_END_DEFAULT,
    VAL_YEAR_DEFAULT,
    load_npz,
    require_temporal_horizon_model,
    temporal_train_val_masks,
)

CHECKS: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def _read_meta(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def audit_horizon(root: Path, train_end: str) -> None:
    meta = _read_meta(root / "models" / "horizon_v16.meta.json")
    chk("horizon meta exists", bool(meta), str(root / "models" / "horizon_v16.meta.json"))
    chk("horizon temporal_val", meta.get("temporal_val") is True, str(meta.get("temporal_val")))
    chk(
        "horizon train_end",
        str(meta.get("train_end", ""))[:10] == train_end[:10],
        str(meta.get("train_end")),
    )
    chk(
        "horizon pipeline_contract",
        meta.get("pipeline_contract") == PIPELINE_CONTRACT_VERSION,
        str(meta.get("pipeline_contract")),
    )
    try:
        require_temporal_horizon_model(root / "models" / "horizon_v16.onnx", train_end=train_end)
        chk("horizon require_temporal_horizon_model", True)
    except Exception as ex:
        chk("horizon require_temporal_horizon_model", False, str(ex)[:80])


def audit_kn2(root: Path, train_end: str, val_year: int) -> None:
    meta = _read_meta(root / "models" / "kn2_trader_v16.meta.json")
    report = _read_meta(root / "data" / "training" / "reports" / "kn2_v16" / "train_report.json")
    chk("kn2 meta/report", bool(meta) or bool(report))
    src = report or meta
    chk("kn2 temporal_val", src.get("temporal_val") is True, str(src.get("temporal_val")))
    chk("kn2 val_year OOS", int(src.get("val_year", 0)) == val_year, str(src.get("val_year")))
    chk(
        "kn2 pipeline_contract",
        src.get("pipeline_contract") == PIPELINE_CONTRACT_VERSION,
        str(src.get("pipeline_contract")),
    )

    npz_path = root / "data" / "clean" / "kn2_training_v16_location.npz"
    if npz_path.is_file():
        data = load_npz(npz_path)
        chk(
            "kn2 NPZ pipeline_contract",
            str(data.get("pipeline_contract", [""])[0]) == PIPELINE_CONTRACT_VERSION,
        )
        chk(
            "kn2 NPZ horizon_temporal_val",
            bool(data.get("horizon_temporal_val", [False])[0]),
        )
    else:
        chk("kn2 location NPZ", False, "missing")


def audit_rl(root: Path, train_through: int) -> None:
    meta_paths = [
        root / "models" / "XAUUSD" / "v16" / "rl_meta.json",
        root / "logs" / "training" / "rl_XAUUSD.json",
    ]
    meta = {}
    for p in meta_paths:
        if p.is_file():
            meta = _read_meta(p)
            break
    chk("rl meta", bool(meta))
    chk(
        "rl pipeline_contract",
        meta.get("pipeline_contract") == PIPELINE_CONTRACT_VERSION,
        str(meta.get("pipeline_contract")),
    )
    chk(
        "rl train_through_year",
        int(meta.get("train_through_year", 0)) == train_through,
        str(meta.get("train_through_year")),
    )


def audit_npz_splits(root: Path, train_end: str, val_year: int) -> None:
    import pandas as pd

    for rel in (
        "data/clean/training_horizon_v16_location.npz",
        "data/clean/kn2_training_v16_location.npz",
    ):
        p = root / rel
        if not p.is_file():
            chk(f"npz split {rel}", False, "missing")
            continue
        data = load_npz(p)
        if "time" not in data:
            chk(f"npz split {rel}", False, "no time")
            continue
        train_m, val_m = temporal_train_val_masks(data["time"], train_end=train_end, val_year=val_year)
        overlap = int((train_m & val_m).sum())
        chk(f"npz no overlap {p.name}", overlap == 0, f"overlap={overlap}")
        chk(f"npz val year {p.name}", int(val_m.sum()) > 500, f"val_bars={int(val_m.sum())}")


def audit_code_guards(root: Path) -> None:
    th = (root / "scripts" / "train_horizon_v16.py").read_text(encoding="utf-8")
    chk("train_horizon forbids no-temporal-val", "no-temporal-val 已禁用" in th)
    kn = (root / "scripts" / "train_kn2_v16.py").read_text(encoding="utf-8")
    chk("train_kn2 temporal_train_val_masks", "temporal_train_val_masks" in kn)
    tu = (root / "zhulong" / "agent" / "training_utils.py").read_text(encoding="utf-8")
    chk("clean_m5 causal bad_tick", "bad_tick_revert_causal" in tu)
    chk("no forward shift(-1) bad_tick", "shift(-1)" not in tu.split("bad_tick")[1][:400] if "bad_tick" in tu else False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--train-end", default=TRAIN_END_DEFAULT)
    parser.add_argument("--val-year", type=int, default=VAL_YEAR_DEFAULT)
    parser.add_argument("--train-through-year", type=int, default=2024)
    parser.add_argument("--pre", action="store_true", help="仅审计代码守卫（训练前）")
    parser.add_argument("--post", action="store_true", help="审计产物 meta/NPZ（训练后）")
    args = parser.parse_args()

    root = Path(args.root)
    print("=" * 60)
    print("V16 TRAINING NO-LEAK CONTRACT AUDIT")
    print(f"contract={PIPELINE_CONTRACT_VERSION} train_end={args.train_end} val_year={args.val_year}")
    print("=" * 60)

    print("\n--- code guards ---")
    audit_code_guards(root)

    if not args.pre:
        print("\n--- horizon ---")
        audit_horizon(root, args.train_end)
        print("\n--- kn2 ---")
        audit_kn2(root, args.train_end, args.val_year)
        print("\n--- rl ---")
        audit_rl(root, args.train_through_year)
        print("\n--- npz temporal splits ---")
        audit_npz_splits(root, args.train_end, args.val_year)

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    total = len(CHECKS)
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{total} PASS")
    fails = [n for n, ok, _ in CHECKS if not ok]
    if fails:
        print("FAIL:", ", ".join(fails))
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
