#!/usr/bin/env python3
"""部署前硬门禁：验收报告 + meta + 产物完整性（禁止未验收模型上实机）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.agent.training_utils import PIPELINE_CONTRACT_VERSION, TRAIN_END_DEFAULT

REQUIRED_ARTIFACTS = (
    "models/horizon_v16.onnx",
    "models/horizon_v16_scaler.pkl",
    "models/horizon_v16.meta.json",
    "models/horizon_v16.pth",
    "models/rl_agent_xau.zip",
    "data/agent_state_scaler_xauusd.json",
)

KN2_ARTIFACTS = (
    "models/kn2_trader_v16.pth",
    "models/kn2_trader_v16.meta.json",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fail(failures: list[str], msg: str) -> None:
    failures.append(msg)
    print(f"  [FAIL] {msg}")


def _pass(msg: str) -> None:
    print(f"  [PASS] {msg}")


def check_horizon_meta(root: Path, failures: list[str]) -> None:
    meta_path = root / "models" / "horizon_v16.meta.json"
    if not meta_path.is_file():
        _fail(failures, "missing_horizon_meta")
        return
    meta = _load_json(meta_path)
    if not meta.get("passed"):
        _fail(failures, "horizon_meta_passed_false")
    else:
        _pass("horizon meta passed=true")
    if meta.get("temporal_val") is not True:
        _fail(failures, "horizon_meta_temporal_val_not_true")
    else:
        _pass("horizon temporal_val=true")
    if meta.get("pipeline_contract") != PIPELINE_CONTRACT_VERSION:
        _fail(
            failures,
            f"horizon_pipeline_contract_{meta.get('pipeline_contract')}_need_{PIPELINE_CONTRACT_VERSION}",
        )
    else:
        _pass(f"horizon pipeline_contract={PIPELINE_CONTRACT_VERSION}")
    train_end = str(meta.get("train_end", ""))[:10]
    if train_end != TRAIN_END_DEFAULT[:10]:
        _fail(failures, f"horizon_train_end_{train_end}_need_{TRAIN_END_DEFAULT[:10]}")
    else:
        _pass(f"horizon train_end={train_end}")
    f1 = float(meta.get("macro_f1", 0))
    if f1 <= 0.50:
        _fail(failures, f"horizon_macro_f1_{f1:.4f}_lte_0.50")
    else:
        _pass(f"horizon macro_f1={f1:.4f}")


def check_acceptance_report(root: Path, failures: list[str]) -> None:
    report_path = root / "data" / "training" / "reports" / "v16" / "acceptance_report.json"
    if not report_path.is_file():
        _fail(failures, "missing_horizon_acceptance_report")
        return
    report = _load_json(report_path)
    if not report.get("passed"):
        _fail(failures, f"horizon_acceptance_failed:{report.get('failures')}")
        return
    _pass("horizon acceptance_report passed")
    sections = report.get("sections") or {}
    val_cls = sections.get("val_classification") or {}
    if not val_cls.get("ok"):
        _fail(failures, "horizon_val_classification_not_ok")
    else:
        detail = val_cls.get("detail") or {}
        metrics = detail.get("val_classification") or {}
        _pass(
            f"val_classification macro_f1={metrics.get('macro_f1')} "
            f"n={metrics.get('n_samples')}"
        )


def check_kn2_if_live(root: Path, cfg: dict, failures: list[str]) -> None:
    kn2 = cfg.get("kn2") or {}
    if not kn2.get("enabled") or kn2.get("shadow_mode"):
        _pass("kn2 shadow/disabled — skip kn2 live gate")
        return
    report_path = root / "data" / "training" / "reports" / "kn2_v16" / "acceptance_report.json"
    if not report_path.is_file():
        _fail(failures, "kn2_live_requires_acceptance_report")
        return
    report = _load_json(report_path)
    if not report.get("passed"):
        _fail(failures, f"kn2_acceptance_failed:{report.get('failures')}")
        return
    _pass("kn2 acceptance_report passed (LIVE)")
    for rel in KN2_ARTIFACTS:
        if not (root / rel).is_file():
            alt = root / rel.replace("models/", "data/")
            if not alt.is_file():
                _fail(failures, f"missing_kn2_artifact:{rel}")


def check_artifacts(root: Path, failures: list[str], require_kn2: bool) -> None:
    for rel in REQUIRED_ARTIFACTS:
        if not (root / rel).is_file():
            _fail(failures, f"missing_artifact:{rel}")
        else:
            _pass(f"artifact {rel}")
    if require_kn2:
        for rel in KN2_ARTIFACTS:
            p = root / rel
            if not p.is_file():
                p = root / rel.replace("models/", "data/")
            if not p.is_file():
                _fail(failures, f"missing_kn2_artifact:{rel}")
            else:
                _pass(f"artifact {rel}")


def main() -> int:
    parser = argparse.ArgumentParser(description="V16 deploy gate — block unaccepted models")
    parser.add_argument("--root", default=str(_ROOT))
    parser.add_argument("--require-kn2-live", action="store_true", help="KN2 必须验收通过且 enabled")
    args = parser.parse_args()

    root = Path(args.root)
    cfg_path = root / "config" / "config_agent.json"
    cfg = _load_json(cfg_path) if cfg_path.is_file() else {}
    kn2_live = args.require_kn2_live or (
        bool((cfg.get("kn2") or {}).get("enabled")) and not bool((cfg.get("kn2") or {}).get("shadow_mode"))
    )

    print("=== V16 Pre-Deploy Gate ===")
    failures: list[str] = []
    check_horizon_meta(root, failures)
    check_acceptance_report(root, failures)
    check_kn2_if_live(root, cfg, failures)
    check_artifacts(root, failures, require_kn2=kn2_live)

    print("=" * 60)
    if failures:
        print(f"DEPLOY BLOCKED ({len(failures)} failures)")
        for f in failures:
            print(f"  - {f}")
        return 2
    print("DEPLOY GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
