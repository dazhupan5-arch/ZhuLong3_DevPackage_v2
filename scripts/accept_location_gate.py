#!/usr/bin/env python3
"""V17 LocationGate 验收。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="models/location_gate/meta.json")
    parser.add_argument("--acceptance", default="config/v17_acceptance.json")
    args = parser.parse_args()

    meta_path = _ROOT / args.meta
    acc_path = _ROOT / args.acceptance
    if not meta_path.is_file():
        print(f"缺少 {meta_path}")
        return 1
    acc = json.loads(acc_path.read_text(encoding="utf-8-sig"))
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    lg_acc = acc.get("location_gate") or {}
    report = meta.get("threshold_report") or {}
    failures: list[str] = []

    if float(meta.get("val_auc", 0)) < float(lg_acc.get("min_val_auc", 0.58)):
        failures.append("val_auc_below_threshold")

    spec = lg_acc.get("at_threshold_60") or {}
    rec = report.get("0.6") or report.get("0.60")
    if rec:
        if float(rec.get("precision", 0)) < float(spec.get("min_precision", 0.60)):
            failures.append("precision_below_60")
        if float(rec.get("coverage", 0)) < float(spec.get("min_coverage", 0.25)):
            failures.append("coverage_below_60")
    else:
        failures.append("missing_threshold_60_report")

    passed = len(failures) == 0
    meta["passed"] = passed
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("PASS" if passed else f"FAIL: {failures}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
