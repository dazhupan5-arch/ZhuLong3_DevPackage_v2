#!/usr/bin/env python3
"""V17 DirectionScorer 验收。"""

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
    parser.add_argument("--meta", default="models/direction_scorer/meta.json")
    parser.add_argument("--acceptance", default="config/v17_acceptance.json")
    args = parser.parse_args()

    meta_path = _ROOT / args.meta
    acc_path = _ROOT / args.acceptance
    if not meta_path.is_file():
        print(f"缺少 {meta_path}")
        return 1
    acc = json.loads(acc_path.read_text(encoding="utf-8-sig"))
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    ds_acc = acc.get("direction_scorer") or {}
    report = meta.get("threshold_report") or {}
    failures: list[str] = []

    for key, spec in ds_acc.items():
        if not key.startswith("threshold_"):
            continue
        thr = key.replace("threshold_", "")
        rec = report.get(thr) or report.get(str(float(thr)))
        if not rec:
            failures.append(f"missing_threshold_report_{thr}")
            continue
        if float(rec.get("direction_accuracy", 0)) < float(spec.get("min_direction_accuracy", 0)):
            failures.append(f"direction_accuracy_below_{thr}")
        if float(rec.get("coverage", 0)) < float(spec.get("min_coverage", 0)):
            failures.append(f"coverage_below_{thr}")

    passed = len(failures) == 0
    meta["passed"] = passed
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("PASS" if passed else f"FAIL: {failures}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
