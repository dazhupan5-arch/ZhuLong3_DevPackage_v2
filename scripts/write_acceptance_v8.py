#!/usr/bin/env python3
"""v8 验收报告 Markdown。"""

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
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()

    root = _ROOT
    report_dir = root / "data" / "training" / "reports" / "v8" / args.symbol
    rep = json.loads((report_dir / "acceptance_report_v8.json").read_text(encoding="utf-8"))
    val = rep["metrics"].get("validation", {})
    test1 = rep["metrics"].get("test1", {})
    thr = rep["metrics"].get("threshold", {})

    lines = [
        "# 烛龙 v8 验收报告 — 多尺度分解 + 双模型集成",
        "",
        f"**结论**: {'通过' if rep['passed'] else '未通过'}",
        "",
        "## 验证集",
        f"- 精确率: {val.get('precision', 0):.3f}",
        f"- 召回率: {val.get('recall', 0):.3f}",
        f"- AUC: {val.get('auc', 0):.4f}",
        f"- 阈值: {thr.get('threshold', 0):.2f}",
        "",
        "## 样本外 test1",
        f"- 胜率: {test1.get('win_rate', 0):.3f}",
        f"- 盈亏比: {test1.get('avg_rr', 0):.3f}",
        f"- 交易数: {test1.get('n_trades', 0)}",
        f"- 最大回撤: {test1.get('max_drawdown', 0):.3f}",
        "",
        "## 失败项",
    ]
    lines.extend(f"- {f}" for f in rep.get("failures", [])) or lines.append("- 无")
    (report_dir / "acceptance_report_v8.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {report_dir / 'acceptance_report_v8.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
