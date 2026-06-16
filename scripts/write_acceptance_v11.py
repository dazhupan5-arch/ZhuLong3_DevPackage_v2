#!/usr/bin/env python3
"""v11 验收报告 Markdown。"""

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
    rd = root / "data" / "training" / "reports" / "v11" / args.symbol
    rep = json.loads((rd / "acceptance_report_v11.json").read_text(encoding="utf-8"))
    val = rep["metrics"].get("validation", {}) if "validation" in rep.get("metrics", {}) else rep["metrics"]
    test1 = rep["metrics"].get("test1", {})
    thr = rep["metrics"].get("thresholds", {})
    clf = (rd / "classification_report.txt").read_text(encoding="utf-8") if (rd / "classification_report.txt").is_file() else ""

    lines = [
        "# 烛龙 v11 验收报告 — 三分类 XGBoost",
        "",
        f"**结论**: {'通过' if rep['passed'] else '未通过'}",
        "",
        "## 验证集",
        f"- 加权精确率: {val.get('precision', thr.get('weighted_precision', 0)):.3f}",
        f"- 做多精确率: {thr.get('long_precision', val.get('long_precision', 0)):.3f}",
        f"- 做空精确率: {thr.get('short_precision', val.get('short_precision', 0)):.3f}",
        f"- 阈值: {thr.get('long_thr', 0):.2f}",
        "",
        "## 样本外 test1",
        f"- 胜率: {test1.get('win_rate', 0):.3f}",
        f"- 盈亏比: {test1.get('avg_rr', 0):.3f}",
        f"- 交易数: {test1.get('n_trades', 0)} (多 {test1.get('n_long', 0)} / 空 {test1.get('n_short', 0)})",
        f"- 日最大信号: {test1.get('max_daily_signals', 0)}",
        f"- 最大回撤: {test1.get('max_drawdown', 0):.3f}",
        "",
        "## 失败项",
    ]
    lines.extend(f"- {f}" for f in rep.get("failures", [])) or lines.append("- 无")
    if clf:
        lines.extend(["", "## 分类报告", "```", clf.strip(), "```"])
    (rd / "acceptance_report_v11.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rd / 'acceptance_report_v11.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
