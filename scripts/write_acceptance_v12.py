#!/usr/bin/env python3
"""v12 验收报告 Markdown。"""

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
    rd = root / "data" / "training" / "reports" / "v12" / args.symbol
    rep = json.loads((rd / "acceptance_report_v12.json").read_text(encoding="utf-8"))
    val = rep["metrics"].get("validation", {})
    test1 = rep["metrics"].get("test1", {})
    params = rep["metrics"].get("params", {})

    lines = [
        "# 烛龙 v12 验收报告 — 双向不对称后处理",
        "",
        f"**结论**: {'通过' if rep['passed'] else '未通过'}",
        "",
        "## 参数（不重训 v11 模型）",
        f"- 做多阈值: {params.get('long_threshold', 0.84)}",
        f"- 做空阈值: {params.get('short_threshold', 0.88)}",
        f"- 模型: {params.get('model_version', 'v11')}（v12 规则 + v11 三分类）",
        f"- 做空趋势: H1 EMA20<EMA50 且 H1 RSI<50",
        f"- 冷却: 多 90min / 空 120min",
        f"- 止损: 多 1.2×ATR / 空 1.0×ATR",
        "",
        "## 验证集",
        f"- 加权精确率: {val.get('precision', 0):.3f}",
        f"- 做多精确率: {val.get('long_precision', 0):.3f}",
        f"- 做空精确率: {val.get('short_precision', 0):.3f}",
        "",
        "## 样本外 test1",
        f"- 总胜率: {test1.get('win_rate', 0):.3f}",
        f"- 做多胜率: {test1.get('long_win_rate', 0):.3f}",
        f"- 做空胜率: {test1.get('short_win_rate', 0):.3f}",
        f"- 盈亏比: {test1.get('avg_rr', 0):.3f}",
        f"- 交易数: {test1.get('n_trades', 0)}",
        f"- 最大回撤: {test1.get('max_drawdown', 0):.3f}",
        "",
        "## vs v11",
        "| 指标 | v11 | v12 |",
        "|------|-----|-----|",
        "| 总胜率 | 53.4% | {:.1f}% |".format(100 * test1.get("win_rate", 0)),
        "| 做空胜率 | 47.6% | {:.1f}% |".format(100 * test1.get("short_win_rate", 0)),
        "",
        "## 失败项",
    ]
    lines.extend(f"- {f}" for f in rep.get("failures", [])) or lines.append("- 无")
    (rd / "acceptance_report_v12.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {rd / 'acceptance_report_v12.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
