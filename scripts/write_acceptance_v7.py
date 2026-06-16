#!/usr/bin/env python3
"""生成 v7 LSTM 验收报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.acceptance import evaluate_lgb_acceptance, get_thresholds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    args = parser.parse_args()

    root = _ROOT
    report_dir = root / "data" / "training" / "reports" / "lstm" / args.symbol
    tune = json.loads((report_dir / "threshold_tune_v7.json").read_text(encoding="utf-8"))
    train_m = json.loads((report_dir / "train_metrics.json").read_text(encoding="utf-8"))
    bt = json.loads((report_dir / "backtest_test_v7.json").read_text(encoding="utf-8"))

    val_m = {
        "precision": tune["best"]["precision"],
        "recall": tune["best"]["recall"],
        "f1": tune["best"].get("f1", 0),
        "n_signals": tune["best"]["n_signals"],
        "signals_per_day": tune["best"]["signals_per_day"],
        "auc": train_m["val_auc"],
    }
    test1_m = bt["backtest"]
    report = evaluate_lgb_acceptance(val_m, test1_m, {}, stage="v7")

    out_json = report_dir / "acceptance_report_v7.json"
    out_json.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 烛龙 v7 验收报告 — LSTM 端到端",
        "",
        f"**结论**: {'通过' if report.passed else '未通过'}",
        "",
        "## 验证集",
        f"- AUC: {train_m['val_auc']:.4f}",
        f"- 精确率 @选中阈值: {val_m['precision']:.3f}",
        f"- 召回率: {val_m['recall']:.3f}",
        f"- 阈值: {tune['best']['threshold']:.2f}",
        "",
        "## 样本外 test",
        f"- 胜率: {test1_m.get('win_rate', 0):.3f}",
        f"- 盈亏比: {test1_m.get('avg_rr', 0):.3f}",
        f"- 交易数: {test1_m.get('n_trades', 0)}",
        f"- 最大回撤: {test1_m.get('max_drawdown', 0):.3f}",
        "",
        "## 失败项",
    ]
    lines.extend(f"- {f}" for f in report.failures) if report.failures else lines.append("- 无")
    (report_dir / "acceptance_report_v7.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"passed={report.passed} failures={report.failures}")
    print(f"report -> {report_dir / 'acceptance_report_v7.md'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
