#!/usr/bin/env python3
"""USOIL v1 验收报告 Markdown。"""

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
    parser.add_argument("--symbol", default="USOIL")
    args = parser.parse_args()

    root = _ROOT
    rd = root / "data" / "training" / "reports" / "oil_v1" / args.symbol
    rep = json.loads((rd / "acceptance_report_oil_v1.json").read_text(encoding="utf-8"))
    val = rep["metrics"].get("validation", {})
    test1 = rep["metrics"].get("test1", {})
    thr = rep["metrics"].get("thresholds", {})
    params = rep["metrics"].get("best_xgb_params", {})

    cfg_path = root / "models" / args.symbol / "v1" / "config_oil_v1.json"
    cfg = {}
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    lines = [
        "# 烛龙原油（USOIL）v1 验收报告",
        "",
        f"**结论**: {'✅ 通过' if rep['passed'] else '❌ 未通过'}",
        "",
        "## 训练配置",
        "- 预测窗口: 90 分钟（18 根 M5）",
        "- 标签阈值: 动态 0.8×ATR(14)，下限 0.30%",
        "- 模型: XGBoost 三分类 (multi:softprob)",
        f"- 特征维度: {len(json.loads((root / 'data/training/oil_v1/USOIL/feature_columns.json').read_text())) if (root / 'data/training/oil_v1/USOIL/feature_columns.json').is_file() else 'N/A'}",
        "",
        "## 后处理参数",
        f"- 做多阈值: {thr.get('long_thr', cfg.get('long_threshold', 0.82))}",
        f"- 做空阈值: {thr.get('short_thr', cfg.get('short_threshold', 0.78))}",
        f"- 止损: 多 {cfg.get('long_sl_atr', 1.5)}×ATR / 空 {cfg.get('short_sl_atr', 1.2)}×ATR",
        f"- 止盈: {cfg.get('tp_atr', 2.5)}×ATR",
        f"- 冷却: {cfg.get('cooldown_bars', 18)} 根 M5（90 分钟）",
        "- EIA 屏蔽: 公布前 30min ~ 后 15min",
        "- 趋势过滤: 仅极端 H1 趋势屏蔽逆向信号",
        "",
        "## 验收标准 vs 实测",
        "| 指标 | 目标 | 实测 |",
        "|------|------|------|",
        f"| 验证加权精确率 | ≥ 50% | {100*val.get('precision', 0):.1f}% |",
        f"| 样本外总胜率 | ≥ 52% | {100*test1.get('win_rate', 0):.1f}% |",
        f"| 做多胜率 | — | {100*test1.get('long_win_rate', 0):.1f}% |",
        f"| 做空胜率 | ≥ 48% | {100*test1.get('short_win_rate', 0):.1f}% |",
        f"| 盈亏比 | ≥ 1.4 | {test1.get('avg_rr', 0):.3f} |",
        f"| 日信号峰值 | ≤ 8 | {test1.get('max_daily_signals', 0)} |",
        f"| 最大回撤 | ≤ 35% | {100*test1.get('max_drawdown', 0):.1f}% |",
        f"| 交易数 | — | {test1.get('n_trades', 0)} |",
        "",
        "## 验证集",
        f"- 加权精确率: {val.get('precision', 0):.3f}",
        f"- 做多精确率: {val.get('long_precision', 0):.3f}",
        f"- 做空精确率: {val.get('short_precision', 0):.3f}",
        f"- 日信号: {val.get('signals_per_day', 0):.1f}",
        "",
        "## 样本外 test1 (2025-H1)",
        f"- 总胜率: {test1.get('win_rate', 0):.3f}",
        f"- 期望收益 (R): {test1.get('expectancy', 0):.3f}",
        f"- 累计 R: {test1.get('total_pnl_r', 0):.1f}",
        "",
        "## XGBoost 最优超参",
    ]
    for k, v in params.items():
        lines.append(f"- {k}: {v}")

    lines.extend(["", "## 失败项"])
    failures = rep.get("failures", [])
    if failures:
        lines.extend(f"- {f}" for f in failures)
    else:
        lines.append("- 无")

    lines.extend([
        "",
        "## 交付物",
        f"- 模型: `models/{args.symbol}/v1/xgb_triple_oil.json`",
        f"- 配置: `models/{args.symbol}/v1/config_oil_v1.json`",
        f"- 元数据: `models/{args.symbol}/v1/oil_v1_meta.pkl`",
        "",
        "## 部署说明",
        "1. 加载 `xgb_triple_oil.json` + `oil_v1_meta.pkl` 中的 feature_columns 与阈值",
        "2. 实时特征需包含库存/波动率/季节性列（与训练一致）",
        "3. EIA 公布时段强制屏蔽信号",
        "4. 与黄金 v12 不同配置，勿混用 config_v12.json",
    ])

    out = rd / "acceptance_report_oil_v1.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
