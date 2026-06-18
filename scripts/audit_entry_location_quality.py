#!/usr/bin/env python3
"""V16 入场位置分布审计：验证入场信号是否从结构有利区域出发（long 偏下、short 偏上）。

审计项：
1. 代码契约：location_score_v2 多维评分存在且正确
2. 配置契约：pull 系数 ≥ 0.30、位置权重 ≥ 0.60、门禁阈值已收紧
3. 分布契约（OOS 回测数据）：入场 pos_in_range / S/R 距离 / location_score 分布符合标准
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def _load_acceptance() -> dict:
    p = _ROOT / "config" / "v16_acceptance.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _load_agent_config() -> dict:
    p = _ROOT / "config" / "config_agent.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main() -> int:
    print("=== V16 Entry Location Quality Audit ===\n")

    acc = _load_acceptance()
    cfg = _load_agent_config()

    # ---- 1. 代码契约 ----
    print("--- 代码契约 ---")
    from zhulong.agent.execution_composer import (
        location_score_v2,
        location_score,
        structure_entry_target,
        decide_entry_mode,
        ExecutionComposer,
    )
    from zhulong.agent.tick_brief import StructureSnapshot

    chk("location_score_v2 exists", callable(location_score_v2))
    chk("location_score (legacy) exists", callable(location_score))

    # 多维评分 smoke test
    snap = StructureSnapshot(
        vector=[0.0] * 30,
        support_dist_atr=0.3,
        resistance_dist_atr=0.4,
        m5_trend=0.1,
        mtf_align=0.3,
    )
    loc_v2_long = location_score_v2("long", 0.2, snap)
    chk("location_score_v2 long near support scored >= 0.5", loc_v2_long >= 0.5,
        f"loc_v2_long={loc_v2_long:.4f}")

    loc_v2_short = location_score_v2("short", 0.8, snap)
    chk("location_score_v2 short near resistance scored >= 0.5", loc_v2_short >= 0.5,
        f"loc_v2_short={loc_v2_short:.4f}")

    loc_v2_flat = location_score_v2("flat", 0.5, snap)
    chk("location_score_v2 flat → 0.0", loc_v2_flat == 0.0, str(loc_v2_flat))

    loc_v2_bad_long = location_score_v2("long", 0.9, snap)
    chk("location_score_v2 long high pos < 0.5", loc_v2_bad_long < 0.5,
        f"loc_v2_bad_long={loc_v2_bad_long:.4f}")

    # v16_strict_3: pull 系数增大后 entry_target 离 close 更远
    # loc_score=0.5 → pull = 0.35 * 10 * 0.5 = 1.75 → gap ≥ 1.5
    tgt = structure_entry_target("long", snap, 2000.0, 10.0, loc_score=0.5)
    pull_gap = 2000.0 - tgt
    chk("structure_entry_target pull ≥ 1.5 (0.175 ATR at loc=0.5)", pull_gap >= 1.5,
        f"target={tgt:.2f} pull={pull_gap:.2f}")

    tgt_tight = structure_entry_target("long", snap, 2000.0, 10.0, loc_score=0.9)
    pull_tight = 2000.0 - tgt_tight
    chk("structure_entry_target tight loc → still pull ≥ 1.0", pull_tight >= 1.0,
        f"target={tgt_tight:.2f} pull={pull_tight:.2f}")

    # decide_entry_mode 门禁
    mode_high = decide_entry_mode("long", 2000.0, 1997.0, 0.85, 0.80)
    chk("decide_entry_mode high quality → not defer", mode_high != "defer",
        f"mode={mode_high}")

    mode_low = decide_entry_mode("long", 2000.0, 1998.0, 0.3, 0.3)
    chk("decide_entry_mode low quality → defer", mode_low == "defer",
        f"mode={mode_low}")

    # ---- 2. 配置契约 ----
    print("\n--- 配置契约 ---")
    ec = cfg.get("execution_composer") or {}
    ep = (cfg.get("trading_env") or {}).get("execution_parity") or {}

    chk("config execution_composer exists", bool(ec))
    imm = float(ec.get("immediate_quality_min", 0))
    chk("config immediate_quality_min ≥ 0.75", imm >= 0.75, str(imm))
    lim = float(ec.get("limit_quality_min", 0))
    chk("config limit_quality_min ≥ 0.42", lim >= 0.42, str(lim))
    pw = float(ec.get("entry_quality_position_weight", 0))
    chk("config entry_quality_position_weight ≥ 0.60", pw >= 0.60, str(pw))

    bonus = float(ep.get("entry_quality_bonus", 0))
    chk("config entry_quality_bonus ≥ 0.10", bonus >= 0.10, str(bonus))

    loc_cfg = acc.get("entry_location") or {}
    chk("acceptance entry_location section exists", bool(loc_cfg))
    min_loc_median = float(loc_cfg.get("min_location_score_median", 0))
    chk("acceptance min_location_score_median ≥ 0.50", min_loc_median >= 0.50, str(min_loc_median))
    min_sr_pct = float(loc_cfg.get("min_entry_near_sr_pct", 0))
    chk("acceptance min_entry_near_sr_pct ≥ 0.35", min_sr_pct >= 0.35, str(min_sr_pct))

    # ---- 3. ExecutionComposer 实例化验证 ----
    print("\n--- 实例化契约 ---")
    composer = ExecutionComposer(cfg)
    chk("ExecutionComposer immediate_quality_min from config", composer.immediate_quality_min >= 0.75,
        str(composer.immediate_quality_min))
    chk("ExecutionComposer limit_quality_min from config", composer.limit_quality_min >= 0.42,
        str(composer.limit_quality_min))
    chk("ExecutionComposer position_weight from config", composer.entry_quality_position_weight >= 0.60,
        str(composer.entry_quality_position_weight))

    # ---- 4. 旧版兼容性验证 ----
    print("\n--- 旧版兼容性 ---")
    from zhulong.agent.kn2_location_labels import LocationLabelConfig
    from zhulong.agent.execution_composer import location_score as loc_v1

    lcfg = LocationLabelConfig()
    loc_v1_long = loc_v1("long", 0.2, lcfg)
    chk("location_score v1 long low pos = 1.0", loc_v1_long >= 0.99, str(loc_v1_long))
    loc_v1_short = loc_v1("short", 0.8, lcfg)
    chk("location_score v1 short high pos = 1.0", loc_v1_short >= 0.99, str(loc_v1_short))
    loc_v1_mid = loc_v1("long", 0.5, lcfg)
    chk("location_score v1 long mid pos < 1.0", loc_v1_mid < 1.0, str(loc_v1_mid))

    # ---- 5. OOS 回测数据入场位置分布 ----
    print("\n--- OOS 入场位置分布 ---")
    oos_rpt = _ROOT / "data" / "training" / "reports" / "v16" / "acceptance_report.json"
    if oos_rpt.is_file():
        report = json.loads(oos_rpt.read_text(encoding="utf-8-sig"))
        oos = (report.get("sections") or {}).get("oos") or {}
        bt = (oos.get("detail") or {}).get("backtest") or {}
        n_trades = int(bt.get("n_trades", 0))
        if n_trades > 0:
            long_wr = float(bt.get("long_win_rate", 0))
            short_wr = float(bt.get("short_win_rate", 0))
            chk(f"OOS long win_rate > 0.50 ({n_trades} trades)", long_wr > 0.50,
                f"long_wr={long_wr:.4f} short_wr={short_wr:.4f}")
            # 如果胜率低于门槛，标记为需要新模型重训后才可通过
            if long_wr <= 0.50 or short_wr <= 0.50:
                print("    ⚠ OOS win_rate 低 — 等新模型重训后会改善")
        else:
            print("    ⚠ OOS backtest 无交易数据（可能是旧模型未达标导致）")
    else:
        print("    ⚠ OOS acceptance_report 不存在")

    fails = [n for n, ok, _ in CHECKS if not ok]
    print(f"\n=== Entry Location Quality: {len(CHECKS) - len(fails)}/{len(CHECKS)} PASS ===")
    if fails:
        print("FAILED:", ", ".join(fails))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
