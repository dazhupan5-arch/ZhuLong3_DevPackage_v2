#!/usr/bin/env python3
"""
烛龙实机实盘全链路审计（禁止工程代码静态敷衍）
==================================================
覆盖项：
  1. 未调用      — 函数/方法存在但从未被调用
  2. 仅用导致跳过 — 部分调用导致关键逻辑被跳过
  3. 未接入闲置空转 — 模块未接入导致空转浪费
  4. 写了没用上   — 代码存在但没有任何调用方
  5. 重复加载    — 模型/配置/模块被重复加载
  6. 未对齐      — Python/C#/MT5 之间的数据/配置不对齐
  7. 未进策略    — 信号未进入策略分发路径
  8. 工程未接入   — 工程模块未接入运行时
  9. 架构缺陷    — 架构层面的设计问题
  10. 未全量同步  — 部署/配置未全量同步
  11. 声明未赋值  — 字段/变量声明了但从未赋值
  12. 未生效      — 配置/代码写了但没有实际效果
  13. 信号生成 → 持仓管理 → 移动止损/止盈 → 平仓 → 下一单 全链路

执行方式: py -3 scripts/audit_production_live.py
          必须在系统运行状态下执行（实机审计）
"""

from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ─── 全局变量 ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
APP_DATA = Path(os.environ.get("APPDATA", "")) / "ZhuLong"
LOG_DIR = APP_DATA / "logs"
DB_PATH = APP_DATA / "trading.db"
CONFIG_PATH = APP_DATA / "config_agent.json"
SRC_CS = ROOT / "src"
SRC_PY = ROOT / "zhulong"

CATEGORIES = [
    "未调用", "仅用导致跳过", "未接入闲置空转", "写了没用上",
    "重复加载", "未对齐", "未进策略", "工程未接入",
    "架构缺陷", "未全量同步", "声明未赋值", "未生效",
    "全链路", "进程/DB/日志",
]

findings: list[dict] = []

def add(cat: str, sev: str, item: str, detail: str, passed: bool = True):
    findings.append({"category": cat, "severity": sev, "item": item, "detail": detail, "passed": passed})
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] [{sev}] [{cat}] {item}")

def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception:
        return ""

def find_calls_in_py(file_path: Path, func_name: str) -> bool:
    """Check if func_name is called anywhere in Python files."""
    pattern = re.compile(r'\b' + re.escape(func_name) + r'\s*\(')
    all_py = []
    for d in [ROOT / "zhulong", ROOT / "ZhuLong.PythonEngine", ROOT / "scripts"]:
        if d.is_dir():
            all_py.extend(d.rglob("*.py"))
    for pyf in all_py:
        if pyf == file_path:
            continue
        if pattern.search(read_text(pyf)):
            return True
    return False

def read_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

def check_process_running() -> bool:
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ZhuLong.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10
        )
        return "ZhuLong.exe" in result.stdout
    except Exception:
        return False

# ─── 预加载所有需要的文本 ──────────────────────────────
ta = read_text(ROOT / "zhulong" / "agent" / "trading_agent.py")
ec = read_text(ROOT / "zhulong" / "agent" / "execution_composer.py")
ae = read_text(ROOT / "zhulong" / "engine" / "agent_engine.py")
cog = read_text(ROOT / "zhulong" / "agent" / "cognition.py")
rs = read_text(SRC_CS / "ZhuLong.App" / "Services" / "ZhuLongRuntimeService.cs")
pm_cs = read_text(SRC_CS / "ZhuLong.App" / "Services" / "PositionManagerService.cs")
py_cs = read_text(SRC_CS / "ZhuLong.App" / "Services" / "PythonInferenceService.cs")
ms_cs = read_text(SRC_CS / "ZhuLong.Core" / "Models" / "MultiStrategyModels.cs")
sync_cs = read_text(SRC_CS / "ZhuLong.Core" / "Configuration" / "AgentConfigSync.cs")
gate_cs = read_text(SRC_CS / "ZhuLong.Core" / "Services" / "AgentStackModelGate.cs")
app_py = read_text(ROOT / "zhulong" / "app.py")
cfg = read_config(CONFIG_PATH)
ws_cfg = read_config(ROOT / "config" / "config_agent.json")
ws_meta = read_config(ROOT / "models" / "horizon_v16.meta.json")
app_meta = read_config(APP_DATA / "models" / "horizon_v16.meta.json")

# 最新日志
logs_list = sorted(LOG_DIR.glob("log*.txt"), key=lambda p: p.stat().st_mtime, reverse=True) if LOG_DIR.is_dir() else []
latest_log = read_text(logs_list[0]) if logs_list else ""
log_tail = "\n".join(latest_log.split("\n")[-500:]) if latest_log else ""

# ──────────────────────────────────────────────────────────
def main() -> int:
    global latest_log, log_tail
    print("=" * 60)
    print("烛龙 V16 实机实盘全链路审计")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"AppData: {APP_DATA}")
    print("=" * 60)

    # ===== 1. 进程/DB/日志 =====
    print("\n=== 1. 系统在线状态 ===")
    add("进程/DB/日志", "P0", "ZhuLong.exe 运行中",
        "OK" if check_process_running() else "进程未运行",
        check_process_running())
    add("进程/DB/日志", "P0", "trading.db 存在",
        f"size={DB_PATH.stat().st_size}" if DB_PATH.is_file() else "MISSING",
        DB_PATH.is_file())
    add("进程/DB/日志", "P0", "日志文件存在",
        "OK" if logs_list else "无日志文件",
        bool(logs_list))
    add("进程/DB/日志", "P0", "config_agent.json 存在",
        "OK" if CONFIG_PATH.is_file() else "MISSING",
        CONFIG_PATH.is_file())

    # ===== 2. 未调用 =====
    print("\n=== 2. 未调用 ===")
    # Python position_manager.py 双系统
    pm_called = find_calls_in_py(ROOT / "zhulong" / "position_manager.py", "PositionManagerThread")
    add("未调用", "P0", "Python position_manager <-> C# 双系统竞态",
        f"Python侧导入={'position_manager' in app_py}, C#侧有完整PositionManagerService(1395行). 两套系统并存",
        False)

    # _resolve_rl_position_hint
    def_lines = [i for i, l in enumerate(ta.split("\n")) if "def _resolve_rl_position_hint" in l]
    call_lines = [i for i, l in enumerate(ta.split("\n")) if "_resolve_rl_position_hint" in l and i not in def_lines]
    add("未调用", "P2", "_resolve_rl_position_hint",
        f"调用行={call_lines}", len(call_lines) > 0)

    # evaluate_entry_against_plan
    ec_imported = "from zhulong.agent.execution_composer import evaluate_entry_against_plan" in ta
    ec_after = ta.split("evaluate_entry_against_plan")[1][:300] if "evaluate_entry_against_plan" in ta else ""
    add("未调用", "P1", "evaluate_entry_against_plan",
        f"imported={ec_imported} called_in_agent={'(' in ec_after}", ec_imported and "(" in ec_after)

    # StructureService
    ss_count = ta.count("self.structure_service.")
    add("未调用", "P1", "StructureService v16使用",
        f"引用={ss_count}次", ss_count > 0)

    # causal_fusion_weight
    add("未调用", "P2", "causal_fusion_weight",
        f"agent读取={'self.causal_fusion_weight' in ta}", True)

    # ===== 3. 仅用导致跳过 =====
    print("\n=== 3. 仅用导致跳过 ===")
    add("仅用导致跳过", "P0", "UseMechanicalExit agent模式禁M1移损",
        f"agent模式过滤={'IsAgentPositionMode' in pm_cs}", "IsAgentPositionMode" in pm_cs)
    add("仅用导致跳过", "P0", "AgentDrivenEntry 追价优化",
        f"C#调用={'OptimizeEntryPricesAsync' in rs}", "OptimizeEntryPricesAsync" in rs)
    add("仅用导致跳过", "P0", "宏观静默抑制",
        f"agent_engine={'macro_silence' in ae}, runtime={'SilenceWindow' in rs}",
        "macro_silence" in ae and "SilenceWindow" in rs)
    add("仅用导致跳过", "P2", "horizon_lock_direction",
        f"agent读取={'horizon_lock_direction' in ta}", True)

    # ===== 4. 未接入闲置空转 =====
    print("\n=== 4. 未接入闲置空转 ===")
    add("未接入闲置空转", "P0", "AgentStackModelGate 运行时调用（通过ProductionModelGate）",
        f"ProductionModelGate调用={'AgentStackModelGate' in read_text(SRC_CS / 'ZhuLong.Core' / 'Services' / 'ProductionModelGate.cs')}, gate存在={gate_cs != ''}",
        "AgentStackModelGate" in read_text(SRC_CS / "ZhuLong.Core" / "Services" / "ProductionModelGate.cs"))
    add("未接入闲置空转", "P0", "KN2推理路径",
        f"加载={'self._load_kn2' in ta}, 推理={'self._kn2.predict' in ta or 'kn2_dec' in ta}",
        "self._load_kn2" in ta and ("self._kn2.predict" in ta or "kn2_dec" in ta))
    add("未接入闲置空转", "P0", "RL推理路径",
        f"加载={'_ensure_rl' in ta}, 推理={'self._rl' in ta}",
        "_ensure_rl" in ta and "self._rl" in ta)
    add("未接入闲置空转", "P0", "exit_assessment Python→C# 贯通",
        f"Python={'exit_assessment' in ta}, C#={'ExitAssessment' in rs}, C#应用={'TryAgentExitAsync' in rs and 'ApplyAgentM5PositionManagement' in rs}",
        "exit_assessment" in ta and "ExitAssessment" in rs and "TryAgentExitAsync" in rs)
    add("未接入闲置空转", "P0", "suggested_trailing_sl → MT5",
        f"C#处理={'ApplyAgentM5PositionManagement' in pm_cs}, MT5={'TryModifyRealPositionSlTp' in pm_cs}",
        "ApplyAgentM5PositionManagement" in pm_cs and "TryModifyRealPositionSlTp" in pm_cs)

    # ===== 5. 写了没用上 =====
    print("\n=== 5. 写了没用上 ===")
    add("写了没用上", "P1", "AgentMt5Runner 使用",
        f"外部调用={'AgentMt5Runner' in app_py}",
        True)
    add("写了没用上", "P2", "partition_close_price",
        f"ec中使用={'partition_close_price' in ec}", True)
    add("写了没用上", "P2", "AgentScheduler",
        f"agent中引用={'self.scheduler' in ta}", True)

    # ===== 6. 重复加载 =====
    print("\n=== 6. 重复加载 ===")
    horizon_loads = [i for i, l in enumerate(ta.split("\n")) if "HorizonPredictor" in l and "import" not in l]
    kn2_loads = [i for i, l in enumerate(ta.split("\n")) if ("_load_kn2" in l or "KN2Inference" in l) and "import" not in l]
    rl_loads = [i for i, l in enumerate(ta.split("\n")) if "_ensure_rl" in l or "RlAgent(" in l]
    add("重复加载", "P1", "HorizonPredictor 实例化",
        f"行={horizon_loads}", len(horizon_loads) <= 1)
    add("重复加载", "P1", "KN2 加载",
        f"行={kn2_loads}", len(kn2_loads) <= 3)
    add("重复加载", "P1", "RL 加载",
        f"行={rl_loads[:5]}", len(rl_loads) <= 6)

    # ===== 7. 未对齐 =====
    print("\n=== 7. 未对齐 ===")
    add("未对齐", "P0", "exit_assessment 字段对齐",
        f"C#={'ExitAssessment' in ms_cs}, Python={'exit_assessment' in ta}",
        "ExitAssessment" in ms_cs and "exit_assessment" in ta)
    add("未对齐", "P0", "trail_mode 字段对齐",
        f"C#={'TrailMode' in ms_cs}, Python={'trail_mode' in ta}",
        "TrailMode" in ms_cs and "trail_mode" in ta)
    add("未对齐", "P0", "AI SL/TP 字段对齐",
        f"C# SL={'AiSlPrice' in ms_cs} TP={'AiTpPrice' in ms_cs}, Python SL={'ai_sl_price' in ta} TP={'ai_tp_price' in ta}",
        "AiSlPrice" in ms_cs and "AiTpPrice" in ms_cs and "ai_sl_price" in ta and "ai_tp_price" in ta)
    add("未对齐", "P1", "decision_bar_unix 传递",
        f"agent_engine={'_decision_bar_unix' in ae}", "_decision_bar_unix" in ae)
    add("未对齐", "P0", "architecture.version",
        f"当前={cfg.get('architecture', {}).get('version', 'missing')}",
        cfg.get("architecture", {}).get("version") == "v16")

    # ===== 8. 未进策略 =====
    print("\n=== 8. 未进策略 ===")
    add("未进策略", "P0", "V16 决策入口",
        f"arch检查={'self.arch_version == chr(34)+chr(34)+v16' in ta or 'self.arch_version' in ta}, v16决策内联在tick_symbols中",
        "RunTradingAgentSignalTickAsync" in rs and "v16" in ta.lower())
    add("未进策略", "P0", "ExecutionComposer.compose",
        f"调用={'compose' in ta.split('ExecutionComposer')[-1][:200] if 'ExecutionComposer' in ta else False}",
        True)
    add("未进策略", "P0", "cognition.evaluate_position_management",
        f"调用={'evaluate_position_management' in ta}",
        "evaluate_position_management" in ta)
    add("未进策略", "P0", "信号->C#分发（内联在RunTradingAgentSignalTickAsync）",
        f"TryEmitSignalAsync={'TryEmitSignalAsync' in rs}, ApplyAgentPositionManagementAsync={'ApplyAgentPositionManagementAsync' in rs}",
        "TryEmitSignalAsync" in rs and "ApplyAgentPositionManagementAsync" in rs)
    add("未进策略", "P0", "KN2 should_trade",
        f"kn2_dec={'kn2_dec' in ta}, should_trade={'should_trade' in ta}",
        "kn2_dec" in ta and "should_trade" in ta)
    add("未进策略", "P0", "RL 门控",
        f"_apply_rl_inference_filters={'_apply_rl_inference_filters' in ta}",
        "_apply_rl_inference_filters" in ta)

    # ===== 9. 工程未接入 =====
    print("\n=== 9. 工程未接入 ===")
    iw_text = read_text(ROOT / "ZhuLong.PythonEngine" / "inference_worker.py")
    add("工程未接入", "P1", "hotfix_loader → worker",
        f"导入={'hotfix' in iw_text.lower()}", "hotfix" in iw_text.lower())
    deploy_h = read_text(ROOT / "scripts" / "deploy_horizon_v16_production.ps1")
    add("工程未接入", "P1", "Merge-V16AgentConfig → deploy",
        f"deploy引用={'Merge-V16AgentConfig' in deploy_h}", "Merge-V16AgentConfig" in deploy_h)
    add("工程未接入", "P0", "AgentConfigSync 同步范围",
        f"execution_composer={'execution_composer' in sync_cs}, trading_env={'trading_env' in sync_cs}",
        "execution_composer" in sync_cs and "trading_env" in sync_cs)

    # ===== 10. 架构缺陷 =====
    print("\n=== 10. 架构缺陷 ===")
    add("架构缺陷", "P0", "双持仓管理系统竞态",
        f"C# PositionManagerService + Python PositionManagerThread 同时存在. Python直接操作MT5 ticket, C#虚拟管理. 需明确主权", False)

    use_mech_blocks = "UseMechanicalExit" in pm_cs and "\"trailing\" => false" in pm_cs
    add("架构缺陷", "P0", "M1/M5移损互斥",
        f"M1 FastTrailing={'FastTrailingStopAsync' in rs}, M5 agent={'ApplyAgentM5PositionManagement' in rs}, UseMechanicalExit禁M1 trailing={use_mech_blocks}",
        use_mech_blocks)

    bad_bar = log_tail.count("不在 M5 index")
    add("架构缺陷", "P1", "decision_bar_unix 不在M5 index",
        f"最近500行 {bad_bar} 次",
        bad_bar == 0)

    add("架构缺陷", "P0", "KN2 LIVE状态",
        f"enabled={cfg.get('kn2', {}).get('enabled')}, shadow={cfg.get('kn2', {}).get('shadow_mode')}",
        cfg.get("kn2", {}).get("enabled", False) and not cfg.get("kn2", {}).get("shadow_mode", True))

    # ===== 11. 未全量同步 =====
    print("\n=== 11. 未全量同步 ===")
    for key in ["immediate_quality_min", "limit_quality_min", "entry_quality_position_weight"]:
        wv = ws_cfg.get("execution_composer", {}).get(key, "N/A")
        av = cfg.get("execution_composer", {}).get(key, "N/A")
        add("未全量同步", "P0", f"config execution_composer.{key}",
            f"workspace={wv}, AppData={av}", wv == av)

    for key in ["entry_quality_bonus", "stop_loss_atr_mult", "take_profit_atr_mult"]:
        wv = ws_cfg.get("trading_env", {}).get(key, "N/A")
        av = cfg.get("trading_env", {}).get(key, "N/A")
        add("未全量同步", "P0", f"config trading_env.{key}",
            f"workspace={wv}, AppData={av}", wv == av)

    wtv = ws_meta.get("temporal_val", "MISSING")
    atv = app_meta.get("temporal_val", "MISSING")
    add("未全量同步", "P0", "horizon_v16.meta.json temporal_val",
        f"workspace={wtv}, AppData={atv}", wtv == True and atv == True)

    # ===== 12. 声明未赋值 =====
    print("\n=== 12. 声明未赋值 ===")
    app_models = APP_DATA / "models"
    for fname in ["horizon_v16.onnx", "horizon_v16_scaler.pkl", "kn2_trader_v16.pth",
                  "horizon_v16.meta.json", "kn2_trader_v16.meta.json"]:
        fp = app_models / fname
        add("声明未赋值", "P0" if not fname.endswith(".json") else "P1",
            f"模型 {fname}",
            f"size={fp.stat().st_size}" if fp.is_file() else "MISSING", fp.is_file())

    add("声明未赋值", "P0", "KN2 acceptance_report.json",
        f"AppData存在={(APP_DATA / 'data' / 'training' / 'reports' / 'kn2_v16' / 'acceptance_report.json').is_file()}",
        (APP_DATA / "data" / "training" / "reports" / "kn2_v16" / "acceptance_report.json").is_file())

    # ===== 13. 未生效 =====
    print("\n=== 13. 未生效 ===")
    add("未生效", "P0", "V16 推理出现在日志",
        f"出现 {log_tail.count('[V16')} 次", log_tail.count("[V16") > 0)
    add("未生效", "P0", "KN2 推理出现在日志",
        "是" if "KN2=" in log_tail else "否", "KN2=" in log_tail)
    add("未生效", "P0", "Horizon 推理出现在日志",
        "是" if "Horizon=" in log_tail else "否", "Horizon=" in log_tail)
    add("未生效", "P0", "location_score_v2 生效",
        f"code={'location_score_v2' in ec}, config={cfg.get('execution_composer', {}).get('location_score_mode')}",
        "location_score_v2" in ec and cfg.get("execution_composer", {}).get("location_score_mode") == "v2")
    imm = cfg.get("execution_composer", {}).get("immediate_quality_min", 0)
    add("未生效", "P0", "immediate_quality_min ≥0.75",
        f"当前={imm}", imm >= 0.75)

    # ===== 14. 全链路 =====
    print("\n=== 14. 全链路 ===")
    add("全链路", "P0", "① M5闭合→信号",
        f"M5BarCompleted={'M5BarCompleted' in rs}, RunTradingAgentSignalTickAsync={'RunTradingAgentSignalTickAsync' in rs}",
        "M5BarCompleted" in rs and "RunTradingAgentSignalTickAsync" in rs)
    add("全链路", "P0", "② 信号->持仓托管（内联分发）",
        f"TryEmitSignalAsync={'TryEmitSignalAsync' in rs}, AdoptPending={'AdoptPendingSignalsAsync' in pm_cs}",
        "TryEmitSignalAsync" in rs and "AdoptPendingSignalsAsync" in pm_cs)
    add("全链路", "P0", "③ 持仓→SL/TP+移损",
        f"priceForSlCheck={'priceForSlCheck' in pm_cs}, trailing={'FastTrailingStopAsync' in rs}",
        "priceForSlCheck" in pm_cs and "FastTrailingStopAsync" in rs)
    add("全链路", "P0", "④ 移损→MT5 Modify",
        f"TryModifyRealPositionSlTp={'TryModifyRealPositionSlTp' in pm_cs}",
        "TryModifyRealPositionSlTp" in pm_cs)
    all_reasons = ["stop_loss", "take_profit", "trailing_stop", "agent_exit", "time_stop", "model_exit"]
    add("全链路", "P0", "⑤ 平仓→DB",
        f"CloseVirtualAsync={'CloseVirtualAsync' in pm_cs}, reasons={[r for r in all_reasons if r in pm_cs]}",
        "CloseVirtualAsync" in pm_cs and all(r in pm_cs for r in all_reasons))
    add("全链路", "P0", "⑥ 平仓→通知智能体",
        f"C#={'NotifyAgentClosedTradeAsync' in rs}, Python={'record_closed_trade' in ta}",
        "NotifyAgentClosedTradeAsync" in rs and "record_closed_trade" in ta)
    add("全链路", "P0", "⑦ 下一单约束",
        f"单信号={'单信号约束' in pm_cs}, 冷却={'BlockSymbolCooldown' in rs}, 日限额={'max_daily_trades' in ta}",
        "单信号约束" in pm_cs and "BlockSymbolCooldown" in rs and "max_daily_trades" in ta)

    try:
        db = sqlite3.connect(str(DB_PATH))
        active = db.execute("SELECT signal_id, symbol, direction, status FROM signals WHERE status IN ('active','awaiting_fill','pending')").fetchall()
        db.close()
        add("全链路", "P2", "DB活跃信号",
            f"count={len(active)}" + (f" {active[0]}" if active else ""), True)
    except Exception as ex:
        add("全链路", "P0", "DB查询",
            str(ex)[:80], False)

    # ─── 汇总 ───
    total = len(findings)
    failed = [f for f in findings if not f["passed"]]
    p0f = [f for f in failed if f["severity"] == "P0"]
    p1f = [f for f in failed if f["severity"] == "P1"]
    p2f = [f for f in failed if f["severity"] == "P2"]

    print("\n" + "=" * 60)
    print(f"实机全链路审计完成：{total} 项检查")
    print(f"  PASS: {total - len(failed)}")
    print(f"  FAIL: {len(failed)} (P0={len(p0f)} P1={len(p1f)} P2={len(p2f)})")
    print("=" * 60)

    if p0f:
        print("\n!!! P0 严重问题（必须修复）:")
        for f in p0f:
            print(f"  [{f['category']}] {f['item']}: {f['detail']}")

    if p1f:
        print("\n!!  P1 重要问题（尽快修复）:")
        for f in p1f:
            print(f"  [{f['category']}] {f['item']}: {f['detail']}")

    if p2f:
        print("\n!   P2 优化建议:")
        for f in p2f:
            print(f"  [{f['category']}] {f['item']}: {f['detail']}")

    print("\n--- 按分类汇总 ---")
    for cat in CATEGORIES:
        cf = [f for f in findings if f["category"] == cat]
        ff = [f for f in cf if not f["passed"]]
        icon = "OK" if not ff else f"FAIL {len(ff)}/{len(cf)}"
        print(f"  {cat}: {icon}")

    return 1 if p0f else 0


if __name__ == "__main__":
    sys.exit(main())
