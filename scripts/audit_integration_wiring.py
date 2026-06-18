#!/usr/bin/env python3
"""工程集成审计：未调用 / 未接入 / CLI 与 C# 对齐。"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
CLI = ROOT / "ZhuLong.PythonEngine" / "inference_cli.py"


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig") if p.is_file() else ""


def cli_cmds() -> set[str]:
    text = read(CLI)
    m = re.search(r'handlers\s*=\s*\{([^}]+)\}', text, re.S)
    if not m:
        return set()
    return set(re.findall(r'"([a-z_]+)"\s*:', m.group(1)))


def cs_invokes_cmd(cmd: str) -> bool:
    pat = f'cmd = "{cmd}"' 
    pat2 = f'["cmd"] = "{cmd}"'
    for f in SRC.rglob("*.cs"):
        t = read(f)
        if pat in t or pat2 in t:
            return True
    return False


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    rs = read(SRC / "ZhuLong.App" / "Services" / "ZhuLongRuntimeService.cs")
    py = read(SRC / "ZhuLong.App" / "Services" / "PythonInferenceService.cs")

    checks.append(("ModelsMissing.Invoke", "ModelsMissing?.Invoke" in rs, "应触发 UI 弹窗"))
    checks.append(("NotifyAgentClosedTrade", "NotifyAgentClosedTradeAsync" in rs, "平仓回灌智能体"))
    checks.append(("NotifyAgentSignalEmitted", "NotifyAgentSignalEmittedAsync" in rs, "C# 确认发信号后计数"))
    checks.append(("AgentRecordClosedTrade CLI", "agent_record_trade" in read(CLI), "inference_cli 命令"))
    checks.append(("AgentRecordSignal CLI", "agent_record_signal" in read(CLI), "inference_cli 日计数"))
    checks.append(("direction_sign payload", "direction_sign" in rs, "KN2 持仓编码"))
    checks.append(("AgentInferenceSnapHelper", (ROOT / "src/ZhuLong.Core/Services/AgentInferenceSnapHelper.cs").is_file(), "Horizon 优先 snap"))
    checks.append(("AgentStackModelGate", (ROOT / "src/ZhuLong.Core/Services/AgentStackModelGate.cs").is_file(), "V16 模型门禁"))
    checks.append(("inject_draw 实现", "inject_draw 测试信号已发送" in rs, "非空 stub"))
    checks.append(("draw_payload 解析", "DrawPayloadJson" in read(SRC / "ZhuLong.Core/Models/MultiStrategyModels.cs"), "Python 绘图载荷"))
    checks.append(("macro_silence agent", "macro_silence" in read(ROOT / "zhulong/engine/agent_engine.py"), "智能体读宏观静默"))
    checks.append(("macro_features agent", "macro_features" in read(ROOT / "zhulong/engine/agent_engine.py"), "8维宏观接入 agent_tick"))
    checks.append(("hotfix_loader", (ROOT / "ZhuLong.PythonEngine/hotfix_loader.py").is_file(), "AppData 热更新覆盖"))
    checks.append(("inference_worker", (ROOT / "ZhuLong.PythonEngine/inference_worker.py").is_file(), "常驻 Worker 脚本"))

    py_hotfix = [
        "zhulong/agent/trading_agent.py",
        "zhulong/engine/agent_engine.py",
        "zhulong/utils/py_syntax_gate.py",
        "ZhuLong.PythonEngine/inference_cli.py",
        "ZhuLong.PythonEngine/inference_worker.py",
        "ZhuLong.PythonEngine/hotfix_loader.py",
    ]
    compile_ok = True
    compile_err = ""
    for rel in py_hotfix:
        p = ROOT / rel
        try:
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
        except SyntaxError as ex:
            compile_ok = False
            compile_err = f"{rel}: {ex.msg} line {ex.lineno}"
            break
    checks.append(("Python hotfix ast.parse", compile_ok, compile_err or "只读语法门禁"))

    checks.append(("ValidateModelsAsync", "ValidateModelsAsync" in py, "legacy validate 接入"))
    checks.append(("agent/multi 互斥", "与智能体互斥" in rs or "与多策略互斥" in rs, "模式互斥"))
    cog = read(ROOT / "zhulong/agent/cognition.py")
    checks.append(("regime.detect 3-tuple", "_, regime_metrics = self.regime.detect" not in cog, "持仓管理解包"))
    checks.append(("_regime_metrics helper", "def _regime_metrics" in cog, "统一 regime metrics"))
    ta = read(ROOT / "zhulong/agent/trading_agent.py")
    checks.append(("_resolve_entry_sl_tp", "def _resolve_entry_sl_tp" in ta, "KN2+结构开仓SL/TP"))
    checks.append(("tick_symbols multi", "for sym in ordered" in ta, "多品种 tick"))
    checks.append(("_has_open_position", "def _has_open_position" in ta, "持仓 duplicate 绕过"))
    checks.append(("is_filled gate", "filled is False" in ta or "is_filled" in ta.split("_has_open_position")[1][:400], "挂单不算持仓"))
    checks.append(("record_signal_emitted", "def record_signal_emitted" in ta, "C# 确认后日计数"))
    checks.append(("persist last_bar", '"last_bar"' in ta and "daily_trade_counts" in ta, "Worker 重启状态"))
    checks.append(("C# skip Skipped", "if (r.Skipped)" in rs, "duplicate 不撤销挂单"))
    checks.append(("_last_bar after tick", "self._last_bar[symbol] = bar_key" in ta.split("self._save_state()")[-1], "成功后锁 bar"))
    checks.append(("causal graph_path wired", "graph_path=graph_path" in ta, "因果图配置"))
    checks.append(("resolve_agent_config_path", "def resolve_agent_config_path" in read(ROOT / "zhulong/utils/paths.py"), "AppData 配置"))
    checks.append(("PnL→R SL distance", "ProfitPctToR" in rs, "平仓 R 换算"))
    checks.append(("持仓失败严重日志", "持仓中智能体 tick 失败" in rs, "OpenManagedCount 告警"))

    for cmd in sorted(cli_cmds()):
        if cmd in ("predict", "warmup", "validate"):
            need = cmd in ("warmup", "validate")
        else:
            need = cmd.startswith("agent_") or cmd.endswith("_tick")
        if need:
            checks.append((f"C# 调用 cmd={cmd}", cs_invokes_cmd(cmd), "CLI/C# 对齐"))

    failed = [c for c in checks if not c[1]]
    print("=== 工程集成审计 ===")
    for name, ok, note in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} — {note}")
    print(f"\n合计 {len(checks)} 项，失败 {len(failed)} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
