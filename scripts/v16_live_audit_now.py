#!/usr/bin/env python3
"""V16 实机审计：模型文件 / warmup / tick / JSON / DB 托管链路。"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong"
CFG = APPDATA / "config_agent.json"
LOG = APPDATA / "logs" / f"log-{date.today().strftime('%Y%m%d')}.txt"

sys.path.insert(0, str(INSTALL / "ZhuLong.PythonEngine"))
sys.path.insert(0, str(INSTALL))
os.environ["ZHULONG_INSTALL_DIR"] = str(INSTALL)

checks: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print("=" * 72)
    print("V16 LIVE MACHINE AUDIT")
    print(f"Install: {INSTALL}")
    print(f"Config:  {CFG}")
    print("=" * 72)

    # 1. Deploy files
    print("\n--- 1. V16 deploy / models ---")
    exe = INSTALL / "ZhuLong.exe"
    if exe.is_file():
        chk("ZhuLong.exe", True, exe.stat().st_size // 1024 // 1024 and str(getattr(exe, "version", "")) or "present")
    else:
        chk("ZhuLong.exe", False, "missing")
    for m in (
        "horizon_v16.onnx",
        "horizon_v16_scaler.pkl",
        "kn2_trader_v16.pth",
        "rl_agent_xau.zip",
        "causal_coef.pkl",
    ):
        p = INSTALL / "models" / m
        chk(f"model/{m}", p.is_file(), f"{p.stat().st_size}B" if p.is_file() else "MISSING")

    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    chk("config arch=v16", cfg.get("architecture", {}).get("version") == "v16")
    chk("config use_rl", cfg.get("use_rl") is True)
    chk("config kn2 enabled", cfg.get("kn2", {}).get("enabled") is True)
    chk("config execution_composer", "execution_composer" in cfg)
    chk("config execution_parity", bool((cfg.get("trading_env") or {}).get("execution_parity", {}).get("enabled")))
    chk("config XAUUSD only", cfg.get("symbols", {}).get("USOIL", {}).get("enabled") is False)

    ta = (INSTALL / "zhulong/agent/trading_agent.py").read_text(encoding="utf-8")
    chk("_rf bugfix in PF install", "self._rf(confidence)" in ta)
    chk("ExecutionComposer in install", (INSTALL / "zhulong/agent/execution_composer.py").is_file())
    chk("preserve_working_intent in install", "preserve_working_intent" in ta)
    chk("evaluate_entry_against_plan wired", "evaluate_entry_against_plan" in ta)
    chk("_rl_sizing_action in install", "_rl_sizing_action" in ta)

    appdata_ta = APPDATA / "zhulong/agent/trading_agent.py"
    if appdata_ta.is_file():
        at = appdata_ta.read_text(encoding="utf-8")
        chk("AppData hotfix _rf", "self._rf(confidence)" in at and '"confidence": _rf(' not in at)

    # 2. Python stack probe
    print("\n--- 2. V16 stack probe (warmup + tick) ---")
    from inference_cli import _cmd_agent_warmup  # noqa: E402

    t0 = time.time()
    warm = _cmd_agent_warmup({"root": str(INSTALL), "config_path": str(CFG), "preload_engine": True})
    warm_ms = int((time.time() - t0) * 1000)
    chk("agent_warmup ok", warm.get("ok") is True, f"{warm_ms}ms {warm.get('error', '')[:60]}")
    if warm.get("ok"):
        chk("horizon_ready", warm.get("horizon_ready") is True)
        chk("kn2_ready", warm.get("kn2_ready") is True)
        chk("rl_ready", warm.get("rl_ready") is True)
        chk("engine_preloaded", warm.get("engine_preloaded") is True)
        chk("no deferred", warm.get("deferred") == [])

    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    from zhulong.engine.agent_engine import run_agent_tick  # noqa: E402
    from zhulong.utils.json_safe import dumps_strict  # noqa: E402

    n = 300
    idx = pd.date_range("2026-06-01", periods=n, freq="5min", tz="UTC")
    close = 4300 + np.cumsum(np.random.randn(n) * 0.5)
    m5 = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000},
        index=idx,
    )

    # 2b. 持仓管理路径探针（先于 flat tick，避免 duplicate_bar 跳过持仓字段）
    print("\n--- 2b. Position management tick probe ---")
    t0 = time.time()
    pos_out = run_agent_tick(
        {"XAUUSD": m5},
        {
            "config_path": str(CFG),
            "symbols": ["XAUUSD"],
            "primary_symbol": "XAUUSD",
            "m5_includes_forming": False,
            "open_positions": [
                {
                    "symbol": "XAUUSD",
                    "direction": "sell",
                    "entry": 4319.67,
                    "sl": 4329.57,
                    "tp": 4303.14,
                    "profit_pct": -0.05,
                    "peak_profit_pct": 0.02,
                    "hold_seconds": 300,
                }
            ],
        },
        root=INSTALL,
    )
    pos_ms = int((time.time() - t0) * 1000)
    chk("agent_tick w/ position ok", pos_out.get("ok") is True, f"{pos_ms}ms {str(pos_out.get('error', ''))[:60]}")
    if pos_out.get("ok"):
        pr = (pos_out.get("results") or [{}])[0]
        chk("position mgmt exit_assessment", "exit_assessment" in pr, f"exit={pr.get('exit_assessment')}")
        chk("position mgmt trail_mode", "trail_mode" in pr, f"trail={pr.get('trail_mode')}")

    t0 = time.time()
    out = run_agent_tick(
        {"XAUUSD": m5},
        {
            "config_path": str(CFG),
            "symbols": ["XAUUSD"],
            "primary_symbol": "XAUUSD",
            "m5_includes_forming": False,
        },
        root=INSTALL,
    )
    tick_ms = int((time.time() - t0) * 1000)
    chk("agent_tick ok", out.get("ok") is True, f"{tick_ms}ms {str(out.get('error', ''))[:60]}")
    r = (out.get("results") or [{}])[0]
    if out.get("ok"):
        layers = {
            "horizon": f"{r.get('horizon_direction')}({r.get('horizon_confidence')})",
            "cognition": f"{r.get('cognition_direction')}({r.get('cognition_confidence')})",
            "rl": r.get("rl_raw_action"),
            "final": r.get("action"),
            "kn2": f"{r.get('kn2_should_trade')}({r.get('kn2_confidence')})",
            "rl_loaded": r.get("rl_loaded"),
        }
        chk("V16 layers present", bool(r.get("horizon_direction")) or bool(r.get("cognition_direction")), str(layers))
        sig = r.get("signal") or {}
        chk("signal dict", isinstance(sig, dict), f"dir={sig.get('direction')} reason={sig.get('reject_reason') or r.get('filter_reason')}")
        meta = (sig.get("metadata") or {}) if isinstance(sig, dict) else {}
        chk("entry_mode in tick info", "entry_mode" in r or bool(meta.get("entry_mode")), f"mode={r.get('entry_mode') or meta.get('entry_mode')}")
        try:
            dumps_strict(out)
            chk("JSON strict parse", True)
        except Exception as ex:
            chk("JSON strict parse", False, str(ex)[:80])

    cog_src = (INSTALL / "zhulong/agent/cognition.py").read_text(encoding="utf-8", errors="replace")
    chk("regime unpack fix in install", "_, regime_metrics = self.regime.detect" not in cog_src)
    ta_src = (INSTALL / "zhulong/agent/trading_agent.py").read_text(encoding="utf-8", errors="replace")
    chk("_resolve_entry_sl_tp in install", "def _resolve_entry_sl_tp" in ta_src)

    # 3. DB / custody chain
    print("\n--- 3. Signal custody (DB) ---")
    db = APPDATA / "trading.db"
    chk("trading.db", db.is_file())
    if db.is_file():
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute("SELECT status, COUNT(*) FROM signals GROUP BY status")
        by_status = dict(cur.fetchall())
        print(f"    signals: {by_status}")
        active = by_status.get("pending", 0) + by_status.get("active", 0) + by_status.get("awaiting_fill", 0)
        chk("no stale active custody", active == 0, f"active_pipeline={active}")
        cur.execute(
            "SELECT signal_id, symbol, direction, status, created_at FROM signals "
            "WHERE status IN ('pending','active','awaiting_fill') ORDER BY created_at DESC LIMIT 3"
        )
        rows = cur.fetchall()
        if rows:
            for row in rows:
                print(f"    pipeline: {row}")
        cur.execute("SELECT signal_id, close_reason, close_time FROM trades ORDER BY close_time DESC LIMIT 3")
        trades = cur.fetchall()
        print(f"    recent trades: {len(trades)} rows")
        con.close()

    # 4. Live log slice (today)
    print("\n--- 4. Live log (today) ---")
    if LOG.is_file():
        text = LOG.read_text(encoding="utf-8", errors="replace")
        has_v16_tick = "[V16·Horizon]" in text or "[ExecutionComposer]" in text
        has_nameerror = "NameError: name '_rf'" in text
        has_warmup_ok = "V16 全栈热加载完成" in text or "V16 全栈已就绪" in text
        has_custody = "托管" in text
        has_composer = "[ExecutionComposer]" in text
        chk("log: V16 tick seen today", has_v16_tick)
        chk("log: _rf NameError absent", not has_nameerror, "NameError in log" if has_nameerror else "clean")
        chk("log: custody path wired", has_custody, "no custody keywords")
        chk("log: ExecutionComposer seen (after 3.1.36)", has_composer or not has_warmup_ok,
            "composer log pending first tick after upgrade" if not has_composer else "ok")
    else:
        chk("log file today", False, str(LOG))

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    print(f"RESULT: {passed}/{total} PASS")
    fails = [n for n, ok, d in checks if not ok]
    if fails:
        print("FAIL items:", ", ".join(fails))
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
