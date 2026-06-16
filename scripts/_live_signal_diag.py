#!/usr/bin/env python3
"""实机信号诊断：DB 近期信号 + agent tick 门控原因。"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CN = timezone(timedelta(hours=8))
DB = Path(r"C:\Users\xiaomi\AppData\Roaming\ZhuLong\trading.db")


def check_db() -> None:
    print("=== LIVE DB:", DB, "exists=", DB.is_file())
    if not DB.is_file():
        return
    cn = sqlite3.connect(DB)
    cn.row_factory = sqlite3.Row
    rows = cn.execute(
        "SELECT signal_id, symbol, direction, status, confidence, created_at "
        "FROM signals ORDER BY created_at DESC LIMIT 30"
    ).fetchall()
    print(f"=== RECENT SIGNALS ({len(rows)}) ===")
    for r in rows:
        ts = r["created_at"]
        tstr = datetime.fromtimestamp(ts, CN).strftime("%Y-%m-%d %H:%M") if ts else "?"
        sid = (r["signal_id"] or "")[:55]
        conf = r["confidence"]
        conf_s = f"{conf:.3f}" if conf is not None else "?"
        print(f"  {tstr} | {r['direction']:5} | {r['status']:10} | conf={conf_s} | {sid}")
    week_ago = int((datetime.now(CN) - timedelta(days=7)).timestamp())
    print("=== COUNT last 7d ===")
    for r in cn.execute(
        "SELECT direction, COUNT(*) AS n FROM signals WHERE created_at>=? GROUP BY direction",
        (week_ago,),
    ):
        print(f"  {r['direction']}: {r['n']}")
    cn.close()


def check_agent(root: Path, cfg_path: Path) -> None:
    print("\n=== AGENT CONFIG:", cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    arch = cfg.get("architecture") or {}
    print("  architecture.version:", arch.get("version", "legacy"))
    print("  fallback_strategy:", cfg.get("fallback_strategy"))
    print("  rl min_confidence:", (cfg.get("rl_inference") or {}).get("min_confidence_for_trade"))
    print("  rl action_threshold:", (cfg.get("rl_inference") or {}).get("action_threshold"))
    hp = arch.get("horizon_predictor") or {}
    if hp:
        print("  horizon model:", hp.get("model_path"))
        print("  min_direction_confidence:", hp.get("min_direction_confidence"))

    from zhulong.training.lgb.data_io import load_vendor_csv

    csv_path = root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv"
    if not csv_path.is_file():
        print("  WARN: no M5 CSV for tick sim")
        return
    m5 = load_vendor_csv(csv_path)
    sample = m5.loc["2026-06-10":"2026-06-15"]
    if len(sample) < 300:
        sample = m5.tail(500)
    print(f"  tick sample bars: {len(sample)} ({sample.index[0]} .. {sample.index[-1]})")

    from zhulong.engine.agent_engine import AgentEngine

    engine = AgentEngine(cfg, root=root)
    flat = hold = buy = sell = 0
    reasons: dict[str, int] = {}
    last = None
    for i in range(max(200, len(sample) - 120), len(sample)):
        window = sample.iloc[: i + 1]
        acct = {"_positions": [], "_ticks": {"XAUUSD": {"bid": float(window["close"].iloc[-1]), "ask": float(window["close"].iloc[-1]) + 0.3}}}
        out = engine.tick("XAUUSD", window, acct)
        if not out:
            continue
        sig = out.get("signal") or {}
        d = sig.get("direction", "?")
        if d == "flat":
            flat += 1
            rr = sig.get("reject_reason") or out.get("gate_reason") or "flat"
            reasons[str(rr)] = reasons.get(str(rr), 0) + 1
        elif d == "buy":
            buy += 1
        elif d == "sell":
            sell += 1
        last = out

    print(f"  simulated ticks: flat={flat} buy={buy} sell={sell}")
    print("  flat reasons:", dict(sorted(reasons.items(), key=lambda x: -x[1])[:10]))
    if last:
        print("  last tick summary:")
        for k in (
            "action",
            "cognition_direction",
            "cognition_confidence",
            "gate_reason",
            "knowledge_ready",
            "architecture",
        ):
            if k in last:
                print(f"    {k}: {last[k]}")
        sig = last.get("signal") or {}
        for k in ("direction", "reject_reason", "confidence", "metadata"):
            if k in sig:
                print(f"    signal.{k}: {sig[k]}")


def main() -> int:
    check_db()
    dev_cfg = _ROOT / "config" / "config_agent.json"
    pub_cfg = _ROOT / "publish" / "win-x64" / "config" / "config_agent.json"
    install_cfg = Path(r"C:\Users\xiaomi\AppData\Roaming\ZhuLong\config\config_agent.json")
    print("\n=== CONFIG LOCATIONS ===")
    for p in (install_cfg, pub_cfg, dev_cfg):
        print(f"  {p} exists={p.is_file()}")
    target = install_cfg if install_cfg.is_file() else dev_cfg
    check_agent(_ROOT, target)
    if pub_cfg.is_file() and pub_cfg != target:
        check_agent(_ROOT / "publish" / "win-x64", pub_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
