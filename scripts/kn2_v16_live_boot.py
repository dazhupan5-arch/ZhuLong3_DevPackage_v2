#!/usr/bin/env python3
"""KN2 V16 LIVE 开机实盘场景验证（V16 + Horizon + KN2 全链路）。"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# 以 scripts/xxx.py 启动时，避免 scripts 目录遮蔽 zhulong 包
if sys.path and Path(sys.path[0]).name.lower() == "scripts":
    sys.path.pop(0)
os.chdir(_ROOT)

try:
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
except Exception:
    pass

import torch  # noqa: F401 — 须在 onnxruntime/numpy 之前

APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong"
INSTALL = Path(r"C:\Program Files\ZhuLong")
CFG = APPDATA / "config_agent.json"
MT5_CSV = Path(
    r"C:\Users\xiaomi\AppData\Roaming\MetaQuotes\Terminal\7643C0B96C7AD5841307C9E1EB0B9252\MQL5\Files\bars\XAUUSD_5.csv"
)


def _chk(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def _data_root() -> Path:
    """模型/配置根目录：始终用安装目录（与 C# AppPaths.InstallDir 一致）。"""
    if (INSTALL / "models" / "horizon_v16.onnx").is_file():
        return INSTALL
    if (APPDATA / "models" / "horizon_v16.onnx").is_file():
        return APPDATA
    return _ROOT


def _cli_script() -> tuple[Path, Path]:
    """(inference_cli.py 路径, 执行 root)。"""
    app_cli = APPDATA / "ZhuLong.PythonEngine" / "inference_cli.py"
    if app_cli.is_file():
        return app_cli, _data_root()
    install_cli = INSTALL / "ZhuLong.PythonEngine" / "inference_cli.py"
    if install_cli.is_file():
        return install_cli, INSTALL
    dev_cli = _ROOT / "ZhuLong.PythonEngine" / "inference_cli.py"
    return dev_cli, _ROOT


def run_cli(req: dict) -> dict:
    import importlib.util
    import os

    cli, root = _cli_script()
    if not cli.is_file():
        return {"ok": False, "error": f"missing_inference_cli:{cli}"}

    req = dict(req)
    req["root"] = str(root)

    for p in (str(root), str(INSTALL), str(APPDATA), str(_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(APPDATA), str(INSTALL), str(INSTALL / "ZhuLong.PythonEngine")]
    )

    spec = importlib.util.spec_from_file_location("zhulong_inference_cli", str(cli))
    if spec is None or spec.loader is None:
        return {"ok": False, "error": "cli_spec_failed"}
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cmd = str(req.get("cmd", ""))
    handlers = {
        "agent_validate": getattr(mod, "_cmd_agent_validate", None),
        "agent_tick": getattr(mod, "_cmd_agent_tick", None),
    }
    fn = handlers.get(cmd)
    if fn is None:
        return {"ok": False, "error": f"unknown_cmd:{cmd}"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn, req)
        try:
            return fut.result(timeout=90 if cmd == "agent_tick" else 30) or {}
        except FuturesTimeout:
            return {"ok": False, "error": f"{cmd}_timeout"}


def main() -> int:
    import pandas as pd

    print("=== KN2 V16 LIVE Boot Verification ===\n")
    ok_all = True

    if not CFG.is_file():
        ok_all &= _chk("config_agent.json", False, str(CFG))
        return 1

    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    kn2 = cfg.get("kn2") or {}
    arch = (cfg.get("architecture") or {}).get("version", "")
    ok_all &= _chk("architecture=v16", arch == "v16", arch)
    ok_all &= _chk("kn2.enabled=true (LIVE)", kn2.get("enabled") is True)
    ok_all &= _chk("kn2.shadow_mode=false", kn2.get("shadow_mode") is False)
    ok_all &= _chk("kn2.model_path", "kn2_trader_v16" in str(kn2.get("model_path", "")))

    kn2_pth = APPDATA / "models" / "kn2_trader_v16.pth"
    if not kn2_pth.is_file():
        kn2_pth = INSTALL / "models" / "kn2_trader_v16.pth"
    ok_all &= _chk("kn2 model on disk", kn2_pth.is_file(), str(kn2_pth))

    root = _data_root()
    v = run_cli({"cmd": "agent_validate", "config_path": str(CFG), "quick": True})
    ok_all &= _chk("agent_validate", bool(v.get("ok")), str(v.get("error", v.get("architecture", ""))))
    if v.get("ok"):
        print(f"       arch={v.get('architecture')} kn2_ready={v.get('kn2_ready', 'n/a')}")

    csv = Path(r"C:\Users\xiaomi\Desktop\XAUUSD5.csv")
    if not csv.is_file():
        csv = MT5_CSV
    if not csv.is_file():
        print("\n  [WARN] 无 M5 CSV，跳过 agent_tick")
        return 0 if ok_all else 2

    df = pd.read_csv(csv, header=None, names=["date", "time", "open", "high", "low", "close", "volume"])
    df = df.dropna(subset=["open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    if str(df.iloc[0]["date"]).lower().startswith("timestamp") or str(df.iloc[0]["open"]).lower() == "open":
        df = df.iloc[1:].reset_index(drop=True)
    first_date = str(df.iloc[0]["date"])
    if first_date.isdigit() and len(first_date) >= 10:
        ts = pd.to_numeric(df["date"], errors="coerce")
        df["datetime"] = pd.to_datetime(ts, unit="s", utc=True)
    else:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), utc=True)
    df = df.sort_values("datetime").tail(300)
    bars = [
        [int(r.datetime.timestamp()), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume or 0)]
        for r in df.itertuples()
    ]

    print("\n--- agent_tick (实盘场景) ---")
    tick = run_cli(
        {
            "cmd": "agent_tick",
            "config_path": str(CFG),
            "symbols": ["XAUUSD"],
            "primary_symbol": "XAUUSD",
            "m5_bars_by_symbol": {"XAUUSD": bars},
        }
    )
    ok_all &= _chk("agent_tick ok", bool(tick.get("ok")), tick.get("error", ""))
    r = (tick.get("results") or [{}])[-1]
    kn2_live = not bool(r.get("kn2_shadow_mode", True))
    ok_all &= _chk("kn2 live mode in tick", kn2_live, f"shadow={r.get('kn2_shadow_mode')}")
    print(
        f"  action={r.get('action')} horizon={r.get('horizon_direction')}({r.get('horizon_confidence')}) "
        f"cognition={r.get('cognition_direction')} filter={r.get('filter_reason')}"
    )
    kn2_dec = r.get("kn2_decision") or r.get("kn2") or {}
    if kn2_dec:
        print(
            f"  kn2: action={kn2_dec.get('action_name', kn2_dec.get('action'))} "
            f"conf={kn2_dec.get('confidence')} should_trade={kn2_dec.get('should_trade')}"
        )
    else:
        print(f"  kn2 fields: shadow={r.get('kn2_shadow_mode')} enabled={r.get('kn2_enabled')}")

    print()
    if ok_all:
        print("=== OVERALL: PASS — KN2 V16 LIVE 就绪 ===")
        return 0
    print("=== OVERALL: FAIL — 见上方 FAIL 项 ===")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
