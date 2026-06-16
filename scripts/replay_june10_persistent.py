#!/usr/bin/env python3
"""2026-06-10 回放 — 单进程持久化 Agent（与 inference_cli 相同 DLL 顺序）。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
CFG = Path.home() / "AppData/Roaming/ZhuLong/config_agent.json"
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
TARGET = "2026-06-10"
WINDOW = 256

os.environ.setdefault("ZHULONG_IMF_CSV_ONLY", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ZHULONG_INSTALL_DIR", str(INSTALL))

# DLL 路径（与 inference_cli 一致）
try:
    _engine = INSTALL / "ZhuLong.PythonEngine"
    for p in (INSTALL, _engine):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from zhulong.utils.win_dll import configure_native_dll_paths

    configure_native_dll_paths()
    add_fn = getattr(os, "add_dll_directory", None)
    if add_fn is not None:
        try:
            add_fn(str(INSTALL))
        except OSError:
            pass
    os.environ["PATH"] = str(INSTALL) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

import onnxruntime  # noqa: F401 — 必须在 torch/zhulong 之前

import pandas as pd

from zhulong.agent.trading_agent import TradingAgent


def m5_df(sl: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame(
        {
            "open": sl["open"].values,
            "high": sl["high"].values,
            "low": sl["low"].values,
            "close": sl["close"].values,
            "volume": sl["volume"].fillna(0).values,
        },
        index=idx,
    )


def main() -> int:
    df = pd.read_csv(CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == TARGET].index.tolist()
    day = df.loc[day_idx]
    o, c = float(day["open"].iloc[0]), float(day["close"].iloc[-1])

    cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
    print("Loading agent (persistent)...", flush=True)
    agent = TradingAgent(config=cfg, root=str(INSTALL))

    pos = None
    trades: list[dict] = []
    decisions: list[dict] = []
    gates: dict[str, int] = {}

    for n, idx in enumerate(day_idx):
        if idx < WINDOW:
            continue
        seg = df.iloc[idx - WINDOW : idx].copy()
        cp = float(seg["close"].iloc[-1])
        dt = seg["datetime"].iloc[-1]
        bar_t = dt.strftime("%H:%M")

        if pos:
            d = 1 if pos["d"] == "buy" else -1
            pnl = (cp - pos["ep"]) / pos["ep"] * d
            pos["pnl"] = pnl
            pos["mfe"] = max(pos.get("mfe", pnl), pnl)
            pos["mae"] = min(pos.get("mae", pnl), pnl)
            pos["bars"] += 1
            if pos["d"] == "buy" and cp <= pos["sl"]:
                trades.append({**pos, "xp": cp, "xr": "SL", "pl": pnl * 100})
                pos = None
            elif pos["d"] == "sell" and cp >= pos["sl"]:
                trades.append({**pos, "xp": cp, "xr": "SL", "pl": pnl * 100})
                pos = None

        acct = {"balance": 10000, "equity": 10000, "_positions": []}
        if pos:
            hm = pos["bars"] * 5
            acct["_positions"] = [{
                "symbol": "XAUUSD",
                "direction": pos["d"],
                "open_price": pos["ep"],
                "sl": pos["sl"],
                "_bars_held": pos["bars"],
                "hold_seconds": hm * 60,
                "time_expired": hm >= 240,
                "max_hold_minutes": 240,
                "profit_pct": pos["pnl"] * 100,
            }]

        rr = agent.tick_symbols({"XAUUSD": m5_df(seg)}, ["XAUUSD"], acct)
        if not rr:
            continue
        r = rr[0]
        sig = r.get("signal") or {}
        act = r.get("action", "?")
        rl = r.get("rl_raw_action", "?")
        dir_ = sig.get("direction", "flat")
        filt = r.get("filter_reason") or sig.get("reject_reason") or ""
        exit_sc = float(r.get("exit_assessment") or 0)

        if filt:
            gates[filt] = gates.get(filt, 0) + 1
        decisions.append({"t": bar_t, "act": act, "rl": rl, "dir": dir_, "filt": filt})

        if pos and act == "close":
            trades.append({**pos, "xp": cp, "xr": "agent_close", "pl": pos["pnl"] * 100})
            pos = None
            continue
        if pos and exit_sc >= 0.65:
            trades.append({**pos, "xp": cp, "xr": f"cog_exit({exit_sc:.2f})", "pl": pos["pnl"] * 100})
            pos = None
        if not pos and dir_ in ("buy", "sell") and not filt:
            pos = {"d": dir_, "ep": cp, "sl": float(sig.get("sl") or 0), "bars": 0, "mfe": 0, "mae": 0, "pnl": 0}
        if pos and dir_ in ("buy", "sell") and dir_ != pos["d"] and not filt:
            trades.append({**pos, "xp": cp, "xr": "flip", "pl": pos["pnl"] * 100})
            pos = None

        if (n + 1) % 50 == 0:
            print(f"  ... {n+1}/{len(day_idx)}", flush=True)

    if pos:
        lp = float(df.loc[day_idx[-1], "close"])
        d = 1 if pos["d"] == "buy" else -1
        trades.append({**pos, "xp": lp, "xr": "EOD", "pl": (lp - pos["ep"]) / pos["ep"] * d * 100})

    acts: dict[str, int] = {}
    rls: dict[str, int] = {}
    for d in decisions:
        acts[d["act"]] = acts.get(d["act"], 0) + 1
        rls[d["rl"]] = rls.get(d["rl"], 0) + 1

    print("=" * 80)
    print(f"  SUMMARY — {TARGET}  ({o:.2f} -> {c:.2f}, {(c-o)/o*100:+.2f}%)")
    print("=" * 80)
    print(f"  Ticks: {len(decisions)}")
    print(f"  Actions: {acts}")
    print(f"  RL raw: {rls}")
    print(f"  Gate blocks: {sorted(gates.items(), key=lambda x: -x[1])[:10]}")
    pb = sum(1 for d in decisions if d["dir"] == "buy" and not d["filt"])
    ps = sum(1 for d in decisions if d["dir"] == "sell" and not d["filt"])
    print(f"  Passed gate: buy={pb} sell={ps}")
    print(f"  Trades: {len(trades)}  Total PnL: {sum(t['pl'] for t in trades):+.2f}%")
    for i, t in enumerate(trades, 1):
        print(f"    #{i} {t['d'].upper()} {t['ep']:.1f}->{t['xp']:.1f} {t['bars']}bars PnL={t['pl']:+.2f}% {t['xr']}")
    if not trades:
        print("  No simulated trades — all signals blocked or flat.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
