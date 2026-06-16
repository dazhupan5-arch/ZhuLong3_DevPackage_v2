#!/usr/bin/env python3
"""回放 2026-06-10 XAUUSD M5 — 经 inference_cli agent_tick（与 ZhuLong.exe 同路径）。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path.home() / "AppData" / "Roaming" / "ZhuLong"
CFG = APPDATA / "config_agent.json"
CLI = INSTALL / "ZhuLong.PythonEngine" / "inference_cli.py"
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
TARGET_DATE = "2026-06-10"
WINDOW = 256
PYTHON = sys.executable


def run_tick(bars: list, positions: list | None = None) -> dict:
    req = {
        "cmd": "agent_tick",
        "root": str(INSTALL),
        "config_path": str(CFG),
        "symbols": ["XAUUSD"],
        "primary_symbol": "XAUUSD",
        "m5_includes_forming": False,
        "m5_bars_by_symbol": {"XAUUSD": bars},
    }
    if positions:
        req["account"] = {"balance": 10000, "equity": 10000, "_positions": positions}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fin:
        json.dump(req, fin, ensure_ascii=False)
        req_path = fin.name
    out_path = req_path + ".out"
    proc = subprocess.run(
        [PYTHON, str(CLI), "--input", req_path, "--output", out_path],
        cwd=str(INSTALL),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr[:200]}
    if not Path(out_path).is_file():
        return {"ok": False, "error": "no output"}
    return json.loads(Path(out_path).read_text(encoding="utf-8-sig"))


def main() -> int:
    df = pd.read_csv(CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    day = df[df["datetime"].dt.strftime("%Y-%m-%d") == TARGET_DATE]
    if day.empty:
        print(f"No data for {TARGET_DATE}")
        return 1

    day_idx = day.index.tolist()
    o, c = float(day["open"].iloc[0]), float(day["close"].iloc[-1])
    chg = (c - o) / o * 100

    print("=" * 105)
    print(f"  KN1+PPO REPLAY — {TARGET_DATE}  (via inference_cli, production install)")
    print(f"  Day: {o:.2f} -> {c:.2f} ({chg:+.2f}%)  Hi={day['high'].max():.2f} Lo={day['low'].min():.2f}")
    print(f"  Bars: {len(day_idx)} M5  |  ~{len(day_idx)} subprocess ticks (may take several min)")
    print("=" * 105)
    print(
        f"{'Time':>8s}  {'Close':>8s}  {'Act':>8s}  {'RL':>8s}  {'Cog':>5s}  "
        f"{'CConf':>5s}  {'Dir':>4s}  {'Conf':>5s}  {'Exit':>4s}  {'Gate':<26s}  Pos"
    )
    print("-" * 105)

    pos = None
    trades: list[dict] = []
    decisions: list[dict] = []
    gate_counts: dict[str, int] = {}
    errors = 0

    for n, idx in enumerate(day_idx):
        if idx < WINDOW:
            continue
        seg = df.iloc[idx - WINDOW : idx]
        cp = float(seg["close"].iloc[-1])
        dt = seg["datetime"].iloc[-1]
        bar_t = dt.strftime("%H:%M")

        if pos:
            direction = 1 if pos["d"] == "buy" else -1
            pnl = (cp - pos["ep"]) / pos["ep"] * direction
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

        bars = [
            [
                int(r.datetime.timestamp()),
                float(r.open),
                float(r.high),
                float(r.low),
                float(r.close),
                float(r.volume if pd.notna(r.volume) else 0),
            ]
            for r in seg.itertuples()
        ]

        positions_payload = None
        if pos:
            hold_min = pos["bars"] * 5
            positions_payload = [{
                "symbol": "XAUUSD",
                "direction": pos["d"],
                "open_price": pos["ep"],
                "sl": pos["sl"],
                "tp": pos.get("tp", 0),
                "_bars_held": pos["bars"],
                "hold_seconds": hold_min * 60,
                "time_expired": hold_min >= 240,
                "max_hold_minutes": 240,
                "profit_pct": pos["pnl"] * 100,
            }]

        tick = run_tick(bars, positions_payload)
        if not tick.get("ok"):
            errors += 1
            if errors <= 3:
                print(f"  ERR {bar_t}: {tick.get('error', '')[:80]}")
            continue

        r = (tick.get("results") or [{}])[-1]
        sig = r.get("signal") or {}
        act = r.get("action") or "?"
        rl = r.get("rl_raw_action") or "?"
        cog_dir = (r.get("cognition_direction") or "?")[:5]
        cog_conf = float(r.get("cognition_confidence") or 0)
        dir_ = sig.get("direction", "flat")
        conf = float(sig.get("confidence") or 0)
        filt = r.get("filter_reason") or sig.get("reject_reason") or ""
        exit_sc = float(r.get("exit_assessment") or 0)
        sl_p = float(sig.get("sl") or 0)
        tp_p = float(sig.get("tp") or 0)

        if filt:
            gate_counts[filt] = gate_counts.get(filt, 0) + 1
        decisions.append({"t": bar_t, "act": act, "rl": rl, "dir": dir_, "filt": filt})

        ps = f"{pos['d'].upper()} {pos['bars']}b {pos['pnl']*100:+.2f}%" if pos else "FLAT"
        mark = ""
        if not pos and dir_ in ("buy", "sell"):
            mark = " <<< ENTRY" if not filt else " (blocked)"

        # 只打印有事件或每 hour 的 bar，避免刷屏；同时打印全部 entry/exit
        interesting = (
            mark or (pos and exit_sc >= 0.5) or act == "close"
            or dir_ in ("buy", "sell") or (n % 12 == 0)
        )
        if interesting:
            print(
                f"{bar_t:>8s}  {cp:8.2f}  {act:>8s}  {rl:>8s}  {cog_dir:>5s}  "
                f"{cog_conf:5.2f}  {dir_:>4s}  {conf:5.2f}  {exit_sc:4.2f}  "
                f"{filt[:26]:<26s}  {ps}{mark}"
            )

        if pos and act == "close":
            trades.append({**pos, "xp": cp, "xr": "agent_close", "pl": pos["pnl"] * 100})
            pos = None
            continue
        if pos and exit_sc >= 0.65:
            trades.append({**pos, "xp": cp, "xr": f"cog_exit({exit_sc:.2f})", "pl": pos["pnl"] * 100})
            pos = None
        if not pos and dir_ in ("buy", "sell") and not filt:
            pos = {"d": dir_, "ep": cp, "sl": sl_p, "tp": tp_p, "bars": 0, "mfe": 0, "mae": 0, "pnl": 0}
        if pos and dir_ in ("buy", "sell") and dir_ != pos["d"] and not filt:
            trades.append({**pos, "xp": cp, "xr": "flip", "pl": pos["pnl"] * 100})
            pos = None

        if (n + 1) % 30 == 0:
            print(f"  ... progress {n+1}/{len(day_idx)} bars", flush=True)

    if pos:
        lp = float(day["close"].iloc[-1])
        direction = 1 if pos["d"] == "buy" else -1
        pnl = (lp - pos["ep"]) / pos["ep"] * direction
        trades.append({**pos, "xp": lp, "xr": "EOD", "pl": pnl * 100})

    print()
    print("=" * 105)
    print("  SUMMARY — 2026-06-10")
    print("=" * 105)
    acts: dict[str, int] = {}
    rl_counts: dict[str, int] = {}
    for d in decisions:
        acts[d["act"]] = acts.get(d["act"], 0) + 1
        rl_counts[d["rl"]] = rl_counts.get(d["rl"], 0) + 1
    print(f"  Ticks OK: {len(decisions)}  errors: {errors}")
    print(f"  Final actions: {acts}")
    print(f"  RL raw distribution: {rl_counts}")
    if gate_counts:
        print(f"  Gate blocks: {sorted(gate_counts.items(), key=lambda x: -x[1])[:10]}")

    passed_buy = sum(1 for d in decisions if d["dir"] == "buy" and not d["filt"])
    passed_sell = sum(1 for d in decisions if d["dir"] == "sell" and not d["filt"])
    print(f"  Signals passed gate: buy={passed_buy} sell={passed_sell}")

    total_pl = sum(t["pl"] for t in trades)
    if trades:
        print()
        for i, t in enumerate(trades, 1):
            print(
                f"  #{i} {t['d'].upper():4s} {t['ep']:.2f}->{t['xp']:.2f} "
                f"{t['bars']}bars ({t['bars']*5/60:.1f}h) PnL={t['pl']:+.2f}% reason={t['xr']}"
            )
        print(f"\n  Simulated total PnL: {total_pl:+.2f}%")
    else:
        print("\n  No simulated trades — all signals blocked or flat.")

    print("=" * 105)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
