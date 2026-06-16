#!/usr/bin/env python3
"""快速回放 2026-06-10（单进程 TradingAgent，生产模型）。"""
import json
import sys
from pathlib import Path

import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
CFG = Path.home() / "AppData/Roaming/ZhuLong/config_agent.json"
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
TARGET = "2026-06-10"
WINDOW = 256


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
    sys.path.insert(0, str(INSTALL))
    from zhulong.agent.trading_agent import TradingAgent

    agent = TradingAgent(config=cfg, root=str(INSTALL))
    pos = None
    trades: list[dict] = []
    decisions: list[dict] = []
    gates: dict[str, int] = {}
    acts: dict[str, int] = {}
    rls: dict[str, int] = {}
    samples: list[str] = []

    for idx in day_idx:
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
        cog = (r.get("cognition_direction") or "?")[:5]
        cog_c = float(r.get("cognition_confidence") or 0)
        dir_ = sig.get("direction", "flat")
        filt = r.get("filter_reason") or sig.get("reject_reason") or ""
        exit_sc = float(r.get("exit_assessment") or 0)

        acts[act] = acts.get(act, 0) + 1
        rls[rl] = rls.get(rl, 0) + 1
        if filt:
            gates[filt] = gates.get(filt, 0) + 1
        decisions.append({"t": bar_t, "act": act, "rl": rl, "dir": dir_, "filt": filt})

        mark = ""
        if not pos and dir_ in ("buy", "sell"):
            mark = "ENTRY" if not filt else "blocked"
        if mark or (n := len(decisions)) % 12 == 0 or exit_sc >= 0.5:
            samples.append(
                f"{bar_t} C={cp:.1f} act={act} RL={rl} cog={cog}({cog_c:.2f}) dir={dir_} gate={filt[:24]} {mark}"
            )

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

    if pos:
        lp = float(df.loc[day_idx[-1], "close"])
        d = 1 if pos["d"] == "buy" else -1
        trades.append({**pos, "xp": lp, "xr": "EOD", "pl": (lp - pos["ep"]) / pos["ep"] * d * 100})

    print("=" * 80)
    print(f"  2026-06-10 REPLAY COMPLETE  ({o:.2f} -> {c:.2f}, {(c-o)/o*100:+.2f}%)")
    print("=" * 80)
    print(f"  Bars: {len(decisions)}")
    print(f"  Actions: {acts}")
    print(f"  RL raw: {rls}")
    print(f"  Top gates: {sorted(gates.items(), key=lambda x: -x[1])[:10]}")
    pb = sum(1 for d in decisions if d["dir"] == "buy" and not d["filt"])
    ps = sum(1 for d in decisions if d["dir"] == "sell" and not d["filt"])
    print(f"  Passed gate: buy={pb} sell={ps}")
    print(f"  Trades: {len(trades)}  Total PnL: {sum(t['pl'] for t in trades):+.2f}%")
    for i, t in enumerate(trades, 1):
        print(f"    #{i} {t['d'].upper()} {t['ep']:.1f}->{t['xp']:.1f} {t['bars']}bars PnL={t['pl']:+.2f}% {t['xr']}")
    print("\n  Sample bars:")
    for s in samples[:20]:
        print(f"    {s}")
    if len(samples) > 20:
        print(f"    ... ({len(samples)-20} more)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
