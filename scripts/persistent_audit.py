"""
烛龙3 KA2 持久化顺序审计 — 模拟真实 M5 连续运行
一个 TradingAgent 持续喂 bar，GRU 状态不重置
"""
import torch  # MUST be before sys.path manipulation (DLL env fix)
import sys, os, json, numpy as np, pandas as pd
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"

sys.path.insert(0, str(INSTALL))  # AFTER torch import to prevent DLL conflict

from zhulong.agent.trading_agent import TradingAgent

# ── Load data ──
csv_path = r"C:\Users\xiaomi\Desktop\XAUUSD5.csv"
df = pd.read_csv(csv_path, header=None, names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime")

def m5_of(sl):
    idx = pd.DatetimeIndex(sl["datetime"], tz="UTC")
    return pd.DataFrame({"open":sl["open"].values,"high":sl["high"].values,
        "low":sl["low"].values,"close":sl["close"].values,
        "volume":sl["volume"].fillna(0).values}, index=idx)

# ── Config ──
cfg_a = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
cfg_r = json.loads((INSTALL / "config.json").read_text(encoding="utf-8-sig"))

# ── Create ONE persistent agent ──
print("Loading agent ...")
agent = TradingAgent(config=cfg_a, root=str(INSTALL))
agent.reset_kn2_hidden()
print(f"KN2 ready: {agent._kn2.is_ready}, kn2_mode: {agent.kn2_mode}, shadow: {agent.kn2_shadow}")

# ── Simulated position tracking ──
pos = None          # None or {"dir":"buy"/"sell", "entry":float, "bars":int, "pnl_pct":0}
entry_log = []      # list of entries
bar_log = []        # per-bar decision log
position_bars = 0   # how many bars we've been in a position

# ── Run sequential bars ──
WINDOW = 256        # bars per tick (KN2 needs context)
START = 600         # skip first 600 bars for warmup
NUM_BARS = 500      # test 500 sequential bars

print(f"\nRunning {NUM_BARS} sequential bars with persistent GRU ...")
print(f"{'bar':>4s}  {'action':>6s}  {'dir':>4s}  {'conf':>6s}  {'SL':>8s}  {'TP':>8s}  {'position':>12s}  {'reject'}")
print("-" * 85)

for i in range(START, START + NUM_BARS):
    seg = df.iloc[i-WINDOW:i].copy()
    cp = float(seg["close"].iloc[-1])
    m5t = m5_of(seg)

    # Build account dict with position state
    acct = {"balance": 10000, "equity": 10000}
    if pos:
        pnl_pct = (cp - pos["entry"]) / pos["entry"] if pos["dir"] == "buy" else (pos["entry"] - cp) / pos["entry"]
        mfe = pos.get("mfe", pnl_pct)
        mae = pos.get("mae", pnl_pct)
        mfe = max(mfe, pnl_pct)
        mae = min(mae, pnl_pct)
        pos["pnl_pct"] = pnl_pct
        pos["mfe"] = mfe
        pos["mae"] = mae
        pos["bars"] += 1
        acct["equity"] = 10000 * (1 + pnl_pct * 0.1)  # 10x leverage on position
        acct["_positions"] = [{
            "symbol":"XAUUSD",
            "direction":pos["dir"],
            "open_price":pos["entry"],
            "sl": pos["entry"] * 0.98 if pos["dir"]=="buy" else pos["entry"] * 1.02,
            "tp": pos["entry"] * 1.02 if pos["dir"]=="buy" else pos["entry"] * 0.98,
            "_bars_held": pos["bars"],
            "_mfe": pos["entry"] * (1 + mfe) if pos["dir"]=="buy" else pos["entry"] * (1 - mfe),
            "_mae": pos["entry"] * (1 + mae) if pos["dir"]=="buy" else pos["entry"] * (1 - mae),
        }]
    else:
        acct["_positions"] = []

    try:
        results = agent.tick_symbols({"XAUUSD": m5t}, ["XAUUSD"], acct)
        if not results:
            bar_log.append({"bar":i, "action":"skipped", "dir":"skipped", "conf":0, "reject":"duplicate_bar"})
            continue

        r = results[0]
        sig = r.get("signal") or {}
        act = r.get("action", "?")
        dir_ = sig.get("direction", "flat")
        conf = sig.get("confidence", 0)
        sl = sig.get("sl", 0)
        tp = sig.get("tp", 0)
        reject = sig.get("reject_reason", "")

        pos_str = f"{pos['dir']}({pos['bars']}b)" if pos else "flat"
        entry_marker = ""

        # Track entries (new position opened)
        if not pos and dir_ in ("buy", "sell"):
            pos = {"dir": dir_, "entry": cp, "bars": 0, "mfe": 0, "mae": 0}
            entry_log.append({"bar": i, "dir": dir_, "entry": cp, "conf": conf, "sl": sl, "tp": tp})
            entry_marker = " <<< ENTRY"

        # Track exits (position closed)
        if pos and act == "hold":
            exit_pnl = (cp - pos["entry"]) / pos["entry"] if pos["dir"]=="buy" else (pos["entry"] - cp) / pos["entry"]
            pos_str += f" → CLOSED! pnl={exit_pnl*100:.2f}%"
            pos = None

        # Check SL/TP hit (simulated)
        if pos:
            if pos["dir"] == "buy" and cp <= pos["entry"] * 0.98:
                pos_str += " SL_HIT"
                pos = None
            elif pos["dir"] == "sell" and cp >= pos["entry"] * 1.02:
                pos_str += " SL_HIT"
                pos = None
            # Don't simulate TP close here - let KN2 decide

        reject_str = f"  [{reject[:30]}]" if reject else ""
        print(f"{i:4d}  {act:>6s}  {dir_:>4s}  {conf:>6.4f}  {sl:>8.2f}  {tp:>8.2f}  {pos_str:>12s}{entry_marker}{reject_str}")

        bar_log.append({
            "bar":i, "action":act, "dir":dir_, "conf":conf,
            "sl":sl, "tp":tp, "reject":reject,
            "has_position": bool(pos),
            "pos_dir": pos["dir"] if pos else None,
            "pos_bars": pos["bars"] if pos else 0,
        })

    except Exception as ex:
        print(f"  BAR {i} ERROR: {ex}")
        break

# ── Summary ──
print("\n" + "=" * 70)
print("  PERSISTENT AUDIT SUMMARY")
print("=" * 70)

total = len([b for b in bar_log if b["action"] != "skipped"])
entries = len(entry_log)
holds = len([b for b in bar_log if b["dir"] == "flat" and not b.get("has_position")])
signals_with_pos = len([b for b in bar_log if b["dir"] in ("buy","sell") and b.get("has_position")])

print(f"  Bars processed: {total}")
print(f"  Entries ({entries}):")
for e in entry_log:
    print(f"    bar={e['bar']:4d}  dir={e['dir']:4s}  entry={e['entry']:.1f}  conf={e['conf']:.4f}  "
          f"SL={e['sl']:.1f}  TP={e['tp']:.1f}")
print(f"  Entry rate: {entries}/{total} = {entries/total*100:.1f}%")
print(f"  Entries per hour (5min bars): {entries / (total * 5 / 60):.1f}/h")
print(f"  Holds without position: {holds}/{total}")

# Per-action distribution
act_dist = {"hold":0,"long":0,"short":0}
for b in bar_log:
    a = b["action"]
    if a in act_dist: act_dist[a] += 1
    elif a is None: act_dist["hold"] += 1
print(f"  KN2 internal actions: hold={act_dist['hold']} long={act_dist['long']} short={act_dist['short']}")

dir_dist = {"flat":0,"buy":0,"sell":0}
for b in bar_log:
    d = b["dir"]
    if d in dir_dist: dir_dist[d] += 1
print(f"  Signal directions: flat={dir_dist['flat']} buy={dir_dist['buy']} sell={dir_dist['sell']}")

# Confidence stats for trade signals
trade_confs = [b["conf"] for b in bar_log if b["dir"] in ("buy","sell")]
if trade_confs:
    print(f"  Trade conf: min={min(trade_confs):.4f} max={max(trade_confs):.4f} "
          f"avg={np.mean(trade_confs):.4f}")

print("=" * 70)
