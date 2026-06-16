"""
烛龙3 深度实机审计 — 全交易情景 (via inference_cli subprocess)
"""
import sys, os, json, numpy as np, pandas as pd, subprocess, time
from pathlib import Path

INSTALL = Path(r"C:\Program Files\ZhuLong")
APPDATA = Path(os.environ["APPDATA"]) / "ZhuLong"
PY = r"C:\Users\xiaomi\AppData\Local\Programs\Python\Python311\python.exe"
CLI = str(INSTALL / "ZhuLong.PythonEngine" / "inference_cli.py")
CFG_AGENT = str(APPDATA / "config_agent.json")

csv_path = r"C:\Users\xiaomi\Desktop\XAUUSD5.csv"
df = pd.read_csv(csv_path, header=None, names=["date","time","open","high","low","close","volume"])
df = df.dropna(subset=["open","high","low","close"])
df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str))
df = df.sort_values("datetime")

checks, warns, bugs = [], [], []

def chk(n, ok, d=""):
    checks.append((n, bool(ok)))
    m = "[OK]" if ok else "[BUG!]"
    print(f"  {m} {n}" + (f" — {d}" if d else ""))
    if not ok: bugs.append((n, d))

def w(msg):
    warns.append(msg)
    print(f"  [WARN] {msg}")

def bars_list(sl):
    """Convert DataFrame slice to list of [unix_time, open, high, low, close, volume]."""
    return [[
        int(pd.Timestamp(sl["datetime"].iloc[i], tz="UTC").timestamp()),
        float(sl["open"].iloc[i]), float(sl["high"].iloc[i]),
        float(sl["low"].iloc[i]), float(sl["close"].iloc[i]),
        float(sl["volume"].iloc[i] if pd.notna(sl["volume"].iloc[i]) else 0.0),
    ] for i in range(len(sl))]

def call(payload):
    tmp = APPDATA / "_audit_req.json"
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out_tmp = APPDATA / "_audit_resp.json"
    subprocess.run([PY, CLI, "--input", str(tmp), "--output", str(out_tmp)],
                   capture_output=True, timeout=120, cwd=str(INSTALL))
    try:
        return json.loads(out_tmp.read_text(encoding="utf-8"))
    except:
        return {"ok": False, "error": f"parse:{out_tmp.read_text(encoding='utf-8')[:200]}"}

def mk_req(bars_slice, acct=None):
    r = {
        "cmd": "agent_tick", "root": str(INSTALL), "config_path": CFG_AGENT,
        "symbols": ["XAUUSD"], "primary_symbol": "XAUUSD",
        "m5_bars_by_symbol": {"XAUUSD": bars_list(bars_slice)},
        "m5_includes_forming": False,
    }
    if acct: r["account"] = acct
    return r

print("=" * 70)
print("  KN2 DEEP AUDIT — inference_cli subprocess")
print(f"  Data: {len(df):,} bars")
print("=" * 70)

# ---- 0. ENV ----
print("\n--- 0. ENV ---")
cfg_a = json.loads((APPDATA / "config_agent.json").read_text(encoding="utf-8-sig"))
cfg_r = json.loads((INSTALL / "config.json").read_text(encoding="utf-8-sig"))
sf, rg = cfg_r.get("signal_filters", {}), cfg_r.get("risk_guard", {})

chk("kn2.enabled", cfg_a["kn2"]["enabled"] is True)
chk("kn2.shadow=false", cfg_a["kn2"]["shadow_mode"] is False)
chk("primary=XAUUSD", cfg_a["primary_symbol"] == "XAUUSD")
chk("model on disk", (INSTALL / "models/kn2_trader.pth").is_file())
chk("KN1 onnx", (INSTALL / "models/knowledge_net.onnx").is_file())
chk("causal_graph.json", (INSTALL / "config/causal_graph.json").is_file())

# ---- 1. VALIDATE ----
print("\n--- 1. agent_validate ---")
t0 = time.time()
v = call({"cmd": "agent_validate", "root": str(INSTALL), "config_path": "config/config_agent.json"})
elapsed = time.time() - t0
print(f"  ({elapsed:.1f}s) ok={v.get('ok')} err={v.get('error','')[:80]}")
chk("validate ok", v.get("ok") is True, v.get("error", ""))
chk("kn2_enabled", v.get("kn2_enabled") is True)
chk("kn2_ready", v.get("kn2_ready") is True)
chk("kn2_dictator", v.get("kn2_dictator") is True)

# ---- 2. SINGLE TICK ----
print("\n--- 2. SINGLE TICK ---")
t0 = time.time()
tick = call(mk_req(df.tail(300)))
elapsed = time.time() - t0
print(f"  ({elapsed:.1f}s) ok={tick.get('ok')}")
chk("tick ok", tick.get("ok") is True, tick.get("error", ""))
res = tick.get("results", [])
chk("has results", len(res) > 0)
r0 = res[0]
chk("kn2_mode", r0.get("kn2_mode") is True)
chk("kn2_ready", r0.get("kn2_ready") is True)
chk("kn2_dictator", r0.get("kn2_dictator") is True)
chk("action present", bool(r0.get("action")))

sig = r0.get("signal") or {}
chk("signal present", bool(sig))
dir_ = sig.get("direction", "flat")
chk("direction valid", dir_ in ("buy","sell","flat"), dir_)
if dir_ in ("buy","sell"):
    chk("SL>0", sig.get("sl", 0) > 0)
    chk("TP>SL", sig.get("tp", 0) > sig.get("sl", 0))
chk("strategy=rl_agent", sig.get("strategy") == "rl_agent", sig.get("strategy","N/A"))

meta = sig.get("metadata", {})
k2 = meta.get("kn2")
chk("metadata.kn2 exists", bool(k2))
if k2:
    print(f"  trace: act={k2.get('action_name')} conf={k2.get('confidence')} "
          f"sz={k2.get('position_size')} dictator={k2.get('dictator')}")
    chk("metadata.kn2.dictator", k2.get("dictator") is True)

# ---- 3. MULTI-BAR (20 ticks) ----
print("\n--- 3. MULTI-BAR (20 sequential ticks) ---")
hist = []
for i in range(20):
    seg = df.iloc[-(500+i):-(500+i-300)].copy()
    try:
        o = call(mk_req(seg))
        if o.get("ok") and o.get("results"):
            ri = o["results"][0]
            si = ri.get("signal") or {}
            hist.append({"a":ri.get("action"),"d":si.get("direction"),
                         "c":si.get("confidence",0),"reject":si.get("reject_reason","")})
    except Exception as ex:
        w(f"tick {i}: {ex}")

acts = {"hold":0,"long":0,"short":0}
for h in hist:
    a = h["a"]
    if a in acts: acts[a] += 1
tot = len(hist)
print(f"  {tot} ticks: hold={acts['hold']} long={acts['long']} short={acts['short']}")
chk("not all same class", len([v for v in acts.values() if v > 0]) >= 2)
chk("some long", acts["long"] > 0)
chk("some short", acts["short"] > 0)
chk("some hold", acts["hold"] > 0)

tr = [h for h in hist if h["d"] in ("buy","sell")]
ho = [h for h in hist if h["d"] == "flat"]
print(f"  trades={len(tr)} holds={len(ho)}")
if len(tr) > tot * 0.7:
    w(f"Overtrading: {len(tr)}/{tot} trades in 20 ticks")
if tr:
    cf = [h["c"] for h in tr]
    print(f"  conf: min={min(cf):.3f} max={max(cf):.3f} avg={np.mean(cf):.3f}")
    fl = sum(1 for j in range(1,len(tr)) if tr[j]["d"]!=tr[j-1]["d"])
    if fl > len(tr)*0.25:
        w(f"{fl} flips in {len(tr)} trades — unstable")

# ---- 4. POSITION SCENARIOS ----
print("\n--- 4. POSITION SCENARIOS ---")
cp = float(df["close"].iloc[-1])
seg0 = df.tail(300)

# No pos
np_r = call(mk_req(seg0))
print(f"  No pos:     action={(np_r.get('results',[{}])[0].get('action','?'))}")

# Long profit
lp_r = call(mk_req(seg0, {"balance":10000,"equity":10050,"open_positions":[{
    "symbol":"XAUUSD","direction":"buy","open_price":cp*0.995,"sl":cp*0.98,"tp":cp*1.02,
    "_bars_held":20,"_mfe":cp*1.025,"_mae":cp*0.99}]}))
print(f"  Long+20b:   action={(lp_r.get('results',[{}])[0].get('action','?'))}")

# Long loss
ll_r = call(mk_req(seg0, {"balance":10000,"equity":9880,"open_positions":[{
    "symbol":"XAUUSD","direction":"buy","open_price":cp*0.995,"sl":cp*0.98,"tp":cp*1.02,
    "_bars_held":3,"_mfe":cp*0.998,"_mae":cp*0.985}]}))
print(f"  Long-3b:    action={(ll_r.get('results',[{}])[0].get('action','?'))}")

# Short
sp_r = call(mk_req(seg0, {"balance":10000,"equity":10030,"open_positions":[{
    "symbol":"XAUUSD","direction":"sell","open_price":cp*1.005,"sl":cp*1.02,"tp":cp*0.98,
    "_bars_held":8,"_mfe":cp*0.99,"_mae":cp*1.01}]}))
print(f"  Short:      action={(sp_r.get('results',[{}])[0].get('action','?'))}")

# ---- 5. SIGNAL FILTERS ----
print("\n--- 5. SIGNAL FILTER COMPATIBILITY ---")
print(f"  prob_threshold={sf.get('prob_threshold')} min_rr={sf.get('min_risk_reward')}")
print(f"  max_concurrent={rg.get('max_concurrent_positions')} sym_cooldown={rg.get('symbol_cooldown_minutes')}min")
w("KN2 strategy='rl_agent' bypasses prob_threshold filter (only checks 'ai_model')")
w("symbol_cooldown=30min blocks KN2 repeat signals")

# ---- 6. BUGS ----
print("\n--- 6. BUG SCAN ---")
w("position_size from KN2 stored but never used in _action_to_signal")
w("Scheduler.apply_action_bias runs AFTER KN2 dictator — can override")
w("Scheduler.on_tick meta_result can override KN2 dictator action")

# ---- 7. SEGMENTS ----
print("\n--- 7. MARKET SEGMENTS ---")
for lb, s, e in [("bull",-200,-140),("range",-400,-300),("2025",-2000,-1900)]:
    sg = df.iloc[s:e]
    r = call(mk_req(sg))
    rs = (r.get("results") or [{}])[0]
    d = (rs.get("signal") or {}).get("direction","?")
    print(f"  {lb:6s}: action={rs.get('action','?'):5s} dir={d:4s} conf={(rs.get('signal') or {}).get('confidence',0):.3f}")

# ---- 8. DEDUP CHECK ----
print("\n--- 8. DEDUP CHECK ---")
o1 = call(mk_req(df.tail(300)))
o2 = call(mk_req(df.tail(300)))
has_both = len(o1.get("results",[]))>0 and len(o2.get("results",[]))>0
chk("both ticks produce results", has_both, "fresh AgentEngine each call — OK")

# ---- SUMMARY ----
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
p = sum(c[1] for c in checks)
t = len(checks)
for n, ok in checks:
    if not ok: print(f"  [BUG!] {n}")
print(f"\n  {p}/{t} checks passed")

if bugs:
    print(f"\n  BUGS ({len(bugs)}):")
    for n, d in bugs: print(f"    - {n}: {d}")
if warns:
    print(f"\n  WARNINGS ({len(warns)}):")
    for i, x in enumerate(warns,1): print(f"    {i}. {x}")

print("=" * 70)
