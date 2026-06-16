#!/usr/bin/env python3
"""诊断 6/10 认知层：KN 原始概率 vs V14 vs 认知方向。"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

INSTALL = Path(r"C:\Program Files\ZhuLong")
CFG = Path.home() / "AppData/Roaming/ZhuLong/config_agent.json"
CSV = Path(__file__).resolve().parent / "_june_multi_bars.csv"
TARGET = "2026-06-10"
WINDOW = 256

os.environ.setdefault("ZHULONG_IMF_CSV_ONLY", "1")
os.environ.setdefault("ZHULONG_INSTALL_DIR", str(INSTALL))
for p in (INSTALL, INSTALL / "ZhuLong.PythonEngine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from zhulong.utils.win_dll import configure_native_dll_paths
configure_native_dll_paths()
import onnxruntime  # noqa

from zhulong.agent.knowledge_net import KnowledgeNetInference, resolve_knowledge_paths
from zhulong.agent.trading_agent import TradingAgent
from zhulong.v14_live import load_v14_bundle, build_live_v14_features, predict_v14
from zhulong.agent.causal_inference import fuse_knowledge_with_causal
from zhulong.strategies.indicators import atr_series

df = pd.read_csv(CSV)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == TARGET].index.tolist()

cfg = json.loads(CFG.read_text(encoding="utf-8-sig"))
kn_path, kn_scaler = resolve_knowledge_paths("XAUUSD", cfg, INSTALL)
kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
v14 = load_v14_bundle("XAUUSD", root=INSTALL)
agent = TradingAgent(config=cfg, root=str(INSTALL))

rows = []
for idx in day_idx:
    if idx < WINDOW:
        continue
    seg = df.iloc[idx - WINDOW : idx].copy()
    dt = seg["datetime"].iloc[-1]
    cp = float(seg["close"].iloc[-1])
    idx_utc = pd.DatetimeIndex(seg["datetime"], tz="UTC")
    m5 = pd.DataFrame({
        "open": seg["open"].values, "high": seg["high"].values,
        "low": seg["low"].values, "close": seg["close"].values,
        "volume": seg["volume"].fillna(0).values,
    }, index=idx_utc)

    struct = agent.structure.compute_latest({"M5": m5})
    feat68 = agent._build_v14_features(m5)
    raw_probs, _ = kn.predict(feat68.reshape(1, -1))
    raw_probs = raw_probs[0].copy()
    prob_row = raw_probs.copy()
    raw_causal = 0.0
    if agent.causal_enabled and agent.causal.is_ready:
        shock = agent.causal.macro_shock_from_bar(struct)
        raw_causal = agent.causal.predict_price_change(shock)
        if agent.causal_fusion_weight > 0:
            fused = fuse_knowledge_with_causal(
                prob_row, raw_causal,
                weight_knowledge=1.0 - agent.causal_fusion_weight,
                weight_causal=agent.causal_fusion_weight,
            )
            prob_row = fused[0] if fused.ndim > 1 else fused

    agent.cognition.rebuild_context_from_m5(m5)

    feat_row, _, _, feats_df = build_live_v14_features(m5=m5)
    v14_sig = predict_v14(v14, feat_row, m5, bar_time=m5.index[-1], feats_df=feats_df)
    v14_p = v14_sig.probabilities

    atr = float(atr_series(m5).iloc[-1])
    bar_key = dt.strftime("%Y-%m-%d %H:%M")
    thought = agent.cognition.process(
        struct, prob_row, raw_causal, cp, atr,
        volume=float(seg["volume"].iloc[-1] or 0),
        bar_timestamp=bar_key,
    )
    cog_dir, regime, instant = agent.cognition.resolve_sticky_direction(
        thought.calibrated_probs, bar_key
    )
    v14_side = {"buy": "long", "sell": "short", "flat": "flat"}.get(v14_sig.direction, v14_sig.direction)

    rows.append({
        "t": dt.strftime("%H:%M"), "close": cp,
        "kn_s": raw_probs[0], "kn_f": raw_probs[1], "kn_l": raw_probs[2],
        "kn_argmax": ["short", "flat", "long"][int(raw_probs.argmax())],
        "v14_side": v14_side, "v14_conf": v14_sig.confidence, "v14_p": v14_p,
        "regime": thought.regime, "instant": instant, "cog": cog_dir,
        "should_trade": thought.should_trade, "conf": thought.confidence,
    })

rdf = pd.DataFrame(rows)
print("=" * 90)
print(f"  June 10 cognition diagnostic  ({len(rdf)} bars)")
print("=" * 90)
print("\n--- KN1 argmax ---")
print(rdf["kn_argmax"].value_counts().to_string())
print("\n--- V14 side ---")
print(rdf["v14_side"].value_counts().to_string())
print("\n--- Cognition sticky (cog) ---")
print(rdf["cog"].value_counts().to_string())
print("\n--- Instant direction ---")
print(rdf["instant"].value_counts().to_string())
print("\n--- Regime ---")
print(rdf["regime"].value_counts().to_string())
print(f"\nMean KN probs: short={rdf['kn_s'].mean():.3f} flat={rdf['kn_f'].mean():.3f} long={rdf['kn_l'].mean():.3f}")
vp = np.vstack([p for p in rdf["v14_p"] if p])
print(f"Mean V14 probs: flat={vp[:,0].mean():.3f} long={vp[:,1].mean():.3f} short={vp[:,2].mean():.3f}")
agree = (rdf["kn_argmax"] == rdf["v14_side"]).mean()
print(f"\nKN argmax == V14 side: {agree:.1%}")
print(f"instant=short but cog=long: {((rdf['instant']=='short')&(rdf['cog']=='long')).sum()}")
print(f"should_trade=True: {rdf['should_trade'].sum()}/{len(rdf)}")
print("\n--- Every 2h sample ---")
for _, r in rdf.iloc[::24].iterrows():
    vp = r["v14_p"] or [0, 0, 0]
    print(f"  {r['t']} C={r['close']:.0f} KN=[{r['kn_s']:.2f},{r['kn_f']:.2f},{r['kn_l']:.2f}]→{r['kn_argmax']} "
          f"V14={r['v14_side']}({r['v14_conf']:.2f}) regime={r['regime']} instant={r['instant']} cog={r['cog']}")
print("=" * 90)
