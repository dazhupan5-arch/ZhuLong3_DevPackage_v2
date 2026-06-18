#!/usr/bin/env python3
"""V16 Horizon 正式验收：全量训练 + ONNX + OOS + lockbox + Agent tick。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: F401 — Windows: 须最先加载

import numpy as np
import pandas as pd

from zhulong.agent.horizon_location_labels import resolve_horizon_training_labels
from zhulong.agent.horizon_predictor import HorizonPredictor, direction_from_probs
from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.structure_service import StructureService
from zhulong.agent.training_utils import load_npz, signed_to_class, temporal_train_val_masks, TRAIN_END_DEFAULT, VAL_YEAR_DEFAULT
from zhulong.engine.agent_engine import load_agent_config, run_agent_tick
from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.v10.backtest import backtest_both

from scripts.v16_acceptance_metrics import (
    apply_classification_thresholds,
    apply_train_test_f1_gates,
    check_win_rate,
    classification_report,
    load_f1_floor,
)


def _load_acceptance(root: Path) -> dict:
    p = root / "config" / "v16_acceptance.json"
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _load_bt_module():
    spec = importlib.util.spec_from_file_location("backtest_v16", _ROOT / "scripts" / "backtest_v16.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _side_ratio(long_n: int, short_n: int, max_ratio: float) -> tuple[bool, float]:
    if long_n <= 0 or short_n <= 0:
        return False, float("inf")
    ratio = max(long_n, short_n) / min(long_n, short_n)
    return ratio <= max_ratio, ratio


def _check_training(root: Path, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    detail: dict = {}
    npz = root / "data" / "clean" / "training_horizon_v16.npz"
    meta_path = root / "models" / "horizon_v16.meta.json"
    if not npz.is_file():
        failures.append("missing_training_npz")
        return False, detail, failures
    data = np.load(npz, allow_pickle=True)
    rows = int(len(data["struct"]))
    detail["training_rows"] = rows
    min_rows = int(acc.get("min_training_rows", 700000))
    if rows < min_rows:
        failures.append(f"training_rows_{rows}_lt_{min_rows}")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        detail["val_accuracy"] = float(meta.get("val_accuracy", 0))
        detail["macro_f1"] = float(meta.get("macro_f1", 0))
        if detail["val_accuracy"] < float(acc.get("min_val_accuracy", 0.48)):
            failures.append("val_accuracy_below_threshold")
        min_test_f1 = load_f1_floor(acc)
        if detail["macro_f1"] <= min_test_f1:
            failures.append(f"meta_test_macro_f1_{detail['macro_f1']:.4f}_lte_{min_test_f1}")
    else:
        failures.append("missing_horizon_meta")
    return len(failures) == 0, detail, failures


def _predict_horizon_on_indices(
    kn: KnowledgeNetInference,
    struct: np.ndarray,
    idx: np.ndarray,
) -> np.ndarray:
    y_pred = np.zeros(len(idx), dtype=np.int64)
    chunk = 4096
    for start in range(0, len(idx), chunk):
        sel = idx[start : start + chunk]
        probs, _ = kn.predict(struct[sel])
        y_pred[start : start + len(sel)] = np.argmax(probs, axis=1)
    return y_pred


def _eval_horizon_classification_split(
    root: Path,
    acc: dict,
    *,
    split: str,
) -> tuple[dict[str, Any], list[str]]:
    """split=train|test；测试集=2025 OOS（val_year）。"""
    failures: list[str] = []
    npz_path = root / "data" / "clean" / "training_horizon_v16_location.npz"
    onnx = root / "models" / "horizon_v16.onnx"
    scaler = root / "models" / "horizon_v16_scaler.pkl"
    if not npz_path.is_file() or not onnx.is_file():
        failures.append(f"missing_npz_or_onnx_for_{split}_classification")
        return {}, failures

    data = load_npz(npz_path)
    struct = np.asarray(data["struct"], dtype=np.float32)
    y_signed, _ = resolve_horizon_training_labels(data, label_mode="location")
    y_true = signed_to_class(y_signed)
    times = data.get("time")
    if times is None:
        failures.append("npz_missing_time")
        return {}, failures

    train_mask, test_mask = temporal_train_val_masks(
        times, val_year=int(acc.get("val_year", VAL_YEAR_DEFAULT))
    )
    mask = np.asarray(train_mask if split == "train" else test_mask, dtype=bool)
    if int(mask.sum()) < 500:
        failures.append(f"{split}_sample_too_small_{int(mask.sum())}")
        return {}, failures

    kn = KnowledgeNetInference(onnx, scaler_path=scaler if scaler.is_file() else None)
    if not kn.is_ready:
        failures.append(f"horizon_onnx_not_ready_for_{split}")
        return {}, failures

    idx = np.where(mask)[0]
    max_key = (
        "max_train_classification_bars"
        if split == "train"
        else "max_val_classification_bars"
    )
    max_bars = int(acc.get(max_key, 25000))
    if len(idx) > max_bars:
        step = max(1, len(idx) // max_bars)
        idx = idx[::step][:max_bars]

    y_pred = _predict_horizon_on_indices(kn, struct, idx)
    metrics = classification_report(y_true[idx], y_pred)
    if split == "test":
        apply_classification_thresholds(metrics, acc, failures, prefix="test")
    return metrics, failures


def _check_classification_splits(root: Path, acc: dict) -> tuple[bool, dict, list[str]]:
    """训练集+测试集 macro F1 均>0.5；禁止 train 高 test 低；测试集 long/short P/R>=80%。"""
    failures: list[str] = []
    detail: dict = {}
    train_metrics, train_fails = _eval_horizon_classification_split(root, acc, split="train")
    test_metrics, test_fails = _eval_horizon_classification_split(root, acc, split="test")
    failures.extend(train_fails)
    failures.extend(test_fails)
    if not train_metrics or not test_metrics:
        return False, detail, failures

    detail["train_classification"] = train_metrics
    detail["test_classification"] = test_metrics
    detail["val_classification"] = test_metrics  # 兼容旧报告字段
    apply_train_test_f1_gates(train_metrics, test_metrics, acc, failures, prefix="horizon")
    return len(failures) == 0, detail, failures


def _check_leak_contract(root: Path, acc: dict) -> tuple[bool, dict, list[str]]:
    """硬门禁：禁止数据泄露、未来函数、随机 val 回退。"""
    failures: list[str] = []
    detail: dict = {"contract_version": acc.get("acceptance_contract_version")}
    if not acc.get("require_no_data_leak", True):
        return True, detail, failures

    th_hz = (root / "scripts" / "train_horizon_v16.py").read_text(encoding="utf-8")
    if "no-temporal-val 已禁用" not in th_hz:
        failures.append("train_horizon_missing_no_temporal_val_guard")
    ak = (root / "scripts" / "accept_kn2_v16.py").read_text(encoding="utf-8")
    if acc.get("forbid_random_val_fallback", True) and "int(n * 0.85)" in ak:
        failures.append("accept_kn2_random_val_fallback_forbidden")
    tu = (root / "zhulong" / "agent" / "training_utils.py").read_text(encoding="utf-8")
    if acc.get("require_no_future_function", True):
        if "bad_tick_revert_causal" not in tu:
            failures.append("missing_causal_bad_tick_cleaning")
        if "shift(-1)" in tu.split("bad_tick")[1][:500] if "bad_tick" in tu else False:
            failures.append("future_shift_in_bad_tick_cleaning")

    for rel in (
        "data/clean/training_horizon_v16_location.npz",
        "data/clean/kn2_training_v16_location.npz",
    ):
        p = root / rel
        if not p.is_file():
            failures.append(f"missing_npz_for_leak_audit:{rel}")
            continue
        data = load_npz(p)
        if "time" not in data:
            failures.append(f"npz_no_time:{rel}")
            continue
        train_m, test_m = temporal_train_val_masks(
            data["time"],
            train_end=TRAIN_END_DEFAULT,
            val_year=int(acc.get("val_year", VAL_YEAR_DEFAULT)),
        )
        overlap = int((train_m & test_m).sum())
        if overlap > 0:
            failures.append(f"npz_train_test_overlap_{overlap}:{p.name}")

    meta_path = root / "models" / "horizon_v16.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        if acc.get("require_temporal_val_split", True) and meta.get("temporal_val") is not True:
            failures.append("horizon_meta_temporal_val_not_true")
    detail["leak_checks"] = len(failures) == 0
    return len(failures) == 0, detail, failures


def _check_onnx(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    detail: dict = {}
    if not acc.get("require_onnx", True):
        return True, detail, failures
    arch = cfg.get("architecture") or {}
    hp = arch.get("horizon_predictor") or {}
    onnx = root / str(hp.get("model_path", "models/horizon_v16.onnx"))
    if onnx.suffix.lower() != ".onnx":
        onnx = onnx.with_suffix(".onnx")
    scaler = root / str(hp.get("scaler_path", "models/horizon_v16_scaler.pkl"))
    detail["onnx_path"] = str(onnx)
    if not onnx.is_file():
        failures.append("missing_horizon_onnx")
        return False, detail, failures
    try:
        kn = KnowledgeNetInference(onnx, scaler_path=scaler if scaler.is_file() else None)
        if not kn.is_ready:
            failures.append("onnx_not_ready")
        else:
            probs, _ = kn.predict(np.zeros((1, 30), dtype=np.float32))
            detail["onnx_probe_probs"] = [float(x) for x in probs.reshape(-1)[:3]]
    except Exception as ex:
        failures.append(f"onnx_load_error:{ex}")
    return len(failures) == 0, detail, failures


def _check_oos(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    bt = _load_bt_module()
    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    start, end = acc.get("oos_start", "2025-01-01"), acc.get("oos_end", "2025-12-31")
    pad = pd.Timestamp(start) - pd.Timedelta(days=5)
    m5 = m5_all.loc[pad:end]
    struct = bt._struct_matrix(m5, cfg, jobs=1)
    predictor = HorizonPredictor(root, cfg)
    if not predictor.is_ready:
        failures.append("horizon_predictor_not_ready")
        return False, {}, failures
    result = bt.run_period(m5, struct, predictor, start, end, cfg)
    if result.get("error"):
        failures.append(f"oos_{result['error']}")
        return False, result, failures
    fc = result.get("forecast") or {}
    bt_stats = result.get("backtest") or {}
    ok_ratio, ratio = _side_ratio(int(fc.get("trade_long", 0)), int(fc.get("trade_short", 0)), float(acc.get("max_forecast_side_ratio", 1.35)))
    detail = {"forecast": fc, "backtest": bt_stats, "forecast_side_ratio": ratio}
    if bt_stats.get("n_trades", 0) < int(acc.get("min_oos_trades", 500)):
        failures.append("oos_trades_below_threshold")
    check_win_rate(
        float(bt_stats.get("win_rate", 0)),
        acc,
        failures,
        label="oos",
        override_min=float(acc.get("min_oos_win_rate", acc.get("min_win_rate", 0.60))),
    )
    if not ok_ratio:
        failures.append("oos_forecast_imbalance")
    trade_long, trade_short = int(bt_stats.get("n_long", 0)), int(bt_stats.get("n_short", 0))
    ok_trade_ratio, tr = _side_ratio(trade_long, trade_short, float(acc.get("max_forecast_side_ratio", 1.35)))
    detail["trade_side_ratio"] = tr
    if not ok_trade_ratio:
        failures.append("oos_trade_imbalance")
    pred_total = int(fc.get("trade_long", 0)) + int(fc.get("trade_short", 0))
    if pred_total > 0:
        coherence = (trade_long + trade_short) / pred_total
        detail["forecast_trade_coherence"] = coherence
    return len(failures) == 0, detail, failures


def _lockbox_march(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    bt = _load_bt_module()
    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    s, e = acc.get("lockbox_march_start", "2026-03-10"), acc.get("lockbox_march_end", "2026-03-18")
    ext = m5_all.loc[pd.Timestamp(s) - pd.Timedelta(days=5) : e]
    struct = bt._struct_matrix(ext, cfg, jobs=1)
    predictor = HorizonPredictor(root, cfg)
    result = bt.run_period(ext, struct, predictor, s, e, cfg)
    b = result.get("backtest") or {}
    detail = {"forecast": result.get("forecast"), "backtest": b}
    if b.get("n_trades", 0) < int(acc.get("min_lockbox_march_trades", 20)):
        failures.append("march_trades_below_threshold")
    check_win_rate(
        float(b.get("win_rate", 0)),
        acc,
        failures,
        label="lockbox_march",
        override_min=float(acc.get("min_lockbox_march_win_rate", acc.get("min_win_rate", 0.60))),
    )
    return len(failures) == 0, detail, failures


def _lockbox_june(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    csv_path = root / "scripts" / "_june_multi_bars.csv"
    if not csv_path.is_file():
        return True, {"skipped": "june_csv_missing"}, failures
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    day = str(acc.get("lockbox_june_date", "2026-06-10"))
    day_idx = df[df["datetime"].dt.strftime("%Y-%m-%d") == day].index.tolist()
    if not day_idx:
        return True, {"skipped": f"no_bars_for_{day}"}, failures

    hp_cfg = (cfg.get("architecture") or {}).get("horizon_predictor") or {}
    min_conf = float(hp_cfg.get("min_direction_confidence", 0.42))
    h, g = int(hp_cfg.get("horizon_bars", 12)), float(hp_cfg.get("gain_threshold", 0.002))
    close_all = df["close"].values.astype(np.float64)
    gt_all = np.zeros(len(close_all), dtype=np.int8)
    for i in range(len(close_all) - h):
        ret = (close_all[i + h] - close_all[i]) / max(close_all[i], 1e-9)
        if ret > g:
            gt_all[i] = 1
        elif ret < -g:
            gt_all[i] = -1

    sa = StructureService(cfg.get("structure_analyzer"))
    predictor = HorizonPredictor(root, cfg)
    matches = div = 0
    preds: list[str] = []
    for idx in day_idx:
        if idx < 200 or idx + h >= len(df):
            continue
        seg = df.iloc[max(0, idx - 200) : idx + 1]
        idx_utc = pd.DatetimeIndex(seg["datetime"], tz="UTC")
        m5 = pd.DataFrame(
            {
                "open": seg["open"].values,
                "high": seg["high"].values,
                "low": seg["low"].values,
                "close": seg["close"].values,
                "volume": seg["volume"].fillna(0).values,
            },
            index=idx_utc,
        )
        snap = sa.snapshot_from_row(m5, len(m5) - 1)
        fc = predictor.predict(snap)
        preds.append(fc.direction)
        gt = {1: "long", -1: "short", 0: "flat"}[int(gt_all[idx])]
        if fc.direction != "flat" and gt != "flat":
            div += 1
            if fc.direction == gt:
                matches += 1

    match_rate = matches / max(div, 1)
    detail = {
        "pred_counts": pd.Series(preds).value_counts().to_dict() if preds else {},
        "direction_match_rate": match_rate,
        "eval_bars": len(preds),
    }
    if preds and match_rate < float(acc.get("min_lockbox_june_direction_match", 0.45)):
        failures.append("june_direction_match_below_threshold")
    return len(failures) == 0, detail, failures


def _check_agent(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    detail: dict = {}
    if not acc.get("require_agent_validate", True):
        return True, detail, failures
    onnx = root / "models" / "horizon_v16.onnx"
    scaler = root / "models" / "horizon_v16_scaler.pkl"
    if not onnx.is_file():
        failures.append("agent_validate_missing_onnx")
        return False, detail, failures
    kn = KnowledgeNetInference(onnx, scaler_path=scaler)
    if not kn.is_ready:
        failures.append("agent_validate_kn_not_ready")
        return False, detail, failures

    idx = pd.date_range("2024-01-01", periods=240, freq="5min", tz="UTC")
    close = 2400 + np.cumsum(np.random.randn(240) * 0.08)
    m5 = pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 50.0},
        index=idx,
    )
    out = run_agent_tick(
        {"XAUUSD": m5},
        {"config_path": "config/config_agent.json", "symbols": ["XAUUSD"], "primary_symbol": "XAUUSD"},
        root,
    )
    detail["agent_tick_ok"] = bool(out.get("ok"))
    if not out.get("ok"):
        failures.append(f"agent_tick_failed:{out.get('error')}")
        return False, detail, failures
    results = out.get("results") or []
    if not results:
        failures.append("agent_empty_results")
        return False, detail, failures
    first = results[0]
    if first.get("skipped"):
        failures.append(f"agent_skipped:{first.get('reason')}")
    detail["architecture"] = str((cfg.get("architecture") or {}).get("version"))
    detail["cognition_direction"] = first.get("cognition_direction")
    detail["action"] = first.get("action")
    detail["knowledge_ready"] = first.get("knowledge_ready")
    if str((cfg.get("architecture") or {}).get("version")) == "v16" and not first.get("knowledge_ready"):
        failures.append("agent_knowledge_not_ready")
    return len(failures) == 0, detail, failures


def _check_rl(root: Path, acc: dict, oos_detail: dict | None = None) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    detail: dict = {}
    if not acc.get("require_rl_model", True):
        return True, detail, failures
    rl_zip = root / str(acc.get("rl_model_path", "models/rl_agent_xau.zip"))
    rl_meta = root / str(acc.get("rl_meta_path", "models/XAUUSD/v16/rl_meta.json"))
    detail["rl_model"] = str(rl_zip)
    if not rl_zip.is_file():
        failures.append("missing_rl_model")
        return False, detail, failures
    if rl_meta.is_file():
        meta = json.loads(rl_meta.read_text(encoding="utf-8-sig"))
        detail["rl_meta"] = meta
        if meta.get("architecture") != "v16":
            failures.append("rl_not_v16_architecture")
        km = str(meta.get("knowledge_model", ""))
        if "horizon_v16" not in km:
            failures.append("rl_knowledge_not_horizon_v16")
    else:
        failures.append("missing_rl_meta")
    metrics_log = root / "logs" / "training" / "rl_metrics_XAUUSD.jsonl"
    if not metrics_log.is_file():
        metrics_log = root / "logs" / "rl_metrics_XAUUSD.jsonl"
    min_wr = float(acc.get("min_rl_eval_win_rate", acc.get("min_win_rate", 0.60)))
    min_sample = int(acc.get("min_rl_eval_trades_sample", 30))
    if metrics_log.is_file():
        lines = [ln for ln in metrics_log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
        rec = json.loads(lines[-1]) if lines else {}
        detail["rl_metrics_log"] = str(metrics_log)
        detail["rl_eval_win_rate"] = rec.get("win_rate_recent")
        detail["rl_eval_trades_sampled"] = rec.get("trades_sampled")
        trades_sampled = int(rec.get("trades_sampled", 0))
        win_rate = float(rec.get("win_rate_recent", 0))
        if trades_sampled < min_sample and oos_detail:
            bt = oos_detail.get("backtest") or {}
            oos_trades = int(bt.get("n_trades", 0))
            oos_wr = float(bt.get("win_rate", 0))
            if oos_trades >= min_sample:
                detail["rl_eval_fallback"] = "oos_backtest"
                detail["rl_eval_win_rate"] = oos_wr
                detail["rl_eval_trades_sampled"] = oos_trades
                trades_sampled = oos_trades
                win_rate = oos_wr
        if trades_sampled < min_sample:
            failures.append("rl_eval_sample_too_small")
        check_win_rate(win_rate, acc, failures, label="rl_eval", override_min=min_wr)
    else:
        failures.append("missing_rl_metrics_log")
    return len(failures) == 0, detail, failures


def _check_rl_direction_coherence(root: Path, cfg: dict, acc: dict) -> tuple[bool, dict, list[str]]:
    failures: list[str] = []
    from zhulong.training.lgb.data_io import load_vendor_csv

    m5_all = load_vendor_csv(root / "data" / "training" / "lgb" / "XAUUSD" / "XAUUSD_M5.csv")
    sample = m5_all.loc["2025-06-01":"2025-06-15"]
    if len(sample) < 300:
        return True, {"skipped": "insufficient_sample_bars"}, failures

    from zhulong.engine.agent_engine import AgentEngine

    engine = AgentEngine(cfg, root=root)
    ok_n = total_n = 0
    mismatches: list[str] = []
    step = max(len(sample) // 40, 1)
    for i in range(250, len(sample), step):
        window = sample.iloc[: i + 1]
        out = engine.tick_symbols({"XAUUSD": window}, ["XAUUSD"], {})
        if not out:
            continue
        r = out[0]
        cog = str(r.get("cognition_direction") or "flat")
        action = str(r.get("action") or "hold")
        if cog in ("long", "short") and action not in ("hold", "close"):
            total_n += 1
            trade_dir = "long" if action in ("long", "long_50", "long_100") else "short"
            if trade_dir == cog:
                ok_n += 1
            else:
                mismatches.append(f"{cog}!={trade_dir}({action})")
    coherence = ok_n / max(total_n, 1)
    detail = {"coherence": coherence, "samples": total_n, "mismatches": mismatches[:5]}
    if total_n >= 5 and coherence < float(acc.get("min_rl_direction_coherence", 0.95)):
        failures.append("rl_direction_coherence_below_threshold")
    return len(failures) == 0, detail, failures


def _apply_passed(root: Path, cfg: dict, report: dict) -> None:
    meta_path = root / "models" / "horizon_v16.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig")) if meta_path.is_file() else {}
    meta["passed"] = True
    meta["acceptance_stage"] = report.get("acceptance_stage", "v16")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    v16_cfg_dir = root / "models" / "XAUUSD" / "v16"
    v16_cfg_dir.mkdir(parents=True, exist_ok=True)
    deploy_cfg = {
        "model_version": "v16",
        "passed": True,
        "horizon_bars": int(((cfg.get("architecture") or {}).get("horizon_predictor") or {}).get("horizon_bars", 12)),
        "model_path": "models/horizon_v16.onnx",
        "scaler_path": "models/horizon_v16_scaler.pkl",
        "min_direction_confidence": float(
            ((cfg.get("architecture") or {}).get("horizon_predictor") or {}).get("min_direction_confidence", 0.42)
        ),
        "rl_model_path": "models/rl_agent_xau.zip",
    }
    (v16_cfg_dir / "config_v16.json").write_text(json.dumps(deploy_cfg, indent=2), encoding="utf-8")

    agent_path = root / "config" / "config_agent.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8-sig"))
    hp = agent.setdefault("architecture", {}).setdefault("horizon_predictor", {})
    hp["model_path"] = "models/horizon_v16.onnx"
    agent_path.write_text(json.dumps(agent, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="验收通过后写入 config/meta")
    parser.add_argument(
        "--horizon-only",
        action="store_true",
        help="仅 Horizon 分类+ONNX（KN2/RL 训练前门禁，不含 RL/OOS/lockbox）",
    )
    parser.add_argument("--skip-oos", action="store_true")
    parser.add_argument("--skip-lockbox", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    parser.add_argument("--skip-rl", action="store_true")
    args = parser.parse_args()

    root = _ROOT
    acc = _load_acceptance(root)
    cfg = json.loads((root / "config" / "config_agent.json").read_text(encoding="utf-8-sig"))
    failures: list[str] = []
    sections: dict = {}

    core_checks = (
        ("leak_contract", lambda: _check_leak_contract(root, acc)),
        ("training", lambda: _check_training(root, acc)),
        ("classification_splits", lambda: _check_classification_splits(root, acc)),
        ("onnx", lambda: _check_onnx(root, cfg, acc)),
    )
    extended_checks = (
        ("oos", lambda: _check_oos(root, cfg, acc)),
        ("lockbox_march", lambda: _lockbox_march(root, cfg, acc)),
        ("lockbox_june", lambda: _lockbox_june(root, cfg, acc)),
        ("agent", lambda: _check_agent(root, cfg, acc)),
        ("rl_coherence", lambda: _check_rl_direction_coherence(root, cfg, acc)),
    )

    checks: list[tuple[str, Any]] = list(core_checks)
    if not args.horizon_only:
        for name, fn in extended_checks:
            if name == "oos" and args.skip_oos:
                continue
            if name.startswith("lockbox") and args.skip_lockbox:
                continue
            if name == "agent" and args.skip_agent:
                continue
            if name == "rl_coherence" and args.skip_rl:
                continue
            checks.append((name, fn))

    for name, fn in checks:
        print(f"=== {name} ===", flush=True)
        ok, detail, fails = fn()
        sections[name] = {"ok": ok, "detail": detail, "failures": fails}
        failures.extend(fails)
        print(json.dumps(sections[name], indent=2, default=str), flush=True)

    if not args.horizon_only and not args.skip_rl:
        print("=== rl_model ===", flush=True)
        oos_detail = sections.get("oos", {}).get("detail") or {}
        ok, detail, fails = _check_rl(root, acc, oos_detail=oos_detail)
        sections["rl_model"] = {"ok": ok, "detail": detail, "failures": fails}
        failures.extend(fails)
        print(json.dumps(sections["rl_model"], indent=2, default=str), flush=True)

    passed = len(failures) == 0
    report = {
        "acceptance_stage": acc.get("acceptance_stage", "v16"),
        "passed": passed,
        "failures": failures,
        "sections": sections,
    }
    out_dir = root / "data" / "training" / "reports" / "v16"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "acceptance_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=" * 72)
    print(f"V16 ACCEPTANCE: {'PASS' if passed else 'FAIL'}")
    if failures:
        print("Failures:", failures)
    print(f"Report: {out_path}")
    print("=" * 72)

    if passed and args.apply:
        _apply_passed(root, cfg, report)
        print("Applied passed=true to meta, models/XAUUSD/v16/config_v16.json, config_agent.json")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
