"""
模型推理：加载 transformer_encoder.pth、xgb_classifier.json、xgb_regressor.json、scaler.pkl。
G1 MVP：expected_return = confidence * historical_avg_gain
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from zhulong.utils.paths import model_dir_for_symbol

logger = logging.getLogger(__name__)

_TransformerEncoderCls = None


def _manifest_is_v14(manifest: dict) -> bool:
    """正式模型：V14 XGBoost 方向预测（XAUUSD / USOIL）。"""
    mv = str(manifest.get("model_version", "")).lower()
    stage = str(manifest.get("acceptance_stage", "")).lower()
    mode = str(manifest.get("classifier_mode", "")).lower()
    return mv == "v14" or stage == "v14" or mode in ("xau_v14", "oil_v14")


def _transformer_encoder_class():
    """延迟加载 torch — V14 tabular 推理不需要 PyTorch。"""
    global _TransformerEncoderCls
    if _TransformerEncoderCls is not None:
        return _TransformerEncoderCls

    import torch
    import torch.nn as nn

    class TransformerEncoder(nn.Module):
        def __init__(self, feature_dim: int = 30, d_model: int = 128, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(feature_dim, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, 60, d_model) * 0.1)
            layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.output_proj = nn.Linear(d_model, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_proj(x) + self.pos_encoding[:, : x.size(1), :]
            x = self.transformer(x)
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            return self.output_proj(x)

    _TransformerEncoderCls = TransformerEncoder
    return _TransformerEncoderCls


def TransformerEncoder(*args, **kwargs):
    """兼容 create_demo_models / 旧脚本（延迟加载 torch）。"""
    return _transformer_encoder_class()(*args, **kwargs)


class InferenceEngine:
    ARTIFACTS = (
        "transformer_encoder.pth",
        "xgb_regressor.json",
        "scaler.pkl",
    )
    ARTIFACTS_DUAL = ("xgb_classifier_long.json", "xgb_classifier_short.json")
    ARTIFACTS_LEGACY = ("xgb_classifier.json",)

    def __init__(self, config: dict) -> None:
        self._cfg = config
        self._models: dict[str, dict] = {}
        self._gain_history: dict[str, deque] = {}
        self._window = int(config.get("historical_avg_gain_window", 100))

    def validate_symbol_models(self, symbol: str) -> bool:
        d = model_dir_for_symbol(symbol)
        manifest_path = d / "manifest.json"
        if not manifest_path.is_file():
            logger.error("品种 %s 缺少 manifest.json", symbol)
            return False

        import json

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as ex:
            logger.error("manifest 解析失败 %s: %s", symbol, ex)
            return False

        feature_mode = manifest.get("feature_mode", "transformer")
        classifier_mode = manifest.get("classifier_mode", "legacy")
        if classifier_mode == "oil_v1":
            kind = manifest.get("kind", "")
            if kind == "demo":
                logger.error("品种 %s oil_v1 不可用 demo", symbol)
                return False
            if not manifest.get("acceptance_passed"):
                logger.error("品种 %s 模型未通过验收，拒绝加载", symbol)
                return False
            from zhulong.oil_v1_live import validate_oil_v1_artifacts

            return validate_oil_v1_artifacts(symbol)

        if _manifest_is_v14(manifest):
            from zhulong.v14_live import validate_v14_artifacts

            return validate_v14_artifacts(symbol)

        if feature_mode == "transformer":
            missing = [a for a in self.ARTIFACTS if not (d / a).is_file()]
            if missing:
                logger.error("品种 %s 缺少模型文件: %s", symbol, missing)
                return False
        else:
            for a in ("scaler.pkl", "xgb_regressor.json"):
                if not (d / a).is_file():
                    logger.error("品种 %s 缺少模型文件: %s", symbol, a)
                    return False

        kind = manifest.get("kind", "")
        if kind == "demo":
            dual_ok = all((d / a).is_file() for a in self.ARTIFACTS_DUAL)
            legacy_ok = (d / "xgb_classifier.json").is_file()
            if not dual_ok and not legacy_ok:
                logger.error("品种 %s demo 缺少分类器文件", symbol)
                return False
            return True

        if not manifest.get("acceptance_passed"):
            logger.error("品种 %s 模型未通过验收，拒绝加载", symbol)
            return False

        mode = manifest.get("classifier_mode", "legacy")
        if mode == "oil_v1":
            from zhulong.oil_v1_live import validate_oil_v1_artifacts

            return validate_oil_v1_artifacts(symbol)
        if _manifest_is_v14(manifest):
            from zhulong.v14_live import validate_v14_artifacts

            return validate_v14_artifacts(symbol)
        if mode == "dual_binary":
            dual_missing = [a for a in self.ARTIFACTS_DUAL if not (d / a).is_file()]
            if dual_missing:
                logger.error("品种 %s 缺少双分类器: %s", symbol, dual_missing)
                return False
        elif not (d / "xgb_classifier.json").is_file():
            logger.error("品种 %s 缺少 xgb_classifier.json", symbol)
            return False
        return True

    def load(self, symbol: str) -> None:
        if not self.validate_symbol_models(symbol):
            raise FileNotFoundError(f"models/{symbol} 模型不完整，拒绝加载（G3）")

        d = model_dir_for_symbol(symbol)
        import json

        manifest = {}
        mp = d / "manifest.json"
        if mp.is_file():
            manifest = json.loads(mp.read_text(encoding="utf-8"))

        feature_mode = manifest.get("feature_mode", "transformer")
        mode = manifest.get("classifier_mode", "legacy")

        if mode == "oil_v1":
            from zhulong.oil_v1_live import load_oil_v1_bundle

            bundle = load_oil_v1_bundle(symbol)
            self._models[symbol] = {
                "mode": "oil_v1",
                "feature_mode": "oil_v1_tabular",
                "oil_v1": bundle,
            }
            self._gain_history.setdefault(symbol, deque(maxlen=self._window))
            logger.info("已加载 oil v1 模型: %s", symbol)
            return

        if _manifest_is_v14(manifest):
            from zhulong.v14_live import load_v14_bundle

            bundle = load_v14_bundle(symbol, model_subdir="v14")
            self._models[symbol] = {
                "mode": "v14",
                "feature_mode": "v13_tabular",
                "v14": bundle,
                "prob_threshold": float(manifest.get("long_threshold", 0.70)),
            }
            self._gain_history.setdefault(symbol, deque(maxlen=self._window))
            logger.info("已加载 v14 模型: %s", symbol)
            return

        scaler = joblib.load(d / "scaler.pkl")
        reg = xgb.XGBRegressor()
        reg.load_model(str(d / "xgb_regressor.json"))

        encoder = None
        feature_dim = int(manifest.get("feature_dim", self._cfg.get("feature_dim_5min", 30)))
        if feature_mode == "transformer":
            import torch

            TransformerEncoder = _transformer_encoder_class()
            state = torch.load(d / "transformer_encoder.pth", map_location="cpu")
            if "input_proj.weight" in state:
                feature_dim = int(state["input_proj.weight"].shape[1])
            encoder = TransformerEncoder(feature_dim=feature_dim)
            encoder.load_state_dict(state)
            encoder.eval()

        mode = manifest.get("classifier_mode", "legacy")
        if mode == "dual_binary":
            clf_long = xgb.XGBClassifier()
            clf_short = xgb.XGBClassifier()
            clf_long.load_model(str(d / "xgb_classifier_long.json"))
            clf_short.load_model(str(d / "xgb_classifier_short.json"))
            clf = None
        else:
            clf = xgb.XGBClassifier()
            clf.load_model(str(d / "xgb_classifier.json"))
            clf_long = clf_short = None

        self._models[symbol] = {
            "encoder": encoder,
            "scaler": scaler,
            "clf": clf,
            "clf_long": clf_long,
            "clf_short": clf_short,
            "reg": reg,
            "mode": mode,
            "feature_mode": feature_mode,
            "feature_dim": feature_dim,
            "prob_threshold": float(manifest.get("prob_threshold", self._cfg.get("prob_threshold", 0.60))),
        }
        self._gain_history.setdefault(symbol, deque(maxlen=self._window))
        logger.info("已加载模型: %s", symbol)

    def _transform_seq(self, symbol: str, seq: np.ndarray) -> np.ndarray:
        scaler = self._models[symbol]["scaler"]
        flat = seq.reshape(-1, seq.shape[-1])
        scaled = scaler.transform(flat).reshape(seq.shape)
        return scaled.astype(np.float32)

    def predict(
        self,
        symbol: str,
        seq: np.ndarray,
        hourly: np.ndarray,
        macro: np.ndarray,
        mtf: np.ndarray | None = None,
        m5: pd.DataFrame | None = None,
    ) -> dict:
        if symbol not in self._models:
            self.load(symbol)

        m = self._models[symbol]
        if m.get("mode") == "oil_v1":
            from zhulong.oil_v1_live import predict_oil_v1

            return predict_oil_v1(symbol, m["oil_v1"], m5=m5)

        if m.get("mode") == "v14":
            from zhulong.v14_live import build_live_v14_features, predict_v14

            bundle = m["v14"]
            if m5 is None or len(m5) < 30:
                raise RuntimeError(f"{symbol} V14 推理需要 M5 数据")
            row, cols, m5_df, feats_df = build_live_v14_features(symbol, m5=m5)
            sig = predict_v14(bundle, row, m5_df, feats_df=feats_df)
            direction = 1 if sig.direction == "buy" else (-1 if sig.direction == "sell" else 0)
            proba = sig.probabilities or [0.0, 0.0, 0.0]
            return {
                "direction": direction,
                "confidence": float(sig.confidence),
                "entry_offset": 0.0,
                "expected_return": float(sig.confidence) * 0.25 if direction != 0 else 0.0,
                "probabilities": proba,
            }

        m = self._models[symbol]
        fd = int(m.get("feature_dim", seq.shape[-1]))
        if seq.shape[-1] > fd:
            seq = seq[:, :fd].astype(np.float32, copy=False)

        if m.get("feature_mode") == "sequence_stats":
            from zhulong.feature_engine import build_fused_row

            window = seq.astype(np.float32)
            if mtf is None:
                mtf = np.zeros(6, dtype=np.float32)
            raw = build_fused_row(window, mtf, hourly.astype(np.float32))
            fused = m.scaler.transform(raw.reshape(1, -1))
        else:
            import torch

            seq_s = self._transform_seq(symbol, seq)
            with torch.no_grad():
                emb = m["encoder"](torch.from_numpy(seq_s).unsqueeze(0)).numpy()[0]
            fused = np.concatenate([emb, hourly, macro]).reshape(1, -1)

        if m.get("mode") == "dual_binary":
            p_long = float(m["clf_long"].predict_proba(fused)[0, 1])
            p_short = float(m["clf_short"].predict_proba(fused)[0, 1])
            threshold = float(m.get("prob_threshold", self._cfg.get("prob_threshold", 0.60)))
            short_max = 0.35
            if p_long >= threshold and p_long >= p_short and p_short <= short_max:
                direction, confidence = 1, p_long
            elif p_short >= threshold and p_short > p_long and p_long <= short_max:
                direction, confidence = -1, p_short
            else:
                direction, confidence = 0, max(p_long, p_short, 1.0 - p_long - p_short)
            proba = [p_short, max(0.05, 1.0 - p_long - p_short), p_long]
        else:
            proba = m["clf"].predict_proba(fused)[0]
            class_idx = int(np.argmax(proba))
            direction_map = {0: -1, 1: 0, 2: 1}
            direction = direction_map.get(class_idx, 0)
            confidence = float(proba[class_idx])

        entry_offset = 0.0
        if direction != 0:
            entry_offset = float(m["reg"].predict(fused)[0])

        expected_return = self._expected_return(symbol, direction, confidence, fused, m)

        return {
            "direction": direction,
            "confidence": confidence,
            "entry_offset": entry_offset,
            "expected_return": expected_return,
            "probabilities": proba.tolist(),
        }

    def _expected_return(self, symbol: str, direction: int, confidence: float, fused, m) -> float:
        if self._cfg.get("use_xgb_expected_return") and "reg_er" in m:
            return float(m["reg_er"].predict(fused)[0])

        hist = self._gain_history[symbol]
        if hist:
            avg_gain = float(np.mean(list(hist)))
        else:
            avg_gain = 0.25  # 默认 0.25% 占位，训练后由历史正例替换
        return confidence * avg_gain

    def record_positive_gain(self, symbol: str, gain_pct: float) -> None:
        self._gain_history.setdefault(symbol, deque(maxlen=self._window)).append(abs(gain_pct))
