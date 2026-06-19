"""L2 1 小时方向预测：对称规则，预测即方向。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.rl_agent import _resolve_model_path
from zhulong.agent.tick_brief import HorizonForecast, StructureSnapshot


def direction_from_probs(
    short_p: float,
    flat_p: float,
    long_p: float,
    *,
    min_confidence: float = 0.42,
    flat_scale: float = 1.0,
    dir_margin: float = 0.0,
) -> tuple[str, float, float, float]:
    """对称：long vs short 谁大跟谁；未达门槛 → flat。可选 flat_scale/dir_margin 验证集校准。"""
    s, f, l = float(short_p), float(flat_p) * float(flat_scale), float(long_p)
    total = max(s + f + l, 1e-9)
    s, f, l = s / total, f / total, l / total
    trade_conf = max(l, s)
    if dir_margin > 0:
        if trade_conf < f + dir_margin:
            return "flat", l, s, f
    elif trade_conf < min_confidence:
        return "flat", l, s, f
    if l > s:
        return "long", l, s, f
    if s > l:
        return "short", l, s, f
    return "flat", l, s, f


def _load_horizon_calibration(root: Path, model_path: Path) -> dict[str, float]:
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.is_file():
        return {}
    try:
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        cal = meta.get("calibration") or {}
        return {
            "flat_scale": float(cal.get("flat_scale", 1.0)),
            "dir_margin": float(cal.get("dir_margin", 0.0)),
        }
    except Exception:
        return {}


class HorizonPredictor:
    def __init__(
        self,
        root: Path,
        config: dict[str, Any],
    ) -> None:
        self.root = root
        arch = config.get("architecture") or {}
        hp = arch.get("horizon_predictor") or {}
        self.horizon_bars = int(hp.get("horizon_bars", 12))
        self.gain_threshold = float(hp.get("gain_threshold", 0.002))
        self.min_confidence = float(hp.get("min_direction_confidence", 0.42))
        self.model_id = str(hp.get("model_id", "horizon_v16"))

        model_path = _resolve_model_path(str(hp.get("model_path", "models/horizon_v16.pth")), root)
        scaler_path = _resolve_model_path(str(hp.get("scaler_path", "models/horizon_v16_scaler.pkl")), root)
        onnx = model_path.with_suffix(".onnx")
        load_path = onnx if onnx.is_file() else model_path
        pth = model_path.with_suffix(".pth")
        self._calibration = _load_horizon_calibration(root, model_path)
        self._last_embedding: np.ndarray | None = None
        self._kn: KnowledgeNetInference | None = None
        self.resolved_model_path = load_path.resolve() if load_path.is_file() else model_path.resolve()
        self.resolved_scaler_path = scaler_path.resolve() if scaler_path.is_file() else None
        self.load_error: str | None = None
        if not load_path.is_file() and not pth.is_file():
            self.load_error = f"model_missing:{model_path}"
        else:
            import logging

            log = logging.getLogger(__name__)
            sc = scaler_path if scaler_path.is_file() else None
            candidates: list[tuple[Path, bool]] = []
            if load_path.is_file():
                allow_pt = load_path.suffix.lower() in {".pth", ".pt"}
                candidates.append((load_path, allow_pt))
            if pth.is_file() and all(pth.resolve() != c[0].resolve() for c in candidates):
                candidates.append((pth, True))
            if not candidates and pth.is_file():
                candidates.append((pth, True))
            try:
                for cand, allow_pt in candidates:
                    try:
                        kn = KnowledgeNetInference(
                            cand, scaler_path=sc, allow_pytorch=allow_pt
                        )
                    except TypeError:
                        kn = KnowledgeNetInference(cand, scaler_path=sc)
                    if kn is not None and kn.is_ready:
                        self._kn = kn
                        if allow_pt:
                            log.warning(
                                "Horizon 使用 PyTorch fallback: %s (ONNX 不可用)",
                                cand,
                            )
                        break
                if self._kn is None or not self._kn.is_ready:
                    kn_err = getattr(self._kn, "_onnx_load_error", None) if self._kn else None
                    self.load_error = kn_err or self.load_error or "onnx_session_not_ready"
            except Exception as ex:
                self.load_error = f"{type(ex).__name__}:{ex}"
                log.warning("Horizon model load failed: %s", ex)
                self._kn = None

    @property
    def is_ready(self) -> bool:
        return self._kn is not None and self._kn.is_ready

    def predict(self, snapshot: StructureSnapshot) -> HorizonForecast:
        x = np.asarray(snapshot.vector, dtype=np.float32).reshape(1, -1)
        if self._kn is not None and self._kn.is_ready:
            probs, emb_arr = self._kn.predict(x)
            if emb_arr is not None:
                self._last_embedding = np.asarray(emb_arr, dtype=np.float32).reshape(-1)
            else:
                self._last_embedding = np.zeros(32, dtype=np.float32)
            p = probs[0] if probs.ndim > 1 else probs
            if p.size >= 3:
                short_p, flat_p, long_p = float(p[0]), float(p[1]), float(p[2])
            else:
                short_p, flat_p, long_p = 0.33, 0.34, 0.33
        else:
            short_p, flat_p, long_p = self._heuristic(snapshot)
            self._last_embedding = np.zeros(32, dtype=np.float32)

        direction, pu, pd, pf = direction_from_probs(
            short_p,
            flat_p,
            long_p,
            min_confidence=self.min_confidence,
            flat_scale=self._calibration.get("flat_scale", 1.0),
            dir_margin=self._calibration.get("dir_margin", 0.0),
        )
        conf = max(pu, pd) if direction != "flat" else pf
        return HorizonForecast(
            horizon_bars=self.horizon_bars,
            gain_threshold=self.gain_threshold,
            prob_up=pu,
            prob_down=pd,
            prob_flat=pf,
            direction=direction,
            confidence=conf,
            model_id=self.model_id,
        )

    @staticmethod
    def _heuristic(snap: StructureSnapshot) -> tuple[float, float, float]:
        t = snap.m5_trend
        a = snap.mtf_align
        score = np.clip(t + 0.3 * a, -1.0, 1.0)
        if score > 0.15:
            return 0.2, 0.25, 0.55
        if score < -0.15:
            return 0.55, 0.25, 0.2
        return 0.33, 0.34, 0.33
