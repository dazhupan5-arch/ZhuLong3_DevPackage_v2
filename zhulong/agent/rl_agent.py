"""RL 智能体推理封装（PPO 加载 + 确定性决策）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from zhulong.agent.state_builder import LEGACY_STATE_DIM, STATE_DIM


def _resolve_model_path(rel: str | Path, root: Path) -> Path:
    from zhulong.utils.paths import appdata_dir

    p = Path(rel)
    if p.is_absolute():
        return p
    install = root / p
    if install.is_file():
        return install
    appdata = appdata_dir() / p
    if appdata.is_file():
        return appdata
    return install

logger = logging.getLogger(__name__)

ACTION_NAMES = ["hold", "long", "short", "short_50", "short_100", "close"]


class RlAgent:
    def __init__(
        self,
        model_path: str | Path,
        *,
        deterministic: bool = True,
        symbol: str = "XAUUSD",
    ) -> None:
        self.model_path = Path(model_path)
        self.deterministic = deterministic
        self.symbol = symbol.upper()
        self._model = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.is_file() and not Path(str(self.model_path) + ".zip").is_file():
            logger.warning("RL 模型缺失 %s", self.model_path)
            return
        try:
            from stable_baselines3 import PPO

            p = str(self.model_path)
            if not p.endswith(".zip") and Path(p + ".zip").is_file():
                p = p + ".zip"
            self._model = PPO.load(p, device="cpu")
            logger.info("RL 模型已加载 %s", p)
        except Exception as ex:
            logger.warning("RL 加载失败: %s", ex)
            self._model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def observation_dim(self) -> int:
        if self._model is None:
            return STATE_DIM
        try:
            return int(self._model.observation_space.shape[0])
        except Exception:
            return LEGACY_STATE_DIM

    def predict(self, state: np.ndarray) -> tuple[int, float]:
        """返回 (action_id, confidence_placeholder)。"""
        if self._model is None:
            return 0, 0.5
        dim = self.observation_dim
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.size > dim:
            arr = arr[:dim]
        elif arr.size < dim:
            arr = np.pad(arr, (0, dim - arr.size))
        action, _ = self._model.predict(arr.reshape(1, -1), deterministic=self.deterministic)
        return int(action[0]), 0.85

    def action_name(self, action_id: int) -> str:
        if 0 <= action_id < len(ACTION_NAMES):
            return ACTION_NAMES[action_id]
        return "hold"


def _symbol_block(symbol: str, config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("symbols") or {}).get(symbol.strip().upper()) or {}


def resolve_rl_model_path(symbol: str, config: dict[str, Any], root: Path) -> Path:
    sym = symbol.upper()
    sym_cfg = _symbol_block(sym, config)
    rl_sym = sym_cfg.get("rl") or {}
    rl_cfg = config.get("rl") or {}
    if sym == "USOIL":
        candidates = [
            rl_sym.get("model_path"),
            rl_cfg.get("model_path_oil"),
            rl_cfg.get("model_oil"),
            "models/rl_agent_oil",
            "models/rl_agent_oil.zip",
        ]
    else:
        candidates = [
            rl_sym.get("model_path"),
            rl_cfg.get("model_path_xau"),
            rl_cfg.get("model_path"),
            "models/rl_agent_xau",
            "models/rl_agent_xau.zip",
            "models/XAUUSD/rl_agent_xau",
        ]
    for rel in candidates:
        if not rel:
            continue
        p = _resolve_model_path(rel, root)
        if p.is_file() or p.with_suffix(".zip").is_file():
            return p
        if p.is_dir() and (p / "policy.pth").is_file():
            return p
    rel = candidates[2] if len(candidates) > 2 and candidates[2] else "models/rl_agent_xau"
    p = Path(rel)
    return p if p.is_absolute() else root / p


def resolve_knowledge_paths(symbol: str, config: dict[str, Any], root: Path) -> tuple[Path, Path | None]:
    arch = config.get("architecture") or {}
    if str(arch.get("version")) == "v16":
        hp = arch.get("horizon_predictor") or {}
        model = hp.get("model_path") or "models/horizon_v16.onnx"
        scaler = hp.get("scaler_path") or "models/horizon_v16_scaler.pkl"
        mp = _resolve_model_path(model, root)
        onnx_c = mp if mp.suffix.lower() == ".onnx" else mp.with_suffix(".onnx")
        if onnx_c.is_file():
            mp = onnx_c
        elif not mp.is_file():
            pth = mp.with_suffix(".pth") if mp.suffix.lower() != ".pth" else mp
            if pth.is_file():
                mp = pth
        sp = _resolve_model_path(scaler, root)
        return mp, sp

    kn_cfg = config.get("knowledge_net") or {}
    sym = symbol.upper()
    sym_cfg = _symbol_block(sym, config)
    kn_sym = sym_cfg.get("knowledge_net") or {}
    if sym == "USOIL":
        model = (
            kn_sym.get("model_path")
            or kn_cfg.get("model_path_oil")
            or "models/knowledge_net_oil.onnx"
        )
        scaler = kn_sym.get("scaler_path") or kn_cfg.get("scaler_path_oil") or "models/knowledge_scaler_oil.pkl"
        fallback = (
            kn_sym.get("model_path_pytorch")
            or kn_cfg.get("model_path_oil_pytorch")
            or "models/knowledge_net_oil.pth"
        )
    else:
        model = kn_sym.get("model_path") or kn_cfg.get("model_path") or "models/knowledge_net.onnx"
        scaler = kn_sym.get("scaler_path") or kn_cfg.get("scaler_path") or "models/knowledge_scaler.pkl"
        fallback = kn_sym.get("model_path_pytorch") or kn_cfg.get("model_path_pytorch") or "models/knowledge_net.pth"
    mp = Path(model)
    mp = mp if mp.is_absolute() else root / mp
    fb = Path(fallback)
    fb = fb if fb.is_absolute() else root / fb
    onnx_candidate = mp if mp.suffix.lower() == ".onnx" else mp.with_suffix(".onnx")
    if onnx_candidate.is_file():
        mp = onnx_candidate
    elif not mp.is_file() and fb.is_file():
        mp = fb
    sp = Path(scaler)
    return mp, (sp if sp.is_absolute() else root / sp)
