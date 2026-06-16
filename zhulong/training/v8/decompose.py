"""VMD / CEEMDAN 多尺度分解（离线，结果对齐 M5）。"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_K = 6
DEFAULT_ALPHA = 2000.0


def _vmd_series(close: np.ndarray, K: int = DEFAULT_K, alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    from vmdpy import VMD

    u, _, _ = VMD(close.astype(np.float64), alpha, 0, K, 0, 1, 1e-7)
    return u  # (K, N)


def decompose_h4_to_m5(
    m5: pd.DataFrame,
    K: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    rule: str = "4h",
) -> pd.DataFrame:
    """
    在 H4 收盘价上运行 VMD，再 forward-fill 到 M5。
    比全量 M5 CEEMDAN 快几个数量级，适合 10 年数据。
    """
    h4 = (
        m5["close"]
        .resample(rule, label="right", closed="right")
        .last()
        .dropna()
    )
    if len(h4) < K + 10:
        raise ValueError(f"not enough H4 bars for VMD: {len(h4)}")

    logger.info("VMD on %s bars (K=%s)", len(h4), K)
    u = _vmd_series(h4.to_numpy(), K=K, alpha=alpha)
    imf_h4 = pd.DataFrame({f"IMF_{i + 1}": u[i] for i in range(K)}, index=h4.index)
    imf_h4["residual"] = h4.to_numpy() - u.sum(axis=0)
    imf_m5 = imf_h4.reindex(m5.index, method="ffill")
    logger.info("IMF columns aligned to M5: %s", list(imf_m5.columns))
    return imf_m5


def decompose_ceemdan_chunked(
    close: np.ndarray,
    chunk: int = 4096,
    step: int = 2048,
    max_imf: int = 6,
) -> pd.DataFrame:
    """分块 CEEMDAN（可选，较慢）。"""
    from PyEMD import CEEMDAN

    n = len(close)
    acc = np.zeros((max_imf + 1, n), dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    ce = CEEMDAN()
    for start in range(0, n - chunk + 1, step):
        end = start + chunk
        seg = close[start:end]
        imfs = ce.ceemdan(seg)
        k = min(max_imf, len(imfs))
        for i in range(k):
            acc[i, start:end] += imfs[i]
        acc[max_imf, start:end] += seg - imfs.sum(axis=0)
        weight[start:end] += 1.0
    weight = np.where(weight <= 0, 1.0, weight)
    cols = {f"IMF_{i + 1}": acc[i] / weight for i in range(max_imf)}
    cols["residual"] = acc[max_imf] / weight
    return pd.DataFrame(cols)


def save_decomposition(imf_m5: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imf_m5.to_parquet(path)
    logger.info("wrote decomposition %s (%s rows)", path, len(imf_m5))


def load_decomposition(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
