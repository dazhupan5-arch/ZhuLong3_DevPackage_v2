"""LSTM 滑窗数据集：60 根 OHLCV → 24 根盈亏标签。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from zhulong.training.lgb.data_io import load_vendor_csv
from zhulong.training.lgb.labels_profit import generate_profit_labels
from zhulong.training.lgb.splits import split_indices

logger = logging.getLogger(__name__)

SEQ_LEN = 60
MAX_HOLD_BARS = 24
FEATURES = ["open", "high", "low", "close", "tick_volume"]


def _ohlcv_matrix(m5: pd.DataFrame) -> np.ndarray:
    vol = m5["volume"] if "volume" in m5.columns else m5.get("tick_volume", 0)
    return np.column_stack(
        [
            m5["open"].to_numpy(dtype=np.float32),
            m5["high"].to_numpy(dtype=np.float32),
            m5["low"].to_numpy(dtype=np.float32),
            m5["close"].to_numpy(dtype=np.float32),
            vol.to_numpy(dtype=np.float32),
        ]
    )


def zscore_windows(X: np.ndarray) -> np.ndarray:
    """对每个样本窗口、每个特征通道做 Z-score。"""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def build_windows(
    m5: pd.DataFrame,
    labels: pd.Series,
    seq_len: int = SEQ_LEN,
    max_hold: int = MAX_HOLD_BARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 X (N, seq_len, 5), y (N,), bar_indices (N,) 对齐 m5 行号。
    窗口结束于 bar i；标签取自 labels.iloc[i]。
    """
    ohlcv = _ohlcv_matrix(m5)
    lab = labels.reindex(m5.index).fillna(0).astype(np.int8).to_numpy()
    n = len(m5)

    # windows[k] = ohlcv[k : k+seq_len]，结束于 bar k+seq_len-1
    raw = sliding_window_view(ohlcv, seq_len, axis=0)  # (n-seq_len+1, 5, seq_len)
    windows = np.transpose(raw, (0, 2, 1)).astype(np.float32)  # (n-seq_len+1, seq_len, 5)
    end_ix = np.arange(seq_len - 1, n, dtype=np.int64)
    y = lab[end_ix]
    # 末尾 max_hold 根无法可靠模拟
    valid = end_ix < (n - max_hold)
    windows = windows[valid]
    y = y[valid]
    end_ix = end_ix[valid]
    return zscore_windows(windows), y.astype(np.float32), end_ix


def prepare_lstm_splits(
    m5_path: Path,
    labels_path: Path | None,
    out_dir: Path,
    symbol: str = "XAUUSD",
    seq_len: int = SEQ_LEN,
    max_hold: int = MAX_HOLD_BARS,
) -> dict[str, int]:
    m5 = load_vendor_csv(m5_path)
    if labels_path and labels_path.is_file():
        lab_df = pd.read_csv(labels_path, index_col=0, parse_dates=True)
        labels = lab_df["label"]
        logger.info("labels from %s pos=%.2f%%", labels_path, 100 * (labels == 1).mean())
    else:
        lab_df = generate_profit_labels(m5, max_hold_bars=max_hold)
        labels = lab_df["label"]

    X_all, y_all, bar_ix = build_windows(m5, labels, seq_len, max_hold)
    times = m5.index[bar_ix]
    splits = split_indices(m5.index)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    split_map = {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test1,
    }
    for name, ix in split_map.items():
        mask = times.isin(ix)
        X, y, t = X_all[mask], y_all[mask], times[mask]
        t_sec = (t.asi8 // 1_000_000_000).astype(np.int64)
        np.savez_compressed(
            out_dir / f"{name}.npz",
            X=X,
            y=y,
            times=t_sec,
        )
        counts[name] = len(y)
        logger.info(
            "%s: n=%s pos=%.2f%%",
            name,
            len(y),
            100.0 * y.mean() if len(y) else 0.0,
        )

    meta = {
        "symbol": symbol,
        "seq_len": seq_len,
        "max_hold_bars": max_hold,
        "features": FEATURES,
        "normalization": "per_window_zscore",
        "counts": counts,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return counts
