"""多周期结构特征（30 维，确定性算法）。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from zhulong.strategies.indicators import adx_series, atr_series, ema

FEATURE_DIM = 30

FEATURE_NAMES: tuple[str, ...] = (
    # M5 结构（0–20）
    "m5_trend",
    "m5_adx",
    "m5_price_to_ema",
    "m5_support_dist",
    "m5_resistance_dist",
    "m5_support_strength",
    "m5_resistance_strength",
    "double_top",
    "double_bottom",
    "head_shoulders_top",
    "inverse_head_shoulders",
    "rsi_bull_div",
    "rsi_bear_div",
    "macd_bull_div",
    "macd_bear_div",
    "volume_ratio",
    "volatility_regime",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    # 多周期结构（21–29）
    "m15_trend",
    "m15_adx",
    "h1_trend",
    "h1_adx",
    "h1_rsi",
    "h4_trend",
    "mtf_trend_align",
    "h1_support_dist",
    "h4_resistance_dist",
)


def resample_ohlcv(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols = [c for c in agg if c in m5.columns]
    return m5[cols].resample(rule).agg({c: agg[c] for c in cols}).dropna()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd = ema12 - ema26
    signal = ema(macd, 9)
    return macd, signal


def zigzag_points(m5: pd.DataFrame, atr_mult: float = 0.5) -> list[tuple[pd.Timestamp, float, str]]:
    """返回 (time, price, 'peak'|'valley') 列表。"""
    if len(m5) < 20:
        return []
    atr = atr_series(m5).bfill().fillna(m5["close"] * 0.001)
    threshold = float(atr.iloc[-1]) * atr_mult
    if threshold <= 0:
        threshold = float(m5["close"].iloc[-1]) * 0.001

    pts: list[tuple[pd.Timestamp, float, str]] = []
    direction = 0  # 1 up, -1 down
    last_extreme_idx = 0
    last_extreme_price = float(m5["close"].iloc[0])

    for i in range(1, len(m5)):
        high = float(m5["high"].iloc[i])
        low = float(m5["low"].iloc[i])

        if direction >= 0 and high - last_extreme_price >= threshold:
            if direction == -1:
                pts.append((m5.index[last_extreme_idx], last_extreme_price, "valley"))
            direction = 1
            last_extreme_idx = i
            last_extreme_price = high
        elif direction <= 0 and last_extreme_price - low >= threshold:
            if direction == 1:
                pts.append((m5.index[last_extreme_idx], last_extreme_price, "peak"))
            direction = -1
            last_extreme_idx = i
            last_extreme_price = low

    if direction == 1:
        pts.append((m5.index[last_extreme_idx], last_extreme_price, "peak"))
    elif direction == -1:
        pts.append((m5.index[last_extreme_idx], last_extreme_price, "valley"))
    return pts


def cluster_levels(prices: list[float], ref_price: float, eps_pct: float = 0.02) -> list[tuple[float, int]]:
    if len(prices) < 2:
        return [(p, 1) for p in prices[:3]]
    arr = np.array(prices, dtype=float).reshape(-1, 1)
    eps = max(ref_price * eps_pct, 1e-6)
    labels = DBSCAN(eps=eps, min_samples=2).fit(arr).labels_
    clusters: dict[int, list[float]] = {}
    for lbl, p in zip(labels, prices):
        if lbl < 0:
            continue
        clusters.setdefault(int(lbl), []).append(float(p))
    out = [(float(np.mean(v)), len(v)) for v in clusters.values()]
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:3]


def _nearest_support_resistance(
    close: float, atr: float, peaks: list[tuple[pd.Timestamp, float, str]]
) -> tuple[float, float, float, float]:
    valleys = [p for _, p, t in peaks if t == "valley"]
    peak_prices = [p for _, p, t in peaks if t == "peak"]
    ref = close if close > 0 else 1.0
    supports = cluster_levels(valleys[-100:], ref) if valleys else []
    resistances = cluster_levels(peak_prices[-100:], ref) if peak_prices else []

    sup = max((s for s, _ in supports if s <= close), default=close - atr)
    res = min((r for r, _ in resistances if r >= close), default=close + atr)
    sup_str = float(supports[0][1] / 5.0) if supports else 0.0
    res_str = float(resistances[0][1] / 5.0) if resistances else 0.0
    if atr <= 0:
        atr = ref * 0.001
    return sup, res, min(sup_str, 1.0), min(res_str, 1.0)


def _pattern_flags(zig: list[tuple[pd.Timestamp, float, str]], close: float) -> tuple[float, float, float, float]:
    peaks = [(t, p) for t, p, k in zig if k == "peak"]
    valleys = [(t, p) for t, p, k in zig if k == "valley"]
    tol = close * 0.005 if close > 0 else 0.01

    double_top = 0.0
    double_bottom = 0.0
    if len(peaks) >= 2:
        p1, p2 = peaks[-2][1], peaks[-1][1]
        if abs(p1 - p2) < tol and len(valleys) >= 1:
            double_top = 1.0
    if len(valleys) >= 2:
        v1, v2 = valleys[-2][1], valleys[-1][1]
        if abs(v1 - v2) < tol and len(peaks) >= 1:
            double_bottom = 1.0

    hs_top = 0.0
    hs_inv = 0.0
    if len(peaks) >= 3 and len(valleys) >= 2:
        left, head, right = peaks[-3][1], peaks[-2][1], peaks[-1][1]
        if head > left and head > right and abs(left - right) < tol * 2:
            hs_top = 1.0
    if len(valleys) >= 3 and len(peaks) >= 2:
        lv, mv, rv = valleys[-3][1], valleys[-2][1], valleys[-1][1]
        if mv < lv and mv < rv and abs(lv - rv) < tol * 2:
            hs_inv = 1.0
    return double_top, double_bottom, hs_top, hs_inv


def _divergence(zig: list, indicator: pd.Series, kind: str) -> tuple[float, float]:
    """kind: rsi or macd — 比较最近两个同向极值。"""
    peaks = [(t, p) for t, p, z in zig if z == "peak"][-3:]
    valleys = [(t, p) for t, p, z in zig if z == "valley"][-3:]
    bull, bear = 0.0, 0.0
    if len(valleys) >= 2:
        t1, p1 = valleys[-2]
        t2, p2 = valleys[-1]
        if p2 < p1:
            try:
                i1 = float(indicator.loc[t1])
                i2 = float(indicator.loc[t2])
                if i2 > i1:
                    bull = 1.0
            except (KeyError, TypeError):
                pass
    if len(peaks) >= 2:
        t1, p1 = peaks[-2]
        t2, p2 = peaks[-1]
        if p2 > p1:
            try:
                i1 = float(indicator.loc[t1])
                i2 = float(indicator.loc[t2])
                if i2 < i1:
                    bear = 1.0
            except (KeyError, TypeError):
                pass
    return bull, bear


def _tf_window(tf: pd.DataFrame, ts: pd.Timestamp, lookback: int, min_bars: int = 30) -> pd.DataFrame | None:
    if tf is None or tf.empty:
        return None
    sub = tf.loc[:ts]
    if len(sub) < min_bars:
        return None
    return sub.tail(lookback)


def _tf_structure_features(window: pd.DataFrame, zigzag_atr_mult: float) -> tuple[float, float, float, float, float]:
    """单周期结构摘要：trend, adx_n, rsi_n, sup_dist, res_dist。"""
    close = float(window["close"].iloc[-1])
    atr_s = atr_series(window)
    atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else close * 0.001
    ema20 = float(ema(window["close"], 20).iloc[-1])
    ema50 = float(ema(window["close"], 50).iloc[-1])
    trend = float(np.clip((ema20 - ema50) / max(ema50, 1e-9), -1, 1))

    adx_val = adx_series(window).iloc[-1]
    adx_n = 0.0 if pd.isna(adx_val) else float(adx_val) / 100.0

    rsi_val = _rsi(window["close"]).iloc[-1]
    rsi_n = 0.5 if pd.isna(rsi_val) else float(rsi_val) / 100.0

    zig = zigzag_points(window, zigzag_atr_mult)
    sup, res, _, _ = _nearest_support_resistance(close, atr, zig)
    sup_dist = (close - sup) / max(atr, 1e-9)
    res_dist = (res - close) / max(atr, 1e-9)
    return trend, adx_n, rsi_n, sup_dist, res_dist


def _mtf_trend_align(m5_trend: float, h1_trend: float) -> float:
    if abs(m5_trend) < 1e-6 or abs(h1_trend) < 1e-6:
        return 0.0
    return float(np.clip(m5_trend * h1_trend, -1, 1))


_MP_M5: pd.DataFrame | None = None
_MP_SA: StructureAnalyzer | None = None
_MP_MTF: dict[str, pd.DataFrame] | None = None


def _parallel_worker_init(m5: pd.DataFrame, cfg: dict[str, Any]) -> None:
    global _MP_M5, _MP_SA, _MP_MTF
    _MP_M5 = m5.sort_index()
    _MP_SA = StructureAnalyzer(cfg)
    _MP_MTF = _MP_SA._build_mtf_context(_MP_M5)


def _parallel_worker_chunk(indices: list[int]) -> list[tuple[int, np.ndarray]]:
    assert _MP_M5 is not None and _MP_SA is not None and _MP_MTF is not None
    return [(i, _MP_SA.compute_row(_MP_M5, i, mtf=_MP_MTF)) for i in indices]


class StructureAnalyzer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.periods = cfg.get("periods", ["M5", "M15", "H1", "H4"])
        self.zigzag_atr_mult = float(cfg.get("zigzag_atr_mult", 0.5))
        self.lookback = int(cfg.get("lookback", 200))

    def _prepare_mtf(self, multi: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        m5 = multi.get("M5")
        if m5 is None or m5.empty:
            return multi
        m5 = m5.sort_index().copy()
        out["M5"] = m5
        if "M15" not in multi or multi["M15"].empty:
            out["M15"] = resample_ohlcv(m5, "15min")
        else:
            out["M15"] = multi["M15"].sort_index()
        if "H1" not in multi or multi["H1"].empty:
            out["H1"] = resample_ohlcv(m5, "1h")
        else:
            out["H1"] = multi["H1"].sort_index()
        if "H4" not in multi or multi["H4"].empty:
            out["H4"] = resample_ohlcv(m5, "4h")
        else:
            out["H4"] = multi["H4"].sort_index()
        return out

    def _build_mtf_context(self, m5: pd.DataFrame) -> dict[str, pd.DataFrame]:
        m5 = m5.sort_index()
        return {
            "M15": resample_ohlcv(m5, "15min"),
            "H1": resample_ohlcv(m5, "1h"),
            "H4": resample_ohlcv(m5, "4h"),
        }

    def _compute_mtf_features(
        self,
        ts: pd.Timestamp,
        m5_trend: float,
        mtf: dict[str, pd.DataFrame],
    ) -> np.ndarray:
        m15 = _tf_window(mtf.get("M15"), ts, self.lookback, min_bars=20)
        h1 = _tf_window(mtf.get("H1"), ts, self.lookback, min_bars=30)
        h4 = _tf_window(mtf.get("H4"), ts, self.lookback, min_bars=20)

        m15_trend = m15_adx = 0.0
        h1_trend = h1_adx = h1_rsi = h1_sup = 0.0
        h4_trend = h4_res = 0.0

        if m15 is not None:
            m15_trend, m15_adx, _, _, _ = _tf_structure_features(m15, self.zigzag_atr_mult)
        if h1 is not None:
            h1_trend, h1_adx, h1_rsi, h1_sup, _ = _tf_structure_features(h1, self.zigzag_atr_mult)
        if h4 is not None:
            h4_trend, _, _, _, h4_res = _tf_structure_features(h4, self.zigzag_atr_mult)

        align = _mtf_trend_align(m5_trend, h1_trend)
        return np.array(
            [m15_trend, m15_adx, h1_trend, h1_adx, h1_rsi, h4_trend, align, h1_sup, h4_res],
            dtype=np.float32,
        )

    def compute_row(
        self,
        m5: pd.DataFrame,
        idx: int,
        mtf: dict[str, pd.DataFrame] | None = None,
    ) -> np.ndarray:
        """单根 M5 的结构特征（30 维）。"""
        window = m5.iloc[max(0, idx - self.lookback + 1) : idx + 1]
        if len(window) < 30:
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        close = float(window["close"].iloc[-1])
        atr_s = atr_series(window)
        atr = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else close * 0.001
        ema20 = float(ema(window["close"], 20).iloc[-1])
        ema50 = float(ema(window["close"], 50).iloc[-1])
        trend = np.clip((ema20 - ema50) / max(ema50, 1e-9), -1, 1)

        adx_val = adx_series(window).iloc[-1]
        adx_n = 0.0 if pd.isna(adx_val) else float(adx_val) / 100.0
        price_to_ema = (close - ema20) / max(atr, 1e-9)

        zig = zigzag_points(window, self.zigzag_atr_mult)
        sup, res, sup_s, res_s = _nearest_support_resistance(close, atr, zig)
        sup_dist = (close - sup) / max(atr, 1e-9)
        res_dist = (res - close) / max(atr, 1e-9)

        dt, db, hs, ihs = _pattern_flags(zig[-20:], close)
        rsi = _rsi(window["close"])
        macd_line, _ = _macd(window["close"])
        rsi_bull, rsi_bear = _divergence(zig, rsi, "rsi")
        macd_bull, macd_bear = _divergence(zig, macd_line, "macd")

        vol = window["volume"] if "volume" in window.columns else pd.Series(1.0, index=window.index)
        vol_ma = vol.rolling(20).mean().iloc[-1]
        vol_ratio = float(vol.iloc[-1] / vol_ma) if vol_ma and not pd.isna(vol_ma) else 1.0

        atr_hist = atr_s.dropna()
        if len(atr_hist) >= 10:
            pct = float(atr_hist.iloc[-1]) / max(float(atr_hist.quantile(0.5)), 1e-9)
            vol_regime = 0.0 if pct < 0.8 else (2.0 if pct > 1.2 else 1.0)
        else:
            vol_regime = 1.0

        ts = window.index[-1]
        hour = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        row = np.zeros(FEATURE_DIM, dtype=np.float32)
        row[0] = trend
        row[1] = adx_n
        row[2] = price_to_ema
        row[3] = sup_dist
        row[4] = res_dist
        row[5] = sup_s
        row[6] = res_s
        row[7:11] = [dt, db, hs, ihs]
        row[11:15] = [rsi_bull, rsi_bear, macd_bull, macd_bear]
        row[15] = vol_ratio
        row[16] = vol_regime
        row[17:21] = [hour_sin, hour_cos, dow_sin, dow_cos]

        if mtf is not None:
            row[21:30] = self._compute_mtf_features(ts, float(trend), mtf)

        np.nan_to_num(row, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return row

    def compute_all(
        self,
        m5: pd.DataFrame,
        progress_every: int = 10000,
        n_jobs: int = 0,
    ) -> np.ndarray:
        """全历史逐 bar 结构特征（离线训练用，不截断序列）。"""
        import logging
        import os

        log = logging.getLogger(__name__)
        m5 = m5.sort_index()
        n = len(m5)
        if n < 30:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)

        workers = 1 if n_jobs <= 0 else int(n_jobs)
        if workers <= 1 or n < 500:
            mtf = self._build_mtf_context(m5)
            rows = np.zeros((n, FEATURE_DIM), dtype=np.float32)
            for i in range(n):
                rows[i] = self.compute_row(m5, i, mtf=mtf)
                if progress_every and i > 0 and i % progress_every == 0:
                    msg = f"结构特征 {i} / {n} ({100.0 * i / n:.1f}%)"
                    log.info(msg)
                    print(msg, flush=True)
            return rows

        from concurrent.futures import ProcessPoolExecutor

        cfg = {
            "periods": self.periods,
            "zigzag_atr_mult": self.zigzag_atr_mult,
            "lookback": self.lookback,
        }
        chunk_n = max(200, (n + workers - 1) // workers)
        chunks = [list(range(i, min(i + chunk_n, n))) for i in range(0, n, chunk_n)]
        log.info("结构特征并行计算: %d 根, %d workers, %d chunks", n, workers, len(chunks))
        rows = np.zeros((n, FEATURE_DIM), dtype=np.float32)
        done = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_parallel_worker_init,
            initargs=(m5, cfg),
        ) as pool:
            for part in pool.map(_parallel_worker_chunk, chunks):
                for i, row in part:
                    rows[i] = row
                done += len(part)
                if progress_every and done % progress_every < chunk_n:
                    msg = f"结构特征 {done} / {n} ({100.0 * done / n:.1f}%)"
                    log.info(msg)
                    print(msg, flush=True)
        return rows

    def compute(self, multi_timeframe_data: dict[str, pd.DataFrame]) -> np.ndarray:
        mtf = self._prepare_mtf(multi_timeframe_data)
        m5 = mtf.get("M5")
        if m5 is None or len(m5) < 30:
            return np.zeros((0, FEATURE_DIM), dtype=np.float32)
        ctx = {k: mtf[k] for k in ("M15", "H1", "H4") if k in mtf}
        rows = [self.compute_row(m5, i, mtf=ctx) for i in range(len(m5))]
        return np.stack(rows, axis=0)

    def compute_latest(self, multi_timeframe_data: dict[str, pd.DataFrame]) -> np.ndarray:
        mtf = self._prepare_mtf(multi_timeframe_data)
        m5 = mtf.get("M5")
        if m5 is None or len(m5) < 30:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        ctx = {k: mtf[k] for k in ("M15", "H1", "H4") if k in mtf}
        return self.compute_row(m5, len(m5) - 1, mtf=ctx)
