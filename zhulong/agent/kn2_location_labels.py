"""KN2 结构位置感知标签：从有利区域出发的结构 Triple-Barrier。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from zhulong.agent.state_builder import infer_regime_from_struct
from zhulong.agent.structure_analyzer import FEATURE_NAMES

REGIME_CODES = {
    "ranging": 0,
    "trending_up": 1,
    "trending_down": 2,
    "choppy": 3,
    "breakout_up": 4,
    "breakout_down": 5,
    "unknown": 6,
}


@dataclass
class LocationLabelConfig:
    max_support_dist: float = 0.5
    max_resistance_dist: float = 0.5
    min_support_strength: float = 0.3
    min_resistance_strength: float = 0.3
    min_resistance_dist_long: float = 0.35
    min_support_dist_short: float = 0.35
    pos_range_long_max: float = 0.35
    pos_range_short_min: float = 0.65
    ranging_block_long_above: float = 0.6
    ranging_block_short_below: float = 0.4
    sl_buffer_atr: float = 0.3
    sl_floor_atr: float = 1.0
    sl_cap_atr: float = 2.5
    tp_buffer_atr: float = 0.25
    tp_floor_atr: float = 1.5
    tp_cap_atr: float = 4.0
    max_hold_bars: int = 48
    min_rr: float = 1.2


def compute_pos_in_range(close: np.ndarray, window: int = 12) -> np.ndarray:
    """最近 window 根收盘价在区间中的相对位置 0~1。"""
    n = len(close)
    out = np.full(n, 0.5, dtype=np.float32)
    if n < 2:
        return out
    w = max(3, window)
    for i in range(n):
        start = max(0, i - w + 1)
        seg = close[start : i + 1]
        lo, hi = float(seg.min()), float(seg.max())
        rng = hi - lo
        if rng <= 1e-9:
            out[i] = 0.5
        else:
            out[i] = float(np.clip((close[i] - lo) / rng, 0.0, 1.0))
    return out


def regime_code_array(struct: np.ndarray) -> np.ndarray:
    n = len(struct)
    codes = np.zeros(n, dtype=np.int8)
    for i in range(n):
        name = infer_regime_from_struct(struct[i])
        codes[i] = REGIME_CODES.get(name, 6)
    return codes


def build_entry_masks(
    struct: np.ndarray,
    pos_in_range: np.ndarray,
    regime: np.ndarray,
    cfg: LocationLabelConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """返回 long_candidate, short_candidate 布尔掩码及候选计数。"""
    n = len(struct)
    sup_d = struct[:, 3]
    res_d = struct[:, 4]
    sup_str = struct[:, 5]
    res_str = struct[:, 6]
    trend = struct[:, 0]
    mtf = struct[:, 27] if struct.shape[1] > 27 else np.zeros(n)

    bull_pat = (
        (struct[:, 8] > 0.5)
        | (struct[:, 10] > 0.5)
        | (struct[:, 11] > 0.5)
        | (struct[:, 13] > 0.5)
    )
    bear_pat = (
        (struct[:, 7] > 0.5)
        | (struct[:, 9] > 0.5)
        | (struct[:, 12] > 0.5)
        | (struct[:, 14] > 0.5)
    )

    near_sup = (sup_d <= cfg.max_support_dist) & (sup_str >= cfg.min_support_strength)
    near_res = (res_d <= cfg.max_resistance_dist) & (res_str >= cfg.min_resistance_strength)
    low_range = pos_in_range <= cfg.pos_range_long_max
    high_range = pos_in_range >= cfg.pos_range_short_min

    long_base = near_sup | low_range | bull_pat
    short_base = near_res | high_range | bear_pat

    long_excl = res_d <= cfg.min_resistance_dist_long
    short_excl = sup_d <= cfg.min_support_dist_short

    is_ranging = regime == REGIME_CODES["ranging"]
    long_excl |= is_ranging & (pos_in_range > cfg.ranging_block_long_above)
    short_excl |= is_ranging & (pos_in_range < cfg.ranging_block_short_below)

    is_up = np.isin(regime, [REGIME_CODES["trending_up"], REGIME_CODES["breakout_up"]])
    is_down = np.isin(regime, [REGIME_CODES["trending_down"], REGIME_CODES["breakout_down"]])

    long_base &= ~is_down | (trend > -0.05)
    short_base &= ~is_up | (trend < 0.05)

    long_base &= is_up | is_ranging | (regime == REGIME_CODES["choppy"]) | bull_pat
    short_base &= is_down | is_ranging | (regime == REGIME_CODES["choppy"]) | bear_pat

    if struct.shape[1] > 27:
        long_base &= (mtf >= -0.2) | is_ranging | bull_pat
        short_base &= (mtf <= 0.2) | is_ranging | bear_pat

    long_cand = long_base & ~long_excl
    short_cand = short_base & ~short_excl

    stats = {
        "long_candidate": int(long_cand.sum()),
        "short_candidate": int(short_cand.sum()),
        "both_candidate": int((long_cand & short_cand).sum()),
    }
    return long_cand, short_cand, stats


def _long_sl_tp(
    close: float,
    atr: float,
    sup_dist: float,
    res_dist: float,
    cfg: LocationLabelConfig,
) -> tuple[float, float, float, float]:
    support_px = close - sup_dist * atr
    resistance_px = close + res_dist * atr
    atr_sl = close - cfg.sl_floor_atr * atr
    if support_px > 0 and support_px < close:
        sl_px = min(support_px - cfg.sl_buffer_atr * atr, atr_sl)
    else:
        sl_px = atr_sl
    sl_px = max(sl_px, close - cfg.sl_cap_atr * atr)

    if resistance_px > close:
        tp_px = max(resistance_px + cfg.tp_buffer_atr * atr, close + cfg.tp_floor_atr * atr)
    else:
        tp_px = close + cfg.tp_floor_atr * atr
    tp_px = min(tp_px, close + cfg.tp_cap_atr * atr)

    sl_mult = (close - sl_px) / max(atr, 1e-9)
    tp_mult = (tp_px - close) / max(atr, 1e-9)
    return sl_px, tp_px, sl_mult, tp_mult


def _short_sl_tp(
    close: float,
    atr: float,
    sup_dist: float,
    res_dist: float,
    cfg: LocationLabelConfig,
) -> tuple[float, float, float, float]:
    support_px = close - sup_dist * atr
    resistance_px = close + res_dist * atr
    atr_sl = close + cfg.sl_floor_atr * atr
    if resistance_px > close:
        sl_px = max(resistance_px + cfg.sl_buffer_atr * atr, atr_sl)
    else:
        sl_px = atr_sl
    sl_px = min(sl_px, close + cfg.sl_cap_atr * atr)

    if support_px > 0 and support_px < close:
        tp_px = min(support_px - cfg.tp_buffer_atr * atr, close - cfg.tp_floor_atr * atr)
    else:
        tp_px = close - cfg.tp_floor_atr * atr
    tp_px = max(tp_px, close - cfg.tp_cap_atr * atr)

    sl_mult = (sl_px - close) / max(atr, 1e-9)
    tp_mult = (close - tp_px) / max(atr, 1e-9)
    return sl_px, tp_px, sl_mult, tp_mult


try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _scan_location_labels(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        struct: np.ndarray,
        long_cand: np.ndarray,
        short_cand: np.ndarray,
        sl_buffer: float,
        sl_floor: float,
        sl_cap: float,
        tp_buffer: float,
        tp_floor: float,
        tp_cap: float,
        max_hold: int,
        min_rr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = close.shape[0]
        actions = np.zeros(n, dtype=np.int32)
        should_trade = np.zeros(n, dtype=np.float32)
        sl_mults = np.full(n, sl_floor, dtype=np.float32)
        tp_mults = np.full(n, tp_floor, dtype=np.float32)
        block_reason = np.zeros(n, dtype=np.int8)

        limit = n - max_hold
        for t in prange(limit):
            if not long_cand[t] and not short_cand[t]:
                block_reason[t] = 1
                continue

            c = close[t]
            a = atr[t]
            if a <= 0:
                block_reason[t] = 2
                continue

            sup_d = struct[t, 3]
            res_d = struct[t, 4]
            long_ok = False
            short_ok = False
            long_rr = 0.0
            short_rr = 0.0
            l_sl_m = sl_floor
            l_tp_m = tp_floor
            s_sl_m = sl_floor
            s_tp_m = tp_floor

            if long_cand[t]:
                support_px = c - sup_d * a
                resistance_px = c + res_d * a
                atr_sl = c - sl_floor * a
                if support_px > 0.0 and support_px < c:
                    sl_px = min(support_px - sl_buffer * a, atr_sl)
                else:
                    sl_px = atr_sl
                if sl_px < c - sl_cap * a:
                    sl_px = c - sl_cap * a
                if resistance_px > c:
                    tp_px = max(resistance_px + tp_buffer * a, c + tp_floor * a)
                else:
                    tp_px = c + tp_floor * a
                if tp_px > c + tp_cap * a:
                    tp_px = c + tp_cap * a

                hit_tp = False
                hit_sl = False
                for fwd in range(t + 1, min(t + max_hold + 1, n)):
                    if high[fwd] >= tp_px:
                        hit_tp = True
                    if low[fwd] <= sl_px:
                        hit_sl = True
                    if hit_tp and hit_sl:
                        break
                    if hit_sl and not hit_tp:
                        break
                    if hit_tp and not hit_sl:
                        break

                if hit_tp and not hit_sl:
                    sl_m = (c - sl_px) / a
                    tp_m = (tp_px - c) / a
                    if sl_m > 1e-6 and tp_m / sl_m >= min_rr:
                        long_ok = True
                        long_rr = tp_m / sl_m
                        l_sl_m = sl_m
                        l_tp_m = tp_m

            if short_cand[t]:
                support_px = c - sup_d * a
                resistance_px = c + res_d * a
                atr_sl = c + sl_floor * a
                if resistance_px > c:
                    sl_px = max(resistance_px + sl_buffer * a, atr_sl)
                else:
                    sl_px = atr_sl
                if sl_px > c + sl_cap * a:
                    sl_px = c + sl_cap * a
                if support_px > 0.0 and support_px < c:
                    tp_px = min(support_px - tp_buffer * a, c - tp_floor * a)
                else:
                    tp_px = c - tp_floor * a
                if tp_px < c - tp_cap * a:
                    tp_px = c - tp_cap * a

                hit_tp = False
                hit_sl = False
                for fwd in range(t + 1, min(t + max_hold + 1, n)):
                    if low[fwd] <= tp_px:
                        hit_tp = True
                    if high[fwd] >= sl_px:
                        hit_sl = True
                    if hit_tp and hit_sl:
                        break
                    if hit_sl and not hit_tp:
                        break
                    if hit_tp and not hit_sl:
                        break

                if hit_tp and not hit_sl:
                    sl_m = (sl_px - c) / a
                    tp_m = (c - tp_px) / a
                    if sl_m > 1e-6 and tp_m / sl_m >= min_rr:
                        short_ok = True
                        short_rr = tp_m / sl_m
                        s_sl_m = sl_m
                        s_tp_m = tp_m

            if long_ok and short_ok:
                if long_rr >= short_rr:
                    actions[t] = 1
                    sl_mults[t] = l_sl_m
                    tp_mults[t] = l_tp_m
                else:
                    actions[t] = 2
                    sl_mults[t] = s_sl_m
                    tp_mults[t] = s_tp_m
                should_trade[t] = 1.0
            elif long_ok:
                actions[t] = 1
                sl_mults[t] = l_sl_m
                tp_mults[t] = l_tp_m
                should_trade[t] = 1.0
            elif short_ok:
                actions[t] = 2
                sl_mults[t] = s_sl_m
                tp_mults[t] = s_tp_m
                should_trade[t] = 1.0
            else:
                block_reason[t] = 3

        return actions, should_trade, sl_mults, tp_mults, block_reason

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def generate_location_labels(
    df: pd.DataFrame,
    struct: np.ndarray,
    cfg: LocationLabelConfig | None = None,
    progress_every: int = 0,
) -> dict[str, np.ndarray]:
    """生成结构位置感知 KN2 标签。"""
    cfg = cfg or LocationLabelConfig()
    n = min(len(df), len(struct))
    struct = np.asarray(struct[:n], dtype=np.float32)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64) if "high" in df.columns else close.copy()
    low = df["low"].values.astype(np.float64) if "low" in df.columns else close.copy()
    atr = df["atr"].values.astype(np.float64) if "atr" in df.columns else np.full(n, close[0] * 0.001)

    pos_in_range = compute_pos_in_range(close.astype(np.float32))
    regime = regime_code_array(struct)
    long_cand, short_cand, _ = build_entry_masks(struct, pos_in_range, regime, cfg)

    if _HAS_NUMBA:
        actions, should_trade, sl_mults, tp_mults, block_reason = _scan_location_labels(
            close,
            high,
            low,
            atr,
            struct,
            long_cand,
            short_cand,
            cfg.sl_buffer_atr,
            cfg.sl_floor_atr,
            cfg.sl_cap_atr,
            cfg.tp_buffer_atr,
            cfg.tp_floor_atr,
            cfg.tp_cap_atr,
            cfg.max_hold_bars,
            cfg.min_rr,
        )
    else:
        actions, should_trade, sl_mults, tp_mults, block_reason = _scan_location_labels_python(
            df, struct, long_cand, short_cand, pos_in_range, cfg, progress_every
        )

    return {
        "action": actions.astype(np.int32),
        "should_trade": should_trade.astype(np.float32),
        "position_size": should_trade.astype(np.float32),
        "sl_atr_mult": sl_mults.astype(np.float32),
        "tp_atr_mult": tp_mults.astype(np.float32),
        "long_candidate": long_cand.astype(np.uint8),
        "short_candidate": short_cand.astype(np.uint8),
        "pos_in_range": pos_in_range,
        "regime_code": regime,
        "block_reason": block_reason.astype(np.int8),
    }


def _scan_location_labels_python(
    df: pd.DataFrame,
    struct: np.ndarray,
    long_cand: np.ndarray,
    short_cand: np.ndarray,
    pos_in_range: np.ndarray,
    cfg: LocationLabelConfig,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = df["atr"].values
    actions = np.zeros(n, dtype=np.int32)
    should_trade = np.zeros(n, dtype=np.float32)
    sl_mults = np.full(n, cfg.sl_floor_atr, dtype=np.float32)
    tp_mults = np.full(n, cfg.tp_floor_atr, dtype=np.float32)
    block_reason = np.zeros(n, dtype=np.int8)

    for t in range(n - cfg.max_hold_bars):
        if progress_every and t > 0 and t % progress_every == 0:
            print(f"  location labels {t:,}/{n:,}")
        if not long_cand[t] and not short_cand[t]:
            block_reason[t] = 1
            continue
        c, a = float(close[t]), float(atr[t])
        if a <= 0:
            block_reason[t] = 2
            continue
        sup_d, res_d = float(struct[t, 3]), float(struct[t, 4])
        long_ok = short_ok = False
        long_rr = short_rr = 0.0
        l_sl_m = l_tp_m = s_sl_m = s_tp_m = cfg.sl_floor_atr

        if long_cand[t]:
            sl_px, tp_px, l_sl_m, l_tp_m = _long_sl_tp(c, a, sup_d, res_d, cfg)
            hit_tp = hit_sl = False
            for fwd in range(t + 1, min(t + cfg.max_hold_bars + 1, n)):
                if high[fwd] >= tp_px:
                    hit_tp = True
                if low[fwd] <= sl_px:
                    hit_sl = True
                if hit_tp or hit_sl:
                    break
            if hit_tp and not hit_sl and l_sl_m > 1e-6 and l_tp_m / l_sl_m >= cfg.min_rr:
                long_ok = True
                long_rr = l_tp_m / l_sl_m

        if short_cand[t]:
            sl_px, tp_px, s_sl_m, s_tp_m = _short_sl_tp(c, a, sup_d, res_d, cfg)
            hit_tp = hit_sl = False
            for fwd in range(t + 1, min(t + cfg.max_hold_bars + 1, n)):
                if low[fwd] <= tp_px:
                    hit_tp = True
                if high[fwd] >= sl_px:
                    hit_sl = True
                if hit_tp or hit_sl:
                    break
            if hit_tp and not hit_sl and s_sl_m > 1e-6 and s_tp_m / s_sl_m >= cfg.min_rr:
                short_ok = True
                short_rr = s_tp_m / s_sl_m

        if long_ok and short_ok:
            if long_rr >= short_rr:
                actions[t] = 1
                sl_mults[t], tp_mults[t] = l_sl_m, l_tp_m
            else:
                actions[t] = 2
                sl_mults[t], tp_mults[t] = s_sl_m, s_tp_m
            should_trade[t] = 1.0
        elif long_ok:
            actions[t] = 1
            sl_mults[t], tp_mults[t] = l_sl_m, l_tp_m
            should_trade[t] = 1.0
        elif short_ok:
            actions[t] = 2
            sl_mults[t], tp_mults[t] = s_sl_m, s_tp_m
            should_trade[t] = 1.0
        else:
            block_reason[t] = 3
    return actions, should_trade, sl_mults, tp_mults, block_reason


def summarize_location_labels(
    labels: dict[str, np.ndarray],
    struct: np.ndarray,
    times: pd.DatetimeIndex | None = None,
    legacy_labels: dict[str, np.ndarray] | None = None,
    cfg: LocationLabelConfig | None = None,
) -> dict[str, Any]:
    """P0 分布报告。"""
    cfg = cfg or LocationLabelConfig()
    n = len(labels["action"])
    actions = labels["action"]
    st = labels["should_trade"] > 0.5
    regime = labels.get("regime_code", np.zeros(n, dtype=np.int8))
    pos = labels.get("pos_in_range", np.full(n, 0.5, dtype=np.float32))
    inv_regime = {v: k for k, v in REGIME_CODES.items()}

    action_counts = {
        "hold": int((actions == 0).sum()),
        "long": int((actions == 1).sum()),
        "short": int((actions == 2).sum()),
    }
    report: dict[str, Any] = {
        "total_bars": n,
        "should_trade_pct": round(float(st.mean()) * 100, 3),
        "action_counts": action_counts,
        "action_pct": {k: round(v / max(n, 1) * 100, 3) for k, v in action_counts.items()},
        "config": asdict(cfg),
        "feature_layout": list(FEATURE_NAMES),
        "numba": _HAS_NUMBA,
    }

    long_m = actions == 1
    short_m = actions == 2
    if long_m.any():
        report["long_labeled"] = {
            "count": int(long_m.sum()),
            "mean_support_dist": round(float(struct[long_m, 3].mean()), 4),
            "mean_resistance_dist": round(float(struct[long_m, 4].mean()), 4),
            "mean_pos_in_range": round(float(pos[long_m].mean()), 4),
            "mean_sl_atr_mult": round(float(labels["sl_atr_mult"][long_m].mean()), 4),
            "mean_tp_atr_mult": round(float(labels["tp_atr_mult"][long_m].mean()), 4),
        }
    if short_m.any():
        report["short_labeled"] = {
            "count": int(short_m.sum()),
            "mean_support_dist": round(float(struct[short_m, 3].mean()), 4),
            "mean_resistance_dist": round(float(struct[short_m, 4].mean()), 4),
            "mean_pos_in_range": round(float(pos[short_m].mean()), 4),
            "mean_sl_atr_mult": round(float(labels["sl_atr_mult"][short_m].mean()), 4),
            "mean_tp_atr_mult": round(float(labels["tp_atr_mult"][short_m].mean()), 4),
        }

    by_regime: dict[str, Any] = {}
    for code, name in inv_regime.items():
        m = regime == code
        if not m.any():
            continue
        by_regime[name] = {
            "bars": int(m.sum()),
            "should_trade_pct": round(float(st[m].mean()) * 100, 3),
            "long": int((actions[m] == 1).sum()),
            "short": int((actions[m] == 2).sum()),
        }
    report["by_regime"] = by_regime

    cand = labels.get("long_candidate")
    if cand is not None:
        lc = cand.astype(bool)
        sc = labels["short_candidate"].astype(bool)
        report["candidates"] = {
            "long_candidate_pct": round(float(lc.mean()) * 100, 3),
            "short_candidate_pct": round(float(sc.mean()) * 100, 3),
            "long_cand_to_label_pct": round(float((st & lc).sum()) / max(lc.sum(), 1) * 100, 3),
            "short_cand_to_label_pct": round(float((st & sc).sum()) / max(sc.sum(), 1) * 100, 3),
        }

    br = labels.get("block_reason")
    if br is not None:
        report["block_reasons"] = {
            "not_candidate": int((br == 1).sum()),
            "bad_atr": int((br == 2).sum()),
            "barrier_fail": int((br == 3).sum()),
        }

    if legacy_labels is not None:
        la = legacy_labels["action"]
        report["vs_legacy_fixed_barrier"] = {
            "legacy_should_trade_pct": round(float(legacy_labels["should_trade"].mean()) * 100, 3),
            "legacy_long": int((la == 1).sum()),
            "legacy_short": int((la == 2).sum()),
            "location_only_long": int(((actions == 1) & (la != 1)).sum()),
            "location_only_short": int(((actions == 2) & (la != 2)).sum()),
            "legacy_only_long": int(((la == 1) & (actions != 1)).sum()),
            "legacy_only_short": int(((la == 2) & (actions != 2)).sum()),
        }

    if times is not None and len(times) == n:
        for year in (2024, 2025):
            ym = times.year == year
            if not ym.any():
                continue
            report[f"year_{year}"] = {
                "bars": int(ym.sum()),
                "should_trade_pct": round(float(st[ym].mean()) * 100, 3),
                "long": int((actions[ym] == 1).sum()),
                "short": int((actions[ym] == 2).sum()),
            }

    return report


def evaluate_structure_entry_gate(
    struct_row: np.ndarray,
    pos_in_range: float,
    regime: str,
    direction: str,
    cfg: LocationLabelConfig | None = None,
) -> tuple[bool, str]:
    """实盘结构位置门控：long 需在有利区域，否则拒绝开仓。"""
    direction = (direction or "flat").strip().lower()
    if direction not in ("long", "short"):
        return True, ""
    cfg = cfg or LocationLabelConfig()
    regime_code = np.array([REGIME_CODES.get(regime, 6)], dtype=np.int8)
    struct = np.asarray(struct_row, dtype=np.float32).reshape(1, -1)
    pos = np.array([float(pos_in_range)], dtype=np.float32)
    long_c, short_c, _ = build_entry_masks(struct, pos, regime_code, cfg)
    if direction == "long":
        if bool(long_c[0]):
            return True, ""
        res_d = float(struct[0, 4]) if struct.shape[1] > 4 else 99.0
        if res_d <= cfg.min_resistance_dist_long:
            return False, "structure_gate:near_resistance_no_long"
        if regime == "ranging" and pos_in_range > cfg.ranging_block_long_above:
            return False, "structure_gate:ranging_upper_range"
        return False, "structure_gate:long_location_unfavorable"
    if bool(short_c[0]):
        return True, ""
    sup_d = float(struct[0, 3]) if struct.shape[1] > 3 else 99.0
    if sup_d <= cfg.min_support_dist_short:
        return False, "structure_gate:near_support_no_short"
    if regime == "ranging" and pos_in_range < cfg.ranging_block_short_below:
        return False, "structure_gate:ranging_lower_range"
    return False, "structure_gate:short_location_unfavorable"


def labels_from_npz(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """从 NPZ 读取 P0 预生成的 loc_* 标签。"""
    required = ("loc_action", "loc_should_trade", "loc_sl_atr_mult", "loc_tp_atr_mult")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"NPZ 缺少位置标签字段: {missing}；请先 prepare_kn2_v16_location_labels.py")
    action = np.asarray(data["loc_action"], dtype=np.int32)
    should_trade = np.asarray(data["loc_should_trade"], dtype=np.float32)
    return {
        "action": action,
        "should_trade": should_trade,
        "position_size": np.asarray(
            data.get("loc_position_size", should_trade), dtype=np.float32
        ),
        "sl_atr_mult": np.asarray(data["loc_sl_atr_mult"], dtype=np.float32),
        "tp_atr_mult": np.asarray(data["loc_tp_atr_mult"], dtype=np.float32),
    }


def load_kn2_v16_labels(
    data: dict[str, Any],
    df: pd.DataFrame,
    market_feat: np.ndarray,
    *,
    label_mode: str = "auto",
    progress_every: int = 0,
) -> tuple[dict[str, np.ndarray], str]:
    """
    加载 KN2 训练标签。

    label_mode:
      - auto: NPZ 含 loc_action 则用 location，否则 legacy
      - location: 必须用 loc_*
      - legacy: 现场 generate_kn2_training_labels
    """
    from zhulong.agent.trading_env_kn2 import generate_kn2_training_labels

    mode = (label_mode or "auto").strip().lower()
    has_loc = "loc_action" in data
    if mode == "auto":
        mode = "location" if has_loc else "legacy"
    if mode == "location":
        labels = labels_from_npz(data)
        ver = "location_v1"
        if "loc_label_version" in data:
            raw = data["loc_label_version"]
            ver = str(raw[0] if hasattr(raw, "__len__") and len(raw) else raw)
        return labels, ver
    if mode != "legacy":
        raise ValueError(f"未知 label_mode: {label_mode}")
    labels = generate_kn2_training_labels(
        df,
        market_feat,
        progress_every=progress_every,
    )
    return labels, "legacy_fixed_barrier"


def replay_bar_diagnosis(
    struct_row: np.ndarray,
    pos_in_range: float,
    regime_name: str,
    cfg: LocationLabelConfig | None = None,
) -> dict[str, Any]:
    """单笔 bar 诊断（用于类似 4339 震荡高位追多场景）。"""
    cfg = cfg or LocationLabelConfig()
    regime = np.array([REGIME_CODES.get(regime_name, 6)], dtype=np.int8)
    pos = np.array([pos_in_range], dtype=np.float32)
    struct = np.asarray(struct_row, dtype=np.float32).reshape(1, -1)
    long_c, short_c, stats = build_entry_masks(struct, pos, regime, cfg)
    sf = struct[0]
    return {
        "regime": regime_name,
        "pos_in_range": pos_in_range,
        "m5_support_dist": round(float(sf[3]), 4),
        "m5_resistance_dist": round(float(sf[4]), 4),
        "m5_trend": round(float(sf[0]), 4),
        "long_candidate": bool(long_c[0]),
        "short_candidate": bool(short_c[0]),
        "candidate_stats": stats,
        "verdict": (
            "would_block_long"
            if regime_name == "ranging" and pos_in_range > cfg.ranging_block_long_above
            else ("long_candidate" if long_c[0] else "hold")
        ),
    }
