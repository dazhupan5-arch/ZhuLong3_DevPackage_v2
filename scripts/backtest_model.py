#!/usr/bin/env python3
"""XGBoost v12 离线滚动回测与指标评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from zhulong.training.lgb.backtest import _atr_series, max_drawdown_r, simulate_trade  # noqa: E402
from zhulong.training.v11.train import proba_to_directions  # noqa: E402
from zhulong.training.v12.backtest import V12_LONG_SL, V12_SHORT_SL, apply_short_trend_filter  # noqa: E402


def load_m5_parquet(symbol: str, root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "training" / "v8" / symbol / "m5.parquet",
        root / "data" / "training" / "oil_v1" / symbol / "m5.parquet",
    ]
    for p in candidates:
        if p.is_file():
            df = pd.read_parquet(p)
            if "time" in df.columns:
                df = df.set_index("time")
            return df.sort_index()
    raise FileNotFoundError(f"未找到 {symbol} M5 数据，请先运行 prepare 脚本")


def extract_features_simple(m5: pd.DataFrame, i: int, window: int = 60) -> np.ndarray | None:
    """占位：生产环境应使用 build_v8_features；此处用收益率序列做 smoke test。"""
    sl = m5.iloc[max(0, i - window) : i]
    if len(sl) < window:
        return None
    ret = sl["close"].pct_change().fillna(0).values[-window:]
    return ret.reshape(1, -1)


def backtest_from_proba(
    m5: pd.DataFrame,
    directions: np.ndarray,
    times: pd.DatetimeIndex,
    *,
    sl_long: float = V12_LONG_SL,
    sl_short: float = V12_SHORT_SL,
) -> dict:
    atr = _atr_series(m5)
    rs: list[float] = []
    wins = losses = 0
    for d, t in zip(directions, times):
        if d == 0 or t not in m5.index:
            continue
        idx = m5.index.get_loc(t)
        if isinstance(idx, slice):
            continue
        entry = float(m5.loc[t, "close"])
        a = float(atr.iloc[idx])
        if a <= 0:
            continue
        sl_mult = sl_long if d > 0 else sl_short
        future = m5.iloc[idx + 1 : idx + 13]
        if future.empty:
            continue
        r = simulate_trade(
            d,
            entry,
            a,
            future["high"].values,
            future["low"].values,
            future["close"].values,
            max_bars=12,
            sl_mult=sl_mult,
        )
        rs.append(r)
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1

    if not rs:
        return {"trades": 0, "win_rate": 0.0, "avg_r": 0.0, "max_dd_r": 0.0, "sharpe": 0.0}

    arr = np.array(rs)
    win_rate = wins / len(rs)
    avg_win = float(arr[arr > 0].mean()) if (arr > 0).any() else 0.0
    avg_loss = float(abs(arr[arr < 0].mean())) if (arr < 0).any() else 1e-9
    return {
        "trades": len(rs),
        "win_rate": round(win_rate, 4),
        "avg_r": round(float(arr.mean()), 4),
        "profit_factor": round(avg_win / avg_loss, 3) if avg_loss > 0 else 0.0,
        "max_dd_r": round(max_drawdown_r(arr), 4),
        "sharpe": round(float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252 / 12)), 3),
    }


def run_backtest(
    symbol: str,
    start: str | None,
    end: str | None,
    long_thr: float = 0.84,
    short_thr: float = 0.88,
    root: Path | None = None,
) -> dict:
    root = root or _ROOT
    m5 = load_m5_parquet(symbol, root)
    if start:
        m5 = m5[m5.index >= pd.Timestamp(start)]
    if end:
        m5 = m5[m5.index <= pd.Timestamp(end)]
    if len(m5) < 500:
        raise ValueError("M5 样本过短")

    try:
        import joblib
        import xgboost as xgb

        model_dir = root / "models" / symbol
        model = xgb.XGBClassifier()
        model.load_model(str(model_dir / "xgb_triple.json"))
        cols = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        meta = joblib.load(model_dir / "v12_meta.pkl")
        from zhulong.live_v8_features import build_live_v8_row

        use_model = True
    except Exception as ex:
        print(f"[!] 模型加载失败，使用随机方向 smoke test: {ex}")
        use_model = False
        cols = meta = model = None

    directions_list: list[int] = []
    times_list: list[pd.Timestamp] = []
    step = 1
    warmup = 400
    for i in range(warmup, len(m5), step):
        t = m5.index[i]
        if use_model:
            window = m5.iloc[: i + 1]
            row, feat_cols, _ = build_live_v8_row(symbol, m5=window.tail(min(len(window), 2000)))
            x = row.reshape(1, -1)
            if feat_cols != cols:
                idx = [feat_cols.index(c) for c in cols if c in feat_cols]
                x = row[idx].reshape(1, -1)
            proba = model.predict_proba(x)[0]
            dirs = proba_to_directions(proba.reshape(1, -1), long_thr, short_thr)
            d = int(dirs[0])
            feats_df = pd.DataFrame(x, columns=cols[: x.shape[1]], index=[t])
            filtered = apply_short_trend_filter(window, feats_df, pd.DatetimeIndex([t]), dirs)
            d = int(filtered[0])
        else:
            d = 0
        directions_list.append(d)
        times_list.append(t)

    metrics = backtest_from_proba(
        m5, np.array(directions_list), pd.DatetimeIndex(times_list)
    )
    metrics.update(
        {
            "symbol": symbol,
            "bars": len(m5),
            "long_threshold": long_thr,
            "short_threshold": short_thr,
            "start": str(m5.index[0]),
            "end": str(m5.index[-1]),
        }
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--long-thr", type=float, default=0.84)
    parser.add_argument("--short-thr", type=float, default=0.88)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    metrics = run_backtest(
        args.symbol, args.start, args.end, args.long_thr, args.short_thr
    )
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
