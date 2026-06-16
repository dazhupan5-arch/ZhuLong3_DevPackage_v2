#!/usr/bin/env python3
"""
生成模拟的 XAUUSD M1 数据（2026-06-08 ~ 2026-06-10）。
v3: 中等波动率 + 明确的趋势阶段，使AI可检测到模式
"""

import csv
import math
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

SYMBOL = "XAUUSD"
START = datetime(2026, 6, 7, 23, 0, tzinfo=timezone.utc)
END   = datetime(2026, 6, 10, 21, 0, tzinfo=timezone.utc)
BASE_PRICE = 2350.0


def generate_m1_data():
    bars = []
    current = START
    price = BASE_PRICE

    # 按日构造趋势
    days_since_start = 0
    trend = 0.0

    while current < END:
        if current.weekday() == 5 or (current.weekday() == 6 and current.hour < 11):
            current += timedelta(minutes=1)
            continue
        if current.weekday() == 4 and current.hour >= 21:
            current += timedelta(minutes=1)
            continue

        hour = current.hour
        minute = current.minute

        # 每天/每时段切换趋势
        elapsed_hours = (current - START).total_seconds() / 3600
        if elapsed_hours < 10:
            target_trend = 0.1          # 小幅上涨
        elif elapsed_hours < 20:
            target_trend = -0.15        # 回调
        elif elapsed_hours < 30:
            target_trend = 0.0          # 震荡
        elif elapsed_hours < 40:
            target_trend = 0.25         # 强上涨（可以测试移动止损）
        elif elapsed_hours < 50:
            target_trend = -0.2         # 下跌
        else:
            target_trend = 0.1          # 尾盘整理

        # 趋势平滑
        trend += (target_trend - trend) * 0.01

        # 日内波动
        if 0 <= hour < 6:
            vol = 0.08
        elif 6 <= hour < 8:
            vol = 0.12
        elif 8 <= hour < 12:
            vol = 0.15
        elif 12 <= hour < 14:
            vol = 0.12
        elif 14 <= hour < 20:
            vol = 0.18
        else:
            vol = 0.10

        # 价格变动
        noise = random.gauss(0, vol * 0.5)
        change = trend * 0.1 + noise
        price += change

        # OHLC
        o = price
        h = o + abs(random.gauss(0, vol * 0.6))
        l = o - abs(random.gauss(0, vol * 0.6))
        c = o + random.gauss(change * 0.4, vol * 0.4)
        h = max(h, o, c)
        l = min(l, o, c)

        bar = {
            "time_unix": int(current.timestamp()),
            "time_str": current.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": SYMBOL,
            "open": round(o, 2),
            "high": round(h, 2),
            "low": round(l, 2),
            "close": round(c, 2),
            "volume": int(random.randint(500, 3000)),
        }
        bars.append(bar)
        current += timedelta(minutes=1)

    return bars


def write_csv(bars, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "time_unix", "time_str", "symbol", "open", "high", "low", "close", "volume"
        ])
        writer.writeheader()
        writer.writerows(bars)
    low = min(b['low'] for b in bars)
    high = max(b['high'] for b in bars)
    print(f"已生成 {len(bars)} 根 M1 K线 → {path}")
    print(f"价格区间: ${low:.2f} ~ ${high:.2f}  ({(high-low)/low*100:.2f}%)")
    print(f"时间范围: {bars[0]['time_str']} ~ {bars[-1]['time_str']}")


if __name__ == "__main__":
    from pathlib import Path
    output_dir = Path(__file__).resolve().parent.parent / "simulation" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "M1_export.csv"
    bars = generate_m1_data()
    write_csv(bars, output_path)
    print(f"\n运行: py -3 scripts/replay_simulation.py --csv {output_path} --speed 500")
