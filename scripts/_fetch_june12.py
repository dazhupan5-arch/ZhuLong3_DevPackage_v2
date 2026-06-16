import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

mt5.initialize()

symbol = "XAUUSD"
# Get M5 bars starting from June 12 00:00 UTC
start = datetime(2026, 6, 12, 0, 0, 0, tzinfo=timezone.utc)
utc_from = int(start.timestamp())

rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M5, utc_from, 500)

if rates is None or len(rates) == 0:
    print("No bars returned for June 12!")
    # Try getting the last 1000 bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)
    if rates is not None:
        df = pd.DataFrame(rates)
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        print(f"\nLast 500 bars range: {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
        last_dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())[-5:]
        for d in last_dates:
            cnt = len(df[df['dt'].dt.strftime('%Y-%m-%d') == d])
            print(f"  {d}: {cnt} bars")
else:
    df = pd.DataFrame(rates)
    df['dt'] = pd.to_datetime(df['time'], unit='s')
    print(f"Got {len(df)} bars for June 12 XAUUSD M5")
    print(f"Range: {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
    print(f"\nSample data:")
    print(df[['dt','open','high','low','close','tick_volume']].head(10))
    print("...")
    print(df[['dt','open','high','low','close','tick_volume']].tail(10))
    print(f"\nUnique dates: {sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())}")

mt5.shutdown()
