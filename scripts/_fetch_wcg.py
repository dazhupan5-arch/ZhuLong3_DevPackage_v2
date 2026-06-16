import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

# Connect to WCG terminal specifically
wcg_path = r"C:\Program Files\WCG Group MT5 Terminal\terminal64.exe"
print(f"Connecting to WCG: {wcg_path}")

if mt5.initialize(path=wcg_path):
    info = mt5.terminal_info()
    print(f"Connected: {info.name}")
    print(f"Build: {info.build}")
    print(f"Data path: {info.data_path}")
    
    symbol = "XAUUSD"
    # Get the absolute latest 500 bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        print(f"\nLatest bar: {df['dt'].iloc[-1]} Close: {df['close'].iloc[-1]}")
        print(f"Earliest: {df['dt'].iloc[0]}")
        
        jun12 = df[df['dt'].dt.strftime('%Y%m%d') == '20260612']
        print(f"\nJune 12 bars: {len(jun12)}")
        if len(jun12) > 0:
            print(f"Range: {jun12['dt'].iloc[0]} -> {jun12['dt'].iloc[-1]}")
        
        # Show last 7 days
        dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())[-7:]
        for d in dates:
            cnt = len(df[df['dt'].dt.strftime('%Y-%m-%d') == d])
            wd = pd.Timestamp(d).strftime('%A')
            print(f"  {d} ({wd}): {cnt} bars")
    else:
        print("No rates returned!")

    mt5.shutdown()
else:
    print(f"Failed to initialize WCG: {mt5.last_error()}")
