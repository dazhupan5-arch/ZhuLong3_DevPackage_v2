import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

# Try TMGM Live terminal by specifying its path
paths = [
    r"C:\Program Files\TMGM MT5 Terminal\terminal64.exe",  # TMGM (common)
    r"C:\Program Files\TradeMaxGlobal MT5\terminal64.exe",
    None  # default (WCG)
]

for path in paths:
    print(f"\n{'='*60}")
    print(f"Trying: {path or 'default (WCG)'}")
    mt5.shutdown()
    if path:
        if not mt5.initialize(path=path):
            print("  Failed to initialize")
            continue
    else:
        if not mt5.initialize():
            print("  Failed to initialize")
            continue
    
    info = mt5.terminal_info()
    print(f"  Connected to: {info.name} at {info.path}")
    
    symbol = "XAUUSD"
    # Get last 300 bars
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['dt'] = pd.to_datetime(df['time'], unit='s')
        print(f"  Latest bar: {df['dt'].iloc[-1]}")
        print(f"  Close: {df['close'].iloc[-1]}")
        
        # Check for June 12 data
        jun12 = df[df['dt'].dt.strftime('%Y%m%d') == '20260612']
        print(f"  June 12 bars: {len(jun12)}")
        
        # Show last 5 unique dates
        dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())[-5:]
        for d in dates:
            cnt = len(df[df['dt'].dt.strftime('%Y-%m-%d') == d])
            print(f"    {d}: {cnt} bars")
    else:
        print("  No bars returned!")

mt5.shutdown()
print("\nDone.")
