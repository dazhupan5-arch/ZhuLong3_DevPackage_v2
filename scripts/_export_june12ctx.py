import MetaTrader5 as mt5
import pandas as pd

wcg_path = r"C:\Program Files\WCG Group MT5 Terminal\terminal64.exe"
mt5.initialize(path=wcg_path)

rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 800)
df = pd.DataFrame(rates)
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('datetime')
df = df.rename(columns={"tick_volume": "volume"})

print(f"Total bars: {len(df)}")
print(f"Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

# Count by date
dates = sorted(df['datetime'].dt.strftime('%Y-%m-%d').unique())
for d in dates:
    cnt = len(df[df['datetime'].dt.strftime('%Y-%m-%d') == d])
    print(f"  {d}: {cnt} bars")

# Separate June 11 + June 12
jun11 = df[df['datetime'].dt.strftime('%Y-%m-%d').isin(['2026-06-11', '2026-06-12'])]
print(f"\nJune 11+12 total: {len(jun11)} bars")

# Save
jun11.to_csv(r"D:\trae_projects\ZhuLong3_DevPackage_v2\scripts\_june12_ctx_bars.csv", index=False)
print(f"Saved to _june12_ctx_bars.csv")

mt5.shutdown()
