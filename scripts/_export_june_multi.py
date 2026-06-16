import MetaTrader5 as mt5
import pandas as pd

wcg_path = r"C:\Program Files\WCG Group MT5 Terminal\terminal64.exe"
mt5.initialize(path=wcg_path)

# Pull all available recent bars
rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 5000)
df = pd.DataFrame(rates)
df['datetime'] = pd.to_datetime(df['time'], unit='s')
df = df.sort_values('datetime').reset_index(drop=True)
df = df.rename(columns={"tick_volume": "volume"})

print(f"Total bars: {len(df)}")
print(f"Range: {df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]}")

# Analyze each trading day
df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
daily = df.groupby('date').agg(
    open=('open', 'first'),
    close=('close', 'last'),
    high=('high', 'max'),
    low=('low', 'min'),
    bars=('datetime', 'count'),
    volume=('volume', 'sum'),
).reset_index()

daily['range_pct'] = (daily['high'] - daily['low']) / daily['open'] * 100
daily['change_pct'] = (daily['close'] - daily['open']) / daily['open'] * 100
daily['day_of_week'] = pd.to_datetime(daily['date']).dt.day_name()

print(f"\n{'Date':>12s}  {'Day':>9s}  {'Bars':>5s}  {'Open':>8s}  {'Close':>8s}  {'High':>8s}  {'Low':>8s}  {'Chg%':>7s}  {'Rng%':>7s}")
print("-" * 85)
for _, r in daily.iterrows():
    print(f"{r['date']:>12s}  {r['day_of_week']:>9s}  {r['bars']:>5.0f}  {r['open']:>8.1f}  {r['close']:>8.1f}  {r['high']:>8.1f}  {r['low']:>8.1f}  {r['change_pct']:>+6.2f}%  {r['range_pct']:>6.2f}%")

# Save all data
df.to_csv(r"D:\trae_projects\ZhuLong3_DevPackage_v2\scripts\_june_multi_bars.csv", index=False)
print(f"\nSaved {len(df)} bars to _june_multi_bars.csv")

mt5.shutdown()
