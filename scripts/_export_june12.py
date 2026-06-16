import MetaTrader5 as mt5
import pandas as pd

wcg_path = r"C:\Program Files\WCG Group MT5 Terminal\terminal64.exe"
mt5.initialize(path=wcg_path)

rates = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M5, 0, 500)
df = pd.DataFrame(rates)
df['dt'] = pd.to_datetime(df['time'], unit='s')

# Filter to June 12 only
jun12 = df[df['dt'].dt.strftime('%Y%m%d') == '20260612'].copy()
jun12 = jun12.sort_values('time')

# Save as timestamp CSV compatible with ZhuLong format
jun12.to_csv(r"D:\trae_projects\ZhuLong3_DevPackage_v2\scripts\_june12_bars.csv", index=False)
print(f"Exported {len(jun12)} bars for June 12, 2026")
print(f"Range: {jun12['dt'].iloc[0]} -> {jun12['dt'].iloc[-1]}")
print(f"Price range: {jun12['low'].min():.2f} - {jun12['high'].max():.2f}")
print(f"First close: {jun12['close'].iloc[0]:.2f}, Last close: {jun12['close'].iloc[-1]:.2f}")

mt5.shutdown()
