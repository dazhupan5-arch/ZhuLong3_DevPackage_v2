import pandas as pd

csv_path = r"C:\Program Files\ZhuLong\data\training\lgb\XAUUSD\XAUUSD_M5.csv"
df = pd.read_csv(csv_path)
df['dt'] = pd.to_datetime(df['timestamp'], unit='s')

print("Total bars:", len(df))
print("First:", df['dt'].iloc[0])
print("Last:", df['dt'].iloc[-1])

friday = df[df['dt'].dt.strftime('%Y%m%d') == '20260612']
print(f"\nJune 12 bars: {len(friday)}")
if len(friday) > 0:
    print(f"Range: {friday['dt'].iloc[0]} -> {friday['dt'].iloc[-1]}")
    print(f"Sample:\n{friday[['dt','open','high','low','close','volume']].head(5)}")
    print(f"...\n{friday[['dt','open','high','low','close','volume']].tail(5)}")
else:
    print("NO June 12 data.")
    last_dates = sorted(df['dt'].dt.strftime('%Y-%m-%d').unique())[-10:]
    for d in last_dates:
        cnt = len(df[df['dt'].dt.strftime('%Y-%m-%d') == d])
        wd = pd.Timestamp(d).strftime('%A')
        print(f"  {d} ({wd}): {cnt} bars")
