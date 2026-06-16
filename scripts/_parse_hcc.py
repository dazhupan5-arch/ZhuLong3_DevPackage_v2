import struct
from datetime import datetime, timezone

hcc_path = r"C:\Users\xiaomi\AppData\Roaming\MetaQuotes\Terminal\FBDFAF8C1BBCC753E56F56955E7DF012\bases\WCGGroup-Server\history\XAUUSD\2026.hcc"

with open(hcc_path, 'rb') as f:
    data = f.read()

print(f"File size: {len(data)} bytes")

# MT5 HCC header is 304 bytes
HEADER_SIZE = 304
BAR_SIZE = 60  # each bar record

header = data[:HEADER_SIZE]
version = struct.unpack_from('<i', header, 0)[0]
period = struct.unpack_from('<i', header, 80)[0]
digits = struct.unpack_from('<i', header, 84)[0]

print(f"Version: {version}, Period: {period}, Digits: {digits}")

body = data[HEADER_SIZE:]
num_bars = len(body) // BAR_SIZE
print(f"Total bars: {num_bars}")

# Read all bars
bars = []
for i in range(num_bars):
    offset = i * BAR_SIZE
    ctm, op, hi, lo, cl, vol, spread, rvol = struct.unpack_from('<qdddddqid', body, offset)
    bars.append({
        'time': ctm,
        'open': op,
        'high': hi,
        'low': lo,
        'close': cl,
        'volume': vol,
        'spread': spread,
        'real_vol': rvol
    })

print(f"First bar: dt={datetime.fromtimestamp(bars[0]['time'], tz=timezone.utc)}, close={bars[0]['close']}")
print(f"Last bar:  dt={datetime.fromtimestamp(bars[-1]['time'], tz=timezone.utc)}, close={bars[-1]['close']}")

# Check for June 12
jun12_bars = [b for b in bars if datetime.fromtimestamp(b['time'], tz=timezone.utc).strftime('%Y%m%d') == '20260612']
print(f"\nJune 12 bars: {len(jun12_bars)}")
if jun12_bars:
    print(f"Range: {datetime.fromtimestamp(jun12_bars[0]['time'], tz=timezone.utc)} -> {datetime.fromtimestamp(jun12_bars[-1]['time'], tz=timezone.utc)}")
    for b in jun12_bars[:3]:
        print(f"  {datetime.fromtimestamp(b['time'], tz=timezone.utc)} O={b['open']} H={b['high']} L={b['low']} C={b['close']} V={b['volume']}")
    if len(jun12_bars) > 3:
        print(f"  ... ({len(jun12_bars)} bars)")
        for b in jun12_bars[-3:]:
            print(f"  {datetime.fromtimestamp(b['time'], tz=timezone.utc)} O={b['open']} H={b['high']} L={b['low']} C={b['close']} V={b['volume']}")
else:
    # Show last 10 bars
    print("\nLast 10 bars:")
    for b in bars[-10:]:
        print(f"  {datetime.fromtimestamp(b['time'], tz=timezone.utc)} C={b['close']}")
