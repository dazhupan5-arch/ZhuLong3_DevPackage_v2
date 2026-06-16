using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

/// <summary>统一行情快照：管道 M1 与 MT5 tick 合并，供穿价检测与撮合共用。</summary>
public sealed class MarketSnapshotStore
{
    private readonly object _lock = new();
    private readonly Dictionary<string, SymbolMarketSnapshot> _bySymbol = new(StringComparer.OrdinalIgnoreCase);

    public void UpdateFromBar(M1Bar bar)
    {
        lock (_lock)
        {
            if (!_bySymbol.TryGetValue(bar.Symbol, out var snap))
            {
                snap = new SymbolMarketSnapshot { Symbol = bar.Symbol };
                _bySymbol[bar.Symbol] = snap;
            }

            snap.BarTime = bar.Time;
            snap.BarOpen = bar.Open;
            snap.BarHigh = bar.High;
            snap.BarLow = bar.Low;
            snap.BarClose = bar.Close;
            snap.HasBar = true;
            snap.LastBarUpdateUtc = DateTime.UtcNow;
        }
    }

    public void UpdateFromTick(string symbol, double bid, double ask, long tickTime = 0)
    {
        if (string.IsNullOrWhiteSpace(symbol))
            return;

        lock (_lock)
        {
            if (!_bySymbol.TryGetValue(symbol, out var snap))
            {
                snap = new SymbolMarketSnapshot { Symbol = symbol };
                _bySymbol[symbol] = snap;
            }

            if (bid > 0) snap.Bid = bid;
            if (ask > 0) snap.Ask = ask;
            if (tickTime > 0) snap.TickTime = tickTime;
            snap.HasTick = bid > 0 || ask > 0;
            snap.LastTickUpdateUtc = DateTime.UtcNow;
        }
    }

    public SymbolMarketSnapshot? Get(string symbol)
    {
        lock (_lock)
            return _bySymbol.TryGetValue(symbol, out var snap) ? snap.Clone() : null;
    }

    public sealed class SymbolMarketSnapshot
    {
        public string Symbol { get; init; } = "";
        public double Bid { get; set; }
        public double Ask { get; set; }
        public long TickTime { get; set; }
        public bool HasTick { get; set; }
        public DateTime LastTickUpdateUtc { get; set; }

        public DateTime BarTime { get; set; }
        public double BarOpen { get; set; }
        public double BarHigh { get; set; }
        public double BarLow { get; set; }
        public double BarClose { get; set; }
        public bool HasBar { get; set; }
        public DateTime LastBarUpdateUtc { get; set; }

        public SymbolMarketSnapshot Clone() => new()
        {
            Symbol = Symbol,
            Bid = Bid,
            Ask = Ask,
            TickTime = TickTime,
            HasTick = HasTick,
            LastTickUpdateUtc = LastTickUpdateUtc,
            BarTime = BarTime,
            BarOpen = BarOpen,
            BarHigh = BarHigh,
            BarLow = BarLow,
            BarClose = BarClose,
            HasBar = HasBar,
            LastBarUpdateUtc = LastBarUpdateUtc,
        };
    }
}

/// <summary>限价撮合：tick 与 M1 穿价统一判定。</summary>
public static class IntentFillMatcher
{
    public static bool TryMatchFill(
        string direction,
        double targetEntry,
        MarketSnapshotStore.SymbolMarketSnapshot? snap,
        out double fillPrice,
        out string source)
    {
        fillPrice = 0;
        source = "";
        if (targetEntry <= 0 || snap is null)
            return false;

        if (direction == "buy")
        {
            if (snap.Ask > 0 && snap.Ask <= targetEntry)
            {
                fillPrice = snap.Ask;
                source = "tick";
                return true;
            }

            if (snap.HasBar && snap.BarLow > 0 && snap.BarLow <= targetEntry)
            {
                fillPrice = targetEntry;
                if (snap.Ask > 0 && snap.Ask < fillPrice)
                    fillPrice = snap.Ask;
                source = "M1穿价";
                return true;
            }
        }
        else if (direction == "sell")
        {
            if (snap.Bid > 0 && snap.Bid >= targetEntry)
            {
                fillPrice = snap.Bid;
                source = "tick";
                return true;
            }

            if (snap.HasBar && snap.BarHigh > 0 && snap.BarHigh >= targetEntry)
            {
                fillPrice = targetEntry;
                if (snap.Bid > 0 && snap.Bid > fillPrice)
                    fillPrice = snap.Bid;
                source = "M1穿价";
                return true;
            }
        }

        return false;
    }
}
