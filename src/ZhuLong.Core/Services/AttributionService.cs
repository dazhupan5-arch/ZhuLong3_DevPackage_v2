using Microsoft.EntityFrameworkCore;
using ZhuLong.Core.Data;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

public sealed class AttributionService
{
    private readonly IDbContextFactory<ZhuLongDbContext> _factory;

    public AttributionService(IDbContextFactory<ZhuLongDbContext> factory) => _factory = factory;

    public async Task<AttributionSummary> LoadSummaryAsync(int tradeLimit = 30, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await TradesSchemaMigrator.EnsureReadyAsync(db, ct: ct);
        await SignalsSchemaMigrator.EnsureReadyAsync(db, ct: ct);

        List<TradeEntity> trades;
        try
        {
            trades = await db.Trades
                .Where(t => t.CloseTime != null)
                .OrderByDescending(t => t.CloseTime)
                .Take(tradeLimit)
                .ToListAsync(ct);
        }
        catch (Exception)
        {
            return new AttributionSummary();
        }

        var closed = trades.Select(MapTrade).ToList();
        var wins = closed.Count(t => t.IsWin == 1);
        var total = closed.Count;
        var winRate = total > 0 ? (double)wins / total : 0;
        var avgPnl = total > 0 ? closed.Average(t => t.PnlPercent ?? 0) : 0;
        var grossWin = closed.Where(t => (t.PnlPercent ?? 0) > 0).Sum(t => t.PnlPercent ?? 0);
        var grossLoss = Math.Abs(closed.Where(t => (t.PnlPercent ?? 0) < 0).Sum(t => t.PnlPercent ?? 0));
        var pf = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? 99 : 0;

        var signals = await db.Signals
            .Where(s => closed.Select(c => c.SignalId).Contains(s.SignalId))
            .ToDictionaryAsync(s => s.SignalId, s => s.Confidence, ct);

        var bins = new[]
        {
            ("0.60-0.70", 0.60, 0.70),
            ("0.70-0.80", 0.70, 0.80),
            ("0.80-0.90", 0.80, 0.90),
            ("0.90-1.00", 0.90, 1.01),
        };
        var binRows = bins.Select(b =>
        {
            var inBin = closed.Where(t =>
                signals.TryGetValue(t.SignalId, out var c) && c >= b.Item2 && c < b.Item3).ToList();
            var n = inBin.Count;
            return new AttributionBinRow
            {
                BinLabel = b.Item1,
                Count = n,
                WinRate = n > 0 ? (double)inBin.Count(x => x.IsWin == 1) / n : 0,
                AvgPnlPct = n > 0 ? inBin.Average(x => x.PnlPercent ?? 0) : 0,
            };
        }).ToList();

        var chronological = closed.OrderBy(t => t.CloseTime).ToList();
        var cumulative = new List<double>();
        double sum = 0;
        foreach (var t in chronological)
        {
            sum += t.PnlPercent ?? 0;
            cumulative.Add(sum);
        }

        return new AttributionSummary
        {
            TotalTrades = total,
            WinRate = winRate,
            AvgPnlPct = avgPnl,
            ProfitFactor = pf,
            RecentTrades = closed,
            ConfidenceBins = binRows,
            CumulativePnlPct = cumulative.ToArray(),
        };
    }

    private static TradeModel MapTrade(TradeEntity e) => new()
    {
        TradeId = e.TradeId,
        SignalId = e.SignalId,
        OpenTime = e.OpenTime,
        OpenPrice = e.OpenPrice,
        CloseTime = e.CloseTime,
        ClosePrice = e.ClosePrice,
        PnlPoints = e.PnlPoints,
        PnlPercent = e.PnlPercent,
        IsWin = e.IsWin,
        CloseReason = e.CloseReason,
    };
}
