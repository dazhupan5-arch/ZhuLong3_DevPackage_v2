using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using ZhuLong.Core.Data;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

public sealed class DatabaseService
{
    private readonly IDbContextFactory<ZhuLongDbContext> _factory;
    private readonly ILogger<DatabaseService> _logger;

    public DatabaseService(IDbContextFactory<ZhuLongDbContext> factory, ILogger<DatabaseService> logger)
    {
        _factory = factory;
        _logger = logger;
    }

    public async Task EnsureCreatedAsync(CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await db.Database.EnsureCreatedAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
    }

    private async Task EnsureSchemaReadyAsync(ZhuLongDbContext db, CancellationToken ct)
    {
        await SignalsSchemaMigrator.EnsureReadyAsync(db, _logger, ct);
        await TradesSchemaMigrator.EnsureReadyAsync(db, _logger, ct);
        await MacroEventsSchemaMigrator.EnsureReadyAsync(db, _logger, ct);
    }

    public async Task SaveSignalAsync(SignalModel signal, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        db.Signals.Add(new SignalEntity
        {
            SignalId = signal.SignalId,
            Timestamp = signal.Timestamp,
            Symbol = signal.Symbol,
            Direction = signal.Direction,
            EntryPrice = signal.EntryPrice,
            StopLoss = signal.StopLoss,
            TakeProfit = signal.TakeProfit,
            Confidence = signal.Confidence,
            ExpectedReturn = signal.ExpectedReturn,
            MagicNumber = signal.MagicNumber,
            CommentHint = signal.CommentHint,
            Strategy = signal.Strategy,
            Status = signal.Status,
            ParamsSnapshot = signal.ParamsSnapshot,
            AttributionJson = signal.AttributionJson,
            CreatedAt = signal.CreatedAt,
        });
        await db.SaveChangesAsync(ct);
    }

    public async Task<IReadOnlyList<SignalModel>> GetRecentSignalsAsync(int limit = 50, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        try
        {
            return await db.Signals
                .Where(s => s.Status == "pending" || s.Status == "active" || s.Status == "awaiting_fill")
                .OrderByDescending(s => s.CreatedAt)
                .Take(limit)
                .Select(s => new SignalModel
                {
                    SignalId = s.SignalId,
                    Timestamp = s.Timestamp,
                    Symbol = s.Symbol,
                    Direction = s.Direction,
                    EntryPrice = s.EntryPrice,
                    StopLoss = s.StopLoss,
                    TakeProfit = s.TakeProfit,
                    Confidence = s.Confidence,
                    ExpectedReturn = s.ExpectedReturn,
                    MagicNumber = s.MagicNumber,
                    CommentHint = s.CommentHint,
                    Strategy = s.Strategy,
                    Status = s.Status,
                    CreatedAt = s.CreatedAt,
                })
                .ToListAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "读取 signals 失败");
            return Array.Empty<SignalModel>();
        }
    }

    public async Task<IReadOnlyList<SignalModel>> GetRecentClosedSignalsAsync(int limit = 20, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        try
        {
            var signals = await db.Signals
                .Where(s => s.Status != "pending" && s.Status != "active" && s.Status != "awaiting_fill")
                .OrderByDescending(s => s.CreatedAt)
                .Take(limit * 3)
                .ToListAsync(ct);
            if (signals.Count == 0)
                return Array.Empty<SignalModel>();

            var signalIds = signals.Select(s => s.SignalId).ToList();
            var trades = await db.Trades
                .Where(t => signalIds.Contains(t.SignalId))
                .ToListAsync(ct);
            var tradeBySignal = trades
                .GroupBy(t => t.SignalId)
                .ToDictionary(
                    g => g.Key,
                    g => g.OrderByDescending(t => t.CloseTime ?? 0).First());

            return signals.Select(s =>
            {
                tradeBySignal.TryGetValue(s.SignalId, out var trade);
                return new SignalModel
                {
                    SignalId = s.SignalId,
                    Timestamp = s.Timestamp,
                    Symbol = s.Symbol,
                    Direction = s.Direction,
                    EntryPrice = s.EntryPrice,
                    StopLoss = s.StopLoss,
                    TakeProfit = s.TakeProfit,
                    Confidence = s.Confidence,
                    ExpectedReturn = s.ExpectedReturn,
                    MagicNumber = s.MagicNumber,
                    CommentHint = s.CommentHint,
                    Strategy = s.Strategy,
                    Status = s.Status,
                    CreatedAt = s.CreatedAt,
                    CloseReason = trade?.CloseReason ?? "",
                    PnlPercent = trade?.PnlPercent,
                    CloseTime = trade?.CloseTime,
                };
            })
            .OrderByDescending(s => s.CloseTime ?? s.CreatedAt)
            .Take(limit)
            .ToList();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "读取 closed signals 失败");
            return Array.Empty<SignalModel>();
        }
    }

    public async Task<IReadOnlyList<SignalModel>> GetActiveSignalsAsync(int limit = 50, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        try
        {
            return await db.Signals
                .Where(s => (s.Status == "active" || s.Status == "awaiting_fill")
                    && (s.Direction == "buy" || s.Direction == "sell"))
                .OrderByDescending(s => s.CreatedAt)
                .Take(limit)
                .Select(s => new SignalModel
                {
                    SignalId = s.SignalId,
                    Timestamp = s.Timestamp,
                    Symbol = s.Symbol,
                    Direction = s.Direction,
                    EntryPrice = s.EntryPrice,
                    StopLoss = s.StopLoss,
                    TakeProfit = s.TakeProfit,
                    Confidence = s.Confidence,
                    ExpectedReturn = s.ExpectedReturn,
                    MagicNumber = s.MagicNumber,
                    CommentHint = s.CommentHint,
                    Strategy = s.Strategy,
                    Status = s.Status,
                    CreatedAt = s.CreatedAt,
                })
                .ToListAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "读取 active signals 失败");
            return Array.Empty<SignalModel>();
        }
    }

    public async Task<IReadOnlyList<SignalModel>> GetPendingSignalsAsync(int limit = 30, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        try
        {
            return await db.Signals
                .Where(s => s.Status == "pending")
                .OrderByDescending(s => s.CreatedAt)
                .Take(limit)
                .Select(s => new SignalModel
                {
                    SignalId = s.SignalId,
                    Timestamp = s.Timestamp,
                    Symbol = s.Symbol,
                    Direction = s.Direction,
                    EntryPrice = s.EntryPrice,
                    StopLoss = s.StopLoss,
                    TakeProfit = s.TakeProfit,
                    Confidence = s.Confidence,
                    ExpectedReturn = s.ExpectedReturn,
                    MagicNumber = s.MagicNumber,
                    CommentHint = s.CommentHint,
                    Strategy = s.Strategy,
                    Status = s.Status,
                    CreatedAt = s.CreatedAt,
                })
                .ToListAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "读取 pending signals 失败");
            return Array.Empty<SignalModel>();
        }
    }

    public async Task LogPositionEventAsync(string signalId, string eventType, double? price = null,
        double? volume = null, double? oldSl = null, double? newSl = null, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        db.PositionEvents.Add(new PositionEventEntity
        {
            SignalId = signalId,
            EventTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            EventType = eventType,
            Price = price,
            Volume = volume,
            OldSl = oldSl,
            NewSl = newSl,
        });
        await db.SaveChangesAsync(ct);
    }

    public async Task SaveTradeAsync(TradeModel trade, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        db.Trades.Add(new TradeEntity
        {
            SignalId = trade.SignalId,
            OpenTime = trade.OpenTime,
            OpenPrice = trade.OpenPrice,
            CloseTime = trade.CloseTime,
            ClosePrice = trade.ClosePrice,
            PnlPoints = trade.PnlPoints,
            PnlPercent = trade.PnlPercent,
            IsWin = trade.IsWin,
            CloseReason = trade.CloseReason,
        });
        await db.SaveChangesAsync(ct);
        _logger.LogInformation("交易已记录 signal={Signal} pnl={Pnl:F2}% reason={Reason}",
            trade.SignalId, trade.PnlPercent, trade.CloseReason);
    }

    public async Task UpdateSignalStatusAsync(string signalId, string status, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        var row = await db.Signals.FirstOrDefaultAsync(s => s.SignalId == signalId, ct);
        if (row is null) return;
        row.Status = status;
        await db.SaveChangesAsync(ct);
    }

    public async Task UpdateSignalEntryAsync(string signalId, double entryPrice, CancellationToken ct = default)
    {
        await UpdateSignalPlanAsync(signalId, entryPrice, null, null, ct);
    }

    public async Task UpdateSignalPlanAsync(
        string signalId,
        double? entryPrice,
        double? stopLoss,
        double? takeProfit,
        CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        var row = await db.Signals.FirstOrDefaultAsync(s => s.SignalId == signalId, ct);
        if (row is null) return;
        if (entryPrice.HasValue) row.EntryPrice = entryPrice.Value;
        if (stopLoss.HasValue) row.StopLoss = stopLoss.Value;
        if (takeProfit.HasValue) row.TakeProfit = takeProfit.Value;
        await db.SaveChangesAsync(ct);
    }

    public async Task<double> GetTodayClosedPnlPercentAsync(CancellationToken ct = default)
    {
        var start = new DateTimeOffset(DateTime.UtcNow.Date, TimeSpan.Zero).ToUnixTimeSeconds();
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        var trades = await db.Trades
            .Where(t => t.CloseTime != null && t.CloseTime >= start)
            .Select(t => t.PnlPercent ?? 0)
            .ToListAsync(ct);
        return trades.Sum();
    }

    /// <summary>品种最近有效信号发射时间（Unix UTC），供风控冷却恢复。</summary>
    public async Task<DateTime?> GetLastEmittedSignalUtcAsync(string symbol, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(symbol))
            return null;

        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        var sym = symbol.Trim();
        var lastUnix = await db.Signals
            .Where(s => s.Symbol == sym)
            .Where(s => s.Direction == "buy" || s.Direction == "sell")
            .Where(s => s.Status != "rejected" && s.Status != "intent_cancelled" && s.Status != "expired")
            .Select(s => (long?)s.CreatedAt)
            .MaxAsync(ct);

        if (lastUnix is null or <= 0)
            return null;

        return DateTimeOffset.FromUnixTimeSeconds(lastUnix.Value).UtcDateTime;
    }

    public async Task<IReadOnlyList<TradeModel>> GetRecentTradesAsync(int limit = 30, CancellationToken ct = default)
    {
        await using var db = await _factory.CreateDbContextAsync(ct);
        await EnsureSchemaReadyAsync(db, ct);
        return await db.Trades
            .OrderByDescending(t => t.CloseTime)
            .Take(limit)
            .Select(t => new TradeModel
            {
                TradeId = t.TradeId,
                SignalId = t.SignalId,
                OpenTime = t.OpenTime,
                OpenPrice = t.OpenPrice,
                CloseTime = t.CloseTime,
                ClosePrice = t.ClosePrice,
                PnlPoints = t.PnlPoints,
                PnlPercent = t.PnlPercent,
                IsWin = t.IsWin,
                CloseReason = t.CloseReason,
            })
            .ToListAsync(ct);
    }
}
