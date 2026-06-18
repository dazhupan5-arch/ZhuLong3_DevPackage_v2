using System.Text.Json;
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

        var signalIds = closed.Select(c => c.SignalId).ToList();
        var signals = await db.Signals
            .Where(s => signalIds.Contains(s.SignalId))
            .ToDictionaryAsync(s => s.SignalId, ct);

        var rows = closed.Select(t =>
        {
            signals.TryGetValue(t.SignalId, out var sig);
            return new AttributionRow(
                t,
                sig?.Confidence ?? 0,
                sig?.AttributionJson);
        }).ToList();

        var confidenceBins = BuildConfidenceBins(rows);
        var horizonBins = BuildLayerBins(rows, snap => $"horizon={ReadSnap(snap, "horizon_direction", "?")}");
        var regimeBins = BuildLayerBins(rows, snap => $"regime={ReadSnap(snap, "cognition_regime", "unknown")}");
        var gateBins = BuildLayerBins(rows, snap =>
        {
            var fr = ReadSnap(snap, "filter_reason", "");
            return $"gate={(string.IsNullOrWhiteSpace(fr) ? "none" : fr)}";
        });
        var kn2Bins = BuildLayerBins(rows, snap =>
        {
            var shadow = ReadSnapBool(snap, "kn2_shadow_mode");
            var should = ReadSnapBool(snap, "kn2_should_trade");
            var label = should ? "allow" : shadow ? "veto" : "na";
            return $"kn2={label}";
        });
        var tune = BuildTuneSuggestions(horizonBins, regimeBins, confidenceBins, total, winRate);

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
            ConfidenceBins = confidenceBins,
            HorizonBins = horizonBins,
            RegimeBins = regimeBins,
            GateBins = gateBins,
            Kn2Bins = kn2Bins,
            TuneSuggestions = tune,
            CumulativePnlPct = cumulative.ToArray(),
        };
    }

    private sealed record AttributionRow(TradeModel Trade, double Confidence, string? AttributionJson);

    private static List<AttributionBinRow> BuildConfidenceBins(List<AttributionRow> rows)
    {
        var bins = new[]
        {
            ("0.60-0.70", 0.60, 0.70),
            ("0.70-0.80", 0.70, 0.80),
            ("0.80-0.90", 0.80, 0.90),
            ("0.90-1.00", 0.90, 1.01),
        };
        return bins.Select(b =>
        {
            var inBin = rows.Where(r => r.Confidence >= b.Item2 && r.Confidence < b.Item3).ToList();
            var n = inBin.Count;
            return new AttributionBinRow
            {
                BinLabel = b.Item1,
                Count = n,
                WinRate = n > 0 ? (double)inBin.Count(x => (x.Trade.PnlPercent ?? 0) > 0) / n : 0,
                AvgPnlPct = n > 0 ? inBin.Average(x => x.Trade.PnlPercent ?? 0) : 0,
            };
        }).ToList();
    }

    private static List<AttributionLayerRow> BuildLayerBins(
        List<AttributionRow> rows,
        Func<JsonElement?, string> labelFn)
    {
        var groups = rows.GroupBy(r => labelFn(ParseSnap(r.AttributionJson)));
        return groups.Select(g =>
        {
            var n = g.Count();
            var wins = g.Count(x => (x.Trade.PnlPercent ?? 0) > 0);
            return new AttributionLayerRow
            {
                Layer = g.Key.Split('=')[0],
                Label = g.Key,
                Count = n,
                WinRate = n > 0 ? (double)wins / n : 0,
                AvgPnlPct = n > 0 ? g.Average(x => x.Trade.PnlPercent ?? 0) : 0,
            };
        }).OrderByDescending(x => x.Count).ToList();
    }

    private static List<AttributionTuneRow> BuildTuneSuggestions(
        List<AttributionLayerRow> horizon,
        List<AttributionLayerRow> regime,
        List<AttributionBinRow> confidence,
        int total,
        double winRate)
    {
        var outRows = new List<AttributionTuneRow>();
        if (total >= 5 && winRate < 0.45)
        {
            outRows.Add(new AttributionTuneRow
            {
                Key = "meta_learning",
                Reason = $"整体胜率 {winRate * 100:F1}% (n={total})",
                Action = "已启用 AgentScheduler 在线偏置适应",
                Priority = "high",
            });
        }
        foreach (var b in regime.Where(x => x.Label.Contains("ranging") && x.Count >= 5 && x.WinRate < 0.42))
        {
            outRows.Add(new AttributionTuneRow
            {
                Key = "execution_gates.structure_location_gate",
                Reason = $"{b.Label} 胜率 {b.WinRate * 100:F1}%",
                Action = "保持或收紧 structure_location_gate",
                Priority = "high",
            });
        }
        foreach (var b in horizon.Where(x => x.Count >= 5 && !x.Label.Contains("flat") && x.WinRate < 0.45))
        {
            outRows.Add(new AttributionTuneRow
            {
                Key = "horizon_min_confidence",
                Reason = $"{b.Label} 胜率 {b.WinRate * 100:F1}%",
                Action = "运行 calibrate_horizon_v16.py",
                Priority = "medium",
            });
        }
        foreach (var b in confidence.Where(x => x.BinLabel.Contains("0.55-0.65") && x.Count >= 5 && x.WinRate < 0.45))
        {
            outRows.Add(new AttributionTuneRow
            {
                Key = "rl_inference.min_confidence_for_trade",
                Reason = $"中低置信区间胜率 {b.WinRate * 100:F1}%",
                Action = "提高 min_confidence_for_trade",
                Priority = "medium",
            });
        }
        return outRows;
    }

    private static JsonElement? ParseSnap(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try
        {
            using var doc = JsonDocument.Parse(json);
            return doc.RootElement.Clone();
        }
        catch
        {
            return null;
        }
    }

    private static string ReadSnap(JsonElement? snap, string key, string fallback)
    {
        if (snap is null || snap.Value.ValueKind != JsonValueKind.Object) return fallback;
        if (!snap.Value.TryGetProperty(key, out var el)) return fallback;
        return el.ValueKind switch
        {
            JsonValueKind.String => el.GetString() ?? fallback,
            JsonValueKind.Number => el.GetDouble().ToString("F4"),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => fallback,
        };
    }

    private static bool ReadSnapBool(JsonElement? snap, string key)
    {
        if (snap is null || snap.Value.ValueKind != JsonValueKind.Object) return false;
        return snap.Value.TryGetProperty(key, out var el) && el.ValueKind == JsonValueKind.True;
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
