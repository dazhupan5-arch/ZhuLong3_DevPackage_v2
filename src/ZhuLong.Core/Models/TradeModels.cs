using ZhuLong.Core;

namespace ZhuLong.Core.Models;

public sealed class TradeModel
{
    public long TradeId { get; init; }
    public string SignalId { get; init; } = "";
    public long OpenTime { get; init; }
    public double OpenPrice { get; init; }
    public long? CloseTime { get; init; }
    public double? ClosePrice { get; init; }
    public double? PnlPoints { get; init; }
    public double? PnlPercent { get; init; }
    public int? IsWin { get; init; }
    public string? CloseReason { get; init; }
    public string OpenTimeText => ChinaTime.Format(DateTimeOffset.FromUnixTimeSeconds(OpenTime), "yyyy-MM-dd HH:mm");
    public string PnlText => PnlPercent.HasValue ? $"{PnlPercent:F2}%" : "—";
}

public sealed class AttributionBinRow
{
    public string BinLabel { get; init; } = "";
    public int Count { get; init; }
    public double WinRate { get; init; }
    public double AvgPnlPct { get; init; }
    public string WinRateText => $"{WinRate * 100:F1}%";
    public string AvgPnlText => $"{AvgPnlPct:F2}%";
}

public sealed class AttributionSummary
{
    public int TotalTrades { get; init; }
    public double WinRate { get; init; }
    public double AvgPnlPct { get; init; }
    public double ProfitFactor { get; init; }
    public IReadOnlyList<TradeModel> RecentTrades { get; init; } = [];
    public IReadOnlyList<AttributionBinRow> ConfidenceBins { get; init; } = [];
    public double[] CumulativePnlPct { get; init; } = [];
}
