using ZhuLong.Core;

namespace ZhuLong.Core.Models;

public sealed class M1Bar
{
    public required string Symbol { get; init; }
    public DateTime Time { get; init; }
    public double Open { get; init; }
    public double High { get; init; }
    public double Low { get; init; }
    public double Close { get; init; }
    public double Volume { get; init; }
}

public sealed record OhlcBar
{
    public DateTime Time { get; init; }
    public double Open { get; init; }
    public double High { get; init; }
    public double Low { get; init; }
    public double Close { get; init; }
    public double Volume { get; init; }
}

public sealed class InferenceResult
{
    /// <summary>交易方向（无信号时为 0/flat）。</summary>
    public int Direction { get; init; }
    /// <summary>展示用置信度（Horizon/认知/信号最强分量，flat 时也保留真实值）。</summary>
    public double Confidence { get; init; }
    public string HorizonDirection { get; init; } = "";
    public double HorizonConfidence { get; init; }
    public string CognitionDirection { get; init; } = "";
    public double CognitionConfidence { get; init; }
    public string RlAction { get; init; } = "";
    public double EntryOffset { get; init; }
    public double ExpectedReturn { get; init; }
}

public sealed class SignalModel
{
    public string SignalId { get; init; } = "";
    public long Timestamp { get; init; }
    public string Symbol { get; init; } = "";
    public string Direction { get; init; } = "";
    public double EntryPrice { get; init; }
    public double StopLoss { get; init; }
    public double TakeProfit { get; init; }
    public double Confidence { get; init; }
    public double ExpectedReturn { get; init; }
    public int MagicNumber { get; init; }
    public string CommentHint { get; init; } = "";
    /// <summary>策略标识：ai_model / trend_system / spread_hedge / grid_system</summary>
    public string Strategy { get; init; } = "";
    public string StrategyDisplay => StrategyNames.Display(Strategy);
    public string Status { get; set; } = "pending";
    public string CloseReason { get; set; } = "";
    /// <summary>平仓盈亏（%），来自 trades 表或实时平仓事件。</summary>
    public double? PnlPercent { get; set; }
    /// <summary>实际平仓时间（Unix UTC），来自 trades.close_time。</summary>
    public long? CloseTime { get; set; }
    public string? ParamsSnapshot { get; init; }
    public string? AttributionJson { get; init; }
    public long CreatedAt { get; init; }
    public string CreatedAtText => ChinaTime.Format(DateTimeOffset.FromUnixTimeSeconds(CreatedAt), "yyyy-MM-dd HH:mm:ss");
    public string ClosedAtText => CloseTime.HasValue
        ? ChinaTime.Format(DateTimeOffset.FromUnixTimeSeconds(CloseTime.Value), "yyyy-MM-dd HH:mm:ss")
        : "—";
    private bool HasTradeRecord => CloseTime.HasValue || PnlPercent.HasValue;
    public string StatusDisplay => Status switch
    {
        "pending" => "⏳ 待执行",
        "awaiting_fill" => "⏳ 挂单意图",
        "active" => "📈 持仓中",
        "expired" => "⏰ 已过期",
        "stop_loss" => "🛑 止损",
        "take_profit" => "🏁 止盈",
        "trailing_stop" => "📉 移动止损",
        "profit_drawdown" => "📉 浮盈回撤",
        "time_stop" => "⏱ 时间止损",
        "model_exit" => "🤖 模型出场",
        "agent_exit" => "🤖 智能体出场",
        "rejected" => "⛔ 已拒绝",
        "intent_cancelled" => "🚫 未成交·已撤销",
        "normal_close" when !HasTradeRecord => "🚫 未成交·已撤销",
        "normal_close" => "📋 平仓",
        _ => Status
    };
    /// <summary>已平仓列表：盈亏展示（pnl_percent 已是百分数，如 1.05 = +1.05%）。</summary>
    public string PnlDisplay => PnlPercent.HasValue
        ? $"{PnlPercent.Value:+0.00;-0.00;0.00}%"
        : "—";
    /// <summary>当前托管列表：补充说明（过期原因等），不含机器码 reason。</summary>
    public string CloseDetailDisplay
    {
        get
        {
            if (string.IsNullOrWhiteSpace(CloseReason)) return "";
            return IsMachineCloseReason(CloseReason.Trim()) ? "" : CloseReason.Trim();
        }
    }

    private static bool IsMachineCloseReason(string reason) => reason switch
    {
        "stop_loss" or "take_profit" or "trailing_stop" or "trailing"
            or "profit_drawdown" or "time_stop" or "model_exit" or "agent_exit"
            or "external_close" or "normal_close" or "closed" => true,
        _ => false
    };
}

public sealed class ManagedPositionModel
{
    public long Ticket { get; init; }
    public string SignalId { get; init; } = "";
    public string Symbol { get; init; } = "";
    public string Direction { get; init; } = "";
    public double EntryPrice { get; init; }
    public double StopLoss { get; init; }
    public double Volume { get; init; }
    public double ProfitPct { get; init; }
    public string TrailingState { get; init; } = "";
    public bool IsManaged { get; init; } = true;
    public bool IsFilled { get; init; } = true;
}
