namespace ZhuLong.Core.Models;

public sealed class MultiStrategyTickResult
{
    public string Symbol { get; init; } = "";
    public string MarketState { get; init; } = "";
    public string ActiveStrategy { get; init; } = "";
    public MultiStrategySignalPayload? Signal { get; init; }
    public bool Skipped { get; init; }
    public string? SkipReason { get; init; }
    public string? RejectReason { get; init; }
    public double? Adx { get; init; }
    public double? AtrRatio { get; init; }
    /// <summary>认知引擎平仓评分 0~1（≥0.65 建议平仓）。</summary>
    public double ExitAssessment { get; init; }
    public string? ExitReason { get; init; }
    public double AiSlPrice { get; init; }
    public double AiTpPrice { get; init; }
    /// <summary>M5 智能体 trail 模式：hold / run / tighten。</summary>
    public string? TrailMode { get; init; }
    public double SuggestedTrailingSl { get; init; }
    public string? PositionMgmtReason { get; init; }
    public string? CognitionRegime { get; init; }
    public double CognitionRegimeConfidence { get; init; }
    /// <summary>RL PPO 原始动作（认知否决时仍记录）。</summary>
    public string? RlRawAction { get; init; }
    /// <summary>智能体最终动作（hold/long/short 等）。</summary>
    public string? AgentAction { get; init; }
    /// <summary>认知层主方向 long/short/flat。</summary>
    public string? CognitionDirection { get; init; }
    public double CognitionConfidence { get; init; }
    public string? FilterReason { get; init; }
    public string? Architecture { get; init; }
    public string? HorizonDirection { get; init; }
    public double HorizonConfidence { get; init; }
    public double HorizonMinConfidence { get; init; }
    public bool Kn2ShouldTrade { get; init; }
    public bool Kn2Advisory { get; init; }
    public string? Kn2Action { get; init; }
    public double Kn2Confidence { get; init; }
    public bool Kn2ShadowMode { get; init; }
    /// <summary>Python draw_payload JSON（优先用于图表绘制）。</summary>
    public string? DrawPayloadJson { get; init; }
    public string? AttributionJson { get; init; }
}

public sealed class MultiStrategySignalPayload
{
    public string Strategy { get; init; } = "";
    public string Symbol { get; init; } = "";
    public string Direction { get; init; } = "";
    public double Confidence { get; init; }
    public double Entry { get; init; }
    public double Sl { get; init; }
    public double Tp { get; init; }
    public string SignalId { get; init; } = "";
    public string? RejectReason { get; init; }
}
