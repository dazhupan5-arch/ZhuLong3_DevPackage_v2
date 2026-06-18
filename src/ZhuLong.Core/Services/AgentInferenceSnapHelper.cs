using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

/// <summary>智能体 tick 结果 → InferenceSnapshot；日志与 UI 必须反映各层真实输出，禁止 flat 时 conf=0 掩盖。</summary>
public static class AgentInferenceSnapHelper
{
    public static InferenceResult ToInferenceSnap(MultiStrategyTickResult r)
    {
        var direction = ResolveDirection(r);
        var confidence = ResolveConfidence(r);
        return new InferenceResult
        {
            Direction = direction,
            Confidence = confidence,
            HorizonDirection = r.HorizonDirection ?? "",
            HorizonConfidence = r.HorizonConfidence,
            CognitionDirection = r.CognitionDirection ?? "",
            CognitionConfidence = r.CognitionConfidence,
            RlAction = r.AgentAction ?? r.RlRawAction ?? "",
        };
    }

    public static int ResolveDirection(MultiStrategyTickResult r)
    {
        var h = r.HorizonDirection?.Trim().ToLowerInvariant();
        if (h == "long") return 1;
        if (h == "short") return -1;

        var cog = r.CognitionDirection?.Trim().ToLowerInvariant();
        if (cog == "long") return 1;
        if (cog == "short") return -1;

        var sig = r.Signal?.Direction?.Trim().ToLowerInvariant();
        if (sig == "buy") return 1;
        if (sig == "sell") return -1;
        return 0;
    }

    /// <summary>展示用置信度：取 Horizon / 认知 / 信号中最强分量，与是否 flat 无关。</summary>
    public static double ResolveConfidence(MultiStrategyTickResult r)
    {
        var best = 0.0;
        if (r.HorizonConfidence > best) best = r.HorizonConfidence;
        if (r.CognitionConfidence > best) best = r.CognitionConfidence;
        if (r.Signal?.Confidence > best) best = r.Signal.Confidence;
        return best;
    }

    public static string FormatPrimaryTickLog(string symbol, MultiStrategyTickResult r)
    {
        var rl = string.IsNullOrWhiteSpace(r.RlRawAction) ? "—" : r.RlRawAction;
        var action = string.IsNullOrWhiteSpace(r.AgentAction) ? "—" : r.AgentAction;
        var regime = string.IsNullOrWhiteSpace(r.CognitionRegime) ? "—" : r.CognitionRegime;
        var cogDir = string.IsNullOrWhiteSpace(r.CognitionDirection) ? "—" : r.CognitionDirection;
        var cogConf = r.CognitionConfidence;
        var filterNote = string.IsNullOrWhiteSpace(r.FilterReason) ? "" : $" 门控={r.FilterReason}";
        var arch = r.Architecture ?? "";
        var isV16 = string.Equals(arch, "v16", StringComparison.OrdinalIgnoreCase);
        var logPrefix = StrategyNames.AgentLogPrefix(arch);
        if (isV16)
        {
            var hDir = string.IsNullOrWhiteSpace(r.HorizonDirection) ? cogDir : r.HorizonDirection;
            var hConf = StrategyNames.FormatHorizonConfidence(
                r.HorizonConfidence > 0 ? r.HorizonConfidence : cogConf,
                r.HorizonMinConfidence > 0 ? r.HorizonMinConfidence : 0.48);
            var kn2Label = r.Kn2Advisory
                ? "advisory"
                : (r.Kn2ShouldTrade ? "trade" : "hold");
            var kn2Note = r.Kn2ShadowMode || !string.IsNullOrWhiteSpace(r.Kn2Action)
                ? $" KN2={kn2Label}({r.Kn2Confidence:F2})"
                : "";
            return $"{logPrefix} {symbol} Horizon={hDir}({hConf}) 认知={cogDir}({cogConf:F2}) RL={rl} 最终={action} 行情={regime}{filterNote}{kn2Note} 策略={StrategyNames.AgentStrategyLabel(arch)}";
        }

        return $"{logPrefix} {symbol} 认知={cogDir}({cogConf:F2}) RL={rl} 最终={action} 行情={regime}{filterNote} 策略={StrategyNames.LogLabel(r.ActiveStrategy)}";
    }

    public static (
        int opinionDir,
        double displayConf,
        string horizonDir,
        string cognitionDir,
        string rl,
        string finalAction,
        double horizonMin,
        string? architecture) BuildOpinionPublishArgs(MultiStrategyTickResult r)
    {
        var cogDir = r.CognitionDirection ?? "flat";
        var opinionDir = cogDir switch
        {
            "long" => 1,
            "short" => -1,
            _ => 0,
        };
        var displayConf = ResolveConfidence(r);
        if (displayConf <= 0 && r.HorizonConfidence > 0)
            displayConf = r.HorizonConfidence;
        var horizonDir = string.IsNullOrWhiteSpace(r.HorizonDirection) ? cogDir : r.HorizonDirection;
        var horizonMin = r.HorizonMinConfidence > 0 ? r.HorizonMinConfidence : 0.48;
        var rl = string.IsNullOrWhiteSpace(r.RlRawAction) ? "—" : r.RlRawAction;
        var finalAction = string.IsNullOrWhiteSpace(r.AgentAction) ? "—" : r.AgentAction;
        return (opinionDir, displayConf, horizonDir, cogDir, rl, finalAction, horizonMin, r.Architecture);
    }
}
