namespace ZhuLong.Core;

public static class StrategyNames
{
    public static string Display(string? strategyId) => strategyId switch
    {
        "ai_model" => "AI模型",
        "trend_system" => "趋势系统",
        "spread_hedge" => "金油对冲",
        "grid_system" => "ATR网格",
        "rl_agent" => "RL智能体",
        "scheduler_ai" => "自动调度",
        "" or null => "—",
        _ => strategyId,
    };

    public static string DisplayAgentStack(string? architecture) => architecture switch
    {
        "v16" => "Horizon v16",
        _ => Display("rl_agent"),
    };

    public static string AgentLogPrefix(string? architecture) => architecture switch
    {
        "v16" => "[V16·Horizon]",
        _ => "[RL智能体]",
    };

    public static string AgentStrategyLabel(string? architecture) => architecture switch
    {
        "v16" => "V16 · Horizon+RL",
        _ => "RL智能体",
    };

    public static string LogLabel(string? strategyId)
    {
        var id = strategyId ?? "";
        if (string.IsNullOrEmpty(id)) return "—";
        return $"{Display(id)}({id})";
    }

    public static string DisplayMarketState(string? state) => state switch
    {
        "TREND" => "趋势",
        "VOLATILE" => "高波动",
        "RANGE" => "震荡",
        "" or null => "—",
        _ => state,
    };

    public static string FormatHorizonConfidence(double conf, double minConf)
    {
        if (conf <= 0 && minConf <= 0) return "—";
        var mark = conf >= minConf ? "≥" : "<";
        return $"{conf:F2}{mark}{minConf:F2}";
    }
}
