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
}
