using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>config.json 基本合法性校验（Phase 0 JSON Schema 轻量替代）。</summary>
public static class ConfigValidator
{
    public static IReadOnlyList<string> Validate(AppSettings s)
    {
        var errors = new List<string>();
        var f = s.SignalFilters;
        if (f is not null)
        {
            if (f.ProbThreshold is < 0 or > 1) errors.Add("signal_filters.prob_threshold 须在 0~1");
            if (f.MinRiskReward < 0) errors.Add("signal_filters.min_risk_reward 不能为负");
            if (f.CooldownMinutes < 0) errors.Add("signal_filters.cooldown_minutes 不能为负");
        }

        var pm = s.PositionManagement;
        if (pm is not null)
        {
            if (pm.MaxHoldMinutes <= 0) errors.Add("position_management.max_hold_minutes 须 > 0");
            if (pm.OrderRetryMax < 1) errors.Add("position_management.order_retry_max 须 >= 1");
        }

        var rg = s.RiskGuard;
        if (rg is not null && rg.Enabled)
        {
            if (rg.MaxDailyLossPct <= 0) errors.Add("risk_guard.max_daily_loss_pct 须 > 0");
            if (rg.MaxConcurrentPositions < 1) errors.Add("risk_guard.max_concurrent_positions 须 >= 1");
        }

        return errors;
    }
}
