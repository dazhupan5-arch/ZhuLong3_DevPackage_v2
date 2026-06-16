using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>日损上限、最大持仓、品种冷却（IMPLEMENTATION_PLAN §4.1 P1）。</summary>
public sealed class RiskGuardService
{
    private readonly Dictionary<string, DateTime> _lastSignalUtc = new(StringComparer.OrdinalIgnoreCase);

    public void RecordSignalEmitted(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
            return;
        _lastSignalUtc[symbol.Trim()] = DateTime.UtcNow;
    }

    /// <summary>启动时从 SQLite 恢复各品种最近信号时间，避免重启绕过冷却。</summary>
    public async Task RestoreFromDatabaseAsync(
        DatabaseService database,
        IEnumerable<string> symbols,
        CancellationToken ct = default)
    {
        foreach (var symbol in symbols)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                continue;
            var key = symbol.Trim();
            var lastUtc = await database.GetLastEmittedSignalUtcAsync(key, ct);
            if (!lastUtc.HasValue)
                continue;
            if (!_lastSignalUtc.TryGetValue(key, out var cached) || lastUtc.Value > cached)
                _lastSignalUtc[key] = lastUtc.Value;
        }
    }

    public string? BlockNewSignal(
        AppSettings settings,
        int openPositionCount,
        int pendingSignalCount,
        double todayClosedPnlPct)
    {
        var rg = settings.RiskGuard;
        if (rg is null || !rg.Enabled) return null;

        if (todayClosedPnlPct <= -Math.Abs(rg.MaxDailyLossPct))
            return $"日损已达 {todayClosedPnlPct:F2}%（上限 -{rg.MaxDailyLossPct:F2}%）";

        if (openPositionCount >= rg.MaxConcurrentPositions)
            return $"持仓数 {openPositionCount} 已达上限 {rg.MaxConcurrentPositions}";

        if (pendingSignalCount >= rg.MaxPendingSignals)
            return $"待匹配信号 {pendingSignalCount} 已达上限 {rg.MaxPendingSignals}";

        return null;
    }

    public string? BlockSymbolCooldown(AppSettings settings, string symbol)
    {
        var rg = settings.RiskGuard;
        if (rg is null || !rg.Enabled || rg.SymbolCooldownMinutes <= 0) return null;
        if (string.IsNullOrWhiteSpace(symbol))
            return null;
        var key = symbol.Trim();
        if (!_lastSignalUtc.TryGetValue(key, out var last)) return null;

        var elapsed = DateTime.UtcNow - last;
        if (elapsed.TotalMinutes < rg.SymbolCooldownMinutes)
        {
            var remain = rg.SymbolCooldownMinutes - (int)elapsed.TotalMinutes;
            return $"{key} 冷却中（剩余约 {remain} 分钟）";
        }
        return null;
    }
}
