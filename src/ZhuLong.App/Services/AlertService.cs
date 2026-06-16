using System.Net.Http.Json;
using Microsoft.Extensions.Logging;

namespace ZhuLong.App.Services;

/// <summary>关键事件告警（日志 + 可选 Webhook）。</summary>
public sealed class AlertService
{
    private readonly ILogger<AlertService> _logger;
    private readonly UserSecretsStore _secrets;

    public AlertService(ILogger<AlertService> logger, UserSecretsStore secrets)
    {
        _logger = logger;
        _secrets = secrets;
    }

    public event Action<string>? AlertRaised;

    public void Raise(string level, string message) =>
        RaiseEvent("generic", level, message);

    public void RaiseEvent(string eventType, string level, string message, object? extra = null)
    {
        var line = $"[{level}] {message}";
        _logger.LogWarning("{Alert}", line);
        AlertRaised?.Invoke(line);
        _ = TryWebhookAsync(eventType, level, message, extra);
    }

    public void RaiseDailyLoss(double todayPnlPct, double limitPct) =>
        RaiseEvent("daily_loss", "CRITICAL",
            $"日损 {todayPnlPct:F2}% 已达上限 -{limitPct:F2}%",
            new { today_pnl_pct = todayPnlPct, limit_pct = limitPct });

    public void RaiseMt5Disconnected() =>
        RaiseEvent("mt5_disconnect", "ERROR", "MT5 连接丢失");

    public void RaiseMt5Reconnecting() =>
        RaiseEvent("mt5_reconnect", "WARN", "MT5 断线，重连中…");

    public void RaiseOrderFailed(string action, long ticket, string detail) =>
        RaiseEvent("order_failed", "ERROR", $"{action}失败 ticket={ticket}: {detail}",
            new { action, ticket, detail });

    private async Task TryWebhookAsync(string eventType, string level, string message, object? extra)
    {
        var url = _secrets.ResolveAlertWebhookUrl();
        if (string.IsNullOrWhiteSpace(url)) return;
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(8) };
            await http.PostAsJsonAsync(url, new
            {
                app = "ZhuLong",
                event_type = eventType,
                level,
                text = message,
                message,
                timestamp = DateTime.UtcNow.ToString("o"),
                extra,
            });
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Webhook 告警发送失败");
        }
    }
}
