using System.Globalization;
using ZhuLong.Core;

namespace ZhuLong.Core.Macro;

/// <summary>经济日历时间解析：API 多为 UTC，CSV/Investing 为本地时间。</summary>
public static class MacroEventTime
{
    /// <summary>Finnhub/FMP 字段按 UTC 解析并转为北京时间。</summary>
    public static DateTime? ParseApiUtc(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        if (DateTime.TryParse(
                raw,
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal,
                out var utc))
            return ChinaTime.ToBeijing(new DateTimeOffset(utc, TimeSpan.Zero)).DateTime;
        return null;
    }

    /// <summary>macro_events.csv：已是北京时间/MT5 服务器本地时间。</summary>
    public static DateTime? ParseLocalCsv(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        return DateTime.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.None, out var dt)
            ? dt
            : null;
    }
}
