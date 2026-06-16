using System.Globalization;

namespace ZhuLong.Core;

/// <summary>界面与报告时间按北京时间（UTC+8）展示。</summary>
public static class ChinaTime
{
    private static readonly Lazy<TimeZoneInfo> ZoneLazy = new(ResolveZone);

    public static TimeZoneInfo Zone => ZoneLazy.Value;

    private static TimeZoneInfo ResolveZone()
    {
        foreach (var id in new[] { "China Standard Time", "Asia/Shanghai" })
        {
            try { return TimeZoneInfo.FindSystemTimeZoneById(id); }
            catch { /* next */ }
        }
        return TimeZoneInfo.CreateCustomTimeZone("CST", TimeSpan.FromHours(8), "CST", "CST");
    }

    public static DateTimeOffset ToBeijing(DateTimeOffset utcOrAny) => TimeZoneInfo.ConvertTime(utcOrAny, Zone);

    public static string Format(DateTimeOffset utcOrAny, string format = "yyyy-MM-dd HH:mm:ss") =>
        ToBeijing(utcOrAny).ToString(format, CultureInfo.InvariantCulture);
}
