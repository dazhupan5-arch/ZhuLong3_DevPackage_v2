using System.Globalization;

namespace ZhuLong.Core;

/// <summary>
/// MT5 K 线时间：内部存 UTC Unix 对应的北京时间（Kind=Unspecified），展示与 M5 分桶均用此轴。
/// </summary>
public static class Mt5Time
{
    /// <summary>UTC Unix 秒 → 北京时间（与 UI / 宏观倒计时同一墙钟）。</summary>
    public static DateTime FromUnixUtcSeconds(long unixUtcSeconds)
    {
        var utc = DateTimeOffset.FromUnixTimeSeconds(unixUtcSeconds);
        return ChinaTime.ToBeijing(utc).DateTime;
    }

    /// <summary>日志 / 界面展示 K 线开盘时刻（北京时间）。</summary>
    public static string FormatBar(DateTime barTimeBeijing, string format = "yyyy-MM-dd HH:mm") =>
        barTimeBeijing.ToString(format, CultureInfo.InvariantCulture);

    public static string FormatBarNow(string format = "HH:mm:ss") =>
        ChinaTime.Format(DateTimeOffset.UtcNow, format);
}
