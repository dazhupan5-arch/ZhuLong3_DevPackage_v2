using System.Reflection;

namespace ZhuLong.Core;

public static class AppMetadata
{
    public static string ProductVersion =>
        Assembly.GetExecutingAssembly().GetName().Version?.ToString(3) ?? "1.0.0";

    public static string FormatVersionLine() => $"烛龙 ZhuLong v{ProductVersion}";
}
