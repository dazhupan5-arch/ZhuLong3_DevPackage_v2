namespace ZhuLong.Core.Macro;

/// <summary>宏观事件 impact 归一化与高影响判定。</summary>
public static class MacroImpactHelper
{
    public static bool IsHighImpact(string impact) =>
        ImpactRank(impact) >= 3;

    public static string Normalize(string? impact, string? eventName = null)
    {
        var rank = ImpactRank(impact);
        if (rank >= 3) return "high";
        if (rank == 2) return "medium";
        if (rank == 1) return "low";

        return GuessImpactFromName(eventName ?? "");
    }

    public static int ImpactRank(string? impact)
    {
        if (string.IsNullOrWhiteSpace(impact)) return 0;
        var s = impact.Trim().ToLowerInvariant();
        return s switch
        {
            "high" or "3" => 3,
            "medium" or "mid" or "2" => 2,
            "low" or "1" => 1,
            _ => int.TryParse(s, out var n) ? n switch
            {
                >= 3 => 3,
                2 => 2,
                1 => 1,
                _ => 0,
            } : 0,
        };
    }

    public static string GuessImpactFromName(string name)
    {
        var n = name.ToLowerInvariant();
        if (n.Contains("fomc") || n.Contains("nonfarm") || n.Contains("non farm") || n.Contains("nfp") ||
            (n.Contains("payroll") && !n.Contains("private") && !n.Contains("government") &&
             !n.Contains("manufacturing") && !n.Contains("adp")))
            return "high";
        if (n.Contains("cpi") || n.Contains("gdp") || n.Contains("eia") && n.Contains("crude"))
            return "high";
        if (n.Contains("gdp") || n.Contains("pmi") || n.Contains("retail"))
            return "medium";
        return "low";
    }

    public static bool IsTier1Event(string name)
    {
        var n = name.ToLowerInvariant();
        return n.Contains("fomc") || n.Contains("nonfarm") || n.Contains("non farm") ||
               (n.Contains("payroll") && !n.Contains("private") && !n.Contains("government") &&
                !n.Contains("manufacturing") && !n.Contains("adp")) ||
               n.Contains("cpi") || (n.Contains("eia") && n.Contains("crude"));
    }

    public static bool SameEventFamily(string a, string b)
    {
        var na = a.ToLowerInvariant();
        var nb = b.ToLowerInvariant();
        if (IsNfpHeadline(na) && IsNfpHeadline(nb)) return true;
        if (na.Contains("fomc") && nb.Contains("fomc")) return true;
        if (na.Contains("cpi") && nb.Contains("cpi")) return true;
        if (na.Contains("eia") && na.Contains("crude") && nb.Contains("eia") && nb.Contains("crude"))
            return true;
        return false;
    }

    private static bool IsNfpHeadline(string n) =>
        (n.Contains("nonfarm") || n.Contains("non farm")) &&
        n.Contains("payroll") &&
        !n.Contains("private") && !n.Contains("government") && !n.Contains("manufacturing") &&
        !n.Contains("adp");
}
