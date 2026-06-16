using System.Text.Json;
using ZhuLong.Core.Macro;

namespace ZhuLong.Core.Services;

/// <summary>
/// 宏观 8 维特征合成：经济日历 + fred_latest.json + sentiment.json。
/// 维度: [hours_to_next/48, next_impact, hours_since/48, just_happened,
///        next_event_type, fred_composite, gold_silver_norm, llm_sentiment]
/// </summary>
public static class MacroFeatureBuilder
{
    public const int FeatureDim = 8;

    public static float[] Build(
        IReadOnlyList<MacroEventRecord> events,
        FredSnapshot? fred,
        SentimentSnapshot? sentiment,
        DateTime? now = null)
    {
        now ??= DateTime.Now;
        var future = events.Where(e => e.EventTime >= now.Value).OrderBy(e => e.EventTime).ToList();
        var past = events.Where(e => e.EventTime < now.Value).OrderBy(e => e.EventTime).ToList();

        var hoursToNext = future.Count > 0
            ? Math.Min((future[0].EventTime - now.Value).TotalHours, 999)
            : 999.0;
        var nextImpact = future.Count > 0 ? ImpactValue(future[0].Impact) : 0f;
        var hoursSince = past.Count > 0
            ? Math.Min((now.Value - past[^1].EventTime).TotalHours, 999)
            : 999.0;
        var justHappened = hoursSince <= 1.0 ? 1f : 0f;
        var nextType = future.Count > 0 ? EventTypeCode(future[0].EventName) : 0f;

        var fredScore = fred?.CompositeScore ?? 0.5f;
        var gsNorm = sentiment?.GoldSilverNorm ?? 0.5f;
        var llmSent = sentiment?.OverallSentiment ?? 0.5f;

        return
        [
            (float)Math.Min(hoursToNext / 48.0, 1.0),
            nextImpact,
            (float)Math.Min(hoursSince / 48.0, 1.0),
            justHappened,
            nextType,
            fredScore,
            gsNorm,
            (float)llmSent,
        ];
    }

    public static FredSnapshot? LoadFred(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var updated = root.TryGetProperty("updated_at", out var u) ? u.GetString() : null;
            var latest = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
            foreach (var prop in root.EnumerateObject())
            {
                if (prop.Name is "updated_at" or "meta") continue;
                if (prop.Value.ValueKind != JsonValueKind.Array) continue;
                var arr = prop.Value.EnumerateArray().ToList();
                if (arr.Count == 0) continue;
                var last = arr[^1];
                if (last.TryGetProperty("value", out var v) && v.TryGetDouble(out var d))
                    latest[prop.Name] = d;
            }
            return new FredSnapshot(updated, latest);
        }
        catch
        {
            return null;
        }
    }

    public static SentimentSnapshot? LoadSentiment(string path)
    {
        if (!File.Exists(path)) return null;
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            var root = doc.RootElement;
            var gs = root.TryGetProperty("gold_silver_ratio", out var g) && g.TryGetDouble(out var gd) ? gd : 80.0;
            var xau = root.TryGetProperty("xauusd_sentiment", out var x) && x.TryGetDouble(out var xd) ? xd : 0.5;
            var oil = root.TryGetProperty("usoil_sentiment", out var o) && o.TryGetDouble(out var od) ? od : 0.5;
            var overall = root.TryGetProperty("overall_sentiment", out var s) && s.TryGetDouble(out var sd)
                ? sd
                : (xau + oil) / 2.0;
            return new SentimentSnapshot(gs, xau, oil, overall);
        }
        catch
        {
            return null;
        }
    }

    public static float ImpactValue(string impact) => impact.ToLowerInvariant() switch
    {
        "high" or "3" => 1f,
        "medium" or "mid" or "2" => 0.5f,
        "low" or "1" => 0.25f,
        _ => 0.5f,
    };

    /// <summary>事件类型编码 0~1（FOMC/NFP/CPI 等关键词）。</summary>
    public static float EventTypeCode(string name)
    {
        var n = name.ToLowerInvariant();
        if (n.Contains("fomc") || n.Contains("fed") && n.Contains("rate")) return 1.0f;
        if (n.Contains("nonfarm") || n.Contains("nfp") || n.Contains("payroll")) return 0.9f;
        if (n.Contains("cpi") || n.Contains("inflation")) return 0.85f;
        if (n.Contains("gdp")) return 0.75f;
        if (n.Contains("oil") || n.Contains("eia") || n.Contains("crude")) return 0.7f;
        if (n.Contains("pmi") || n.Contains("ism")) return 0.6f;
        return 0.4f;
    }
}

public sealed class FredSnapshot
{
    public string? UpdatedAt { get; }
    public IReadOnlyDictionary<string, double> Latest { get; }

    public FredSnapshot(string? updatedAt, IReadOnlyDictionary<string, double> latest)
    {
        UpdatedAt = updatedAt;
        Latest = latest;
    }

    /// <summary>宏观背景综合分 0~1（失业率↓、CPI 温和、利率稳定 → 中性 0.5）。</summary>
    public float CompositeScore
    {
        get
        {
            var score = 0.5;
            if (Latest.TryGetValue("UNRATE", out var unrate))
                score += (5.0 - Math.Clamp(unrate, 3.0, 8.0)) / 20.0;
            if (Latest.TryGetValue("FEDFUNDS", out var ff))
                score += (2.5 - Math.Clamp(ff, 0.0, 6.0)) / 20.0;
            if (Latest.TryGetValue("T10YIE", out var inf))
                score += (2.5 - Math.Clamp(inf, 0.5, 4.0)) / 20.0;
            return (float)Math.Clamp(score, 0, 1);
        }
    }
}

public sealed class SentimentSnapshot
{
    public double GoldSilverRatio { get; }
    public double XauusdSentiment { get; }
    public double UsoilSentiment { get; }
    public double OverallSentiment { get; }

    public SentimentSnapshot(double gs, double xau, double oil, double overall)
    {
        GoldSilverRatio = gs;
        XauusdSentiment = xau;
        UsoilSentiment = oil;
        OverallSentiment = overall;
    }

    /// <summary>金银比归一化（典型 60~100）。</summary>
    public float GoldSilverNorm =>
        (float)Math.Clamp((GoldSilverRatio - 60.0) / 40.0, 0, 1);
}
