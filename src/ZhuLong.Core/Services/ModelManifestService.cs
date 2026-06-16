using System.Text.Json;

namespace ZhuLong.Core.Services;

public sealed class ModelManifestInfo
{
    public required string Symbol { get; init; }
    public string ModelVersion { get; init; } = "";
    public string TrainedAt { get; init; } = "";
    public string Stage { get; init; } = "";
    public bool AcceptancePassed { get; init; }
    public double? Test1WinRate { get; init; }
    public double? Test1ShortWinRate { get; init; }
    public double? ValPrecision { get; init; }
    public int? Test1Trades { get; init; }
    public double? ProfitFactor { get; init; }
    public double? MaxDrawdownR { get; init; }
    public string Note { get; init; } = "";

    public string FormatBlock()
    {
        var lines = new List<string>
        {
            $"{Symbol} · {ModelVersion} ({Stage})",
            $"训练日期 {TrainedAt} · 验收 {(AcceptancePassed ? "通过" : "未通过")}",
        };
        if (Test1WinRate is not null)
            lines.Add($"样本外胜率 {Test1WinRate.Value:P1}" +
                      (Test1ShortWinRate is not null ? $"（空 {Test1ShortWinRate.Value:P1}）" : ""));
        if (ValPrecision is not null)
            lines.Add($"验证精确率 {ValPrecision.Value:P1}");
        if (Test1Trades is not null)
            lines.Add($"样本外交易 {Test1Trades} 笔");
        if (ProfitFactor is not null)
            lines.Add($"盈亏比 {ProfitFactor.Value:F2}");
        if (MaxDrawdownR is not null)
            lines.Add($"最大回撤 {MaxDrawdownR.Value:F2}R");
        if (!string.IsNullOrWhiteSpace(Note))
            lines.Add(Note);
        return string.Join(Environment.NewLine, lines);
    }
}

public static class ModelManifestService
{
    public static IReadOnlyList<ModelManifestInfo> ReadInstalled(IEnumerable<string> symbols)
    {
        var list = new List<ModelManifestInfo>();
        foreach (var sym in symbols.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            var info = TryRead(sym);
            if (info is not null)
                list.Add(info);
        }
        return list;
    }

    public static ModelManifestInfo? TryRead(string symbol)
    {
        var dir = AppPaths.ModelDir(symbol);
        var manifestPath = Path.Combine(dir, "manifest.json");
        if (!File.Exists(manifestPath))
            return null;

        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(manifestPath));
            var root = doc.RootElement;
            var metrics = root.TryGetProperty("metrics", out var m) ? m : default;

            int? test1Trades = null;
            double? profitFactor = null;
            double? maxDrawdownR = null;
            var summaryPath = Path.Combine(dir, "acceptance_summary.json");
            if (File.Exists(summaryPath))
            {
                using var sdoc = JsonDocument.Parse(File.ReadAllText(summaryPath));
                var s = sdoc.RootElement;
                test1Trades = ReadInt(s, "test1_trades");
                profitFactor = ReadDouble(s, "profit_factor");
                maxDrawdownR = ReadDouble(s, "max_drawdown_r");
            }

            return new ModelManifestInfo
            {
                Symbol = symbol,
                ModelVersion = root.TryGetProperty("model_version", out var mv) ? mv.GetString() ?? "" : "",
                TrainedAt = root.TryGetProperty("trained_at", out var ta) ? ta.GetString() ?? "" : "",
                Stage = root.TryGetProperty("acceptance_stage", out var st) ? st.GetString() ?? "" : "",
                AcceptancePassed = root.TryGetProperty("acceptance_passed", out var ap) && ap.GetBoolean(),
                Test1WinRate = ReadDouble(metrics, "test1_win_rate"),
                Test1ShortWinRate = ReadDouble(metrics, "test1_short_win_rate"),
                ValPrecision = ReadDouble(metrics, "val_precision"),
                Test1Trades = test1Trades,
                ProfitFactor = profitFactor,
                MaxDrawdownR = maxDrawdownR,
                Note = root.TryGetProperty("note", out var note) ? note.GetString() ?? "" : "",
            };
        }
        catch
        {
            return null;
        }
    }

    private static double? ReadDouble(JsonElement parent, string name)
    {
        if (parent.ValueKind != JsonValueKind.Object || !parent.TryGetProperty(name, out var el))
            return null;
        return el.ValueKind switch
        {
            JsonValueKind.Number => el.GetDouble(),
            JsonValueKind.String when double.TryParse(el.GetString(), out var d) => d,
            _ => null,
        };
    }

    private static int? ReadInt(JsonElement parent, string name)
    {
        if (parent.ValueKind != JsonValueKind.Object || !parent.TryGetProperty(name, out var el))
            return null;
        return el.ValueKind switch
        {
            JsonValueKind.Number => el.GetInt32(),
            JsonValueKind.String when int.TryParse(el.GetString(), out var i) => i,
            _ => null,
        };
    }
}
