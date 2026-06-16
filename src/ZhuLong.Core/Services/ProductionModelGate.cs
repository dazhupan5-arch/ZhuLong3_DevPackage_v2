using System.Text.Json;
using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>正式模型门禁：manifest 须 acceptance_passed=true 且非 demo。</summary>
public static class ProductionModelGate
{
    public sealed record CheckResult(
        bool Ready,
        bool AllReady,
        IReadOnlyList<string> ReadySymbols,
        IReadOnlyList<string> PendingReasons)
    {
        public string Summary => AllReady
            ? $"正式模型已就绪：{string.Join(", ", ReadySymbols)}"
            : ReadySymbols.Count > 0
                ? $"部分就绪：{string.Join(", ", ReadySymbols)}（{string.Join("；", PendingReasons)}）"
                : PendingReasons.Count > 0
                    ? string.Join("；", PendingReasons)
                    : "未配置品种";
    }

    public static CheckResult Check(AppSettings settings)
    {
        var symbols = ModelConfigSync.ResolveSymbols(settings);
        var ready = new List<string>();
        var pending = new List<string>();

        foreach (var symbol in symbols)
        {
            var manifestPath = Path.Combine(AppPaths.ModelDir(symbol), "manifest.json");
            if (!File.Exists(manifestPath))
            {
                pending.Add($"{symbol}：缺少 manifest.json");
                continue;
            }

            try
            {
                var text = File.ReadAllText(manifestPath);
                if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
                using var doc = JsonDocument.Parse(text);
                var root = doc.RootElement;

                if (root.TryGetProperty("kind", out var kindEl) &&
                    string.Equals(kindEl.GetString(), "demo", StringComparison.OrdinalIgnoreCase))
                {
                    pending.Add($"{symbol}：当前为演示占位模型");
                    continue;
                }

                if (!root.TryGetProperty("acceptance_passed", out var passedEl) ||
                    passedEl.ValueKind != JsonValueKind.True)
                {
                    pending.Add($"{symbol}：训练产物未通过验收（acceptance_passed）");
                    continue;
                }

                if (!TryResolveMissingArtifact(symbol, out var missingArtifact))
                {
                    pending.Add($"{symbol}：缺少 {Path.GetFileName(missingArtifact!)}");
                    continue;
                }

                ready.Add(symbol);
            }
            catch (Exception ex)
            {
                pending.Add($"{symbol}：manifest 无效（{ex.Message}）");
            }
        }

        var allReady = ready.Count == symbols.Count && pending.Count == 0;
        var anyReady = ready.Count > 0;
        return new CheckResult(anyReady, allReady, ready, pending);
    }

    private static bool TryResolveMissingArtifact(string symbol, out string? missingPath)
    {
        missingPath = RequiredArtifacts(symbol).FirstOrDefault(f => !File.Exists(f));
        return missingPath is null;
    }

    private static IEnumerable<string> RequiredArtifacts(string symbol)
    {
        var dir = AppPaths.ModelDir(symbol);
        var manifestPath = Path.Combine(dir, "manifest.json");
        yield return manifestPath;

        if (IsV14Manifest(manifestPath))
        {
            foreach (var path in V14ArtifactCandidates(dir))
                yield return path;
            yield break;
        }

        if (IsOilV1Manifest(Path.Combine(dir, "manifest.json")))
        {
            yield return Path.Combine(dir, "v1", "xgb_triple_oil.json");
            yield return Path.Combine(dir, "v1", "oil_v1_meta.pkl");
            yield return Path.Combine(dir, "v1", "feature_columns.json");
            yield break;
        }

        yield return Path.Combine(dir, "scaler.pkl");
        yield return Path.Combine(dir, "xgb_regressor.json");
        yield return Path.Combine(dir, "transformer_encoder.pth");
    }

    private static IEnumerable<string> V14ArtifactCandidates(string dir)
    {
        var v14Dir = Path.Combine(dir, "v14");
        yield return Path.Combine(v14Dir, "xgb_v14.json");
        yield return Path.Combine(v14Dir, "feature_columns.json");
        if (File.Exists(Path.Combine(v14Dir, "v14_meta.pkl")))
            yield return Path.Combine(v14Dir, "v14_meta.pkl");
        else
            yield return Path.Combine(v14Dir, "v12_meta.pkl");
    }

    private static bool IsV14Manifest(string manifestPath)
    {
        if (!File.Exists(manifestPath))
            return false;
        try
        {
            var text = File.ReadAllText(manifestPath);
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            using var doc = JsonDocument.Parse(text);
            var root = doc.RootElement;
            if (root.TryGetProperty("model_version", out var verEl) &&
                string.Equals(verEl.GetString(), "v14", StringComparison.OrdinalIgnoreCase))
                return true;
            if (root.TryGetProperty("acceptance_stage", out var stageEl) &&
                string.Equals(stageEl.GetString(), "v14", StringComparison.OrdinalIgnoreCase))
                return true;
            if (root.TryGetProperty("classifier_mode", out var modeEl))
            {
                var mode = modeEl.GetString();
                if (string.Equals(mode, "xau_v14", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(mode, "oil_v14", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
        }
        catch
        {
            /* ignore */
        }

        return false;
    }

    private static bool IsOilV1Manifest(string manifestPath)
    {
        return string.Equals(GetClassifierMode(manifestPath), "oil_v1", StringComparison.OrdinalIgnoreCase);
    }

    private static string? GetClassifierMode(string manifestPath)
    {
        if (!File.Exists(manifestPath))
            return null;
        try
        {
            var text = File.ReadAllText(manifestPath);
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            using var doc = JsonDocument.Parse(text);
            return doc.RootElement.TryGetProperty("classifier_mode", out var modeEl)
                ? modeEl.GetString()
                : null;
        }
        catch
        {
            return null;
        }
    }
}
