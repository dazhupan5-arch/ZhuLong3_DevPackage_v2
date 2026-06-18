using System.Text.Json;
using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Services;

/// <summary>V16 智能体栈门禁（Horizon 验收 + ONNX + 可选 KN2 LIVE）。</summary>
public static class AgentStackModelGate
{
    public static ProductionModelGate.CheckResult Check(AppSettings settings)
    {
        var symbols = ModelConfigSync.ResolveSymbols(settings);
        var ready = new List<string>();
        var pending = new List<string>();
        var configPath = AgentConfigSync.ResolveAgentConfigPath(settings);

        if (!File.Exists(configPath))
        {
            pending.Add($"缺少 config_agent.json（{configPath}）");
            return new ProductionModelGate.CheckResult(false, false, ready, pending);
        }

        JsonDocument? doc = null;
        try
        {
            var text = File.ReadAllText(configPath);
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            doc = JsonDocument.Parse(text);
            var root = doc.RootElement;

            if (root.TryGetProperty("enabled", out var en) && en.ValueKind == JsonValueKind.False)
                pending.Add("config_agent.json enabled=false");

            var arch = root.TryGetProperty("architecture", out var archEl) &&
                       archEl.TryGetProperty("version", out var verEl)
                ? verEl.GetString() ?? "legacy"
                : "legacy";
            if (!string.Equals(arch, "v16", StringComparison.OrdinalIgnoreCase))
                pending.Add($"architecture={arch}（实机需 v16）");

            var hp = root.TryGetProperty("architecture", out var a2) &&
                     a2.TryGetProperty("horizon_predictor", out var hpEl)
                ? hpEl
                : default;
            var onnxRel = hp.ValueKind == JsonValueKind.Object &&
                            hp.TryGetProperty("model_path", out var mp)
                ? mp.GetString() ?? "models/horizon_v16.onnx"
                : "models/horizon_v16.onnx";
            var onnxFile = ResolveBundledFile(onnxRel);
            if (!onnxFile.Exists)
                pending.Add($"缺少 Horizon ONNX: {onnxRel}");
            else if (!HorizonMetaPassed(onnxRel))
                pending.Add("Horizon 未通过验收（horizon_v16.meta passed≠true 或 temporal_val≠true）");

            var kn2Enabled = false;
            var kn2Shadow = true;
            if (root.TryGetProperty("kn2", out var kn2El) && kn2El.ValueKind == JsonValueKind.Object)
            {
                kn2Enabled = kn2El.TryGetProperty("enabled", out var ke) && ke.ValueKind == JsonValueKind.True;
                kn2Shadow = !kn2Enabled ||
                            (kn2El.TryGetProperty("shadow_mode", out var sm) && sm.ValueKind == JsonValueKind.True);
                if (kn2Enabled && !kn2Shadow)
                {
                    var kn2Rel = kn2El.TryGetProperty("model_path", out var kmp)
                        ? kmp.GetString() ?? "models/kn2_trader_v16.pth"
                        : "models/kn2_trader_v16.pth";
                    if (!ResolveBundledFile(kn2Rel).Exists)
                        pending.Add($"KN2 LIVE 缺少模型: {kn2Rel}");
                    else if (!Kn2AcceptancePassed())
                        pending.Add("KN2 LIVE 未通过验收（acceptance_report passed≠true）");
                }
            }

            if (root.TryGetProperty("use_rl", out var ur) && ur.ValueKind == JsonValueKind.True)
            {
                var rlRel = "models/rl_agent_xau.zip";
                if (!ResolveBundledFile(rlRel).Exists)
                    pending.Add($"缺少 RL 模型: {rlRel}");
            }

            foreach (var symbol in symbols)
            {
                if (pending.Count == 0)
                    ready.Add(symbol);
            }
            if (pending.Count > 0 && ready.Count == 0)
                pending.Insert(0, "V16 智能体栈未就绪");
        }
        catch (Exception ex)
        {
            pending.Add($"config_agent 解析失败: {ex.Message}");
        }
        finally
        {
            doc?.Dispose();
        }

        var allReady = ready.Count == symbols.Count && pending.Count == 0;
        return new ProductionModelGate.CheckResult(ready.Count > 0, allReady, ready, pending);
    }

    private static bool HorizonMetaPassed(string onnxRel)
    {
        var metaRel = Path.ChangeExtension(onnxRel, ".meta.json");
        if (string.IsNullOrWhiteSpace(metaRel) || metaRel == onnxRel)
            metaRel = "models/horizon_v16.meta.json";
        var metaPath = ResolveBundledFile(metaRel);
        if (!metaPath.Exists)
            metaPath = ResolveBundledFile("models/horizon_v16.meta.json");
        if (!metaPath.Exists)
            return false;
        try
        {
            var text = File.ReadAllText(metaPath.FullName);
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            using var doc = JsonDocument.Parse(text);
            var root = doc.RootElement;
            if (!root.TryGetProperty("passed", out var p) || p.ValueKind != JsonValueKind.True)
                return false;
            if (!root.TryGetProperty("temporal_val", out var tv) || tv.ValueKind != JsonValueKind.True)
                return false;
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool Kn2AcceptancePassed()
    {
        foreach (var rel in new[]
                 {
                     "data/training/reports/kn2_v16/acceptance_report.json",
                     Path.Combine("data", "training", "reports", "kn2_v16", "acceptance_report.json"),
                 })
        {
            var p = ResolveBundledFile(rel);
            if (!p.Exists)
                continue;
            try
            {
                var text = File.ReadAllText(p.FullName);
                if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
                using var doc = JsonDocument.Parse(text);
                return doc.RootElement.TryGetProperty("passed", out var passed) &&
                       passed.ValueKind == JsonValueKind.True;
            }
            catch
            {
                return false;
            }
        }

        return false;
    }

    private static FileInfo ResolveBundledFile(string rel)
    {
        var p = Path.IsPathRooted(rel) ? new FileInfo(rel) : new FileInfo(Path.Combine(AppPaths.InstallDir, rel));
        if (p.Exists) return p;
        var app = new FileInfo(Path.Combine(AppPaths.AppDataDir, rel));
        if (app.Exists) return app;
        var devRoot = AppPaths.FindDevRoot();
        if (!string.IsNullOrEmpty(devRoot))
        {
            var dev = new FileInfo(Path.Combine(devRoot, rel));
            if (dev.Exists) return dev;
        }
        return p;
    }
}
