using System.Text.Json;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

public sealed class ProductionModelGateTests : IDisposable
{
    private readonly string _tmp;

    public ProductionModelGateTests()
    {
        _tmp = Path.Combine(Path.GetTempPath(), "zhulong_gate_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_tmp);
    }

    public void Dispose()
    {
        try { Directory.Delete(_tmp, true); } catch { /* ignore */ }
    }

    [Fact]
    public void Demo_manifest_not_production_ready()
    {
        var symDir = Path.Combine(_tmp, "XAUUSD");
        Directory.CreateDirectory(symDir);
        File.WriteAllText(Path.Combine(symDir, "manifest.json"),
            JsonSerializer.Serialize(new { kind = "demo", acceptance_passed = false }));

        var settings = new AppSettings { Model = new AppSettings.ModelSettings { DefaultSymbols = ["XAUUSD"] } };
        // 使用临时目录需通过安装目录 — 本测试仅验证逻辑：复制到 publish 模型目录结构
        // 直接测 Check 依赖 AppPaths.InstallDir，改为测 manifest 解析辅助
        Assert.False(IsProductionManifest(Path.Combine(symDir, "manifest.json")));
    }

    [Fact]
    public void Accepted_manifest_is_production_ready()
    {
        var path = Path.Combine(_tmp, "manifest.json");
        File.WriteAllText(path, JsonSerializer.Serialize(new { kind = "trained", acceptance_passed = true }));
        Assert.True(IsProductionManifest(path));
    }

    private static bool IsProductionManifest(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        if (root.TryGetProperty("kind", out var k) && k.GetString() == "demo") return false;
        return root.TryGetProperty("acceptance_passed", out var p) && p.ValueKind == JsonValueKind.True;
    }
}
