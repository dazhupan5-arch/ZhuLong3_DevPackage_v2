using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

public sealed class ModelManifestServiceTests
{
    [Fact]
    public void FormatBlock_renders_acceptance_metrics()
    {
        var info = new ModelManifestInfo
        {
            Symbol = "XAUUSD",
            ModelVersion = "v11+x12_rules",
            TrainedAt = "2026-06-08",
            Stage = "v12",
            AcceptancePassed = true,
            Test1WinRate = 0.557,
            Test1Trades = 97,
            ProfitFactor = 1.65,
            MaxDrawdownR = 0.15,
            Note = "测试备注",
        };

        var text = info.FormatBlock();
        Assert.Contains("XAUUSD", text);
        Assert.Contains("55.7%", text);
        Assert.Contains("97 笔", text);
        Assert.Contains("1.65", text);
    }
}
