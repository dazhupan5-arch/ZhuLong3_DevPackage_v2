namespace ZhuLong.Core.Tests;

/// <summary>模型四件套完整性（G3 门禁）。</summary>
public sealed class ModelArtifactsTests
{
    private static readonly string[] RequiredFiles =
    {
        "transformer_encoder.pth",
        "scaler.pkl",
        "xgb_classifier.json",
        "xgb_regressor.json",
        "manifest.json",
    };

    [Theory]
    [InlineData("XAUUSD")]
    [InlineData("USOIL")]
    public void Symbol_HasAllRequiredArtifacts(string symbol)
    {
        var root = FindRepoRoot();
        var dir = Path.Combine(root, "models", symbol);
        Assert.True(Directory.Exists(dir), $"models/{symbol} 目录不存在");

        foreach (var file in RequiredFiles)
        {
            var path = Path.Combine(dir, file);
            Assert.True(File.Exists(path), $"缺少 models/{symbol}/{file}");
            Assert.True(new FileInfo(path).Length > 0, $"models/{symbol}/{file} 为空");
        }
    }

    private static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 10; i++)
        {
            if (File.Exists(Path.Combine(dir, "config.json")) &&
                Directory.Exists(Path.Combine(dir, "models")))
                return dir;

            dir = Directory.GetParent(dir)?.FullName ?? dir;
        }

        throw new InvalidOperationException("无法定位仓库根目录");
    }
}
