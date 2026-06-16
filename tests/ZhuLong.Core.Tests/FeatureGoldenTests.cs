using System.Security.Cryptography;
using System.Text;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

public class FeatureGoldenTests
{
    private static List<OhlcBar> LoadGoldenM5()
    {
        var path = Path.Combine(FeatureGoldenTestsHelper.FindRepoRoot(), "tests", "fixtures", "golden_m5.csv");
        var lines = File.ReadAllLines(path).Skip(1);
        var bars = new List<OhlcBar>();
        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var p = line.Split(',');
            bars.Add(new OhlcBar
            {
                Time = DateTime.Parse(p[0], null, System.Globalization.DateTimeStyles.AssumeUniversal),
                Open = double.Parse(p[1]),
                High = double.Parse(p[2]),
                Low = double.Parse(p[3]),
                Close = double.Parse(p[4]),
                Volume = double.Parse(p[5]),
            });
        }
        return bars;
    }

    private static string FindRepoRoot() => FeatureGoldenTestsHelper.FindRepoRoot();

    [Fact]
    public void FeaturePipeline_30Dim_WithMtf()
    {
        var bars = LoadGoldenM5();
        var baseFeats = FeatureCalculator.ComputeM5Features(bars);
        var mtf = FeatureMtfCalculator.Compute(bars);
        var padded = FeaturePad.ToModelDim(baseFeats, mtf);

        Assert.Equal(30, padded.GetLength(1));
        Assert.True(padded.GetLength(0) >= 60);
    }

    [Fact]
    public void FeatureHash_StableOnGoldenCsv()
    {
        var bars = LoadGoldenM5();
        var baseFeats = FeatureCalculator.ComputeM5Features(bars);
        var mtf = FeatureMtfCalculator.Compute(bars);
        var padded = FeaturePad.ToModelDim(baseFeats, mtf);

        var hash = Sha256(padded);
        var metaPath = Path.Combine(FindRepoRoot(), "tests", "fixtures", "golden_feature_hash_csharp.json");
        if (!File.Exists(metaPath))
        {
            File.WriteAllText(metaPath, $"{{\"hash\":\"{hash}\"}}");
            return;
        }

        var expected = System.Text.Json.JsonDocument.Parse(File.ReadAllText(metaPath))
            .RootElement.GetProperty("hash").GetString();
        Assert.Equal(expected, hash);
    }

    private static string Sha256(float[,] m)
    {
        var rows = m.GetLength(0);
        var cols = m.GetLength(1);
        var bytes = new byte[rows * cols * 4];
        var offset = 0;
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(offset, 4), m[i, j]);
            offset += 4;
        }
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }
}

public class ConfigSchemaValidatorTests
{
    [Fact]
    public void ValidateFile_AcceptsDefaultConfig()
    {
        var root = FeatureGoldenTestsHelper.FindRepoRoot();
        var cfg = Path.Combine(root, "config.json");
        if (!File.Exists(cfg)) return;

        var errors = ConfigSchemaValidator.ValidateFile(cfg);
        Assert.Empty(errors);
    }
}

internal static class FeatureGoldenTestsHelper
{
    internal static string FindRepoRoot()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 8; i++)
        {
            if (File.Exists(Path.Combine(dir, "config.json")))
                return dir;
            dir = Directory.GetParent(dir)?.FullName ?? dir;
        }
        return AppContext.BaseDirectory;
    }
}
