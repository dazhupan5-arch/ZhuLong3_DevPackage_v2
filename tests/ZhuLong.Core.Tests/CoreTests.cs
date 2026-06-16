using ZhuLong.Core.Configuration;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

public class FeaturePadTests
{
    [Fact]
    public void ToModelDim_Pads22To30()
    {
        var seq = new float[60, FeatureConstants.BaseFeatureDim];
        for (var i = 0; i < 60; i++)
            seq[i, 0] = i;

        var padded = FeaturePad.ToModelDim(seq);

        Assert.Equal(60, padded.GetLength(0));
        Assert.Equal(30, padded.GetLength(1));
        Assert.Equal(5f, padded[5, 0]);
        Assert.Equal(0f, padded[5, 22]);
    }
}

public class FeatureCalculatorTests
{
    [Fact]
    public void ComputeM5Features_Has22Dimensions()
    {
        var bars = Enumerable.Range(0, 80).Select(i => new OhlcBar
        {
            Time = DateTime.UtcNow.AddMinutes(-i * 5),
            Open = 2000 + i,
            High = 2001 + i,
            Low = 1999 + i,
            Close = 2000.5 + i,
            Volume = 100,
        }).Reverse().ToList();

        var f = FeatureCalculator.ComputeM5Features(bars);

        Assert.Equal(80, f.GetLength(0));
        Assert.Equal(FeatureCalculator.FeatureDim, f.GetLength(1));
    }
}

public class SignalGeneratorTests
{
    [Fact]
    public void TryGenerate_ReturnsSignal_WhenFiltersPass()
    {
        var settings = new AppSettings
        {
            SignalFilters = new AppSettings.SignalFilterSettings(),
            SignalGeometry = new AppSettings.SignalGeometrySettings(),
            Mt5 = new AppSettings.Mt5Settings { CommentPrefix = "ZhuLong" },
        };
        var gen = new SignalGeneratorService();
        var inference = new InferenceResult
        {
            Direction = 1,
            Confidence = 0.85,
            EntryOffset = -0.0015,
            ExpectedReturn = 1.0,
        };

        var signal = gen.TryGenerate(settings, "XAUUSD", inference, atrPct: 0.5, closePrice: 2350.0);

        Assert.NotNull(signal);
        Assert.Equal("buy", signal!.Direction);
        Assert.StartsWith("ZhuLong_", signal.CommentHint);
        Assert.True(signal.StopLoss < signal.EntryPrice);
        Assert.True(signal.TakeProfit > signal.EntryPrice);
    }

    [Fact]
    public void TryGenerate_ReturnsNull_WhenConfidenceLow()
    {
        var settings = new AppSettings();
        var gen = new SignalGeneratorService();
        var inference = new InferenceResult { Direction = 1, Confidence = 0.1, EntryOffset = -0.1, ExpectedReturn = 0.1 };

        var signal = gen.TryGenerate(settings, "XAUUSD", inference, 0.5, 2350);

        Assert.Null(signal);
    }
}

public class SymbolMappingTests
{
    [Theory]
    [InlineData("XAUUSDm", "XAUUSD")]
    [InlineData("GOLD", "XAUUSD")]
    [InlineData("CL-OIL", "USOIL")]
    [InlineData("XTIUSD", "USOIL")]
    [InlineData("USOIL", "USOIL")]
    public void ResolveStandardSymbol_MapsBrokerAliases(string raw, string expected)
    {
        var settings = new AppSettings
        {
            Model = new AppSettings.ModelSettings { DefaultSymbols = ["XAUUSD", "USOIL"] },
            SymbolMapping = new Dictionary<string, string>
            {
                ["XAUUSD"] = "XAUUSDm",
                ["USOIL"] = "CL-OIL",
            },
        };

        Assert.Equal(expected, settings.ResolveStandardSymbol(raw));
    }

    [Fact]
    public void ResolveStandardSymbol_ReverseMapping()
    {
        var settings = new AppSettings
        {
            SymbolMapping = new Dictionary<string, string> { ["XAUUSD"] = "XAUUSDm" },
        };

        Assert.Equal("XAUUSD", settings.ResolveStandardSymbol("XAUUSDm"));
    }
}

public class PendingSignalStoreTests
{
    [Fact]
    public void AddAndSnapshot_Works()
    {
        var store = new PendingSignalStore();
        store.Add(new SignalModel { SignalId = "s1", Symbol = "XAUUSD", Direction = "buy" });
        var snap = store.Snapshot();
        Assert.Single(snap);
        Assert.Equal("s1", snap[0].SignalId);
    }
}
