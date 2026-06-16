using ZhuLong.Core.Macro;
using ZhuLong.Core.Services;

namespace ZhuLong.Core.Tests;

public class MacroFeatureBuilderTests
{
    [Fact]
    public void Build_Returns8Dimensions_WithCalendarAndJson()
    {
        var now = new DateTime(2026, 6, 5, 12, 0, 0);
        var events = new List<MacroEventRecord>
        {
            new(now.AddHours(-2), "US CPI", "high", "USD"),
            new(now.AddHours(6), "FOMC Rate Decision", "high", "USD"),
        };
        var fred = new FredSnapshot("2026-06-05", new Dictionary<string, double>
        {
            ["UNRATE"] = 4.0,
            ["FEDFUNDS"] = 4.33,
            ["T10YIE"] = 2.28,
        });
        var sentiment = new SentimentSnapshot(82.5, 0.58, 0.52, 0.55);

        var vec = MacroFeatureBuilder.Build(events, fred, sentiment, now);

        Assert.Equal(8, vec.Length);
        Assert.InRange(vec[0], 0, 1); // hours_to_next normalized
        Assert.Equal(1f, vec[1]);     // next high impact
        Assert.InRange(vec[4], 0.9f, 1f); // FOMC type code
        Assert.InRange(vec[6], 0.4f, 0.7f); // gold/silver norm
    }

    [Fact]
    public void EventTypeCode_RecognizesKeywords()
    {
        Assert.Equal(1f, MacroFeatureBuilder.EventTypeCode("FOMC Rate Decision"));
        Assert.Equal(0.9f, MacroFeatureBuilder.EventTypeCode("US Nonfarm Payrolls"));
        Assert.Equal(0.7f, MacroFeatureBuilder.EventTypeCode("EIA Crude Oil Inventories"));
    }

    [Fact]
    public void LoadFred_FromSampleJson()
    {
        var path = Path.Combine(FindDataDir(), "fred_latest.json");
        if (!File.Exists(path)) return;
        var snap = MacroFeatureBuilder.LoadFred(path);
        Assert.NotNull(snap);
        Assert.True(snap!.Latest.ContainsKey("UNRATE"));
    }

    private static string FindDataDir()
    {
        var dir = AppContext.BaseDirectory;
        for (var i = 0; i < 8; i++)
        {
            var candidate = Path.Combine(dir, "data");
            if (Directory.Exists(candidate)) return candidate;
            dir = Directory.GetParent(dir)?.FullName ?? dir;
        }
        return Path.Combine(AppContext.BaseDirectory, "data");
    }
}
