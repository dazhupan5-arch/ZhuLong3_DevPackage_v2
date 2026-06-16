using ZhuLong.Core.Macro;

namespace ZhuLong.Core.Tests;

public class MacroCalendarTests
{
    [Fact]
    public void ParseApiUtc_ConvertsFinnhubNfpToLocal()
    {
        // Finnhub: 2026-06-05 12:30 UTC = 北京时间 20:30 (UTC+8)
        var local = MacroEventTime.ParseApiUtc("2026-06-05 12:30:00");
        Assert.NotNull(local);
        Assert.Equal(20, local!.Value.Hour);
        Assert.Equal(30, local.Value.Minute);
        Assert.Equal(6, local.Value.Month);
        Assert.Equal(5, local.Value.Day);
    }

    [Fact]
    public void MergeWithCsv_ReplacesWrongApiNfpWithCsvLocalTime()
    {
        var api = new List<MacroEventRecord>
        {
            new(new DateTime(2026, 6, 5, 12, 30, 0), "Non Farm Payrolls", "high", "USD", "finnhub"),
        };
        var csv = new List<MacroEventRecord>
        {
            new(new DateTime(2026, 6, 5, 20, 30, 0), "US Nonfarm Payrolls", "high", "USD", "csv"),
        };

        var merged = MacroCalendarFetcher.MergeWithCsv(
            api,
            new DateTime(2026, 6, 4, 0, 0, 0, DateTimeKind.Utc),
            new DateTime(2026, 6, 8, 0, 0, 0, DateTimeKind.Utc),
            csv);

        var nfp = merged.Single(e => e.EventName.Contains("Payroll", StringComparison.OrdinalIgnoreCase));
        Assert.Equal(20, nfp.EventTime.Hour);
        Assert.Equal("csv", nfp.Source);
    }

    [Fact]
    public void IsHighImpact_AcceptsNumericAndText()
    {
        Assert.True(MacroImpactHelper.IsHighImpact("high"));
        Assert.True(MacroImpactHelper.IsHighImpact("3"));
        Assert.False(MacroImpactHelper.IsHighImpact("medium"));
    }
}
