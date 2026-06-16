using ZhuLong.Core.Configuration;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;
using Xunit;

namespace ZhuLong.Core.Tests;

public sealed class TrailingStopEngineTests
{
    [Fact]
    public void AgentMode_B58111Scenario_DoesNotTightenAboveStructure()
    {
        var pm = new AppSettings.PositionManagementSettings
        {
            TrailingUseAtrMode = true,
            TrailingBreakevenAtrMult = 1.0,
            TrailingTightenAtrMult = 1.5,
            TrailingStepAtrMult = 0.5,
            TrailingStructureBufferAtrMult = 0.3,
            TrailingSwingLookbackBars = 24,
            TrailingUseStructureConstraint = true,
            AgentTrailingWidenFactor = 1.5,
            MinHoldSecondsBeforeTrailing = 60,
        };

        var entry = 4233.64;
        var atr = 4.64;
        var m5 = BuildM5WithSwingLow(4230.0, entry);

        var activate = TrailingStopEngine.Evaluate(
            new TrailingStopEngine.TrailingContext
            {
                Direction = "buy",
                Entry = entry,
                Price = 4244.0,
                Atr = atr,
                BestPrice = entry,
                TrailingSl = 0,
                TrailingActivated = false,
                LastTrailPrice = 0,
                HoldSeconds = 62,
                M5Bars = m5,
            },
            pm,
            agentMode: true);

        Assert.True(activate.TrailingActivated);
        Assert.Equal(entry, activate.TrailingSl);

        var tighten = TrailingStopEngine.Evaluate(
            new TrailingStopEngine.TrailingContext
            {
                Direction = "buy",
                Entry = entry,
                Price = 4244.0,
                Atr = atr,
                BestPrice = activate.BestPrice,
                TrailingSl = activate.TrailingSl,
                TrailingActivated = true,
                LastTrailPrice = activate.LastTrailPrice,
                HoldSeconds = 62,
                M5Bars = m5,
            },
            pm,
            agentMode: true);

        Assert.Equal(entry, tighten.TrailingSl);
    }

    private static List<OhlcBar> BuildM5WithSwingLow(double swingLow, double entry)
    {
        var bars = new List<OhlcBar>();
        var t = DateTime.UtcNow.AddMinutes(-30 * 5);
        for (var i = 0; i < 30; i++)
        {
            var low = entry - 5;
            var high = entry + 3;
            if (i == 20)
            {
                low = swingLow;
                high = swingLow + 2;
            }
            bars.Add(new OhlcBar
            {
                Time = t.AddMinutes(i * 5),
                Open = entry,
                High = high,
                Low = low,
                Close = entry + 1,
                Volume = 1,
            });
        }
        return bars;
    }
}
