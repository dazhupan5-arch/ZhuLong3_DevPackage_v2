using ZhuLong.Core.Features;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Tests;

public sealed class FeatureCacheHistoryTests
{
    [Fact]
    public void AppendHistoryChunk_1000M1_ProducesEnoughM5ForSequence()
    {
        var cache = new FeatureCacheService();
        const string symbol = "XAUUSD";
        var bars = new List<M1Bar>();
        var t0 = new DateTime(2026, 6, 5, 0, 0, 0);
        for (var i = 0; i < 1000; i++)
        {
            var t = t0.AddMinutes(i);
            bars.Add(new M1Bar
            {
                Symbol = symbol,
                Time = t,
                Open = 2400 + i * 0.01,
                High = 2401 + i * 0.01,
                Low = 2399 + i * 0.01,
                Close = 2400.5 + i * 0.01,
                Volume = 10,
            });
        }

        cache.AppendHistoryChunk(symbol, bars[..500], final: false);
        cache.AppendHistoryChunk(symbol, bars[500..], final: true);

        Assert.Equal(1000, cache.GetM1Count(symbol));
        Assert.True(cache.GetM5Count(symbol) >= 80);
        Assert.True(cache.TryGetSequence(symbol, 60, out _, out _, out _));
    }

    [Fact]
    public void AppendHistoryChunk_MergesWithExisting_DoesNotShrinkM5()
    {
        var cache = new FeatureCacheService();
        const string symbol = "XAUUSD";
        var t0 = new DateTime(2026, 6, 8, 0, 0, 0);

        var deep = new List<M1Bar>();
        for (var i = 0; i < 5000; i++)
        {
            var t = t0.AddMinutes(-5000 + i);
            deep.Add(new M1Bar
            {
                Symbol = symbol,
                Time = t,
                Open = 4300,
                High = 4301,
                Low = 4299,
                Close = 4300.5,
                Volume = 1,
            });
        }
        cache.AppendHistoryChunk(symbol, deep, final: true);
        var m5Before = cache.GetM5Count(symbol);
        Assert.True(m5Before >= 400);

        var shallow = new List<M1Bar>();
        for (var i = 0; i < 1000; i++)
        {
            var t = t0.AddMinutes(i);
            shallow.Add(new M1Bar
            {
                Symbol = symbol,
                Time = t,
                Open = 4310,
                High = 4311,
                Low = 4309,
                Close = 4310.5,
                Volume = 1,
            });
        }
        cache.AppendHistoryChunk(symbol, shallow, final: true);

        Assert.True(cache.GetM5Count(symbol) >= m5Before);
        Assert.True(cache.GetM5Count(symbol) >= 400);
    }
}
