using System.Text.Json;
using ZhuLong.Core;
using ZhuLong.Core.Pipes;

namespace ZhuLong.Core.Tests;

public sealed class PipeServerTimeTests
{
    [Fact]
    public void ParseBarTime_UnixSeconds_MatchesLocalFromUtc()
    {
        const long unix = 1_749_312_000L; // 2025-06-07 16:00:00 UTC
        var el = JsonDocument.Parse(unix.ToString()).RootElement;
        var parsed = PipeServer.ParseBarTime(el);
        var expected = Mt5Time.FromUnixUtcSeconds(unix);
        Assert.Equal(expected, parsed);
    }

    [Fact]
    public void ParseBarTime_IsoZ_MatchesLocalFromUtc()
    {
        var el = JsonDocument.Parse("\"2026-06-05T12:00:00Z\"").RootElement;
        var parsed = PipeServer.ParseBarTime(el);
        var expected = Mt5Time.FromUnixUtcSeconds(
            new DateTimeOffset(2026, 6, 5, 12, 0, 0, TimeSpan.Zero).ToUnixTimeSeconds());
        Assert.Equal(expected, parsed);
    }

    [Fact]
    public void ParseBarTime_LegacyMt5String_StillParses()
    {
        var el = JsonDocument.Parse("\"2026.06.08 21:40:00\"").RootElement;
        var parsed = PipeServer.ParseBarTime(el);
        Assert.Equal(new DateTime(2026, 6, 8, 21, 40, 0), parsed);
    }
}
