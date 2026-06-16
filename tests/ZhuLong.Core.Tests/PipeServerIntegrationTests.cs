using System.IO.Pipes;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using ZhuLong.Core.Models;
using ZhuLong.Core.Pipes;

namespace ZhuLong.Core.Tests;

/// <summary>L1-5：命名管道 JSON bar 收发（无需启动 GUI）。</summary>
public sealed class PipeServerIntegrationTests
{
    [Fact]
    public async Task BarReceived_WhenClientSendsM1Json()
    {
        var id = Guid.NewGuid().ToString("N")[..8];
        var dataPipe = $"ZhuLong_TestData_{id}";
        var drawPipe = $"ZhuLong_TestDraw_{id}";

        await using var server = new PipeServer(NullLogger<PipeServer>.Instance, dataPipe, drawPipe);
        M1Bar? received = null;
        server.BarReceived += b => received = b;
        server.Start();

        await Task.Delay(200);

        using var client = new NamedPipeClientStream(".", dataPipe, PipeDirection.Out);
        await client.ConnectAsync(5000);

        var barJson =
            """{"type":"bar","symbol":"XAUUSD","time":"2026-06-05T12:00:00Z","open":2350.0,"high":2351.0,"low":2349.5,"close":2350.5,"volume":120}""" + "\n";
        var bytes = Encoding.UTF8.GetBytes(barJson);
        await client.WriteAsync(bytes);
        await client.FlushAsync();

        var deadline = DateTime.UtcNow.AddSeconds(5);
        while (received is null && DateTime.UtcNow < deadline)
            await Task.Delay(50);

        Assert.NotNull(received);
        Assert.Equal("XAUUSD", received!.Symbol);
        Assert.Equal(2350.5, received.Close, 3);
        Assert.Equal(120, received.Volume, 3);
    }
}
