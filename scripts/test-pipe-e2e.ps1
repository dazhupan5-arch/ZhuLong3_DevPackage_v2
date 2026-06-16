# E2E: 模拟 MT5 客户端（duplex + ready 握手）对接 ZhuLong 或独立 PipeServer
param(
    [switch]$AgainstLiveZhuLong
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$dll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (-not (Test-Path $dll)) {
    & (Join-Path $root 'scripts\build-zhulong-mt5-pipe.ps1')
}

$dllEsc = $dll.Replace('\', '\\')
$cs = @"
using System;
using System.IO;
using System.IO.Pipes;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

public static class PipeE2E
{
    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
    public static extern int ZhuLongPipeConnectV1(string pipeLogicalName, int mode, uint connectTimeoutMs);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
    public static extern int ZhuLongPipeWriteLineV1(int handleId, string line);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
    public static extern int ZhuLongPipeReadLineV1(int handleId, byte[] buffer, int capacity);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall)]
    public static extern int ZhuLongPipePollV1(int handleId);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall)]
    public static extern void ZhuLongPipeDisconnectV1(int handleId);

    public static NamedPipeServerStream CreateDrawServer(string name)
    {
        var sec = new PipeSecurity();
        sec.AddAccessRule(new PipeAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null),
            PipeAccessRights.ReadWrite, AccessControlType.Allow));
        return new NamedPipeServerStream(name, PipeDirection.InOut, 1,
            PipeTransmissionMode.Byte, PipeOptions.Asynchronous, 65536, 65536, sec);
    }

    public static NamedPipeServerStream CreateDataServer(string name)
    {
        var sec = new PipeSecurity();
        sec.AddAccessRule(new PipeAccessRule(new SecurityIdentifier(WellKnownSidType.WorldSid, null),
            PipeAccessRights.ReadWrite, AccessControlType.Allow));
        return new NamedPipeServerStream(name, PipeDirection.In, NamedPipeServerStream.MaxAllowedServerInstances,
            PipeTransmissionMode.Byte, PipeOptions.Asynchronous, 65536, 65536, sec);
    }

    static string ReadLineBlocking(NamedPipeServerStream pipe, CancellationToken ct)
    {
        var buf = new byte[4096];
        var acc = new StringBuilder();
        while (!ct.IsCancellationRequested)
        {
            int n = pipe.ReadAsync(buf, 0, buf.Length, ct).GetAwaiter().GetResult();
            if (n <= 0) return null;
            acc.Append(Encoding.UTF8.GetString(buf, 0, n));
            int nl = acc.ToString().IndexOf('\n');
            if (nl >= 0) return acc.ToString().Substring(0, nl).Trim();
        }
        return null;
    }

    public static int RunStandalone()
    {
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));
        using (var dataSrv = CreateDataServer("ZhuLong_Data"))
        using (var drawSrv = CreateDrawServer("ZhuLong_Drawing"))
        {
            var dataTask = Task.Run(() =>
            {
                dataSrv.WaitForConnectionAsync(cts.Token).GetAwaiter().GetResult();
                using (var r = new StreamReader(dataSrv, Encoding.UTF8))
                    return r.ReadLineAsync().GetAwaiter().GetResult();
            }, cts.Token);

            var drawTask = Task.Run(() =>
            {
                drawSrv.WaitForConnectionAsync(cts.Token).GetAwaiter().GetResult();
                var ready = ReadLineBlocking(drawSrv, cts.Token);
                if (ready == null || ready.IndexOf("ready", StringComparison.OrdinalIgnoreCase) < 0)
                    return "no-ready:" + ready;
                var line = "{\"action\":\"draw_signal\",\"symbol\":\"XAUUSD\",\"signal_id\":\"E2E-1\",\"direction\":\"buy\",\"entry\":2400,\"sl\":2390,\"tp\":2420,\"confidence\":\"0.9\",\"expiry_minutes\":\"60\"}\n";
                var bytes = Encoding.UTF8.GetBytes(line);
                drawSrv.Write(bytes, 0, bytes.Length);
                drawSrv.Flush();
                return "ok";
            }, cts.Token);

            Thread.Sleep(400);

            int hData = ZhuLongPipeConnectV1("ZhuLong_Data", 1, 8000);
            if (hData <= 0) return 10;

            int hDraw = ZhuLongPipeConnectV1("ZhuLong_Drawing", 2, 8000);
            if (hDraw <= 0) { ZhuLongPipeDisconnectV1(hData); return 11; }

            if (ZhuLongPipeWriteLineV1(hDraw, "{\"action\":\"ready\",\"symbol\":\"XAUUSD\"}") != 1)
            { ZhuLongPipeDisconnectV1(hData); ZhuLongPipeDisconnectV1(hDraw); return 12; }

            string bar = "{\"type\":\"bar\",\"symbol\":\"XAUUSD\",\"time\":1717588800,\"open\":1.0,\"high\":2.0,\"low\":0.5,\"close\":1.5,\"volume\":100}";
            if (ZhuLongPipeWriteLineV1(hData, bar) != 1)
            { ZhuLongPipeDisconnectV1(hData); ZhuLongPipeDisconnectV1(hDraw); return 13; }

            var readLine = dataTask.GetAwaiter().GetResult();
            var drawResult = drawTask.GetAwaiter().GetResult();
            if (drawResult != "ok")
            { ZhuLongPipeDisconnectV1(hData); ZhuLongPipeDisconnectV1(hDraw); return 14; }

            var outBuf = new byte[16384];
            int rc = 0;
            for (int i = 0; i < 50 && rc <= 0; i++)
            {
                Thread.Sleep(50);
                rc = ZhuLongPipeReadLineV1(hDraw, outBuf, outBuf.Length);
            }

            ZhuLongPipeDisconnectV1(hData);
            ZhuLongPipeDisconnectV1(hDraw);

            if (string.IsNullOrWhiteSpace(readLine) || readLine.IndexOf("XAUUSD", StringComparison.Ordinal) < 0)
                return 15;
            if (rc != 1) return 16;
            var drawJson = Encoding.UTF8.GetString(outBuf).TrimEnd('\0');
            if (drawJson.IndexOf("draw_signal", StringComparison.Ordinal) < 0)
                return 17;
            return 0;
        }
    }

    public static int RunAgainstLive()
    {
        int hDraw = ZhuLongPipeConnectV1("ZhuLong_Drawing", 2, 12000);
        if (hDraw <= 0) return 21;
        if (ZhuLongPipeWriteLineV1(hDraw, "{\"action\":\"ready\",\"symbol\":\"XAUUSD\",\"chart\":0}") != 1)
        { ZhuLongPipeDisconnectV1(hDraw); return 22; }

        var outBuf = new byte[16384];
        int poll = ZhuLongPipePollV1(hDraw);
        if (poll < 0) { ZhuLongPipeDisconnectV1(hDraw); return 23; }

        Thread.Sleep(500);
        poll = ZhuLongPipePollV1(hDraw);
        if (poll < 0) { ZhuLongPipeDisconnectV1(hDraw); return 24; }

        ZhuLongPipeDisconnectV1(hDraw);
        return 0;
    }
}
"@

Add-Type -TypeDefinition $cs -Language CSharp
$code = if ($AgainstLiveZhuLong) { [PipeE2E]::RunAgainstLive() } else { [PipeE2E]::RunStandalone() }
$mode = if ($AgainstLiveZhuLong) { 'live' } else { 'standalone' }
if ($code -eq 0) {
    Write-Host "PIPE E2E OK (mode=$mode)" -ForegroundColor Green
    exit 0
}
Write-Host "PIPE E2E FAILED exit=$code" -ForegroundColor Red
exit $code
