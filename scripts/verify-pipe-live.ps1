# 用 DLL 模拟 MT5 指标：Data 写 + Drawing 读，对接运行中的 ZhuLong

$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent

$dll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'

if (-not (Test-Path $dll)) { & (Join-Path $root 'scripts\build-zhulong-mt5-pipe.ps1') }



$dllEsc = $dll.Replace('\', '\\')

$cs = @"

using System;

using System.Runtime.InteropServices;

using System.Text;

using System.Threading;



public static class PipeLiveClient

{

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]

    public static extern int ZhuLongPipeConnectV1(string pipeLogicalName, int mode, uint connectTimeoutMs);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]

    public static extern int ZhuLongPipeWriteLineV1(int handleId, string line);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall)]

    public static extern int ZhuLongPipeReadLineV1(int handleId, byte[] buffer, int capacity);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall)]

    public static extern int ZhuLongPipePollV1(int handleId);

    [DllImport("$dllEsc", CallingConvention = CallingConvention.StdCall)]

    public static extern void ZhuLongPipeDisconnectV1(int handleId);



    public static int Run()

    {

        int hData = ZhuLongPipeConnectV1("ZhuLong_Data", 1, 12000);

        if (hData <= 0) return 10;



        int hDraw = ZhuLongPipeConnectV1("ZhuLong_Drawing", 2, 12000);

        if (hDraw <= 0) { ZhuLongPipeDisconnectV1(hData); return 11; }



        string bar = "{\"type\":\"bar\",\"symbol\":\"XAUUSD\",\"time\":1717588800,\"open\":4200,\"high\":4205,\"low\":4198,\"close\":4202,\"volume\":100}";

        if (ZhuLongPipeWriteLineV1(hData, bar) != 1)

        { ZhuLongPipeDisconnectV1(hData); ZhuLongPipeDisconnectV1(hDraw); return 13; }



        // hold draw read 5s like real indicator

        for (int i = 0; i < 50; i++)

        {

            if (ZhuLongPipePollV1(hDraw) < 0) { ZhuLongPipeDisconnectV1(hData); ZhuLongPipeDisconnectV1(hDraw); return 14; }

            Thread.Sleep(100);

        }



        ZhuLongPipeDisconnectV1(hData);

        ZhuLongPipeDisconnectV1(hDraw);

        return 0;

    }

}

"@

Add-Type -TypeDefinition $cs -Language CSharp

$code = [PipeLiveClient]::Run()

if ($code -eq 0) {

    Write-Host 'LIVE PIPE OK: data+draw held 5s against ZhuLong' -ForegroundColor Green

    exit 0

}

Write-Host "LIVE PIPE FAILED exit=$code" -ForegroundColor Red

exit $code

