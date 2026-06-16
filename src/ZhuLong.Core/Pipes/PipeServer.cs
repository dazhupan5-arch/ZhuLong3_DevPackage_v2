using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO.Pipes;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Win32.SafeHandles;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Pipes;

/// <summary>命名管道服务端：上行 M1 数据 + 下行绘图（Out 多实例，避免僵尸连接占坑）。</summary>
public sealed class PipeServer : IAsyncDisposable
{
    private const int MaxDrawQueue = 64;
    private static readonly TimeSpan DrawConnectSettle = TimeSpan.FromMilliseconds(200);
    private static readonly TimeSpan DrawClientProbeInterval = TimeSpan.FromMilliseconds(500);
    private static readonly TimeSpan DrawIdleProbeInterval = TimeSpan.FromSeconds(45);

    private readonly ILogger<PipeServer> _logger;
    private readonly string _dataPipeName;
    private readonly string _drawingPipeName;
    private readonly ConcurrentQueue<string> _drawQueue = new();
    private readonly ConcurrentQueue<string> _drawClearQueue = new();

    private CancellationTokenSource? _cts;
    private Task? _dataTask;
    private Task? _drawTask;

    public PipeServer(ILogger<PipeServer> logger, string dataPipe, string drawingPipe)
    {
        _logger = logger;
        _dataPipeName = NormalizePipeName(dataPipe);
        _drawingPipeName = NormalizePipeName(drawingPipe);
    }

    public event Action<M1Bar>? BarReceived;
    public event Action<string, IReadOnlyList<M1Bar>, bool>? HistoryBarsReceived;
    public event Action? DataClientConnected;
    public event Action? DrawClientConnected;
    /// <summary>MT5 会话通知：symbol, warm（切换周期/热重连）。</summary>
    public event Action<string, bool>? SessionReceived;

    public bool IsDataConnected { get; private set; }
    public bool IsDrawConnected { get; private set; }
    public bool IsListening => _cts is not null;

    public void Start()
    {
        if (_cts is not null)
            return;

        _cts = new CancellationTokenSource();
        _dataTask = Task.Run(() => RunDataLoopAsync(_cts.Token));
        _drawTask = Task.Run(() => RunDrawLoopAsync(_cts.Token));
    }

    public async Task StopAsync()
    {
        if (_cts is null)
            return;

        _cts.Cancel();
        if (_dataTask is not null)
        {
            try { await _dataTask.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            catch { /* ignore */ }
        }

        if (_drawTask is not null)
        {
            try { await _drawTask.ConfigureAwait(false); }
            catch (OperationCanceledException) { }
            catch { /* ignore */ }
        }

        _cts.Dispose();
        _cts = null;
        _dataTask = null;
        _drawTask = null;
        IsDataConnected = false;
        IsDrawConnected = false;
    }

    public Task<bool> SendDrawCommandAsync(object payload, CancellationToken ct = default)
    {
        var line = JsonSerializer.Serialize(payload);
        EnqueueDraw(line);
        return Task.FromResult(IsDrawConnected);
    }

    private void EnqueueDraw(string line)
    {
        var target = line.Contains("\"clear_signal\"", StringComparison.Ordinal)
            || line.Contains("\"clear_all\"", StringComparison.Ordinal)
            ? _drawClearQueue
            : _drawQueue;
        target.Enqueue(line);
        while (target.Count > MaxDrawQueue && target.TryDequeue(out _))
        {
            /* drop oldest */
        }
    }

    private async Task RunDataLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            await using var pipe = CreateServer(_dataPipeName, PipeDirection.In);
            try
            {
                IsDataConnected = false;
                _logger.LogInformation("等待 MT5 连接数据管道 {Pipe}", _dataPipeName);
                await pipe.WaitForConnectionAsync(ct);

                IsDataConnected = true;
                _logger.LogInformation("MT5 已连接数据管道");
                DataClientConnected?.Invoke();

                using var reader = new StreamReader(pipe, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 262144, leaveOpen: true);
                while (!ct.IsCancellationRequested && pipe.IsConnected)
                {
                    var line = await reader.ReadLineAsync(ct);
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    TryParseLine(line);
                }
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "数据管道异常，500ms 后重建");
                await Task.Delay(500, ct);
            }
            finally
            {
                IsDataConnected = false;
            }
        }
    }

    private async Task RunDrawLoopAsync(CancellationToken ct)
    {
        var drawPipeLog = StartupLogPath();

        while (!ct.IsCancellationRequested)
        {
            NamedPipeServerStream? pipe = null;
            try
            {
                pipe = CreateDrawServer(_drawingPipeName);
                IsDrawConnected = false;

                _logger.LogDebug("绘图管道等待 MT5 连接");
                AppendStartupLog(drawPipeLog, $"Drawing pipe: waiting (queue={_drawQueue.Count})");

                await pipe.WaitForConnectionAsync(ct).ConfigureAwait(false);

                var pid = TryGetClientPid(pipe);
                _logger.LogInformation("MT5 已连接绘图管道 pid={Pid}", pid);
                AppendStartupLog(drawPipeLog, $"Drawing pipe: connected pid={pid} queue={_drawQueue.Count}");

                // 给 MT5 指标 OnTimer 留出启动读循环的时间
                await Task.Delay(DrawConnectSettle, ct).ConfigureAwait(false);

                IsDrawConnected = true;
                DrawClientConnected?.Invoke();

                using var clientWatch = WatchDrawClientAsync(pipe, pid, ct);
                await ServiceDrawPipeAsync(pipe, ct).ConfigureAwait(false);
                _logger.LogInformation("绘图管道客户端断开 pid={Pid}", pid);
            }
            catch (OperationCanceledException) { break; }
            catch (IOException ex)
            {
                _logger.LogWarning(ex, "绘图管道 IO 异常，即将重建");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "绘图管道异常，500ms 后重连");
            }
            finally
            {
                IsDrawConnected = false;
                ReleaseDrawPipe(pipe);
            }

            if (!ct.IsCancellationRequested)
            {
                try { await Task.Delay(200, ct).ConfigureAwait(false); }
                catch { break; }
            }
        }
    }

    private static IDisposable WatchDrawClientAsync(NamedPipeServerStream pipe, uint pid, CancellationToken ct)
    {
        if (pid == 0)
            return EmptyDisposable.Instance;

        var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _ = Task.Run(async () =>
        {
            try
            {
                while (!cts.Token.IsCancellationRequested)
                {
                    if (!pipe.IsConnected)
                        break;
                    if (!IsClientProcessAlive(pid))
                    {
                        try { pipe.Disconnect(); }
                        catch { /* ignore */ }
                        break;
                    }
                    await Task.Delay(DrawClientProbeInterval, cts.Token).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException) { }
            catch { /* ignore */ }
        }, cts.Token);

        return cts;
    }

    private static bool IsClientProcessAlive(uint pid)
    {
        try
        {
            using var proc = Process.GetProcessById((int)pid);
            return !proc.HasExited;
        }
        catch
        {
            return false;
        }
    }

    private sealed class EmptyDisposable : IDisposable
    {
        public static readonly EmptyDisposable Instance = new();
        public void Dispose() { }
    }

    private async Task ServiceDrawPipeAsync(NamedPipeServerStream pipe, CancellationToken ct)
    {
        var lastProbeUtc = DateTime.UtcNow;
        while (!ct.IsCancellationRequested)
        {
            if (!pipe.IsConnected)
                return;

            if (_drawClearQueue.IsEmpty && _drawQueue.IsEmpty)
            {
                if (DateTime.UtcNow - lastProbeUtc >= DrawIdleProbeInterval)
                {
                    lastProbeUtc = DateTime.UtcNow;
                    if (!await WriteDrawLineAsync(pipe, "{\"action\":\"ping\"}", ct).ConfigureAwait(false))
                        return;
                }
                await Task.Delay(200, ct).ConfigureAwait(false);
                continue;
            }

            while (TryDequeueDraw(out var line))
            {
                lastProbeUtc = DateTime.UtcNow;
                if (!await WriteDrawLineAsync(pipe, line, ct).ConfigureAwait(false))
                    return;
            }
        }
    }

    private bool TryDequeueDraw(out string line)
    {
        if (_drawClearQueue.TryDequeue(out line))
            return true;
        return _drawQueue.TryDequeue(out line);
    }

    private static async Task<bool> WriteDrawLineAsync(NamedPipeServerStream pipe, string line, CancellationToken ct)
    {
        try
        {
            if (!line.EndsWith('\n'))
                line += "\n";
            var bytes = Encoding.UTF8.GetBytes(line);
            await pipe.WriteAsync(bytes.AsMemory(0, bytes.Length), ct).ConfigureAwait(false);
            await pipe.FlushAsync(ct).ConfigureAwait(false);
            return true;
        }
        catch (IOException)
        {
            return false;
        }
    }

    private static void ReleaseDrawPipe(NamedPipeServerStream? pipe)
    {
        if (pipe is null)
            return;
        try
        {
            if (pipe.IsConnected)
                pipe.Disconnect();
        }
        catch { /* ignore */ }
        try { pipe.Dispose(); }
        catch { /* ignore */ }
    }

    private static uint TryGetClientPid(NamedPipeServerStream pipe)
    {
        try
        {
            if (!GetNamedPipeClientProcessId(pipe.SafePipeHandle, out var pid))
                return 0;
            return pid;
        }
        catch { return 0; }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetNamedPipeClientProcessId(SafePipeHandle pipe, out uint clientProcessId);

    private static string StartupLogPath() =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ZhuLong", "startup.log");

    private static void AppendStartupLog(string path, string message)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var line = $"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss}] [PipeServer] {message}{Environment.NewLine}";
            File.AppendAllText(path, line);
        }
        catch { /* ignore */ }
    }

    private static NamedPipeServerStream CreateServer(string name, PipeDirection direction)
    {
        return NamedPipeServerStreamAcl.Create(
            name,
            direction,
            NamedPipeServerStream.MaxAllowedServerInstances,
            PipeTransmissionMode.Byte,
            PipeOptions.Asynchronous,
            1048576,
            1048576,
            BuildPipeSecurity());
    }

    private static NamedPipeServerStream CreateDrawServer(string name)
    {
        return NamedPipeServerStreamAcl.Create(
            name,
            PipeDirection.Out,
            NamedPipeServerStream.MaxAllowedServerInstances,
            PipeTransmissionMode.Byte,
            PipeOptions.Asynchronous,
            1048576,
            1048576,
            BuildPipeSecurity());
    }

    private static PipeSecurity BuildPipeSecurity()
    {
        var security = new PipeSecurity();
        security.AddAccessRule(new PipeAccessRule(
            new SecurityIdentifier(WellKnownSidType.WorldSid, null),
            PipeAccessRights.ReadWrite,
            AccessControlType.Allow));
        security.AddAccessRule(new PipeAccessRule(
            new SecurityIdentifier(WellKnownSidType.AuthenticatedUserSid, null),
            PipeAccessRights.ReadWrite,
            AccessControlType.Allow));
        return security;
    }

    private void TryParseLine(string line)
    {
        try
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            var type = root.GetProperty("type").GetString();
            if (type == "heartbeat")
                return;

            if (type == "session")
            {
                var warm = root.TryGetProperty("warm", out var w) && w.GetBoolean();
                var symbol = root.TryGetProperty("symbol", out var sym) ? sym.GetString() ?? "" : "";
                _logger.LogInformation("MT5 会话通知 warm={Warm} symbol={Symbol}", warm, symbol);
                SessionReceived?.Invoke(symbol, warm);
                return;
            }

            if (type == "bar")
            {
                BarReceived?.Invoke(ParseBar(root));
                return;
            }

            if (type == "m1_history")
            {
                var symbol = root.GetProperty("symbol").GetString() ?? "";
                var final = root.TryGetProperty("final", out var f) && f.GetBoolean();
                var bars = new List<M1Bar>();
                foreach (var el in root.GetProperty("bars").EnumerateArray())
                    bars.Add(ParseBar(el, symbol));
                HistoryBarsReceived?.Invoke(symbol, bars, final);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "忽略非法管道 JSON: {Line}", line.Length > 120 ? line[..120] : line);
        }
    }

    private static M1Bar ParseBar(JsonElement root, string? symbolOverride = null)
    {
        return new M1Bar
        {
            Symbol = symbolOverride ?? root.GetProperty("symbol").GetString() ?? "",
            Time = ParseBarTime(root.GetProperty("time")),
            Open = root.GetProperty("open").GetDouble(),
            High = root.GetProperty("high").GetDouble(),
            Low = root.GetProperty("low").GetDouble(),
            Close = root.GetProperty("close").GetDouble(),
            Volume = root.TryGetProperty("volume", out var v) ? v.GetDouble() : 0,
        };
    }

    public static DateTime ParseBarTime(JsonElement timeEl)
    {
        if (timeEl.ValueKind == JsonValueKind.Number)
            return Mt5Time.FromUnixUtcSeconds(timeEl.GetInt64());

        var raw = timeEl.GetString() ?? "";
        if (string.IsNullOrWhiteSpace(raw))
            return DateTime.UtcNow;

        if (raw.EndsWith("Z", StringComparison.OrdinalIgnoreCase)
            && DateTimeOffset.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var iso))
            return Mt5Time.FromUnixUtcSeconds(iso.ToUnixTimeSeconds());

        return ParseMt5TimeString(raw);
    }

    private static DateTime ParseMt5TimeString(string raw)
    {
        var formats = new[]
        {
            "yyyy.MM.dd HH:mm:ss",
            "yyyy.MM.dd HH:mm",
            "yyyy-MM-dd HH:mm:ss",
            "yyyy-MM-ddTHH:mm:ss",
        };
        if (DateTime.TryParseExact(raw, formats, CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeUniversal, out var dt))
            return dt;

        if (DateTime.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out dt))
            return dt;

        throw new FormatException($"无法解析 MT5 时间: {raw}");
    }

    private static string NormalizePipeName(string pipe) =>
        pipe.Replace(@"\\.\pipe\", "", StringComparison.OrdinalIgnoreCase)
            .Replace(@"\\.\pipe\\", "", StringComparison.OrdinalIgnoreCase)
            .Trim('\\');

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
    }
}
