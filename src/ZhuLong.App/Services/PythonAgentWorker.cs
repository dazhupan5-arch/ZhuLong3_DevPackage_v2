using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using ZhuLong.Core;

namespace ZhuLong.App.Services;

/// <summary>常驻 Python 推理 Worker（agent_* 命令），避免每 tick 冷启动加载模型。</summary>
public sealed class PythonAgentWorker : IDisposable
{
    private readonly ILogger<PythonAgentWorker> _logger;
    private readonly SemaphoreSlim _gate = new(1, 1);
    private Process? _proc;
    private StreamWriter? _stdin;
    private StreamReader? _stdout;
    private Task? _stderrPump;
    private int _requestId;
    private bool _disposed;
    private string? _lastReadFailureReason;

    public PythonAgentWorker(ILogger<PythonAgentWorker> logger) => _logger = logger;

    public async Task<JsonElement?> SendAsync(
        Dictionary<string, object?> request,
        TimeSpan timeout,
        CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            _lastReadFailureReason = null;
            await EnsureStartedAsync(ct).ConfigureAwait(false);
            request["id"] = Interlocked.Increment(ref _requestId);

            var line = JsonSerializer.Serialize(request);
            await _stdin!.WriteLineAsync(line.AsMemory(), ct).ConfigureAwait(false);
            await _stdin.FlushAsync(ct).ConfigureAwait(false);

            var resp = await ReadJsonResponseAsync(timeout, ct).ConfigureAwait(false);
            if (resp is null)
            {
                var reason = _lastReadFailureReason ?? "timeout";
                _logger.LogError("智能体 Worker 无有效响应 ({Reason}, 限 {Sec}s)", reason, timeout.TotalSeconds);
                await KillWorkerAsync().ConfigureAwait(false);
                throw new InvalidOperationException(
                    reason.Contains("JSON", StringComparison.OrdinalIgnoreCase)
                        ? "智能体 Worker 响应 JSON 损坏（Python 输出含 NaN/Inf 或非 JSON），已终止 Worker"
                        : $"智能体 Worker 响应超时 ({timeout.TotalSeconds:F0}s)");
            }

            return resp;
        }
        finally
        {
            _gate.Release();
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            if (_proc is { HasExited: false } && _stdin is not null)
            {
                var shutdown = JsonSerializer.Serialize(new Dictionary<string, object?>
                {
                    ["cmd"] = "shutdown",
                    ["id"] = Interlocked.Increment(ref _requestId),
                });
                _stdin.WriteLine(shutdown);
                _stdin.Flush();
                _proc.WaitForExit(2000);
            }
        }
        catch { /* best effort */ }
        KillWorkerAsync().GetAwaiter().GetResult();
        _gate.Dispose();
    }

    private async Task<JsonElement?> ReadJsonResponseAsync(TimeSpan timeout, CancellationToken ct)
    {
        var deadline = DateTime.UtcNow + timeout;
        var opts = new JsonDocumentOptions
        {
            AllowTrailingCommas = true,
            CommentHandling = JsonCommentHandling.Skip,
        };

        while (DateTime.UtcNow < deadline)
        {
            var remaining = deadline - DateTime.UtcNow;
            if (remaining <= TimeSpan.Zero)
                break;

            using var lineCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            lineCts.CancelAfter(remaining);

            string? respLine;
            try
            {
                respLine = await _stdout!.ReadLineAsync(lineCts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (!ct.IsCancellationRequested)
            {
                break;
            }

            if (string.IsNullOrWhiteSpace(respLine))
            {
                if (_proc is { HasExited: true })
                {
                    _logger.LogError("智能体 Worker 无响应（进程已退出 exit={Code})", _proc.ExitCode);
                    break;
                }
                continue;
            }

            respLine = respLine.Trim();
            if (!respLine.StartsWith('{'))
            {
                _logger.LogDebug(
                    "Worker 跳过非 JSON stdout: {Line}",
                    respLine.Length > 160 ? respLine[..160] + "…" : respLine);
                continue;
            }

            try
            {
                using var doc = JsonDocument.Parse(respLine, opts);
                var root = doc.RootElement.Clone();
                if (root.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.False)
                {
                    var err = root.TryGetProperty("error", out var errEl) ? errEl.GetString() : "unknown";
                    _logger.LogError("智能体 Worker 命令失败: {Err}", err);
                }
                return root;
            }
            catch (JsonException ex)
            {
                _logger.LogWarning(
                    ex,
                    "Worker JSON 行无效 pos={Pos}，继续等待…",
                    ex.BytePositionInLine);
                // 形如 {"ok": true, "results": ... 的损坏响应不可能变合法，继续等只会拖到超时
                if (respLine.Contains("\"results\"", StringComparison.Ordinal) ||
                    respLine.Contains("\"agent\"", StringComparison.Ordinal))
                {
                    _lastReadFailureReason = "json_corrupt";
                    _logger.LogError(
                        "Worker 响应 JSON 损坏（非超时，多为 NaN/Inf），立即失败 pos={Pos}",
                        ex.BytePositionInLine);
                    await KillWorkerAsync().ConfigureAwait(false);
                    return null;
                }
            }
        }

        _lastReadFailureReason = "timeout";
        return null;
    }

    private async Task EnsureStartedAsync(CancellationToken ct)
    {
        if (_proc is { HasExited: false } && _stdin is not null && _stdout is not null)
            return;

        await KillWorkerAsync().ConfigureAwait(false);

        if (!PythonExecutableResolver.TryResolve(PythonRuntime.ResolveExecutable(), out var exe, out var resolveErr))
            throw new InvalidOperationException(resolveErr ?? "Python 解析失败");

        var script = AppPaths.InferenceWorkerScriptPath;
        if (!File.Exists(script))
            throw new FileNotFoundException("缺少 inference_worker.py", script);

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            WorkingDirectory = AppPaths.InstallDir,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        if (PythonQuickProbe.IsPyLauncher(exe))
            psi.ArgumentList.Add("-3");
        psi.ArgumentList.Add("-u");
        psi.ArgumentList.Add(script);
        PythonInferenceService.ApplySubprocessEnv(psi);

        _proc = Process.Start(psi) ?? throw new InvalidOperationException("无法启动 inference_worker.py");
        _stdin = _proc.StandardInput;
        _stdout = _proc.StandardOutput;
        _stderrPump = Task.Run(() => PumpStderrAsync(_proc.StandardError));

        var ping = await SendPingAsync(ct).ConfigureAwait(false);
        if (ping is null || !ping.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException("智能体 Worker 启动后 ping 失败");
        _logger.LogInformation("智能体 Worker 已启动 pid={Pid}", _proc.Id);
    }

    private async Task<JsonElement?> SendPingAsync(CancellationToken ct)
    {
        var id = Interlocked.Increment(ref _requestId);
        var line = JsonSerializer.Serialize(new Dictionary<string, object?> { ["cmd"] = "ping", ["id"] = id });
        await _stdin!.WriteLineAsync(line.AsMemory(), ct).ConfigureAwait(false);
        await _stdin.FlushAsync(ct).ConfigureAwait(false);
        var respLine = await _stdout!.ReadLineAsync(ct).ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(respLine))
            return null;
        using var doc = JsonDocument.Parse(respLine);
        return doc.RootElement.Clone();
    }

    private async Task PumpStderrAsync(StreamReader stderr)
    {
        try
        {
            while (!_disposed)
            {
                var line = await stderr.ReadLineAsync().ConfigureAwait(false);
                if (line is null) break;
                if (!string.IsNullOrWhiteSpace(line))
                    _logger.LogDebug("agent_worker stderr: {Line}", line);
            }
        }
        catch { /* worker shutting down */ }
    }

    private async Task KillWorkerAsync()
    {
        _stdin = null;
        _stdout = null;
        if (_proc is not null)
        {
            try
            {
                if (!_proc.HasExited)
                    _proc.Kill(entireProcessTree: true);
            }
            catch { /* ignore */ }
            _proc.Dispose();
            _proc = null;
        }
        if (_stderrPump is not null)
        {
            try { await _stderrPump.ConfigureAwait(false); }
            catch { /* ignore */ }
            _stderrPump = null;
        }
    }
}
