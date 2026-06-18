using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using ZhuLong.Core;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;
using Python.Runtime;

namespace ZhuLong.App.Services;

/// <summary>
/// Python 推理：默认走子进程（inference_cli.py），与 WinUI 进程隔离；
/// 避免 pyarrow/arrow.dll 在 Python.NET 内 AccessViolation 导致整进程退出。
/// Python.NET 仅保留给 MT5 桥接使用。
/// </summary>
public sealed class PythonInferenceService : IDisposable
{
    private readonly ILogger<PythonInferenceService> _logger;
    private readonly PythonGilExecutor _gil;
    private readonly PythonAgentWorker _agentWorker;
    private bool _initialized;
    private readonly HashSet<string> _warmedSymbols = new(StringComparer.OrdinalIgnoreCase);
    private readonly SemaphoreSlim _cliGate = new(1, 1);
    private bool _agentStackWarmed;
    private bool _agentEnginePreloaded;

    public PythonInferenceService(
        ILogger<PythonInferenceService> logger,
        PythonGilExecutor gil,
        PythonAgentWorker agentWorker)
    {
        _logger = logger;
        _gil = gil;
        _agentWorker = agentWorker;
    }

    public bool IsReady => _initialized;

    public void Reinitialize()
    {
        // Python.NET 不支持同进程 Shutdown 后再 Initialize（会 PyGILState_Ensure AccessViolation）
        _warmedSymbols.Clear();
        _agentStackWarmed = false;
        _agentEnginePreloaded = false;
        PythonRuntime.InvalidateCache();
        PythonRuntime.DiscoverAndCache(force: true);

        if (_initialized)
        {
            // Runtime.PythonDLL 必须在 PythonEngine.Initialize 之前设置，已初始化后不可再改
            BootstrapPythonRuntimeEnv();
            _logger.LogInformation("Python 运行库已就绪；切换 Python 版本请完全退出烛龙后重开");
            StartupLog.Write("Python 重载：已跳过 Shutdown（同进程不可安全重启 Python.NET）");
            return;
        }

        Initialize();
    }

    public void Initialize()
    {
        if (_initialized) return;

        var pythonDll = AppPaths.FindPythonDll();
        if (string.IsNullOrEmpty(pythonDll))
        {
            var hint = AppPaths.PythonDepsScriptHint;
            throw new InvalidOperationException(
                $"未找到本机 Python（需 PYTHONNET_PYDLL）。请在设置页点「一键修复环境」，或在 PowerShell 执行: {hint}");
        }

        Runtime.PythonDLL = pythonDll;
        BootstrapPythonRuntimeEnv();
        StartupLog.Write("Python.NET 初始化开始 dll=" + pythonDll);

        // 必须在 Py.GIL() 之前 Initialize；在 GIL 内调用会导致 PyGILState_Ensure 崩溃
        if (!PythonEngine.IsInitialized)
        {
            PythonEngine.Initialize();
            PythonEngine.BeginAllowThreads();
        }

        _initialized = true;
        _logger.LogInformation("Python 运行时已就绪（推理=子进程隔离，MT5=Python.NET）");
        StartupLog.Write("Python.NET 初始化完成");
    }

    public InferenceResult Predict(string symbol, float[,] seq, float[] hourly, float[] macro) =>
        PredictAsync(symbol, seq, hourly, macro, null, TimeSpan.FromSeconds(60)).GetAwaiter().GetResult();

    public void WarmupInBackground(IEnumerable<string> symbols)
    {
        if (!_initialized) return;
        var list = symbols.ToArray();
        if (list.Length == 0) return;

        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(90));
            try { await WarmupAsync(list, cts.Token).ConfigureAwait(false); }
            catch (Exception ex) { _logger.LogWarning(ex, "后台模型预热失败"); }
        });
    }

    public bool IsSymbolWarmed(string symbol) =>
        _warmedSymbols.Contains(symbol);

    public async Task WarmupAsync(IEnumerable<string> symbols, CancellationToken ct = default)
    {
        if (!_initialized) return;
        var list = symbols.Where(s => !_warmedSymbols.Contains(s)).ToArray();
        if (list.Length == 0) return;

        _logger.LogInformation("子进程模型预热 symbols={Symbols}", string.Join(",", list));
        var sw = Stopwatch.StartNew();
        var resp = await RunCliAsync(new { cmd = "warmup", symbols = list }, TimeSpan.FromSeconds(90), ct)
            .ConfigureAwait(false);
        if (resp is null || !resp.Value.GetProperty("ok").GetBoolean())
        {
            _logger.LogWarning("模型预热失败: {Err}", ReadCliError(resp));
            return;
        }

        if (resp.Value.TryGetProperty("warmed", out var warmedEl) && warmedEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in warmedEl.EnumerateArray())
            {
                var sym = item.GetString();
                if (!string.IsNullOrEmpty(sym))
                    _warmedSymbols.Add(sym);
            }
        }
        else
        {
            foreach (var sym in list)
                _warmedSymbols.Add(sym);
        }

        if (resp.Value.TryGetProperty("failed", out var failedEl) && failedEl.ValueKind == JsonValueKind.Array
            && failedEl.GetArrayLength() > 0)
        {
            var parts = failedEl.EnumerateArray()
                .Select(e => e.TryGetProperty("symbol", out var s) ? s.GetString() : null)
                .Where(s => !string.IsNullOrEmpty(s));
            _logger.LogWarning("模型预热部分失败: {Failed}", string.Join(", ", parts));
        }

        _logger.LogInformation("子进程模型预热完成 {Ms}ms", sw.ElapsedMilliseconds);
    }

    public async Task<InferenceResult> PredictAsync(
        string symbol, float[,] seq, float[] hourly, float[] macro,
        (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[]? m5Bars = null,
        TimeSpan? timeout = null, CancellationToken ct = default)
    {
        if (!_initialized)
            throw new InvalidOperationException("Python 推理未初始化");

        var padded = FeaturePad.ToModelDim(seq);
        var sw = Stopwatch.StartNew();
        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "predict",
            ["symbol"] = symbol,
            ["seq"] = ToNestedList(padded),
            ["hourly"] = hourly,
            ["macro"] = macro,
            ["m5_bars"] = m5Bars is { Length: > 0 } ? ToM5List(m5Bars) : null,
        };

        var resp = await RunCliAsync(req, timeout ?? TimeSpan.FromSeconds(60), ct).ConfigureAwait(false);
        if (resp is null)
            throw new TimeoutException($"Python 推理超时 >{(timeout ?? TimeSpan.FromSeconds(60)).TotalSeconds:F0}s");

        if (!resp.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException(ReadCliError(resp) ?? "推理子进程失败");

        var result = resp.Value.GetProperty("result");
        _warmedSymbols.Add(symbol);
        _logger.LogInformation("子进程推理完成 {Symbol} {Ms}ms", symbol, sw.ElapsedMilliseconds);
        return new InferenceResult
        {
            Direction = result.GetProperty("direction").GetInt32(),
            Confidence = result.GetProperty("confidence").GetDouble(),
            EntryOffset = result.GetProperty("entry_offset").GetDouble(),
            ExpectedReturn = result.GetProperty("expected_return").GetDouble(),
        };
    }

    public async Task<IReadOnlyList<MultiStrategyTickResult>> MultiStrategyTickAsync(
        IReadOnlyDictionary<string, (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[]> m5BySymbol,
        IReadOnlyList<string> symbols,
        string primarySymbol,
        bool macroSilence,
        string configPath,
        TimeSpan? timeout = null,
        CancellationToken ct = default)
    {
        if (!_initialized)
            throw new InvalidOperationException("Python 推理未初始化");

        var m5Payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (sym, bars) in m5BySymbol)
            m5Payload[sym] = ToM5List(bars);

        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "multi_strategy_tick",
            ["root"] = AppPaths.InstallDir,
            ["config_path"] = configPath,
            ["macro_silence"] = macroSilence,
            ["primary_symbol"] = primarySymbol,
            ["symbols"] = symbols.ToArray(),
            ["m5_bars_by_symbol"] = m5Payload,
        };

        var sw = Stopwatch.StartNew();
        var resp = await RunCliAsync(req, timeout ?? TimeSpan.FromSeconds(90), ct).ConfigureAwait(false);
        if (resp is null)
            throw new TimeoutException($"多策略推理超时 >{(timeout ?? TimeSpan.FromSeconds(90)).TotalSeconds:F0}s");

        if (!resp.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException(ReadCliError(resp) ?? "多策略子进程失败");

        var list = new List<MultiStrategyTickResult>();
        if (!resp.Value.TryGetProperty("results", out var results) || results.ValueKind != JsonValueKind.Array)
            return list;

        var schedulerMode = resp.Value.TryGetProperty("scheduler", out var schedEl) && schedEl.GetBoolean();
        if (schedulerMode)
            _logger.LogInformation("多策略子进程：自动调度 SchedulerEngine 已启用");

        foreach (var item in results.EnumerateArray())
            list.Add(ParseMultiStrategyTickResult(item));

        _logger.LogInformation("子进程多策略完成 {Count} 条 {Ms}ms", list.Count, sw.ElapsedMilliseconds);
        return list;
    }

    public async Task<(IReadOnlyList<MultiStrategyTickResult> Results, string? SkipReason)> AgentTickAsync(
        IReadOnlyDictionary<string, (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[]> m5BySymbol,
        IReadOnlyList<string> symbols,
        string primarySymbol,
        bool macroSilence,
        string configPath,
        IReadOnlyDictionary<string, object>? ticksBySymbol = null,
        IReadOnlyList<Dictionary<string, object>>? openPositions = null,
        TimeSpan? timeout = null,
        CancellationToken ct = default,
        long decisionBarUnix = 0,
        bool m5IncludesForming = true,
        IReadOnlyList<float>? macroFeatures = null)
    {
        if (!_initialized)
            throw new InvalidOperationException("Python 推理未初始化");

        var m5Payload = new Dictionary<string, object?>(StringComparer.OrdinalIgnoreCase);
        foreach (var (sym, bars) in m5BySymbol)
            m5Payload[sym] = ToM5List(bars);

        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "agent_tick",
            ["root"] = AppPaths.InstallDir,
            ["config_path"] = configPath,
            ["macro_silence"] = macroSilence,
            ["primary_symbol"] = primarySymbol,
            ["symbols"] = symbols.ToArray(),
            ["m5_bars_by_symbol"] = m5Payload,
            ["m5_includes_forming"] = m5IncludesForming,
        };
        if (decisionBarUnix > 0)
            req["decision_bar_unix"] = decisionBarUnix;
        if (ticksBySymbol is { Count: > 0 })
            req["ticks_by_symbol"] = ticksBySymbol;
        if (openPositions is { Count: > 0 })
            req["open_positions"] = openPositions;
        if (macroFeatures is { Count: > 0 })
            req["macro_features"] = macroFeatures.ToArray();

        var sw = Stopwatch.StartNew();
        var resp = await RunAgentWorkerAsync(req, timeout ?? TimeSpan.FromSeconds(120), ct).ConfigureAwait(false);
        if (resp is null)
            throw new TimeoutException($"智能体推理超时 >{(timeout ?? TimeSpan.FromSeconds(120)).TotalSeconds:F0}s");

        if (!resp.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException(ReadCliError(resp) ?? "智能体子进程失败");

        var list = new List<MultiStrategyTickResult>();
        if (!resp.Value.TryGetProperty("results", out var results) || results.ValueKind != JsonValueKind.Array)
        {
            var skipOnly = ReadAgentSkipReason(resp);
            if (!string.IsNullOrWhiteSpace(skipOnly))
                _logger.LogWarning("智能体未启用: {Reason} config={Config}", skipOnly, configPath);
            _logger.LogInformation("子进程智能体完成 0 条 {Ms}ms", sw.ElapsedMilliseconds);
            return (list, skipOnly);
        }

        var agentEnabled = resp.Value.TryGetProperty("agent", out var agentEl) && agentEl.GetBoolean();
        if (agentEnabled)
            _logger.LogInformation("子进程：TradingAgent RL 智能体已启用");
        else
        {
            var skipReason = ReadAgentSkipReason(resp);
            _logger.LogWarning("智能体未启用: {Reason} config={Config}", skipReason ?? "unknown", configPath);
            _logger.LogInformation("子进程智能体完成 0 条 {Ms}ms", sw.ElapsedMilliseconds);
            return (list, skipReason);
        }

        foreach (var item in results.EnumerateArray())
            list.Add(ParseMultiStrategyTickResult(item));

        var skipped = list.FirstOrDefault(r => r.Skipped);
        if (skipped is not null)
        {
            var reason = skipped.SkipReason ?? skipped.RejectReason ?? "unknown";
            _logger.LogWarning("智能体 tick 软跳过 {Symbol}: {Reason}", skipped.Symbol, reason);
        }

        _logger.LogInformation("Worker 智能体完成 {Count} 条 {Ms}ms", list.Count, sw.ElapsedMilliseconds);
        return (list, skipped?.SkipReason);
    }

    public async Task AgentWarmupAsync(string configPath, CancellationToken ct = default, bool preloadEngine = true)
    {
        if (_agentStackWarmed && (!preloadEngine || _agentEnginePreloaded))
            return;

        var timeoutSec = preloadEngine ? 300 : 90;
        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "agent_warmup",
            ["root"] = AppPaths.InstallDir,
            ["config_path"] = configPath,
            ["preload_engine"] = preloadEngine,
        };
        var resp = await RunAgentWorkerAsync(req, TimeSpan.FromSeconds(timeoutSec), ct).ConfigureAwait(false);
        if (resp is null)
            throw new InvalidOperationException(
                preloadEngine
                    ? "AgentEngine 预加载超时（>300s），请检查 RL/torch 环境"
                    : "Horizon 模型预加载超时（>90s），请检查 onnxruntime 与 models/horizon_v16.onnx");
        if (!resp.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException(ReadCliError(resp) ?? "agent_warmup 失败");
        _agentStackWarmed = true;
        if (preloadEngine &&
            resp.Value.TryGetProperty("engine_preloaded", out var ep) &&
            ep.ValueKind == JsonValueKind.True)
            _agentEnginePreloaded = true;
    }

    public async Task AgentRecordClosedTradeAsync(
        string symbol, double pnlR, string configPath, CancellationToken ct = default)
    {
        if (!_initialized) return;
        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "agent_record_trade",
            ["root"] = AppPaths.InstallDir,
            ["config_path"] = configPath,
            ["symbol"] = symbol,
            ["pnl_r"] = pnlR,
        };
        var resp = await RunAgentWorkerAsync(req, TimeSpan.FromSeconds(30), ct).ConfigureAwait(false);
        if (resp is null)
            _logger.LogWarning("agent_record_trade 超时 symbol={Symbol}", symbol);
        else if (!resp.Value.GetProperty("ok").GetBoolean())
            _logger.LogWarning("agent_record_trade 失败: {Err}", ReadCliError(resp));
    }

    public async Task AgentRecordSignalEmittedAsync(
        string symbol, string configPath, CancellationToken ct = default)
    {
        if (!_initialized) return;
        var req = new Dictionary<string, object?>
        {
            ["cmd"] = "agent_record_signal",
            ["root"] = AppPaths.InstallDir,
            ["config_path"] = configPath,
            ["symbol"] = symbol,
        };
        var resp = await RunAgentWorkerAsync(req, TimeSpan.FromSeconds(15), ct).ConfigureAwait(false);
        if (resp is null)
            _logger.LogWarning("agent_record_signal 超时 symbol={Symbol}", symbol);
        else if (!resp.Value.GetProperty("ok").GetBoolean())
            _logger.LogWarning("agent_record_signal 失败: {Err}", ReadCliError(resp));
    }

    public async Task AgentValidateAsync(string configPath, CancellationToken ct = default)
    {
        if (!AgentEnvironmentValidator.TryValidateV16(configPath, PythonRuntime.ResolveExecutable(), out var nativeErr))
            throw new InvalidOperationException(nativeErr ?? "智能体环境验证失败");

        var ping = new Dictionary<string, object?> { ["cmd"] = "ping" };
        var resp = await RunAgentWorkerAsync(ping, TimeSpan.FromSeconds(30), ct).ConfigureAwait(false);
        if (resp is null)
            throw new InvalidOperationException("智能体 Worker 未响应（>30s）");
        if (!resp.Value.GetProperty("ok").GetBoolean())
            throw new InvalidOperationException(ReadCliError(resp) ?? "智能体 Worker ping 失败");
    }

    private async Task<JsonElement?> RunAgentWorkerAsync(
        Dictionary<string, object?> request,
        TimeSpan timeout,
        CancellationToken ct)
    {
        if (!_initialized)
            throw new InvalidOperationException("Python 推理未初始化");
        return await _agentWorker.SendAsync(request, timeout, ct).ConfigureAwait(false);
    }

    private static string? ReadAgentSkipReason(JsonElement? resp)
    {
        if (resp is null || resp.Value.ValueKind != JsonValueKind.Object)
            return null;
        if (resp.Value.TryGetProperty("reason", out var reasonEl))
            return reasonEl.GetString();
        return null;
    }

    private static MultiStrategyTickResult ParseMultiStrategyTickResult(JsonElement item)
    {
        MultiStrategySignalPayload? signal = null;
        if (item.TryGetProperty("signal", out var sigEl) && sigEl.ValueKind == JsonValueKind.Object)
        {
            signal = new MultiStrategySignalPayload
            {
                Strategy = sigEl.TryGetProperty("strategy", out var st) ? st.GetString() ?? "" : "",
                Symbol = sigEl.TryGetProperty("symbol", out var sym) ? sym.GetString() ?? "" : "",
                Direction = sigEl.TryGetProperty("direction", out var dir) ? dir.GetString() ?? "" : "",
                Confidence = sigEl.TryGetProperty("confidence", out var conf) ? conf.GetDouble() : 0,
                Entry = sigEl.TryGetProperty("entry", out var en) ? en.GetDouble() : 0,
                Sl = sigEl.TryGetProperty("sl", out var sl) ? sl.GetDouble() : 0,
                Tp = sigEl.TryGetProperty("tp", out var tp) ? tp.GetDouble() : 0,
                SignalId = sigEl.TryGetProperty("signal_id", out var sid) ? sid.GetString() ?? "" : "",
                RejectReason = sigEl.TryGetProperty("reject_reason", out var rr) ? rr.GetString() : null,
            };
        }

        return new MultiStrategyTickResult
        {
            Symbol = item.TryGetProperty("symbol", out var symbol) ? symbol.GetString() ?? "" : "",
            MarketState = item.TryGetProperty("state", out var state) ? state.GetString() ?? "" : "",
            ActiveStrategy = item.TryGetProperty("strategy", out var strategy) ? strategy.GetString() ?? "" : "",
            Signal = signal,
            Skipped = item.TryGetProperty("skipped", out var skipped) && skipped.GetBoolean(),
            SkipReason = item.TryGetProperty("reason", out var reason) ? reason.GetString() : null,
            RejectReason = signal?.RejectReason,
            Adx = item.TryGetProperty("adx", out var adx) ? adx.GetDouble() : null,
            AtrRatio = item.TryGetProperty("atr_ratio", out var atr) ? atr.GetDouble() : null,
            ExitAssessment = item.TryGetProperty("exit_assessment", out var exitScore) ? exitScore.GetDouble() : 0,
            ExitReason = item.TryGetProperty("exit_reason", out var exitReason) ? exitReason.GetString() : null,
            AiSlPrice = item.TryGetProperty("ai_sl_price", out var aiSl) ? aiSl.GetDouble() : 0,
            AiTpPrice = item.TryGetProperty("ai_tp_price", out var aiTp) ? aiTp.GetDouble() : 0,
            TrailMode = item.TryGetProperty("trail_mode", out var tm) ? tm.GetString() : null,
            SuggestedTrailingSl = item.TryGetProperty("suggested_trailing_sl", out var sts)
                ? sts.GetDouble() : 0,
            PositionMgmtReason = item.TryGetProperty("position_mgmt_reason", out var pmr)
                ? pmr.GetString() : null,
            CognitionRegime = item.TryGetProperty("cognition_regime", out var regime) ? regime.GetString() : null,
            CognitionRegimeConfidence = item.TryGetProperty("cognition_regime_confidence", out var regimeConf)
                ? regimeConf.GetDouble() : 0,
            RlRawAction = item.TryGetProperty("rl_raw_action", out var rlRaw) ? rlRaw.GetString() : null,
            AgentAction = item.TryGetProperty("action", out var act) ? act.GetString() : null,
            CognitionDirection = item.TryGetProperty("cognition_direction", out var cogDir) ? cogDir.GetString() : null,
            CognitionConfidence = item.TryGetProperty("cognition_confidence", out var cogConf) ? cogConf.GetDouble() : 0,
            FilterReason = item.TryGetProperty("filter_reason", out var filt) ? filt.GetString() : null,
            Architecture = item.TryGetProperty("architecture", out var arch) ? arch.GetString() : null,
            HorizonDirection = item.TryGetProperty("horizon_direction", out var hDir) ? hDir.GetString() : null,
            HorizonConfidence = item.TryGetProperty("horizon_confidence", out var hConf) ? hConf.GetDouble() : 0,
            HorizonMinConfidence = item.TryGetProperty("horizon_min_confidence", out var hMin) ? hMin.GetDouble() : 0,
            Kn2ShouldTrade = item.TryGetProperty("kn2_should_trade", out var k2t) && k2t.GetBoolean(),
            Kn2Advisory = item.TryGetProperty("kn2_advisory", out var k2adv) && k2adv.GetBoolean(),
            Kn2Action = item.TryGetProperty("kn2_action", out var k2a) ? k2a.GetString() : null,
            Kn2Confidence = item.TryGetProperty("kn2_confidence", out var k2c) ? k2c.GetDouble() : 0,
            Kn2ShadowMode = item.TryGetProperty("kn2_shadow_mode", out var k2s) && k2s.GetBoolean(),
            DrawPayloadJson = item.TryGetProperty("draw_payload", out var drawEl) && drawEl.ValueKind == JsonValueKind.Object
                ? drawEl.GetRawText()
                : null,
            AttributionJson = item.TryGetProperty("attribution", out var attrEl) && attrEl.ValueKind == JsonValueKind.Object
                ? attrEl.GetRawText()
                : null,
        };
    }

    public async Task<bool> ValidateModelsAsync(IEnumerable<string> symbols, CancellationToken ct = default)
    {
        if (!_initialized) return false;
        try
        {
            var resp = await RunCliAsync(new { cmd = "validate", symbols = symbols.ToArray() },
                TimeSpan.FromSeconds(30), ct).ConfigureAwait(false);
            return resp is not null && resp.Value.GetProperty("ok").GetBoolean();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "validate_models 子进程失败");
            return false;
        }
    }

    public bool ValidateModels(IEnumerable<string> symbols)
    {
        try
        {
            return ValidateModelsAsync(symbols, CancellationToken.None).GetAwaiter().GetResult();
        }
        catch
        {
            return false;
        }
    }

    private async Task<JsonElement?> RunCliAsync(object request, TimeSpan timeout, CancellationToken ct)
    {
        if (!PythonExecutableResolver.TryResolve(PythonRuntime.ResolveExecutable(), out var exe, out var resolveErr))
            throw new InvalidOperationException(resolveErr ?? "Python 解析失败");

        var script = AppPaths.InferenceCliScriptPath;
        if (!File.Exists(script))
            throw new FileNotFoundException("缺少 inference_cli.py", script);

        var input = Path.Combine(Path.GetTempPath(), $"zhulong_req_{Guid.NewGuid():N}.json");
        var output = Path.Combine(Path.GetTempPath(), $"zhulong_out_{Guid.NewGuid():N}.json");
        await _cliGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            await File.WriteAllTextAsync(
                input,
                JsonSerializer.Serialize(request),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
                ct).ConfigureAwait(false);

            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            timeoutCts.CancelAfter(timeout);

            var psi = new ProcessStartInfo
            {
                FileName = exe,
                WorkingDirectory = AppPaths.InstallDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (PythonQuickProbe.IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(BuildInferenceCliLaunchCode(script, input, output));
            ApplySubprocessEnv(psi);

            Process? proc = null;
            try
            {
                proc = Process.Start(psi)!;
                var stdoutTask = proc.StandardOutput.ReadToEndAsync(timeoutCts.Token);
                var stderrTask = proc.StandardError.ReadToEndAsync(timeoutCts.Token);
                await Task.WhenAll(
                    proc.WaitForExitAsync(timeoutCts.Token),
                    stdoutTask,
                    stderrTask).ConfigureAwait(false);
                var stderr = await stderrTask.ConfigureAwait(false);
                _ = await stdoutTask.ConfigureAwait(false);

                if (File.Exists(output))
                {
                    var outText = await File.ReadAllTextAsync(output, ct).ConfigureAwait(false);
                    try
                    {
                        using var doc = JsonDocument.Parse(outText);
                        var root = doc.RootElement.Clone();
                        if (root.TryGetProperty("ok", out var okEl) && okEl.ValueKind == JsonValueKind.False)
                        {
                            var err = ReadCliError(root);
                            _logger.LogError("推理子进程失败: {Msg}", err);
                            throw new InvalidOperationException(err ?? "推理子进程失败");
                        }
                        if (root.TryGetProperty("ok", out _))
                            return root;
                    }
                    catch (InvalidOperationException)
                    {
                        throw;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "解析子进程 JSON 输出失败");
                    }
                }

                if (proc.ExitCode != 0 || !File.Exists(output))
                {
                    var msg = await ReadCliFailureMessageAsync(output, stderr, proc.ExitCode, ct).ConfigureAwait(false);
                    _logger.LogError("推理子进程失败: {Msg}", msg);
                    throw new InvalidOperationException(msg);
                }

                var finalText = await File.ReadAllTextAsync(output, ct).ConfigureAwait(false);
                using var finalDoc = JsonDocument.Parse(finalText);
                var finalRoot = finalDoc.RootElement.Clone();
                if (finalRoot.TryGetProperty("ok", out var finalOk) && finalOk.ValueKind == JsonValueKind.False)
                {
                    var err = ReadCliError(finalRoot);
                    throw new InvalidOperationException(err ?? "推理子进程失败");
                }

                return finalRoot;
            }
            finally
            {
                if (proc is { HasExited: false })
                {
                    try { proc.Kill(entireProcessTree: true); } catch { /* best effort */ }
                }
                proc?.Dispose();
            }
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            _logger.LogError("推理子进程超时 ({Sec}s)", timeout.TotalSeconds);
            return null;
        }
        finally
        {
            _cliGate.Release();
            TryDelete(input);
            TryDelete(output);
        }
    }

    private static string BuildInferenceCliLaunchCode(string script, string input, string output)
    {
        static string PyLit(string path) => path.Replace("\\", "\\\\").Replace("'", "\\'");
        return "import runpy,sys; "
               + $"sys.argv=['inference_cli.py','--input',r'{PyLit(input)}','--output',r'{PyLit(output)}']; "
               + $"runpy.run_path(r'{PyLit(script)}', run_name='__main__')";
    }

    internal static void ApplySubprocessEnv(ProcessStartInfo psi)
    {
        var dll = AppPaths.FindPythonDll();
        if (!string.IsNullOrEmpty(dll))
            psi.Environment["PYTHONNET_PYDLL"] = dll;
        psi.Environment["ZHULONG_IMF_CSV_ONLY"] = "1";
        psi.Environment["ZHULONG_ALLOW_YFINANCE"] = "0";
        psi.Environment["TF_CPP_MIN_LOG_LEVEL"] = "3";
        psi.Environment["TF_ENABLE_ONEDNN_OPTS"] = "0";
        psi.Environment["GLOG_minloglevel"] = "3";
        psi.Environment["ZHULONG_DATA_DIR"] = AppPaths.WritableDataDir;
        psi.Environment["ZHULONG_LOGS_DIR"] = AppPaths.LogsDir;
        psi.Environment["ZHULONG_MACRO_DIR"] = Path.Combine(AppPaths.WritableDataDir, "macro");
        psi.Environment["PYTHONIOENCODING"] = "utf-8";
        PrependPythonNativeDllDirs(psi);
        // ===== P1-3: 设置 PYTHONPATH 确保本地 zhulong 包优先 =====
        var installDir = AppPaths.InstallDir;
        var engineDir = AppPaths.PythonEngineDir;
        var appDataDir = AppPaths.AppDataDir;
        var bundledDir = AppPaths.BundledPythonDir;
        // AppData 优先：Python 补丁 + 避免从 Program Files 执行 inference_cli 挂起
        var exists = psi.Environment.TryGetValue("PYTHONPATH", out var existing);
        var paths = new List<string> { appDataDir, installDir, engineDir };
        if (AppPaths.HasBundledPython && Directory.Exists(bundledDir))
            paths.Insert(0, bundledDir);
        if (exists && !string.IsNullOrEmpty(existing))
        {
            paths.AddRange(existing.Split(Path.PathSeparator)
                .Where(p => !string.IsNullOrWhiteSpace(p)));
        }
        psi.Environment["PYTHONPATH"] = string.Join(Path.PathSeparator, paths);
        psi.Environment["PYTHONDONTWRITEBYTECODE"] = "1";
        psi.Environment["ZHULONG_INSTALL_DIR"] = installDir;
        if (AppPaths.HasBundledPython)
            psi.Environment["ZHULONG_BUNDLED_PYTHON"] = "1";
        var pathParts = new List<string>();
        if (AppPaths.HasBundledPython && Directory.Exists(bundledDir))
            pathParts.Add(bundledDir);
        pathParts.Add(installDir);
        if (psi.Environment.TryGetValue("PATH", out var curPath) && !string.IsNullOrWhiteSpace(curPath))
            pathParts.Add(curPath);
        psi.Environment["PATH"] = string.Join(Path.PathSeparator, pathParts);
        // ===== 结束 =====
    }

    private static void PrependPythonNativeDllDirs(ProcessStartInfo psi)
    {
        if (!PythonExecutableResolver.TryResolve(PythonRuntime.ResolveExecutable(), out var exe, out _))
            return;

        var pyDir = Path.GetDirectoryName(exe);
        if (string.IsNullOrEmpty(pyDir))
            return;

        var dirs = new List<string> { pyDir };
        var sitePackages = Path.Combine(pyDir, "Lib", "site-packages");
        foreach (var sub in new[] { "torch\\lib", "onnxruntime\\capi", "numpy.libs" })
        {
            var d = Path.Combine(sitePackages, sub);
            if (Directory.Exists(d))
                dirs.Add(d);
        }

        psi.Environment.TryGetValue("PATH", out var existing);
        existing ??= Environment.GetEnvironmentVariable("PATH") ?? "";
        psi.Environment["PATH"] = string.Join(Path.PathSeparator, dirs) + Path.PathSeparator + existing;
    }

    private static string? ReadCliError(JsonElement? resp)
    {
        if (resp is null) return "无响应";
        if (resp.Value.ValueKind == JsonValueKind.Object && resp.Value.TryGetProperty("error", out var err))
            return err.GetString();
        return "unknown";
    }

    private static async Task<string> ReadCliFailureMessageAsync(
        string outputPath, string stderr, int exitCode, CancellationToken ct)
    {
        if (File.Exists(outputPath))
        {
            try
            {
                var text = await File.ReadAllTextAsync(outputPath, ct).ConfigureAwait(false);
                using var doc = JsonDocument.Parse(text);
                if (doc.RootElement.TryGetProperty("error", out var err))
                {
                    var msg = err.GetString();
                    if (!string.IsNullOrWhiteSpace(msg))
                        return msg!;
                }
                if (doc.RootElement.TryGetProperty("missing", out var missing)
                    && missing.ValueKind == JsonValueKind.Array)
                {
                    var parts = missing.EnumerateArray()
                        .Select(e => e.GetString())
                        .Where(s => !string.IsNullOrWhiteSpace(s));
                    return "missing: " + string.Join("; ", parts);
                }
            }
            catch
            {
                /* ignore parse errors */
            }
        }

        if (!string.IsNullOrWhiteSpace(stderr))
            return stderr.Trim();

        return $"exit={exitCode}";
    }

    private static List<List<float>> ToNestedList(float[,] arr)
    {
        var rows = arr.GetLength(0);
        var cols = arr.GetLength(1);
        var outer = new List<List<float>>(rows);
        for (var i = 0; i < rows; i++)
        {
            var inner = new List<float>(cols);
            for (var j = 0; j < cols; j++)
                inner.Add(arr[i, j]);
            outer.Add(inner);
        }
        return outer;
    }

    private static List<object[]> ToM5List(
        (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[] bars)
    {
        var list = new List<object[]>(bars.Length);
        foreach (var b in bars)
            list.Add([b.TimeUnix, b.Open, b.High, b.Low, b.Close, b.Volume]);
        return list;
    }

    private static void BootstrapPythonRuntimeEnv()
    {
        Environment.SetEnvironmentVariable("ZHULONG_ALLOW_YFINANCE", "0");
        Environment.SetEnvironmentVariable("ZHULONG_IMF_CSV_ONLY", "1");
        Environment.SetEnvironmentVariable("ZHULONG_DATA_DIR", AppPaths.WritableDataDir);
        Environment.SetEnvironmentVariable("ZHULONG_LOGS_DIR", AppPaths.LogsDir);
        var macroDir = Path.Combine(AppPaths.WritableDataDir, "macro");
        Directory.CreateDirectory(macroDir);
        var bundled = Path.Combine(AppPaths.DataDir, "macro", "macro_daily.csv");
        var userCsv = Path.Combine(macroDir, "macro_daily.csv");
        if (!File.Exists(userCsv) && File.Exists(bundled))
        {
            try { File.Copy(bundled, userCsv, overwrite: false); }
            catch { /* ignore */ }
        }
        Environment.SetEnvironmentVariable("ZHULONG_MACRO_DIR", macroDir);
    }

    private void SafeShutdownPythonEngine()
    {
        try
        {
            _gil.Run(() =>
            {
                if (PythonEngine.IsInitialized)
                    PythonEngine.Shutdown();
            });
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "PythonEngine.Shutdown 跳过");
            StartupLog.Write("PythonEngine.Shutdown: " + ex.Message);
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch { /* ignore */ }
    }

    public void Dispose()
    {
        if (!_initialized) return;
        _agentWorker.Dispose();
        SafeShutdownPythonEngine();
        _initialized = false;
    }
}
