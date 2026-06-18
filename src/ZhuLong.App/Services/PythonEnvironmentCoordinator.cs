using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using ZhuLong.Core;
using ZhuLong.Core.Bootstrap;
using ZhuLong.Core.Configuration;

namespace ZhuLong.App.Services;

/// <summary>Python / 宏观 / 配置 一键修复（实机疑难杂症）。</summary>
public sealed class PythonEnvironmentCoordinator
{
    private static readonly TimeSpan PipStepTimeout = TimeSpan.FromMinutes(25);

    private static readonly string[] RequiredModules =
    [
        "MetaTrader5", "fredapi", "onnxruntime", "gymnasium", "stable_baselines3",
        "torch", "xgboost", "pandas", "sklearn", "joblib", "requests",
    ];

    private readonly PythonInferenceService _python;
    private readonly MacroOfflineRefreshService _macroRefresh;
    private DateTimeOffset _lastAgentValidateFailUtc;
    private bool _agentEnvRepairAttempted;

    public PythonEnvironmentCoordinator(PythonInferenceService python, MacroOfflineRefreshService macroRefresh)
    {
        _python = python;
        _macroRefresh = macroRefresh;
    }

    public void RunSelfCheck(Action<string> log)
    {
        log("—— 环境自检 ——");
        log("安装目录: " + AppPaths.InstallDir);
        log("可写数据: " + AppPaths.WritableDataDir);

        var (exe, dll) = PythonRuntime.DiscoverAndCache(force: true);
        if (string.IsNullOrEmpty(dll) || !File.Exists(dll))
            log("[×] PYTHONNET_PYDLL 未找到");
        else
            log("[√] PYTHONNET_PYDLL: " + dll);

        var pyProbe = !string.IsNullOrEmpty(exe) && File.Exists(exe) ? exe : "python";
        if (PythonQuickProbe.TryRunVersionLine(pyProbe, out var resolved, out var ver, out var verErr))
        {
            log("[√] Python: " + resolved);
            log("    " + ver);
        }
        else
            log("[×] Python: " + (verErr ?? "未知"));

        foreach (var mod in RequiredModules)
        {
            if (PythonQuickProbe.TryImportModule(pyProbe, mod, out var err))
                log("[√] import " + mod);
            else
                log("[×] import " + mod + ": " + err);
        }

        if (PythonQuickProbe.TryImportModule(pyProbe, "pyarrow", out var paErr))
            log("[√] import pyarrow");
        else
            log("[!] import pyarrow（可选，有 imf_vmd.csv 时可跳过）: " + paErr);

        log("");
        log("—— V16 智能体真实探针（与开机校验相同）——");
        log("[!] 说明：使用本机 Python 3.10+，安装后请运行 install_python_deps.ps1 或设置页「一键修复环境」");
        try
        {
            var settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
            AgentConfigSync.AlignWithAppSettings(settings);
            if (AppPaths.HasBundledPython && settings.TradingAgent?.Enabled == true)
            {
                var cfg = AgentConfigSync.ResolveAgentConfigPath(settings);
                if (AgentEnvironmentValidator.TryValidateV16(cfg, PythonRuntime.ResolveExecutable(), out var agentErr))
                    log("[√] V16 智能体探针通过（内置 Python + Horizon Session）");
                else
                {
                    log("[×] V16 智能体探针失败: " + agentErr);
                    LogHorizonProbeHints(log, agentErr);
                }
            }
            else if (settings.TradingAgent?.Enabled == true)
            {
                var cfg = AgentConfigSync.ResolveAgentConfigPath(settings);
                if (AgentEnvironmentValidator.TryValidateV16(cfg, pyProbe, out var agentErr))
                    log("[√] V16 智能体探针通过（Horizon Session + scaler + 模型文件，与 Worker 同路径）");
                else
                {
                    log("[×] V16 智能体探针失败: " + agentErr);
                    LogHorizonProbeHints(log, agentErr);
                }
            }
            else
                log("[—] 智能体未启用，跳过 V16 探针");
        }
        catch (Exception ex)
        {
            log("[×] V16 智能体探针异常: " + ex.Message);
        }

        log(_python.IsReady ? "[√] Python.NET 已加载" : "[!] Python.NET 未加载（连接 MT5 时加载）");
        log("修复脚本: " + ResolveDepsScriptPath());
        log("—— 自检结束 ——");
    }

    public async Task<bool> RunOneClickRepairAsync(Action<string> log, CancellationToken ct = default)
    {
        log("—— 一键修复环境（全量）——");

        // 1) 发现 Python 并写入 AppData 缓存
        log("[1/7] 发现 Python…");
        var (exe, dll) = PythonRuntime.DiscoverAndCache(force: true);
        if (string.IsNullOrEmpty(dll) || !File.Exists(dll))
        {
            log("[×] 未找到 Python 3.10+。请先安装并勾选 Add to PATH，再点一次修复。");
            return false;
        }

        if (string.IsNullOrEmpty(exe) || !File.Exists(exe))
        {
            if (!PythonExecutableResolver.TryResolve("python", out exe, out var resolveErr))
            {
                log("[×] 无法启动 Python: " + resolveErr);
                return false;
            }
        }

        var pyExe = Path.GetFullPath(exe);
        log("[√] Python: " + pyExe);
        log("[√] DLL: " + dll);
        Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", dll);

        // 2) 修复 AppData 配置（旧 API2D 域名等）
        log("[2/7] 修复本机配置…");
        PatchUserConfig(log);
        _ = AppBootstrap.EnsureFirstRun();
        log("[√] AppData 数据目录已就绪");

        // 3) pip 安装（C# 直调 + ps1 双保险）
        log("[3/7] pip 安装依赖（需网络；日志会逐行滚动，单步最长 25 分钟）…");
        var pipOk = await RunPipRepairAsync(pyExe, log, ct).ConfigureAwait(false);
        if (!pipOk)
        {
            log("[!] C# pip 未完全成功，尝试 install_python_deps.ps1 …");
            var script = ResolveDepsScriptPath();
            if (!File.Exists(script))
            {
                log("[×] 找不到 " + script);
                return false;
            }

            var psCode = await RunPowerShellAsync(script, pyExe, AppPaths.InstallDir, log, ct).ConfigureAwait(false);
            if (psCode != 0)
            {
                log("[×] install_python_deps.ps1 失败（退出码 " + psCode + "）");
                return false;
            }
        }

        // 4) 逐模块验证（与 pip 同一 python.exe）
        log("[4/7] 验证 Python 依赖（" + pyExe + "）…");
        var failed = new List<string>();
        foreach (var mod in RequiredModules)
        {
            if (PythonQuickProbe.TryImportModule(pyExe, mod, out var err))
                log("  [√] " + mod);
            else
            {
                log("  [×] " + mod + ": " + err);
                failed.Add(mod);
            }
        }

        var hasImfCsv = File.Exists(Path.Combine(AppPaths.InstallDir, "models", "XAUUSD", "imf_vmd.csv"));
        if (PythonQuickProbe.TryImportModule(pyExe, "pyarrow", out var pyarrowErr))
            log("  [√] pyarrow");
        else if (hasImfCsv)
            log("  [!] pyarrow 不可用，将使用 imf_vmd.csv 推理（可继续）");
        else
        {
            log("  [×] pyarrow: " + pyarrowErr);
            failed.Add("pyarrow");
        }

        if (failed.Count > 0)
        {
            log("[!] 缺少包，尝试用同一 Python 补装 requirements_runtime.txt …");
            var runtimeReq = Path.Combine(AppPaths.InstallDir, "requirements_runtime.txt");
            var fixArgs = File.Exists(runtimeReq)
                ? new List<string> { "-m", "pip", "install", "--prefer-binary", "-r", runtimeReq }
                : new List<string>
                {
                    "-m", "pip", "install", "--prefer-binary",
                    "torch", "xgboost", "pandas==2.2.3", "pyarrow==17.0.0", "numpy>=2.0,<3",
                    "scikit-learn", "joblib", "MetaTrader5", "fredapi", "requests",
                };
            var fixCode = await RunPythonAsync(pyExe, fixArgs, AppPaths.InstallDir, log, ct).ConfigureAwait(false);
            if (fixCode == 0)
            {
                failed.Clear();
                foreach (var mod in RequiredModules)
                {
                    if (!PythonQuickProbe.TryImportModule(pyExe, mod, out var err))
                    {
                        log("  [×] " + mod + ": " + err);
                        failed.Add(mod);
                    }
                }

                if (failed.Count == 0 && !PythonQuickProbe.TryImportModule(pyExe, "pyarrow", out pyarrowErr))
                {
                    if (!hasImfCsv)
                    {
                        log("  [×] pyarrow: " + pyarrowErr);
                        failed.Add("pyarrow");
                    }
                }
            }
        }

        if (failed.Count > 0)
        {
            log("[×] 仍缺少: " + string.Join(", ", failed));
            log("    请确认网络/杀毒软件未拦截 pip，或以管理员运行烛龙后再试。");
            return false;
        }

        log("  验证 IMF Parquet/CSV…");
        if (!VerifyImfCacheReadable(log))
        {
            log("[×] 模型 IMF 缓存无法读取（pyarrow/pandas 仍不兼容）");
            return false;
        }
        log("  [√] IMF 缓存可读");

        // 5) 重载 Python.NET
        log("[5/7] 重载 Python 运行库（MT5 用；推理已子进程隔离）…");
        try
        {
            if (_python.IsReady)
                _python.Reinitialize();
            else
                _python.Initialize();
            log("[√] Python 运行库就绪（推理子进程隔离）");
        }
        catch (Exception ex)
        {
            log("[×] Python.NET 失败: " + ex.Message);
            return false;
        }

        log("  验证 V14 模型与 IMF 特征链…");
        if (!VerifyInferencePath(log))
        {
            log("[×] 推理链路未通过（见上方日志）");
            return false;
        }
        log("  [√] 推理链路 OK");

        // 6) 宏观脚本烟测
        log("[6/7] 宏观离线脚本烟测…");
        try
        {
            var (macroOk, macroMsg) = await _macroRefresh.RefreshAllAsync(ct).ConfigureAwait(false);
            log(macroOk ? "[√] " + macroMsg : "[!] " + macroMsg);
        }
        catch (Exception ex)
        {
            log("[!] 宏观刷新异常（可稍后手动刷新）: " + ex.Message);
        }

        // 7) MT5 检查清单
        log("[7/7] MT5 连接清单");
        log("  · 请先打开 MetaTrader 5 并登录");
        log("  · 工具→选项→专家顾问：允许算法交易、允许 DLL");
        log("  · 图表加载 ZhuLongIndicator（M1）");
        log("  · 回到主界面点「连接 MT5」→「开始运行」");

        log("—— 一键修复完成 ——");
        return true;
    }

    private static void PatchUserConfig(Action<string> log)
    {
        var userCfg = Path.Combine(AppPaths.AppDataDir, "config.json");
        if (!File.Exists(userCfg))
            return;

        try
        {
            var text = File.ReadAllText(userCfg);
            if (text.Contains("api.api2d.com", StringComparison.OrdinalIgnoreCase))
            {
                text = text.Replace("https://api.api2d.com/v1/chat/completions",
                    "https://oa.api2d.net/v1/chat/completions", StringComparison.OrdinalIgnoreCase);
                File.WriteAllText(userCfg, text);
                log("  已更新 AppData config：API2D 端点 → oa.api2d.net");
            }

            var node = JsonNode.Parse(text) as JsonObject;
            if (node?["macro"]?["sentiment"] is JsonObject sent &&
                sent["base_url"]?.GetValue<string>() is { } url &&
                url.Contains("api.api2d.com", StringComparison.OrdinalIgnoreCase))
            {
                sent["base_url"] = "https://oa.api2d.net/v1/chat/completions";
                File.WriteAllText(userCfg, node.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
                log("  已修复 macro.sentiment.base_url");
            }
        }
        catch (Exception ex)
        {
            log("[!] 配置修补跳过: " + ex.Message);
        }
    }

    private static bool VerifyImfCacheReadable(Action<string> log)
    {
        var imfParquet = Path.Combine(AppPaths.InstallDir, "models", "XAUUSD", "imf_vmd.parquet");
        var imfCsv = Path.Combine(AppPaths.InstallDir, "models", "XAUUSD", "imf_vmd.csv");
        if (!File.Exists(imfParquet) && !File.Exists(imfCsv))
        {
            log("  [!] 未找到 imf_vmd 缓存，跳过");
            return true;
        }

        var code =
            "import sys\n" +
            "from pathlib import Path\n" +
            $"root = Path(r'{AppPaths.InstallDir.Replace("\\", "\\\\")}')\n" +
            "sys.path.insert(0, str(root))\n" +
            "import pandas as pd\n" +
            "csv = root / 'models' / 'XAUUSD' / 'imf_vmd.csv'\n" +
            "pq = root / 'models' / 'XAUUSD' / 'imf_vmd.parquet'\n" +
            "ok = False\n" +
            "if csv.is_file():\n" +
            "    df = pd.read_csv(csv, index_col=0, parse_dates=True)\n" +
            "    ok = len(df) > 0\n" +
            "    print('IMF_CSV', len(df))\n" +
            "if not ok and pq.is_file():\n" +
            "    from zhulong.utils.parquet_io import read_parquet_safe\n" +
            "    df = read_parquet_safe(pq)\n" +
            "    ok = df is not None and len(df) > 0\n" +
            "    print('IMF_PQ', 0 if df is None else len(df))\n" +
            "if not ok: raise SystemExit('IMF_READ_FAIL')\n";

        if (!PythonExecutableResolver.TryResolve("python", out var exe, out var resolveErr))
        {
            log("  [×] " + resolveErr);
            return false;
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                WorkingDirectory = AppPaths.InstallDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (PythonQuickProbe.IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(code);
            var dll = AppPaths.FindPythonDll();
            if (!string.IsNullOrEmpty(dll))
                psi.Environment["PYTHONNET_PYDLL"] = dll;

            using var p = Process.Start(psi)!;
            var stdout = p.StandardOutput.ReadToEnd().Trim();
            var stderr = p.StandardError.ReadToEnd().Trim();
            p.WaitForExit(120000);
            if (p.ExitCode != 0)
            {
                log("  [×] " + (string.IsNullOrEmpty(stderr) ? stdout : stderr));
                return false;
            }

            log("  " + stdout);
            return true;
        }
        catch (Exception ex)
        {
            log("  [×] " + ex.Message);
            return false;
        }
    }

    private static bool VerifyInferencePath(Action<string> log)
    {
        if (!PythonExecutableResolver.TryResolve("python", out var exe, out var resolveErr))
        {
            log("  [×] " + resolveErr);
            return false;
        }

        var manifest = Path.Combine(AppPaths.InstallDir, "models", "XAUUSD", "manifest.json");
        if (!File.Exists(manifest))
        {
            log("  [×] 缺少 models/XAUUSD/manifest.json，请重新安装烛龙");
            return false;
        }

        var code =
            "import sys\n" +
            "from pathlib import Path\n" +
            $"root = Path(r'{AppPaths.InstallDir.Replace("\\", "\\\\")}')\n" +
            "sys.path.insert(0, str(root))\n" +
            "sys.path.insert(0, str(root / 'ZhuLong.PythonEngine'))\n" +
            "import os; os.environ['ZHULONG_IMF_CSV_ONLY']='1'\n" +
            "from zhulong.live_v8_features import _load_imf_cache\n" +
            "from zhulong.v14_live import validate_v14_artifacts, load_v14_bundle\n" +
            "assert validate_v14_artifacts('XAUUSD', root=root), 'v14 artifacts missing'\n" +
            "bundle = load_v14_bundle('XAUUSD', model_subdir='v14', root=root)\n" +
            "imf = _load_imf_cache('XAUUSD')\n" +
            "assert imf is not None and len(imf) > 0, 'IMF cache empty'\n" +
            "print('INFER_PATH_OK', len(imf), len(bundle['columns']))\n";

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = exe,
                WorkingDirectory = AppPaths.InstallDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            if (PythonQuickProbe.IsPyLauncher(exe))
                psi.ArgumentList.Add("-3");
            psi.ArgumentList.Add("-c");
            psi.ArgumentList.Add(code);
            var dll = AppPaths.FindPythonDll();
            if (!string.IsNullOrEmpty(dll))
                psi.Environment["PYTHONNET_PYDLL"] = dll;

            using var p = Process.Start(psi)!;
            var stdout = p.StandardOutput.ReadToEnd().Trim();
            var stderr = p.StandardError.ReadToEnd().Trim();
            p.WaitForExit(120000);
            if (p.ExitCode != 0)
            {
                log("  [×] " + (string.IsNullOrEmpty(stderr) ? stdout : stderr));
                return false;
            }

            log("  " + stdout);
            return true;
        }
        catch (Exception ex)
        {
            log("  [×] " + ex.Message);
            return false;
        }
    }

    private static async Task<bool> RunPipRepairAsync(string pyExe, Action<string> log, CancellationToken ct)
    {
        var root = AppPaths.InstallDir;
        var steps = new List<string[]>
        {
            new[] { "install", "--upgrade", "pip", "wheel", "setuptools" },
        };

        var runtimeReq = Path.Combine(root, "requirements_runtime.txt");
        if (File.Exists(runtimeReq))
            steps.Add(new[] { "install", "--prefer-binary", "-r", runtimeReq });
        else
            steps.Add(new[] { "install", "--prefer-binary", "torch", "xgboost", "pandas==2.2.3", "pyarrow==17.0.0", "numpy>=2.0,<3", "scikit-learn", "joblib", "MetaTrader5", "fredapi", "requests" });

        steps.Add(new[] { "install", "--prefer-binary", "--upgrade", "MetaTrader5", "fredapi", "requests" });

        foreach (var args in steps)
        {
            log("  >>> pip " + string.Join(' ', args));
            var pipArgs = new List<string> { "-m", "pip" };
            pipArgs.AddRange(args);
            var code = await RunPythonAsync(pyExe, pipArgs, root, log, ct, PipStepTimeout).ConfigureAwait(false);
            if (code != 0)
            {
                log("[×] pip 失败（退出码 " + code + "）");
                return false;
            }

            log("  <<< pip 本步完成");
        }

        return true;
    }

    private static async Task<int> RunPythonAsync(
        string pyExe, IReadOnlyList<string> args, string cwd, Action<string> log, CancellationToken ct,
        TimeSpan? stepTimeout = null)
    {
        var timeout = stepTimeout ?? PipStepTimeout;
        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeoutCts.CancelAfter(timeout);

        var psi = new ProcessStartInfo
        {
            FileName = pyExe,
            WorkingDirectory = cwd,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };

        if (PythonQuickProbe.IsPyLauncher(pyExe))
            psi.ArgumentList.Add("-3");
        foreach (var a in args)
            psi.ArgumentList.Add(a);

        var dll = AppPaths.FindPythonDll();
        if (!string.IsNullOrEmpty(dll))
            psi.Environment["PYTHONNET_PYDLL"] = dll;

        using var p = Process.Start(psi)!;
        var stdoutTask = PumpLinesAsync(p.StandardOutput, line => log("    " + line.TrimEnd('\r')), timeoutCts.Token);
        var stderrTask = PumpLinesAsync(p.StandardError, line => log("    [err] " + line.TrimEnd('\r')), timeoutCts.Token);

        try
        {
            await p.WaitForExitAsync(timeoutCts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            try { p.Kill(entireProcessTree: true); } catch { /* ignore */ }
            log("[×] pip 超时（超过 " + (int)timeout.TotalMinutes + " 分钟）。请检查网络；torch 首次下载可能较慢。");
            return -1;
        }

        await stdoutTask.ConfigureAwait(false);
        await stderrTask.ConfigureAwait(false);
        return p.ExitCode;
    }

    private static async Task PumpLinesAsync(StreamReader reader, Action<string> onLine, CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                var line = await reader.ReadLineAsync(ct).ConfigureAwait(false);
                if (line is null)
                    break;
                onLine(line);
            }
        }
        catch (OperationCanceledException)
        {
            /* timeout */
        }
    }

    /// <summary>启动/调度前：验证智能体子进程（Python 依赖 + 模型 + 试跑 tick）。</summary>
    public async Task TryRefreshMacroOnStartupAsync(CancellationToken ct = default)
    {
        try
        {
            var (ok, msg) = await _macroRefresh.RefreshAllAsync(ct).ConfigureAwait(false);
            if (!ok)
                StartupLog.Write("宏观离线刷新: " + msg);
        }
        catch (Exception ex)
        {
            StartupLog.Write("宏观离线刷新跳过: " + ex.Message);
        }
    }

    public async Task<bool> EnsureAgentReadyAsync(Action<string> log, CancellationToken ct = default)
    {
        var settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
        AgentConfigSync.AlignWithAppSettings(settings);
        var configPath = AgentConfigSync.ResolveAgentConfigPath(settings);
        log("智能体环境校验开始…");
        log("配置文件: " + configPath);

        if (_agentEnvRepairAttempted &&
            _lastAgentValidateFailUtc != default &&
            DateTimeOffset.UtcNow - _lastAgentValidateFailUtc < TimeSpan.FromMinutes(3))
        {
            return false;
        }

        for (var attempt = 0; attempt < 2; attempt++)
        {
            try
            {
                await _python.AgentValidateAsync(configPath, ct).ConfigureAwait(false);
                log("环境探针通过（Horizon InferenceSession + scaler + 模型文件 + Python 依赖）");
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                log($"[×] 环境探针失败: {msg}");
                LogHorizonProbeHints(log, msg);
                if (attempt == 0 && IsRepairableAgentEnvError(msg))
                {
                    log("  检测到 Python/ONNX 依赖问题，正在自动修复…");
                    _agentEnvRepairAttempted = true;
                    if (await TryAutoRepairAgentEnvAsync(log, ct, msg).ConfigureAwait(false))
                    {
                        log("  依赖修复完成，重新校验…");
                        continue;
                    }

                    log("  自动修复未完成，请检查网络或以管理员运行 install_python_deps.ps1");
                }

                _lastAgentValidateFailUtc = DateTimeOffset.UtcNow;
                return false;
            }

            try
            {
                await _python.AgentWarmupAsync(configPath, ct).ConfigureAwait(false);
                log("智能体 V16 全栈热加载完成（Horizon InferenceSession + AgentEngine + KN2 + RL）");
                _lastAgentValidateFailUtc = default;
                _agentEnvRepairAttempted = false;
                return true;
            }
            catch (Exception ex)
            {
                var msg = ex.Message;
                log($"[×] 全栈热加载失败: {msg}");
                if (msg.Contains("timeout", StringComparison.OrdinalIgnoreCase) ||
                    msg.Contains("超时", StringComparison.OrdinalIgnoreCase))
                {
                    log("  提示：Horizon/RL 预加载超时（环境探针已通过，不是「未安装 Python」）");
                    log("  建议：结束残留 python.exe 后重启；或确认 models/ 与 RL 权重可读");
                }
                else if (msg.Contains("engine_preload_failed", StringComparison.OrdinalIgnoreCase) ||
                         msg.Contains("rl_load_failed", StringComparison.OrdinalIgnoreCase))
                {
                    log("  提示：Horizon 已通过，RL/torch 加载失败；请运行 install_python_deps.ps1 或一键修复环境");
                }
                else if (msg.Contains("horizon_not_ready", StringComparison.OrdinalIgnoreCase))
                {
                    LogHorizonProbeHints(log, msg);
                }
                else
                {
                    log("请确认：1) 无残留 python 子进程  2) 安装目录与 AppData 热更新未冲突");
                }

                if (attempt == 0 && IsRepairableAgentEnvError(msg))
                {
                    log("  检测到可修复依赖问题，正在自动升级…");
                    _agentEnvRepairAttempted = true;
                    if (await TryAutoRepairAgentEnvAsync(log, ct, msg).ConfigureAwait(false))
                    {
                        log("  依赖修复完成，重新校验…");
                        continue;
                    }
                }

                _lastAgentValidateFailUtc = DateTimeOffset.UtcNow;
                return false;
            }
        }

        return false;
    }

    private static void LogHorizonProbeHints(Action<string> log, string? msg)
    {
        if (string.IsNullOrEmpty(msg))
            return;

        if (msg.Contains("onnx=", StringComparison.OrdinalIgnoreCase))
        {
            var idx = msg.IndexOf("onnx=", StringComparison.OrdinalIgnoreCase);
            var detail = idx >= 0 ? msg[idx..].Trim() : msg;
            log("  ONNX 详情: " + detail);
        }

        if (msg.Contains("horizon_not_ready", StringComparison.OrdinalIgnoreCase) ||
            msg.Contains("horizon_onnx", StringComparison.OrdinalIgnoreCase) ||
            msg.Contains("InferenceSession", StringComparison.OrdinalIgnoreCase) ||
            msg.Contains("onnx=", StringComparison.OrdinalIgnoreCase))
        {
            log("  提示：模型文件存在但 InferenceSession 创建失败（import onnxruntime 通过 ≠ 可推理）");
            log("  建议：1) 以管理员运行 install_python_deps.ps1 重装 onnxruntime");
            log("        2) 安装 VC++ 2015-2022 x64 运行库");
            log("        3) 删除 %APPDATA%\\ZhuLong\\ZhuLong.PythonEngine 陈旧热更新后重试");
            log("        4) 运行 scripts\\diagnose_horizon_not_ready.ps1 获取完整诊断");
        }
        else if (msg.Contains("missing_python_module", StringComparison.OrdinalIgnoreCase) ||
                 msg.Contains("python_not_found", StringComparison.OrdinalIgnoreCase) ||
                 msg.Contains("python_too_old", StringComparison.OrdinalIgnoreCase))
        {
            log("请确认：1) Python 3.10+  2) install_python_deps.ps1  3) 设置页 Python 路径正确");
        }
        else if (msg.Contains("missing_horizon", StringComparison.OrdinalIgnoreCase))
        {
            log("请确认 models/horizon_v16.onnx 与 horizon_v16_scaler.pkl 存在于安装目录或 AppData");
        }
    }

    private static bool IsRepairableAgentEnvError(string msg) =>
        msg.Contains("rl_load_failed", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("rl_warmup", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("rl_import_failed", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("numpy._core", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("numpy_too_old", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("missing_python_module", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("numpy_version_check_failed", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("horizon_not_ready", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("horizon_onnx_load_failed", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("InferenceSession", StringComparison.OrdinalIgnoreCase) ||
        msg.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase);

    private async Task<bool> TryAutoRepairAgentEnvAsync(Action<string> log, CancellationToken ct, string? triggerMsg = null)
    {
        var (exe, dll) = PythonRuntime.DiscoverAndCache(force: true);
        if (string.IsNullOrEmpty(exe) || !File.Exists(exe))
        {
            if (!PythonExecutableResolver.TryResolve("python", out exe, out var resolveErr))
            {
                log("  [×] 无法定位 Python: " + resolveErr);
                return false;
            }
        }

        var pyExe = Path.GetFullPath(exe);
        if (!string.IsNullOrEmpty(dll) && File.Exists(dll))
            Environment.SetEnvironmentVariable("PYTHONNET_PYDLL", dll);

        log("  使用 Python: " + pyExe);

        var needsOnnx = triggerMsg is not null &&
            (triggerMsg.Contains("horizon_not_ready", StringComparison.OrdinalIgnoreCase) ||
             triggerMsg.Contains("InferenceSession", StringComparison.OrdinalIgnoreCase) ||
             triggerMsg.Contains("onnx=", StringComparison.OrdinalIgnoreCase) ||
             triggerMsg.Contains("horizon_onnx", StringComparison.OrdinalIgnoreCase));
        if (needsOnnx)
        {
            log("  >>> pip install --force-reinstall onnxruntime …");
            var onnxCode = await RunPythonAsync(
                pyExe,
                new List<string> { "-m", "pip", "install", "--prefer-binary", "--force-reinstall", "onnxruntime" },
                AppPaths.InstallDir,
                log,
                ct).ConfigureAwait(false);
            if (onnxCode != 0)
                log("  [!] onnxruntime 重装未完全成功，继续全量修复…");
            else
                log("  [√] onnxruntime 已重装");
        }

        var targeted = new List<string>
        {
            "-m", "pip", "install", "--prefer-binary", "--upgrade",
            "numpy>=2.0,<3.0", "stable-baselines3>=2.2.0", "gymnasium>=0.29.0", "torch>=2.0.0",
        };
        log("  >>> pip install --upgrade numpy/stable-baselines3 …");
        var targetedCode = await RunPythonAsync(pyExe, targeted, AppPaths.InstallDir, log, ct).ConfigureAwait(false);
        if (targetedCode != 0)
            log("  [!] 定向 pip 未完全成功，继续全量 requirements_runtime …");

        if (!await RunPipRepairAsync(pyExe, log, ct).ConfigureAwait(false))
            return false;

        if (!PythonQuickProbe.TryGetNumpyVersion(pyExe, out var npVer, out var npErr))
        {
            log("  [×] numpy 仍不可用: " + npErr);
            return false;
        }

        if (npVer is null || npVer.Major < 2)
        {
            log("  [×] numpy 版本仍 <2: " + npVer);
            return false;
        }

        log("  [√] numpy " + npVer);
        return PythonQuickProbe.TryImportModule(pyExe, "stable_baselines3", out _);
    }

    private static string ResolveDepsScriptPath()
    {
        var install = Path.Combine(AppPaths.InstallDir, "install_python_deps.ps1");
        if (File.Exists(install))
            return install;
        return Path.Combine(AppPaths.FindDevRoot() ?? AppPaths.InstallDir, "scripts", "install_python_deps.ps1");
    }

    private static string ResolvePowerShellExe()
    {
        var pwsh = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
            "PowerShell", "7", "pwsh.exe");
        return File.Exists(pwsh) ? pwsh : "powershell.exe";
    }

    private static async Task<int> RunPowerShellAsync(
        string scriptPath, string pythonExe, string rootDir, Action<string> log, CancellationToken ct)
    {
        var psi = new ProcessStartInfo
        {
            FileName = ResolvePowerShellExe(),
            WorkingDirectory = rootDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
        };
        psi.ArgumentList.Add("-NoProfile");
        psi.ArgumentList.Add("-ExecutionPolicy");
        psi.ArgumentList.Add("Bypass");
        psi.ArgumentList.Add("-File");
        psi.ArgumentList.Add(scriptPath);
        psi.ArgumentList.Add("-PythonExe");
        psi.ArgumentList.Add(pythonExe);
        psi.ArgumentList.Add("-Root");
        psi.ArgumentList.Add(rootDir);

        using var p = Process.Start(psi)!;
        var stdout = await p.StandardOutput.ReadToEndAsync(ct).ConfigureAwait(false);
        var stderr = await p.StandardError.ReadToEndAsync(ct).ConfigureAwait(false);
        await p.WaitForExitAsync(ct).ConfigureAwait(false);

        foreach (var line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            log(line.TrimEnd('\r'));
        foreach (var line in stderr.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            log("[ps] " + line.TrimEnd('\r'));

        return p.ExitCode;
    }
}
