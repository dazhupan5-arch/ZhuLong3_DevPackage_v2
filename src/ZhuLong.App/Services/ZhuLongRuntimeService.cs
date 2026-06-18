using System.Text.Json;
using System.Threading;
using Microsoft.Extensions.Logging;
using ZhuLong.Core;
using ZhuLong.Core.Bootstrap;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;
using ZhuLong.Core.Pipes;
using ZhuLong.Core.Macro;
using ZhuLong.Core.Services;
using ZhuLong.App.Services.Membership;

namespace ZhuLong.App.Services;

public sealed class RuntimeState
{
    public DateTime? LastBarUtc { get; set; }
    public DateTime? LatestM1BarTime { get; set; }
    public string? ActiveSymbol { get; set; }
    public string PrimarySymbol { get; set; } = "XAUUSD";
    public bool Mt5Connected { get; set; }
    public bool ModelsReady { get; set; }
    public bool IsRunning { get; set; }
    public bool PipeDataConnected { get; set; }
    public bool PipeDrawConnected { get; set; }
    public string PipeStatus { get; set; } = "未启动";
    public string ActiveMarketState { get; set; } = "";
    public string ActiveStrategy { get; set; } = "";
    public string AgentArchitecture { get; set; } = "";
    /// <summary>智能体最新方向意见（开机评估/M5 tick，非已发信号）。</summary>
    public string AgentOpinionText { get; set; } = "智能体意见：—";
    public int ManagedPositionCount { get; set; }
    public string SyncHealthText { get; set; } = "托管 —";

    // ===== P0-2: 保留最新 Bid/Ask =====
    public double LatestBid { get; set; }
    public double LatestAsk { get; set; }
    public DateTime LastPriceUpdate { get; set; }
    // ===== P2-2: AI 动态 SL/TP 存储 =====
    public double? AiSuggestedSl { get; set; }
    public double? AiSuggestedTp { get; set; }
    public string? AiExitReason { get; set; }
    // ===== 市场状态 =====
    public bool IsMarketOpen { get; set; } = true;
}

/// <summary>烛龙运行时编排 — 五线程模型（G10）。</summary>
public sealed class ZhuLongRuntimeService : IAsyncDisposable
{
    private readonly ILogger<ZhuLongRuntimeService> _logger;
    private readonly PipeServer _pipeServer;
    private readonly FeatureCacheService _featureCache;
    private readonly SignalGeneratorService _signalGenerator = new();
    private readonly MacroCalendarService _macro;
    private readonly PythonInferenceService _python;
    private readonly Mt5ApiWrapper _mt5;
    private readonly DatabaseService _database;
    private readonly PendingSignalStore _pendingStore;
    private readonly PositionManagerService _positionManager;
    private readonly MarketSnapshotStore _marketSnapshot;
    private readonly RiskGuardService _riskGuard;
    private readonly InferenceSnapshotStore _inferenceSnap;
    private readonly AlertService _alerts;
    private readonly PythonEnvironmentCoordinator _envCoordinator;
    private readonly Dictionary<string, InferenceResult> _lastInference = new();
    private readonly SemaphoreSlim _historyBootstrapGate = new(1, 1);
    private readonly SemaphoreSlim _signalTickGate = new(1, 1);
    private readonly TaskCompletionSource _startupHistoryReady = new();  // P1-1: 启动等待历史数据就绪
    private volatile bool _startupAssessmentDone;  // 启动评估是否已完成
    private volatile bool _startupAssessmentApplied;  // 是否成功产出可采信的智能体意见（非 skip/异常）
    private AppSettings _settings = new();
    private readonly HashSet<string> _warnedNoModelSymbols = new(StringComparer.OrdinalIgnoreCase);
    private bool _productionModelSkipLogged;
    private bool _agentRuntimeReady;
    private bool _coreInitialized;
    private bool _pipesListening;
    private CancellationTokenSource? _cts;
    private Task? _signalLoop;
    private Task? _positionLoop;
    private Task? _mt5Watchdog;
    private int _m1ApiSyncTicks;
    private Task? _expiryLoop;
    private Task? _dailyCleanup;
    private Task? _priceMonitorLoop;
    private Task? _syncRealPosLoop;
    private readonly SemaphoreSlim _replayDrawGate = new(1, 1);
    private readonly HashSet<string> _drawnSignalIds = new(StringComparer.Ordinal);
    private readonly object _drawnLock = new();
    private readonly object _agentBarLock = new();
    private DateTime _lastAgentSignalBarTime = DateTime.MinValue;

    public RuntimeState State { get; } = new();

    public MacroEventRecord? GetNextHighImpactEvent() => _macro.GetNextHighImpactEvent();

    public ZhuLongRuntimeService(
        ILogger<ZhuLongRuntimeService> logger,
        PipeServer pipeServer,
        FeatureCacheService featureCache,
        PythonInferenceService python,
        Mt5ApiWrapper mt5,
        DatabaseService database,
        PendingSignalStore pendingStore,
        PositionManagerService positionManager,
        MarketSnapshotStore marketSnapshot,
        MacroCalendarService macro,
        RiskGuardService riskGuard,
        InferenceSnapshotStore inferenceSnap,
        AlertService alerts,
        PythonEnvironmentCoordinator envCoordinator)
    {
        _logger = logger;
        _pipeServer = pipeServer;
        _featureCache = featureCache;
        _python = python;
        _envCoordinator = envCoordinator;
        _mt5 = mt5;
        _database = database;
        _pendingStore = pendingStore;
        _positionManager = positionManager;
        _marketSnapshot = marketSnapshot;
        _macro = macro;
        _riskGuard = riskGuard;
        _inferenceSnap = inferenceSnap;
        _alerts = alerts;

        _pipeServer.BarReceived += OnBarReceived;
        _pipeServer.HistoryBarsReceived += OnHistoryBarsReceived;
        _pipeServer.DataClientConnected += OnDataPipeConnected;
        _pipeServer.SessionReceived += OnMt5SessionReceived;
        _pipeServer.DrawClientConnected += () =>
        {
            State.PipeDrawConnected = true;
            State.PipeDataConnected = _pipeServer.IsDataConnected;
            State.PipeStatus = FormatPipeStatusText();
            Log("MT5 已连接绘图管道");
            _ = ReplayPendingDrawCommandsAsync();
        };
        _featureCache.M5BarCompleted += (sym, bar) =>
        {
            Log($"已合成新的 5 分钟 K 线 {sym} {Mt5Time.FormatBar(bar.Time)} C={bar.Close:F2}");
            if (!IsTradingAgentEnabled()) return;
            if (!_startupAssessmentDone) return;
            if (!string.Equals(sym, State.PrimarySymbol, StringComparison.OrdinalIgnoreCase)) return;
            lock (_agentBarLock)
            {
                if (bar.Time <= _lastAgentSignalBarTime) return;
                _lastAgentSignalBarTime = bar.Time;
            }
            var cts = _cts;
            if (cts is null || cts.IsCancellationRequested) return;
            _ = Task.Run(async () =>
            {
                try { await RunSignalTickAsync(cts.Token); }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "M5 闭合触发智能体调度失败");
                    Log($"M5 闭合调度异常: {ex.Message}");
                }
            }, cts.Token);
        };
        _positionManager.PositionUpdated += p =>
        {
            UpdateSyncHealth();
            PositionUpdated?.Invoke(p);
        };
        _positionManager.SignalDrawClearRequested += sid => _ = ClearSignalDrawWithRetryAsync(sid);
        _positionManager.ManagedStatusChanged += (id, status) =>
        {
            SignalStatusChanged?.Invoke(id, status, "", null);
            if (string.Equals(status, "active", StringComparison.OrdinalIgnoreCase))
            {
                var managed = _positionManager.GetManagedState(id);
                if (managed is not null && !string.IsNullOrWhiteSpace(managed.Symbol))
                    _riskGuard.RecordSignalEmitted(managed.Symbol);
            }
        };
        _positionManager.PositionClosed += p =>
        {
            var rawReason = p.TrailingState;
            var status = rawReason switch
            {
                "trailing" => "trailing_stop",
                "trailing_stop" => "trailing_stop",
                "profit_drawdown" => "profit_drawdown",
                "stop_loss" => "stop_loss",
                "take_profit" => "take_profit",
                "time_stop" => "time_stop",
                "model_exit" => "model_exit",
                "external_close" => "external_close",
                "closed" => "normal_close",
                _ => rawReason
            };
            _pendingStore.UpdateStatus(p.SignalId, status, rawReason);
            UntrackDrawnSignal(p.SignalId);
            _ = ClearSignalDrawWithRetryAsync(p.SignalId);
            SignalStatusChanged?.Invoke(p.SignalId, status, rawReason, p.ProfitPct);
            UpdateSyncHealth();
            Log($"平仓信号已更新 {p.Symbol} {p.SignalId} → {status}");
            if (IsTradingAgentEnabled() && _agentRuntimeReady && _python.IsReady)
            {
                var pnlR = ProfitPctToR(p);
                _ = NotifyAgentClosedTradeAsync(p.Symbol, pnlR);
            }
        };
        _positionManager.LogEmitted += Log;
        _positionManager.ChartRefreshRequested += state => _ = RefreshManagedSignalDrawAsync(state);
        _mt5.LogEmitted += Log;
    }

    public event Action<string>? LogEmitted;
    public event Action<SignalModel>? SignalCreated;
    public event Action<string, string, string, double?>? SignalStatusChanged;
    public event Action<ManagedPositionModel>? PositionUpdated;
    public event Action<string>? ModelsMissing;
    public event Action<string>? AgentOpinionUpdated;
    public event Action? SignalsHydrated;

    public AppSettings Settings => _settings;

    public void LoadSettings()
    {
        _settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
        EnsureAgentConfigAligned();
    }

    public void ReloadSettingsFromDisk()
    {
        _settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
        EnsureAgentConfigAligned();
        if (ModelConfigSync.EnsureDefaultSymbols(_settings, persist: true))
            Log("已同步推理品种列表（含已安装模型目录）");
        var errors = ConfigValidator.Validate(_settings);
        if (errors.Count > 0)
            Log("配置校验警告: " + string.Join("; ", errors));
        _positionManager.UpdateSettings(_settings);
        _macro.Configure(_settings);
        _mt5.SetDeviation(_settings.Mt5?.Deviation ?? 20);
        var prod = ProductionModelGate.Check(_settings);
        State.ModelsReady = prod.Ready;
        SyncPrimarySymbolState();
        _productionModelSkipLogged = false;
        Log("策略参数已重新加载");
        if (!State.ModelsReady)
            Log($"推理仍暂停 — {prod.Summary}");
        else
            Log(FormatInferenceModeHint(prod));
    }

    public IReadOnlyList<string> GetConfiguredSymbols() =>
        ModelConfigSync.ResolveSymbols(_settings);

    public string GetPrimarySymbol()
    {
        SyncPrimarySymbolState();
        return State.PrimarySymbol;
    }

    public bool GetInferAllReadySymbols() => _settings.Model?.InferAllReadySymbols ?? false;

    public void SetPrimarySymbol(string symbol)
    {
        symbol = symbol.Trim();
        if (string.IsNullOrEmpty(symbol)) return;

        _settings.Model ??= new AppSettings.ModelSettings();
        _settings.Model.PrimarySymbol = symbol;
        _settings.Model.InferAllReadySymbols = false;
        ModelConfigSync.EnsureDefaultSymbols(_settings, persist: false);
        State.PrimarySymbol = symbol;
        PersistSettings();
        var prod = ProductionModelGate.Check(_settings);
        State.ModelsReady = prod.Ready;
        _productionModelSkipLogged = false;
        if (prod.ReadySymbols.Contains(symbol, StringComparer.OrdinalIgnoreCase))
            Log($"已切换推理品种 → {symbol}");
        else
            Log($"切换品种 → {symbol}（模型未就绪: {prod.Summary}）");
    }

    /// <summary>尽早启动命名管道监听（幂等），避免 Python 初始化期间 MT5 指标连不上。</summary>
    public void EnsurePipeServerStarted()
    {
        if (_pipesListening || _pipeServer.IsListening)
        {
            _pipesListening = true;
            return;
        }

        _pipeServer.Start();
        _pipesListening = true;
        State.PipeStatus = "等待 MT5 指标连接管道（数据/绘图）";
        Log("烛龙管道监听已启动（ZhuLong_Data / ZhuLong_Drawing）");
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        // 加载配置
        _settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
        EnsurePipeServerStarted();
        EnsureAgentConfigAligned();
        SyncPrimarySymbolState();
        _positionManager.UpdateSettings(_settings);
        _macro.Configure(_settings);
        var prod = ProductionModelGate.Check(_settings);
        State.ModelsReady = prod.Ready;
        Log("正在初始化系统…");
        if (!prod.Ready)
        {
            Log(prod.Summary);
            ModelsMissing?.Invoke(prod.Summary);
        }

        // 数据库（活跃信号恢复在启动评估之后执行，避免 inferenceSnap 为空时误关 awaiting_fill）
        await _database.EnsureCreatedAsync(ct).ConfigureAwait(false);

        // 宏观日历（拉取 + 自动刷新循环）
        try
        {
            await _macro.InitializeAsync(ct).ConfigureAwait(false);
            _macro.TryLogUpcomingReminder(Log, hours: 24.0);
            if (IsTradingAgentEnabled())
                _ = _envCoordinator.TryRefreshMacroOnStartupAsync(ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "宏观日历初始化失败");
            Log($"宏观日历加载失败: {ex.Message}");
        }

        // Python / 智能体：仅初始化运行时；Horizon 预加载放到 Start() 历史就绪之后（避免与 MT5 补数抢资源）
        try
        {
            _python.Initialize();
            if (IsTradingAgentEnabled())
                Log("智能体模式：点击「开始运行」后将预加载 Horizon 并启用 5 分钟推理");
            else
            {
                var symbols = prod.ReadySymbols.ToArray();
                if (symbols.Length > 0)
                {
                    Log($"加载模型 {string.Join(", ", symbols)}");
                    await _python.WarmupAsync(symbols, ct).ConfigureAwait(false);
                    if (!await _python.ValidateModelsAsync(symbols, ct).ConfigureAwait(false))
                        Log("子进程 validate 未通过，legacy 推理可能失败");
                    _python.WarmupInBackground(symbols);
                }
                else
                {
                    var sup = string.Join(",", prod.ReadySymbols);
                    Log($"尚无已验收模型（已就绪品种: {sup}），模型训练通过后自动启用");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Python 初始化失败");
            Log("Python 初始化失败: " + ex.Message);
        }

        _coreInitialized = true;
    }

    public void Start()
    {
        MembershipHost.Instance.Refresh();
        if (!MembershipHost.Instance.CanUseApp)
        {
            Log("无法启动：试用已结束，请在「设置」页粘贴授权码激活。");
            return;
        }

        if (_cts is not null) return;
        if (!_coreInitialized)
        {
            Log("系统尚未完成初始化，请稍候再点开始运行");
            return;
        }
        _settings = AppSettings.LoadOrCreate(AppPaths.ConfigPath);
        EnsureAgentConfigAligned();
        SyncPrimarySymbolState();
        _positionManager.UpdateSettings(_settings);
        _macro.Configure(_settings);
        _cts = new CancellationTokenSource();
        _positionManager.SetDataPipeConnected(State.PipeDataConnected);
        _ = RestoreRiskGuardCooldownAsync(_cts.Token);
        EnsurePipeServerStarted();
        State.IsRunning = true;
        if (string.IsNullOrWhiteSpace(State.PipeStatus) || State.PipeStatus == "未启动")
            State.PipeStatus = "等待 MT5 指标连接管道（数据/绘图）";

        // ===== P1-1: 修复 — 先启动历史数据预热（MT5 API 拉取），再运行开机评估 =====
        // 必须把 bootstrap 放到 startup Task 之前，并等待其完成
        Task? bootstrapTask = null;
        if (State.Mt5Connected)
        {
            bootstrapTask = BootstrapMissingHistoryAsync(_cts.Token);
            Log("开始历史数据预热（MT5 API）…");
        }
        else
        {
            Log("MT5 未连接，跳过历史数据预热");
            _startupHistoryReady.TrySetResult();  // 标记无需等待
        }

        _ = Task.Run(async () =>
        {
            try
            {
                if (bootstrapTask is not null)
                {
                    try { await bootstrapTask.ConfigureAwait(false); }
                    catch (Exception ex) { _logger.LogWarning(ex, "历史数据预热未完成"); }
                }

                // Step 1: 等待历史数据就绪（最多 60 秒）
                await WaitForHistoryReadyAsync(_cts!.Token, minM5: 30, timeoutSec: 60);

                // Step 1b: 历史就绪后再预加载 Horizon（避免 warmup 与 MT5 补数并发挂死）
                if (IsTradingAgentEnabled() && !_agentRuntimeReady)
                {
                    _agentRuntimeReady = await _envCoordinator.EnsureAgentReadyAsync(Log, _cts!.Token)
                        .ConfigureAwait(false);
                    if (!_agentRuntimeReady)
                        Log("智能体未就绪：V16 全栈热加载失败，5 分钟推理已暂停（详见上方 Python 错误）");
                    else
                        Log("智能体 V16 全栈已就绪（Horizon + KN2 + RL 已在开机时热加载）");
                }

                // Step 2: 开机智能体评估（用到了 M5 数据才能工作）
                await RunStartupAgentAssessmentAsync(_cts!.Token);

				// Step 3: 恢复信号（受开机评估的 _inferenceSnap 影响）
				await HydratePendingSignalsAsync(_cts!.Token);

				_startupAssessmentDone = true;  // 标记启动评估完成

				if (_pipeServer.IsDrawConnected)
                    await ReplayPendingDrawCommandsAsync();
                else
                    Log("绘图管道未就绪，信号绘制待重连");

                UpdateSyncHealth();
                SignalsHydrated?.Invoke();
            }
            catch (Exception ex) { _logger.LogWarning(ex, "启动时恢复 pending 信号失败"); }
        });
        // ===== 结束 =====

        _signalLoop = Task.Run(() => SignalSchedulerAsync(_cts.Token));
        _positionLoop = Task.Run(() => PositionScannerAsync(_cts.Token));
        _mt5Watchdog = Task.Run(() => Mt5WatchdogAsync(_cts.Token));
        _expiryLoop = Task.Run(() => PendingExpiryLoopAsync(_cts.Token));
        _dailyCleanup = Task.Run(() => DailyCleanupLoopAsync(_cts.Token));
        _priceMonitorLoop = Task.Run(() => RealTimePriceMonitorAsync(_cts.Token));
        _syncRealPosLoop = Task.Run(() => SyncRealPositionsLoopAsync(_cts.Token));
        _ = Task.Run(() => RuntimeHeartbeatAsync(_cts.Token));
        Log("请在 MT5 图表加载 ZhuLongIndicator，并确认 MQL5\\Libraries\\ZhuLongMt5Pipe.dll 已手动部署且允许 DLL");
    }

    public async Task StopAsync()
    {
        _cts?.Cancel();
        if (_signalLoop is not null) try { await _signalLoop; } catch { }
        if (_positionLoop is not null) try { await _positionLoop; } catch { }
        if (_mt5Watchdog is not null) try { await _mt5Watchdog; } catch { }
        if (_expiryLoop is not null) try { await _expiryLoop; } catch { }
        if (_dailyCleanup is not null) try { await _dailyCleanup; } catch { }
        if (_priceMonitorLoop is not null) try { await _priceMonitorLoop; } catch { }
        if (_syncRealPosLoop is not null) try { await _syncRealPosLoop; } catch { }
        _cts?.Dispose();
        _cts = null;
        State.IsRunning = false;
        State.PipeDataConnected = false;
        State.PipeDrawConnected = false;
        if (_pipesListening)
        {
            try { await _pipeServer.StopAsync().ConfigureAwait(false); }
            catch (Exception ex) { _logger.LogDebug(ex, "管道停止时异常"); }
            _pipesListening = false;
        }
        State.PipeStatus = "未启动";
        _mt5.Disconnect();
        State.Mt5Connected = false;
        Log("系统关闭");
    }

    public Task<IReadOnlyList<SignalModel>> GetRecentSignalsAsync(CancellationToken ct = default) =>
        _database.GetRecentSignalsAsync(50, ct);

    public Task<IReadOnlyList<SignalModel>> GetRecentClosedSignalsAsync(CancellationToken ct = default) =>
        _database.GetRecentClosedSignalsAsync(20, ct);

    public IReadOnlyList<ManagedPositionModel> GetManagedPositions() => _positionManager.Snapshot();

	// ===== P1-1: 开机智能体评估（修复 — 不等待管道，直接用 MT5 API 预热数据）=====
	private async Task RunStartupAgentAssessmentAsync(CancellationToken ct)
	{
		if (IsTradingAgentEnabled() && !_agentRuntimeReady)
		{
			Log("开机评估跳过：智能体未就绪，保留现有托管/待处理信号（不做观望裁决）");
			return;
		}

		var primary = State.PrimarySymbol;
		var m5Count = _featureCache.GetM5Count(primary);
		if (m5Count < 30)
		{
			Log($"开机评估跳过：{primary} M5={m5Count} 根 < 30，保留现有托管/待处理信号");
			return;
		}

		Log($"开机智能体评估开始：{primary} M5={m5Count}根");
		await _signalTickGate.WaitAsync(ct).ConfigureAwait(false);
		try
		{
			MultiStrategyTickResult? primaryTick = null;
			if (IsTradingAgentEnabled())
			{
				var configPath = ResolveAgentConfigPath();
				var symbols = new[] { primary };
				var m5Map = new Dictionary<string, (long, double, double, double, double, double)[]>(StringComparer.OrdinalIgnoreCase);
				long decisionBarUnix = 0;
				if (_featureCache.TryExportAgentM5Bars(primary, out var bars, out decisionBarUnix))
					m5Map[primary] = bars;

				if (m5Map.Count > 0)
				{
                    var (results, skipReason) = await _python.AgentTickAsync(
                        m5Map, symbols, primary, _macro.IsSilenceWindow(),
                        configPath, BuildAgentTickPayload(symbols), BuildAgentPositionPayload(),
                        TimeSpan.FromSeconds(300), ct,
                        decisionBarUnix: decisionBarUnix,
                        m5IncludesForming: false,
                        macroFeatures: _macro.GetFeatures());
					if (results.Count == 0 && !string.IsNullOrWhiteSpace(skipReason))
						Log($"开机评估：智能体未运行（{skipReason}）");
					primaryTick = results.FirstOrDefault(r =>
						string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase));
					foreach (var r in results)
					{
						if (r.Skipped)
						{
							Log($"开机评估：{r.Symbol} 跳过（{r.RejectReason ?? "skipped"}，不裁决旧信号）");
							continue;
						}
						_inferenceSnap.Set(r.Symbol, AgentInferenceSnapHelper.ToInferenceSnap(r));
						_startupAssessmentApplied = true;
						Log($"开机评估：{AgentInferenceSnapHelper.FormatPrimaryTickLog(r.Symbol, r)}");
					}
				}
			}
			else if (IsMultiStrategyEnabled())
			{
				var configPath = _settings.MultiStrategy?.ConfigPath ?? "config/config_multi_strategy.json";
				var symbols = new[] { primary };
				var m5Map = new Dictionary<string, (long, double, double, double, double, double)[]>(StringComparer.OrdinalIgnoreCase);
				if (_featureCache.TryExportM5Bars(primary, out var bars))
					m5Map[primary] = bars;

				if (m5Map.Count > 0)
				{
					var results = await _python.MultiStrategyTickAsync(
						m5Map, symbols, primary, _macro.IsSilenceWindow(),
						configPath, TimeSpan.FromSeconds(90), ct);
					foreach (var r in results)
					{
						if (r.Signal != null && r.Signal.Direction is "buy" or "sell")
						{
							_inferenceSnap.Set(r.Symbol, new InferenceResult
							{
								Direction = r.Signal.Direction == "buy" ? 1 : -1,
								Confidence = r.Signal.Confidence,
							});
						}
						else
						{
							_inferenceSnap.Set(r.Symbol, new InferenceResult { Direction = 0, Confidence = 0 });
						}
					}
				}
			}

			if (primaryTick is not null)
			{
				var (opinionDir, displayConf, hDir, cogDir, rl, action, hMin, arch) =
					AgentInferenceSnapHelper.BuildOpinionPublishArgs(primaryTick);
				var isV16 = string.Equals(arch, "v16", StringComparison.OrdinalIgnoreCase);
				Log($"开机智能体评估完成：{AgentInferenceSnapHelper.FormatPrimaryTickLog(primary, primaryTick)}");
				if (isV16)
					PublishAgentOpinion(primary, opinionDir, displayConf, hDir, rl, action, hMin, arch, cogDir);
				else
					PublishAgentOpinion(primary, opinionDir, displayConf, cogDir, rl, action, architecture: arch);
			}
			else if (_inferenceSnap.TryGet(primary, out var inf))
			{
				var hDir = string.IsNullOrWhiteSpace(inf.HorizonDirection) ? inf.CognitionDirection : inf.HorizonDirection;
				var hConf = inf.HorizonConfidence > 0 ? inf.HorizonConfidence : inf.Confidence;
				var cogDir = string.IsNullOrWhiteSpace(inf.CognitionDirection) ? "flat" : inf.CognitionDirection;
				var rl = string.IsNullOrWhiteSpace(inf.RlAction) ? "—" : inf.RlAction;
				Log($"开机智能体评估完成：{primary} Horizon={hDir}({hConf:F2}) 认知={cogDir}({inf.CognitionConfidence:F2}) RL={rl} conf={inf.Confidence:F2}");
				PublishAgentOpinion(
					primary, inf.Direction, inf.Confidence, hDir, rl, "—", 0.48, "v16", cogDir);
			}
			else
			{
				Log("开机智能体评估完成（无有效结果）");
			}
		}
		catch (Exception ex)
		{
			_logger.LogWarning(ex, "开机智能体评估失败");
			Log($"开机评估异常：{ex.Message}（不裁决旧信号，托管/待处理将保守恢复）");
		}
		finally
		{
			_signalTickGate.Release();
		}
	}

	/// <summary>等待 M5 历史数据就绪（最多 timeoutSec 秒），用于开机时序协调。</summary>
	private async Task WaitForHistoryReadyAsync(CancellationToken ct, int minM5 = 30, int timeoutSec = 60)
	{
		// 先检查是否已经就绪
		if (_featureCache.GetM5Count(State.PrimarySymbol) >= minM5)
		{
			Log($"历史数据就绪：{State.PrimarySymbol} M5={_featureCache.GetM5Count(State.PrimarySymbol)} 根");
			return;
		}

		// 等待 BootstrapMissingHistoryAsync 完成（通过 TCS 通知）
		var timeout = Task.Delay(TimeSpan.FromSeconds(timeoutSec), ct);
		var completed = await Task.WhenAny(_startupHistoryReady.Task, timeout).ConfigureAwait(false);
		if (completed == timeout)
			Log($"等待历史数据超时（{timeoutSec}s, M5<{minM5}），跳过开机评估，直接恢复信号");

		// 无论是否超时，都检查当前 M5 数量
		if (_featureCache.GetM5Count(State.PrimarySymbol) >= minM5)
			Log($"历史数据就绪：{State.PrimarySymbol} M5={_featureCache.GetM5Count(State.PrimarySymbol)} 根");
		else
			Log($"历史数据不足（{State.PrimarySymbol} M5={_featureCache.GetM5Count(State.PrimarySymbol)} < {minM5}）");
	}
	// ===== 结束 =====

    private async Task HydratePendingSignalsAsync(CancellationToken ct)
    {
        try
        {
            var primary = State.PrimarySymbol;
            var applyAgentFilter = _startupAssessmentApplied;
            var isFlat = applyAgentFilter
                && _inferenceSnap.TryGet(primary, out var snap)
                && snap.Direction == 0;

            var pending = await _database.GetPendingSignalsAsync(30, ct);
            if (pending.Count > 0)
            {
                if (isFlat)
                {
                    Log($"开机评估明确看平，关闭 {pending.Count} 条待处理旧信号");
                    foreach (var sig in pending)
                        await CloseSignalInDbAsync(sig.SignalId, "model_exit", ct);
                }
                else
                {
                    foreach (var sig in pending)
                        _pendingStore.Add(sig);
                    Log($"已从数据库恢复 {pending.Count} 条待处理信号"
                        + (applyAgentFilter ? "" : "（评估未产出意见，不经 KN2/方向过滤）"));
                }
            }

            var active = await _database.GetActiveSignalsAsync(50, ct);
            if (active.Count > 0)
            {
                if (!applyAgentFilter || _inferenceSnap.IsEmpty)
                {
                    var conservative = await _positionManager.RestoreActiveSignalsAsync(
                        active, ct, applyAgentDirectionFilter: false);
                    Log($"保守恢复托管 {conservative}/{active.Count} 条（评估"
                        + (applyAgentFilter ? "无有效意见" : "未成功/已跳过")
                        + "，保留 awaiting_fill，不经方向过滤）");
                    return;
                }

                var restored = await _positionManager.RestoreActiveSignalsAsync(active, ct);
                var skipped = active.Count - restored;
                Log(skipped > 0
                    ? $"已从数据库恢复 {restored} 条托管信号（跳过 {skipped} 条已超最大持仓时间或AI不兼容）"
                    : $"已从数据库恢复 {restored} 条托管中信号（用于图表重绘）");
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "恢复 pending 信号失败");
        }
    }

    public IReadOnlyList<ManagedPositionModel> GetPositionsForDisplay() =>
        _positionManager.Snapshot().OrderByDescending(p => p.Ticket).ToList();

    private void EnqueueClearSignalDraw(string signalId) =>
        _ = ClearSignalDrawWithRetryAsync(signalId);

    private async Task RefreshManagedSignalDrawAsync(PositionManagerService.ManagedState state)
    {
        if (!_positionManager.IsManagingSignal(state.SignalId))
            return;
        try
        {
            var sl = state.TrailingActivated ? state.TrailingSl : state.StopLoss;
            var previousSl = state.TrailingActivated ? state.StopLoss : 0.0;
            await _pipeServer.SendDrawCommandAsync(new
            {
                action = "draw_signal",
                signal_id = state.SignalId,
                symbol = state.Symbol,
                direction = state.Direction,
                entry = state.EntryPrice,
                sl,
                tp = state.TakeProfit,
                previous_sl = previousSl,
            }, CancellationToken.None);
            TrackDrawnSignal(state.SignalId);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "托管信号绘图刷新失败 {Signal}", state.SignalId);
        }
    }

    private void OnDataPipeConnected()
    {
        State.PipeDataConnected = true;
        _positionManager.SetDataPipeConnected(true);
        if (_pipeServer.IsDrawConnected)
            State.PipeDrawConnected = true;
        State.PipeStatus = FormatPipeStatusText();
        Log("MT5 已连接数据管道");
        EnsureFeatureCacheAfterPipeConnect(null);
        if (State.Mt5Connected && _cts is not null)
            _ = BootstrapMissingHistoryAsync(_cts.Token);
        if (_pipeServer.IsDrawConnected)
            _ = ReplayPendingDrawCommandsAsync();
    }

    private void OnMt5SessionReceived(string symbol, bool warm)
    {
        var std = string.IsNullOrWhiteSpace(symbol)
            ? State.PrimarySymbol
            : NormalizeIncomingSymbol(symbol, out _);
        if (warm)
            Log($"MT5 热重连（切换周期）symbol={std} — FeatureCache 保留，实时 M1 恢复");
        else
            Log($"MT5 会话 symbol={std} warm={warm}");
        EnsureFeatureCacheAfterPipeConnect(std);
    }

    /// <summary>热重连后若缓存不足（如烛龙刚重启），用 MT5 API 补历史。</summary>
    private void EnsureFeatureCacheAfterPipeConnect(string? symbolHint)
    {
        var symbol = string.IsNullOrWhiteSpace(symbolHint) ? State.PrimarySymbol : symbolHint;
        if (string.IsNullOrWhiteSpace(symbol))
            return;

        const int minM1 = 120;
        if (_featureCache.GetM1Count(symbol) >= minM1)
            return;

        Log($"管道连接后 {symbol} M1={_featureCache.GetM1Count(symbol)} 根不足，触发 API 补历史");
        var cts = _cts;
        if (cts is null || cts.IsCancellationRequested)
            return;
        _ = BootstrapMissingHistoryAsync(cts.Token);
    }

    private void OnBarReceived(M1Bar bar)
    {
        var std = NormalizeIncomingSymbol(bar.Symbol, out var brokerRaw);
        if (!string.Equals(brokerRaw, std, StringComparison.OrdinalIgnoreCase))
            bar = CloneBar(bar, std);

        _featureCache.Ingest(bar);
        _marketSnapshot.UpdateFromBar(bar);
        State.LastBarUtc = DateTime.UtcNow;
        State.LatestM1BarTime = bar.Time;

        if (!string.Equals(State.ActiveSymbol, std, StringComparison.OrdinalIgnoreCase))
        {
            State.ActiveSymbol = std;
            var prod = ProductionModelGate.Check(_settings);
            if (prod.ReadySymbols.Contains(std, StringComparer.OrdinalIgnoreCase))
                Log($"图表品种切换 → {std}（使用 {std} 模型推理）");
            else if (_warnedNoModelSymbols.Add(std))
                Log($"图表品种切换 → {std}（尚无正式模型，信号仍仅 {string.Join("/", prod.ReadySymbols)}）");
        }

        State.PipeStatus = $"M1 {std} {Mt5Time.FormatBar(bar.Time, "HH:mm")} | 北京 {ChinaTime.Format(DateTimeOffset.UtcNow, "HH:mm:ss")}";
        Log($"M1 {std} {Mt5Time.FormatBar(bar.Time)} C={bar.Close:F2}");

        var cts = _cts;
        if (cts is not null && !cts.IsCancellationRequested)
            _ = ProcessBarPenetrationSafeAsync(bar, cts.Token);
    }

    private async Task ProcessBarPenetrationSafeAsync(M1Bar bar, CancellationToken ct)
    {
        try
        {
            await _positionManager.ProcessBarPenetrationAsync(bar, ct);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "M1 穿价撮合失败 {Symbol}", bar.Symbol);
        }
    }

    private void OnHistoryBarsReceived(string symbol, IReadOnlyList<M1Bar> bars, bool final)
    {
        var std = NormalizeIncomingSymbol(symbol, out _);
        var normalized = string.Equals(std, symbol, StringComparison.OrdinalIgnoreCase)
            ? bars
            : bars.Select(b => CloneBar(b, std)).ToList();

        _featureCache.AppendHistoryChunk(std, normalized, final);
        if (!final) return;

        var m1 = _featureCache.GetM1Count(std);
        var m5 = _featureCache.GetM5Count(std);
        State.PipeStatus = $"历史 M1 已载入 {std}";
        Log($"历史 M1 已载入 {std}：{m1} 根 → 合成 M5 {m5} 根（开机即可推理）");

        if (string.IsNullOrEmpty(State.ActiveSymbol))
            State.ActiveSymbol = std;
    }

    private string NormalizeIncomingSymbol(string brokerOrRaw, out string brokerRaw)
    {
        brokerRaw = brokerOrRaw.Trim();
        var std = _settings.ResolveStandardSymbol(brokerRaw);
        if (!string.Equals(std, brokerRaw, StringComparison.OrdinalIgnoreCase))
            _featureCache.RekeySymbol(brokerRaw, std);
        return std;
    }

    private static M1Bar CloneBar(M1Bar bar, string symbol) => new()
    {
        Symbol = symbol,
        Time = bar.Time,
        Open = bar.Open,
        High = bar.High,
        Low = bar.Low,
        Close = bar.Close,
        Volume = bar.Volume,
    };

    private IReadOnlyList<string> ResolveInferenceSymbols(ProductionModelGate.CheckResult prod)
    {
        var withData = prod.ReadySymbols
            .Where(s => _featureCache.GetM1Count(s) > 0)
            .ToList();

        if (_settings.Model?.InferAllReadySymbols == true)
            return withData;

        SyncPrimarySymbolState();
        var primary = State.PrimarySymbol;
        if (!prod.ReadySymbols.Contains(primary, StringComparer.OrdinalIgnoreCase))
            return [];

        return _featureCache.GetM1Count(primary) > 0 ? (IReadOnlyList<string>)[primary] : [];
    }

    private IReadOnlyList<string> CollectAgentSymbols()
    {
        var enabled = LoadAgentEnabledSymbols();
        var candidates = CollectMultiStrategySymbols();
        if (enabled.Count == 0)
            return candidates;
        return candidates.Where(s => enabled.Contains(s, StringComparer.OrdinalIgnoreCase)).ToList();
    }

    /// <summary>历史 M1 预热 / API 补数品种：智能体模式下跳过 config 中 disabled 的品种。</summary>
    private IReadOnlyList<string> CollectDataPrefetchSymbols()
    {
        var all = ModelConfigSync.ResolveSymbols(_settings);
        if (!IsTradingAgentEnabled())
            return all;
        var enabled = LoadAgentEnabledSymbols();
        if (enabled.Count == 0)
            return all;
        return all.Where(s => enabled.Contains(s, StringComparer.OrdinalIgnoreCase)).ToList();
    }

    private HashSet<string> LoadAgentEnabledSymbols()
    {
        var set = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        try
        {
            var path = ResolveAgentConfigPath();
            if (!File.Exists(path))
                return set;
            using var doc = JsonDocument.Parse(File.ReadAllText(path));
            if (!doc.RootElement.TryGetProperty("symbols", out var symbols) ||
                symbols.ValueKind != JsonValueKind.Object)
                return set;
            foreach (var prop in symbols.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.Object)
                    continue;
                if (!prop.Value.TryGetProperty("enabled", out var en) || en.GetBoolean())
                    set.Add(prop.Name);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "读取 config_agent symbols 失败");
        }
        return set;
    }

    private IReadOnlyList<string> CollectMultiStrategySymbols()
    {
        const int minM5 = 80;
        var list = new List<string>();
        foreach (var sym in ModelConfigSync.ResolveSymbols(_settings))
        {
            if (_featureCache.GetM5Count(sym) >= minM5)
                list.Add(sym);
        }
        return list;
    }

    // ===== P0-2, P2-2: Tick 数据辅助方法 =====
    private double GetBidPrice(string symbol)
    {
        var tick = _mt5.GetTickPrice(symbol);
        return tick?.Bid ?? 0;
    }

    private double GetAskPrice(string symbol)
    {
        var tick = _mt5.GetTickPrice(symbol);
        return tick?.Ask ?? 0;
    }
    // ===== 结束 =====

    // ----- RunTradingAgentSignalTickAsync -----
    private async Task RunTradingAgentSignalTickAsync(CancellationToken ct)
    {
        _macro.TryLogUpcomingReminder(Log, hours: 12.0);
        if (!_agentRuntimeReady)
        {
            _agentRuntimeReady = await _envCoordinator.EnsureAgentReadyAsync(Log, ct).ConfigureAwait(false);
            if (!_agentRuntimeReady)
            {
                Log("智能体调度暂停：子进程环境未就绪");
                return;
            }
        }

        if (!IsMarketOpen())
        {
            Log("休市期间：跳过智能体调度");
            return;
        }

        var msSymbols = CollectAgentSymbols();
        SyncPrimarySymbolState();
        var primary = State.PrimarySymbol;

        Log($"RL 智能体调度开始（{string.Join(", ", msSymbols)}，主品种 {primary}）");
        if (msSymbols.Count == 0)
        {
            if (_featureCache.GetM1Count(primary) == 0)
                Log("智能体等待中：尚无 M1 数据");
            else
                Log($"智能体等待中：M5 不足（需 ≥80 根，当前 {_featureCache.GetM5Count(primary)}）");
            return;
        }

        if (_macro.IsSilenceWindow())
        {
            Log("宏观静默，信号丢弃");
            return;
        }

        var todayPnl = await _database.GetTodayClosedPnlPercentAsync(ct);
        var openCount = _positionManager.OpenManagedCount;
        if (_riskGuard.BlockNewSignal(_settings, openCount, _pendingStore.Count, todayPnl) is { } globalBlock)
        {
            Log($"风控拦截：{globalBlock}");
            return;
        }

        var m5Map = new Dictionary<string, (long, double, double, double, double, double)[]>(StringComparer.OrdinalIgnoreCase);
        long decisionBarUnix = 0;
        foreach (var sym in msSymbols)
        {
            if (_featureCache.TryExportAgentM5Bars(sym, out var bars, out var barUnix))
            {
                m5Map[sym] = bars;
                if (string.Equals(sym, primary, StringComparison.OrdinalIgnoreCase))
                    decisionBarUnix = barUnix;
            }
        }

        if (m5Map.Count == 0)
        {
            Log("智能体跳过：无法导出 M5");
            return;
        }

        if (decisionBarUnix > 0)
            Log($"决策 M5 bar unix={decisionBarUnix}（已闭合 K 线，全机应对齐）");

        var configPath = ResolveAgentConfigPath();
        var ticks = BuildAgentTickPayload(msSymbols);
        var positions = BuildAgentPositionPayload();
        var macroFeatures = _macro.GetFeatures();
        IReadOnlyList<MultiStrategyTickResult> results;
        string? agentSkipReason = null;
        try
        {
            (results, agentSkipReason) = await _python.AgentTickAsync(
                m5Map, msSymbols, primary, _macro.IsSilenceWindow(), configPath,
                ticks, positions,
                TimeSpan.FromSeconds(120), ct,
                decisionBarUnix: decisionBarUnix,
                m5IncludesForming: false,
                macroFeatures: macroFeatures);
        }
        catch (TimeoutException)
        {
            Log("智能体 Python 推理超时");
            return;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "RL 智能体调度失败");
            if (_positionManager.OpenManagedCount > 0)
                Log("【严重】持仓中智能体 tick 失败：M5 持仓管理/移损/智能体平仓已中断，当前仅 MT5 静态 SL/TP 保护");
            Log($"智能体失败: {ex.Message}");
            return;
        }

        if (results.Count == 0)
        {
            Log(string.IsNullOrWhiteSpace(agentSkipReason)
                ? "智能体未返回结果（请检查 config/config_agent.json 是否 enabled=true）"
                : $"智能体未运行：{agentSkipReason}");
            return;
        }

        var primaryResult = results.FirstOrDefault(r =>
            string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase));
        if (primaryResult is not null)
        {
            State.ActiveMarketState = primaryResult.MarketState;
            State.ActiveStrategy = primaryResult.ActiveStrategy;
            State.AgentArchitecture = primaryResult.Architecture ?? "";
            var rl = string.IsNullOrWhiteSpace(primaryResult.RlRawAction) ? "—" : primaryResult.RlRawAction;
            var action = string.IsNullOrWhiteSpace(primaryResult.AgentAction) ? "—" : primaryResult.AgentAction;
            var arch = primaryResult.Architecture ?? "";
            var isV16 = string.Equals(arch, "v16", StringComparison.OrdinalIgnoreCase);
            Log(AgentInferenceSnapHelper.FormatPrimaryTickLog(primary, primaryResult));
            if (primaryResult.AiSlPrice > 0)
                State.AiSuggestedSl = primaryResult.AiSlPrice;
            if (primaryResult.AiTpPrice > 0)
                State.AiSuggestedTp = primaryResult.AiTpPrice;
            var (opinionDir, displayConf, hDir, cogDir, _, _, hMin, _) =
                AgentInferenceSnapHelper.BuildOpinionPublishArgs(primaryResult);
            if (isV16)
                PublishAgentOpinion(primary, opinionDir, displayConf, hDir, rl, action, hMin, arch, cogDir);
            else
                PublishAgentOpinion(primary, opinionDir, displayConf, cogDir, rl, action, architecture: arch);
        }

        foreach (var r in results)
        {
            if (string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase))
                _inferenceSnap.Set(r.Symbol, AgentInferenceSnapHelper.ToInferenceSnap(r));

            if (_positionManager.HasFilledPositionForSymbol(r.Symbol))
            {
                await ApplyAgentPositionManagementAsync(r, ct);
                await TrySendDrawPayloadAsync(r, ct);
                continue;
            }

            if (r.Skipped)
                continue;

            var working = _positionManager.GetWorkingIntent(r.Symbol);
            if (working is not null)
            {
                await ApplyAgentWorkingIntentAsync(r, working, ct);
                if (_positionManager.HasWorkingIntentForSymbol(r.Symbol))
                    continue;
            }

            var payload = r.Signal;
            if (payload is null || payload.Direction is not ("buy" or "sell"))
            {
                var reason = payload?.RejectReason ?? r.RejectReason ?? "无触发";
                if (string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase)
                    || !string.IsNullOrWhiteSpace(reason))
                {
                    var rlNote = string.IsNullOrWhiteSpace(r.RlRawAction) ? "" : $" RL原始={r.RlRawAction}";
                    Log($"未出信号 {r.Symbol} 策略={StrategyNames.LogLabel(r.ActiveStrategy)}：{reason}{rlNote}");
                }
                continue;
            }

            if (_riskGuard.BlockSymbolCooldown(_settings, payload.Symbol) is { } symBlock)
            {
                Log($"风控拦截 {payload.Symbol}：{symBlock}");
                continue;
            }

            var signal = _signalGenerator.TryGenerateFromStrategySignal(_settings, payload, r.AttributionJson);
            if (signal is null)
            {
                Log($"未通过过滤 {payload.Symbol} 策略={StrategyNames.LogLabel(payload.Strategy)}：{_signalGenerator.LastRejectReason ?? "未知"}");
                continue;
            }

            await TryEmitSignalAsync(signal, ct);
            await TrySendDrawPayloadAsync(r, ct);
        }
    }

    /// <summary>M5 驱动挂单意图：同向更新 / hold·反向撤销。</summary>
    private async Task ApplyAgentWorkingIntentAsync(
        MultiStrategyTickResult r,
        PositionManagerService.ManagedState working,
        CancellationToken ct)
    {
        var payload = r.Signal;
        var dir = payload?.Direction?.Trim().ToLowerInvariant() ?? "flat";
        var action = (r.AgentAction ?? dir).Trim().ToLowerInvariant();

        if (dir is "buy" or "sell" && dir == working.Direction)
        {
            if (payload is not null)
            {
                var updated = await _positionManager.UpdateWorkingIntentAsync(working, payload, ct);
                Log(updated
                    ? $"挂单意图更新 {r.Symbol} {dir} entry={payload.Entry:F2} sl={payload.Sl:F2} tp={payload.Tp:F2}"
                    : $"挂单意图延续 {r.Symbol} signal={working.SignalId} 目标≤{working.TargetEntry:F2}");
            }
            else
            {
                Log($"挂单意图延续 {r.Symbol} signal={working.SignalId} 目标≤{working.TargetEntry:F2}");
            }
            return;
        }

        var reason = payload?.RejectReason ?? r.RejectReason ?? action switch
        {
            "hold" => "智能体 hold",
            "flat" => "智能体 flat",
            "close" or "close_only" => "智能体 close",
            _ => $"智能体观望/改向({action}/{dir})"
        };
        await _positionManager.RevokeWorkingIntentAsync(working.SignalId, reason, ct);
    }

    /// <summary>将智能体 exit_assessment 应用到已成交持仓（不含挂单意图）。</summary>
    private async Task ApplyAgentPositionManagementAsync(MultiStrategyTickResult r, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(r.Symbol))
            return;

        var filled = _positionManager.ActiveManagedStates()
            .Where(s => string.Equals(s.Symbol, r.Symbol, StringComparison.OrdinalIgnoreCase) && s.IsFilled)
            .ToList();
        if (filled.Count == 0)
            return;

        var primaryState = filled[0];
        var trail = string.IsNullOrWhiteSpace(r.TrailMode) ? "hold" : r.TrailMode;
        var slHint = r.SuggestedTrailingSl > 0 ? r.SuggestedTrailingSl : r.AiSlPrice;
        Log($"持仓管理 {r.Symbol} signal={primaryState.SignalId} 浮盈={primaryState.LastProfitPct:F2}% " +
            $"trail={trail} sl={slHint:F2} tp={r.AiTpPrice:F2} exit={r.ExitAssessment:F2} " +
            $"{r.PositionMgmtReason ?? r.ExitReason ?? ""}");

        foreach (var state in filled)
        {
            _positionManager.ApplyAgentM5PositionManagement(
                state.SignalId,
                r.SuggestedTrailingSl > 0 ? r.SuggestedTrailingSl : (r.AiSlPrice > 0 ? r.AiSlPrice : null),
                r.AiTpPrice > 0 ? r.AiTpPrice : null,
                r.TrailMode,
                r.PositionMgmtReason ?? r.ExitReason);
        }

        if (r.ExitAssessment >= 0.65)
        {
            await _positionManager.TryAgentExitAsync(
                r.Symbol, r.ExitAssessment, r.ExitReason ?? "智能体评估", ct);
        }
    }

    // ----- RunMultiStrategySignalTickAsync -----
    private async Task RunMultiStrategySignalTickAsync(CancellationToken ct, ProductionModelGate.CheckResult prod)
    {
        var msSymbols = CollectMultiStrategySymbols();
        SyncPrimarySymbolState();
        var primary = State.PrimarySymbol;

        Log($"多策略调度开始（{string.Join(", ", msSymbols)}，主品种 {primary}）");
        if (msSymbols.Count == 0)
        {
            if (!prod.ReadySymbols.Contains(primary, StringComparer.OrdinalIgnoreCase))
                Log($"多策略等待中：主品种 {primary} 模型未就绪");
            else if (_featureCache.GetM1Count(primary) == 0)
                Log("多策略等待中：尚无 M1 数据");
            else
                Log($"多策略等待中：M5 不足（需 ≥80 根，当前 {_featureCache.GetM5Count(primary)}）");
            return;
        }

        if (_macro.IsSilenceWindow())
        {
            Log("宏观静默，信号丢弃");
            return;
        }

        var todayPnl = await _database.GetTodayClosedPnlPercentAsync(ct);
        var openCount = _positionManager.OpenManagedCount;
        if (_riskGuard.BlockNewSignal(_settings, openCount, _pendingStore.Count, todayPnl) is { } globalBlock)
        {
            Log($"风控拦截：{globalBlock}");
            return;
        }

        var m5Map = new Dictionary<string, (long, double, double, double, double, double)[]>(StringComparer.OrdinalIgnoreCase);
        foreach (var sym in msSymbols)
        {
            if (_featureCache.TryExportM5Bars(sym, out var bars))
                m5Map[sym] = bars;
        }

        if (m5Map.Count == 0)
        {
            Log("多策略跳过：无法导出 M5");
            return;
        }

        var configPath = _settings.MultiStrategy?.ConfigPath ?? "config/config_multi_strategy.json";
        IReadOnlyList<MultiStrategyTickResult> results;
        try
        {
            results = await _python.MultiStrategyTickAsync(
                m5Map, msSymbols, primary, _macro.IsSilenceWindow(), configPath,
                TimeSpan.FromSeconds(90), ct);
        }
        catch (TimeoutException)
        {
            Log("多策略 Python 推理超时");
            return;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "多策略调度失败");
            Log($"多策略失败: {ex.Message}");
            return;
        }

        var primaryResult = results.FirstOrDefault(r =>
            string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase));
        if (primaryResult is not null)
        {
            State.ActiveMarketState = primaryResult.MarketState;
            State.ActiveStrategy = primaryResult.ActiveStrategy;
            var adxPart = primaryResult.Adx is { } adx ? $" ADX={adx:F1}" : "";
            var schedPart = string.Equals(primaryResult.ActiveStrategy, "scheduler_ai", StringComparison.OrdinalIgnoreCase)
                ? " [自动调度]"
                : "";
            Log($"[多策略]{schedPart} {primary} 行情={primaryResult.MarketState} 策略={StrategyNames.LogLabel(primaryResult.ActiveStrategy)}{adxPart}");
        }

        foreach (var r in results)
        {
            if (r.Skipped)
                continue;

            var payload = r.Signal;
            if (payload is null || payload.Direction is not ("buy" or "sell"))
            {
                var reason = payload?.RejectReason ?? r.RejectReason ?? "无触发";
                if (string.Equals(r.Symbol, primary, StringComparison.OrdinalIgnoreCase)
                    || !string.IsNullOrWhiteSpace(reason))
                {
                    var rlNote = string.IsNullOrWhiteSpace(r.RlRawAction) ? "" : $" RL原始={r.RlRawAction}";
                    Log($"未出信号 {r.Symbol} 策略={StrategyNames.LogLabel(r.ActiveStrategy)}：{reason}{rlNote}");
                }
                continue;
            }

            if (_riskGuard.BlockSymbolCooldown(_settings, payload.Symbol) is { } symBlock)
            {
                Log($"风控拦截 {payload.Symbol}：{symBlock}");
                continue;
            }

            var signal = _signalGenerator.TryGenerateFromStrategySignal(_settings, payload, r.AttributionJson);
            if (signal is null)
            {
                Log($"未通过过滤 {payload.Symbol} 策略={StrategyNames.LogLabel(payload.Strategy)}：{_signalGenerator.LastRejectReason ?? "未知"}");
                continue;
            }

            await TryEmitSignalAsync(signal, ct);
        }
    }

    private async Task SignalSchedulerAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            // ===== P0-3: 检查市场是否开盘 =====
            var delay = TimeSpan.FromSeconds(300 - (DateTimeOffset.UtcNow.ToUnixTimeSeconds() % 300));
            try { await Task.Delay(delay, ct); } catch { break; }
            
            if (!IsMarketOpen())
            {
                Log("休市期间：跳过信号生成");
                continue;
            }
            // 智能体由 M5BarCompleted 驱动，UTC 定时器与 M1 到达竞态会导致多机决策 bar 不一致
            if (IsTradingAgentEnabled())
                continue;
            
            try { await RunSignalTickAsync(ct); }
            catch (Exception ex)
            {
                _logger.LogError(ex, "信号调度周期异常");
                Log($"信号调度异常: {ex.Message}");
            }
        }
    }

    private bool IsMarketOpen()
    {
        var utc = DateTime.UtcNow;
        if (utc.DayOfWeek == DayOfWeek.Saturday)
            return false;
        if (utc.DayOfWeek == DayOfWeek.Sunday && utc.Hour < 11)
            return false;
        if (utc.DayOfWeek == DayOfWeek.Friday && utc.Hour >= 21)
            return false;
        return true;
    }

    private async Task BootstrapMissingHistoryAsync(CancellationToken ct)
    {
        if (!State.Mt5Connected) return;

        await _historyBootstrapGate.WaitAsync(ct);
        try
        {
            var seqLen = _settings.Model?.SeqLen ?? 60;
            var needM5 = Math.Max(seqLen + 20, 400);
            foreach (var symbol in CollectDataPrefetchSymbols())
            {
                if (_featureCache.GetM5Count(symbol) >= needM5) continue;

                var broker = _settings.ResolveBrokerSymbol(symbol);
                var bars = _mt5.FetchM1History(symbol, broker, 5000);
                if (bars.Count == 0)
                {
                    _logger.LogDebug("MT5 API 无 {Symbol} 历史 M1（未挂指标或品种不可用）", symbol);
                    continue;
                }

                _featureCache.AppendHistoryChunk(symbol, bars, final: true);
                Log($"历史 M1 已载入 {symbol}（MT5 API）：{_featureCache.GetM1Count(symbol)} 根 → 合成 M5 {_featureCache.GetM5Count(symbol)} 根（开机即可推理）");
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "历史 M1 预热失败");
            Log($"历史 M1 预热失败: {ex.Message}");
        }
        finally
        {
            _historyBootstrapGate.Release();
            // P1-1: 标记启动阶段的历史数据就绪（即使失败也让后续按现有数据继续）
            _startupHistoryReady.TrySetResult();

            // 热重连/延迟管道：不重跑开机评估，避免 duplicate_bar/Worker 异常误清托管
            if (_startupAssessmentDone)
                Log("管道/MT5 延迟连接：不重跑开机评估，沿用现有托管与意见");
        }
    }

    private async Task RestoreRiskGuardCooldownAsync(CancellationToken ct)
    {
        try
        {
            var symbols = ModelConfigSync.ResolveSymbols(_settings);
            await _riskGuard.RestoreFromDatabaseAsync(_database, symbols, ct);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "风控冷却状态恢复失败");
        }
    }

    private async Task RunSignalTickAsync(CancellationToken ct)
    {
        if (!await _signalTickGate.WaitAsync(0, ct))
        {
            _logger.LogDebug("信号调度跳过：上一轮尚未结束");
            return;
        }

        try
        {
            if (IsTradingAgentEnabled())
            {
                await RunTradingAgentSignalTickAsync(ct);
                Log("信号调度结束");
                return;
            }

            if (!State.ModelsReady)
            {
                if (!_productionModelSkipLogged)
                {
                    Log("[inference-paused] signal tick skipped: no production models");
                    _productionModelSkipLogged = true;
                }
                return;
            }

            var prod = ProductionModelGate.Check(_settings);
            var configured = ModelConfigSync.ResolveSymbols(_settings);
            var seqLen = _settings.Model?.SeqLen ?? 60;
            var macro = _macro.GetFeatures();

            if (IsMultiStrategyEnabled())
            {
                await RunMultiStrategySignalTickAsync(ct, prod);
                Log("信号调度结束");
                return;
            }

            var symbols = ResolveInferenceSymbols(prod);
            Log($"信号调度开始（品种 {symbols.Count}，就绪模型 {string.Join(",", prod.ReadySymbols)}，图表 {State.ActiveSymbol ?? "—"}）");
            if (symbols.Count == 0) { /* ... unchanged ... */ return; }

            var skipped = configured
                .Except(symbols, StringComparer.OrdinalIgnoreCase)
                .Where(s => _featureCache.GetM1Count(s) > 0)
                .ToList();
            if (skipped.Count > 0) { /* unchanged */ }

            var todayPnl = await _database.GetTodayClosedPnlPercentAsync(ct);
            var openCount = _positionManager.OpenManagedCount;

            if (_riskGuard.BlockNewSignal(_settings, openCount, _pendingStore.Count, todayPnl) is { } globalBlock)
            {
                Log($"风控拦截：{globalBlock}");
                if (openCount > 0 && todayPnl <= -Math.Abs(_settings.RiskGuard?.MaxDailyLossPct ?? 3))
                {
                    Log("⚠ 触发账户风控上限，全部持仓平仓");
                    await _positionManager.CloseAllPositionsAsync("risk_guard", ct);
                }
                return;
            }

            foreach (var symbol in symbols)
            {
                // ... unchanged signal generation loop ...
                if (_macro.IsSilenceWindow()) { Log("宏观静默，信号丢弃"); return; }
                if (_riskGuard.BlockSymbolCooldown(_settings, symbol) is { } symBlock) { Log($"风控拦截 {symbol}：{symBlock}"); continue; }
                if (!_featureCache.TryGetSequence(symbol, seqLen, out var seq, out var hourly, out var atrPct)) { /* ... */ continue; }
                if (!_featureCache.TryGetLatestClose(symbol, out var close)) { /* ... */ continue; }

                if (seq.GetLength(1) < FeatureConstants.ModelFeatureDim)
                    seq = FeaturePad.ToModelDim(seq);
                Log($"推理开始 {symbol} M5={_featureCache.GetM5Count(symbol)} close={close:F2} atr%={atrPct:F3}");

                InferenceResult inference;
                try
                {
                    if (!_featureCache.TryExportM5Bars(symbol, out var m5Bars)) { Log($"信号调度跳过 {symbol}：无法导出 M5 缓存"); continue; }
                    var inferSw = System.Diagnostics.Stopwatch.StartNew();
                    inference = await _python.PredictAsync(symbol, seq, hourly, macro, m5Bars, TimeSpan.FromSeconds(60), ct);
                    inferSw.Stop();
                    _lastInference[symbol] = inference;
                    _inferenceSnap.Set(symbol, inference);
                    Log($"推理完成 {symbol} dir={inference.Direction} conf={inference.Confidence:F2} ({inferSw.ElapsedMilliseconds}ms)");
                }
                catch (TimeoutException)
                {
                    Log($"Python 推理超时，使用上次有效结果或跳过 {symbol}");
                    if (!_lastInference.TryGetValue(symbol, out inference!))
                        continue;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "信号调度失败 {Symbol}", symbol);
                    Log($"信号失败 {symbol}: {ex.Message}");
                    continue;
                }

                var signal = _signalGenerator.TryGenerate(_settings, symbol, inference, atrPct, close);
                if (signal is null) { Log($"未通过过滤 {symbol}：{_signalGenerator.LastRejectReason ?? "未知"}"); continue; }

                await TryEmitSignalAsync(signal, ct);
            }

            Log("信号调度结束");
        }
        finally
        {
            _signalTickGate.Release();
        }
    }

    private async Task PositionScannerAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                if (IsMarketOpen())
                    await _positionManager.ScanAsync(ct);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "持仓扫描异常");
            }
            try { await Task.Delay(1000, ct); } catch { break; }
        }
    }

    private async Task RealTimePriceMonitorAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var symbols = _positionManager.ActiveSymbols();
                foreach (var sym in symbols)
                {
                    if (!State.PipeDataConnected)
                        continue;

                    var tick = _mt5.GetTickPrice(sym);
                    if (tick is null) continue;

                    // ===== P0-2: 保留 Bid/Ask =====
                    State.LatestBid = tick.Bid;
                    State.LatestAsk = tick.Ask;
                    State.LastPriceUpdate = DateTime.UtcNow;

                    _marketSnapshot.UpdateFromTick(sym, tick.Bid, tick.Ask, tick.Time);

                    await _positionManager.OptimizeEntryPricesAsync(sym, tick, State.ActiveMarketState, ct);
                    await _positionManager.FastTrailingStopAsync(sym, tick.Bid, tick.Ask, ct);
                }
            }
            catch (OperationCanceledException) { break; }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "实时报价监控异常");
            }
            try { await Task.Delay(500, ct); } catch { break; }
        }
    }

    // ===== P0-3: 实盘持仓同步循环（3秒间隔）=====
    private async Task SyncRealPositionsLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(3000, ct); } catch { break; }
            try
            {
                if (!State.Mt5Connected) continue;
                await _positionManager.SyncRealPositionsAsync(ct);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "实盘持仓同步异常");
            }
        }
    }
    // ===== 结束 =====

    private async Task PendingExpiryLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(30_000, ct); } catch { break; }
            try { await PurgeExpiredPendingAsync(ct); } catch (Exception ex) { _logger.LogWarning(ex, "信号过期清理失败"); }
        }
    }

    private async Task DailyCleanupLoopAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(TimeSpan.FromHours(1), ct); } catch { break; }
            try
            {
                var removed = _pendingStore.RemoveStaleClosed(maxAgeDays: 1);
                if (removed.Count > 0)
                    Log($"每日清理：移除 {removed.Count} 条已完成超 24 小时的信号");
            }
            catch (Exception ex) { _logger.LogWarning(ex, "每日信号清理失败"); }
        }
    }

    private async Task PurgeExpiredPendingAsync(CancellationToken ct)
    {
        var mins = ResolveSignalExpiryMinutes();
        foreach (var sig in _pendingStore.MarkExpired(mins))
        {
            await _database.UpdateSignalStatusAsync(sig.SignalId, "expired", ct);
            await ClearSignalDrawWithRetryAsync(sig.SignalId);
            SignalStatusChanged?.Invoke(sig.SignalId, "expired", sig.CloseReason, null);
            Log($"信号过期 {sig.Symbol} {sig.SignalId}");
        }
    }

    private async Task RuntimeHeartbeatAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try { await Task.Delay(TimeSpan.FromMinutes(10), ct); }
            catch { break; }

            try
            {
                var lastBar = State.LastBarUtc?.ToLocalTime().ToString("HH:mm:ss") ?? "-";
                StartupLog.Write(
                    $"heartbeat running={State.IsRunning} mt5={State.Mt5Connected} dataPipe={State.PipeDataConnected} drawPipe={State.PipeDrawConnected} " +
                    $"models={State.ModelsReady} lastBar={lastBar}");
            }
            catch { }
        }
    }

    private bool _watchdogDrawWasConnected;

    private async Task Mt5WatchdogAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            var dataNow = _pipeServer.IsDataConnected;
            var drawNow = _pipeServer.IsDrawConnected;
            State.PipeDataConnected = dataNow;
            State.PipeDrawConnected = drawNow;
            _positionManager.SetDataPipeConnected(dataNow);
            if (dataNow || drawNow)
                State.PipeStatus = FormatPipeStatusText();

            if (drawNow && !_watchdogDrawWasConnected)
                _ = ReplayPendingDrawCommandsAsync();
            _watchdogDrawWasConnected = drawNow;

            // ===== P0-3: 检查市场开盘状态 =====
            State.IsMarketOpen = IsMarketOpen();
            if (!State.IsMarketOpen && _positionManager.OpenManagedCount > 0)
            {
                Log("休市期间：暂停持仓管理计时");
            }
            // ===== P0-3 结束 =====

            if (State.IsRunning && !State.Mt5Connected)
            {
                if (_mt5.TryReconnect())
                {
                    State.Mt5Connected = true;
                    Log("MT5 重连成功，数据流继续");
                }
                else _alerts.RaiseMt5Reconnecting();
            }
            else if (State.IsRunning && State.Mt5Connected && !_mt5.Connected)
            {
                State.Mt5Connected = false;
                Log("MT5 连接丢失");
                _alerts.RaiseMt5Disconnected();
            }

            if (State.IsRunning && State.Mt5Connected && ++_m1ApiSyncTicks >= 12)
            {
                _m1ApiSyncTicks = 0;
                try { await SyncRecentM1FromMt5ApiAsync(ct); }
                catch (Exception ex) { _logger.LogDebug(ex, "M1 API 补数跳过"); }
            }

            if (State.IsRunning)
            {
                try { await TryProcessDrawInjectFileAsync(ct); }
                catch (Exception ex) { _logger.LogDebug(ex, "inject_draw 跳过"); }
            }

            try { await Task.Delay(5000, ct); } catch { break; }
        }
    }

    private async Task SyncRecentM1FromMt5ApiAsync(CancellationToken ct)
    {
        const int recentBars = 20;
        foreach (var symbol in CollectDataPrefetchSymbols())
        {
            ct.ThrowIfCancellationRequested();
            var broker = _settings.ResolveBrokerSymbol(symbol);
            var bars = await Task.Run(() => _mt5.FetchM1History(symbol, broker, recentBars), ct);
            if (bars.Count == 0) continue;
            var m5Before = _featureCache.GetM5Count(symbol);
            foreach (var bar in bars)
                _featureCache.Ingest(bar);
            var m5After = _featureCache.GetM5Count(symbol);
            if (m5After != m5Before && bars.Count > 0)
                Log($"M1 API 补数 {symbol}：{bars.Count} 根（最新 {Mt5Time.FormatBar(bars[^1].Time)}，M5={m5After}）");
        }
    }

    private void Log(string message)
    {
        _logger.LogInformation("{Message}", message);
        LogEmitted?.Invoke(message);
    }

    private async Task CloseSignalInDbAsync(string signalId, string status, CancellationToken ct)
    {
        await _database.UpdateSignalStatusAsync(signalId, status, ct);
        EnqueueClearSignalDraw(signalId);
        SignalStatusChanged?.Invoke(signalId, status, "", null);
    }

    private void PublishAgentOpinion(
        string symbol,
        int direction,
        double confidence,
        string? horizonDir = null,
        string? rlAction = null,
        string? finalAction = null,
        double minConfidence = 0,
        string? architecture = null,
        string? cognitionDir = null)
    {
        static string DirLabel(string? dir) => dir switch
        {
            "long" => "多头",
            "short" => "空头",
            _ => "观望",
        };

        var isV16 = string.Equals(architecture, "v16", StringComparison.OrdinalIgnoreCase);
        var hLabel = DirLabel(horizonDir);
        var cLabel = DirLabel(cognitionDir ?? horizonDir);
        var rlLabel = string.IsNullOrWhiteSpace(rlAction) ? "—" : rlAction;
        var finalLabel = string.IsNullOrWhiteSpace(finalAction) ? "—" : finalAction;
        var confNote = minConfidence > 0
            ? StrategyNames.FormatHorizonConfidence(confidence, minConfidence)
            : $"{confidence:F2}";
        var text = isV16
            ? (direction == 0
                ? $"Horizon方向：{symbol} 最终=观望 | Horizon={hLabel} conf={confNote} | 认知={cLabel} | RL={rlLabel} 动作={finalLabel}"
                : $"Horizon方向：{symbol} {hLabel} conf={confNote}（RL={rlLabel} 最终={finalLabel}）")
            : (direction == 0
                ? $"智能体意见：{symbol} 最终=观望 | 认知={cLabel} conf={confidence:F2} | RL={rlLabel} 动作={finalLabel}"
                : $"智能体意见：{symbol} 认知={cLabel} conf={confidence:F2}（RL={rlLabel} 最终={finalLabel}）");
        State.AgentOpinionText = text;
        AgentOpinionUpdated?.Invoke(text);
    }

    private static string DrawInjectPath =>
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ZhuLong", "inject_draw.json");

    private async Task TryProcessDrawInjectFileAsync(CancellationToken ct)
    {
        if (!File.Exists(DrawInjectPath))
            return;
        string text;
        try
        {
            text = await File.ReadAllTextAsync(DrawInjectPath, ct).ConfigureAwait(false);
            File.Delete(DrawInjectPath);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "inject_draw 读取失败");
            return;
        }

        if (string.IsNullOrWhiteSpace(text))
            return;

        try
        {
            using var doc = JsonDocument.Parse(text);
            if (!_pipeServer.IsDrawConnected)
            {
                Log("inject_draw 跳过：绘图管道未连接");
                return;
            }
            await _pipeServer.SendDrawCommandAsync(doc.RootElement, ct).ConfigureAwait(false);
            Log("inject_draw 测试信号已发送到 MT5 图表");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "inject_draw 解析/发送失败");
        }
    }

    private async Task<bool> TryEmitSignalAsync(SignalModel signal, CancellationToken ct)
    {
        _pendingStore.Add(signal);
        await _database.SaveSignalAsync(signal, ct);
        await _positionManager.AdoptPendingNowAsync(ct);

        if (!_positionManager.IsManagingSignal(signal.SignalId))
        {
            await _database.UpdateSignalStatusAsync(signal.SignalId, "rejected", ct);
            SignalStatusChanged?.Invoke(signal.SignalId, "rejected", "未纳入托管", null);
            Log($"信号未托管已拒绝 {signal.Symbol} {signal.SignalId}");
            return false;
        }

        var managed = _positionManager.GetManagedState(signal.SignalId);
        signal.Status = managed?.IsFilled == true ? "active" : "awaiting_fill";

        await ClearOrphanDrawingsAsync(ct);
        await SyncChartDrawingsToManagedAsync(ct);

        SignalCreated?.Invoke(signal);
        SignalStatusChanged?.Invoke(signal.SignalId, signal.Status, "", null);
        Log($"信号生成 {signal.Symbol} {signal.Direction} 策略={StrategyNames.LogLabel(signal.Strategy)} conf={signal.Confidence:F2} entry={signal.EntryPrice:F2} sl={signal.StopLoss:F2} tp={signal.TakeProfit:F2}");
        Log(managed?.IsFilled == true
            ? "信号已成交，进入持仓管理（SL/TP 由智能体跟踪，平仓由 exit_assessment 决策）"
            : "挂单意图已建立（M1 穿价或 tick 到达目标价即撮合成交）");
        _riskGuard.RecordSignalEmitted(signal.Symbol);
        UpdateSyncHealth();
        if (IsTradingAgentEnabled() && _agentRuntimeReady && _python.IsReady)
            _ = NotifyAgentSignalEmittedAsync(signal.Symbol);
        return true;
    }

    private void TrackDrawnSignal(string signalId)
    {
        if (string.IsNullOrWhiteSpace(signalId)) return;
        lock (_drawnLock) { _drawnSignalIds.Add(signalId); }
    }

    private void UntrackDrawnSignal(string signalId)
    {
        if (string.IsNullOrWhiteSpace(signalId)) return;
        lock (_drawnLock) { _drawnSignalIds.Remove(signalId); }
    }

    private async Task ClearSignalDrawWithRetryAsync(string signalId, int attempts = 3)
    {
        if (string.IsNullOrWhiteSpace(signalId))
            return;
        for (var i = 0; i < attempts; i++)
        {
            try
            {
                await _pipeServer.SendDrawCommandAsync(
                    new { action = "clear_signal", signal_id = signalId }, CancellationToken.None);
                break;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "clear_signal 重试 {Attempt} {Signal}", i + 1, signalId);
            }
            if (i < attempts - 1)
                await Task.Delay(2000);
        }
        UntrackDrawnSignal(signalId);
    }

    private void UpdateSyncHealth()
    {
        var managedCount = _positionManager.OpenManagedCount;
        State.ManagedPositionCount = managedCount;
        List<string> drawn;
        lock (_drawnLock) { drawn = _drawnSignalIds.ToList(); }

        if (managedCount == 0 && drawn.Count == 0)
            State.SyncHealthText = "托管 0 笔 · 一致";
        else if (managedCount > 0 && drawn.Count > 0)
            State.SyncHealthText = $"托管 {managedCount} 笔 · 图表 {drawn.Count} · 一致";
        else if (managedCount == 0 && drawn.Count > 0)
            State.SyncHealthText = $"⚠ 无托管但图表登记 {drawn.Count} 笔";
        else
            State.SyncHealthText = $"⚠ 托管 {managedCount} 笔但图表未绘制";
    }

    /// <summary>按需清理：仅对已知 signal_id 发 clear_signal，不占用通道做周期性全清。</summary>
    private async Task ClearOrphanDrawingsAsync(CancellationToken ct = default)
    {
        var managedIds = _positionManager.ActiveManagedStates()
            .Select(s => s.SignalId)
            .ToHashSet(StringComparer.Ordinal);
        List<string> orphans;
        lock (_drawnLock)
        {
            orphans = _drawnSignalIds.Where(id => !managedIds.Contains(id)).ToList();
        }
        if (orphans.Count == 0)
            return;

        Log($"清理幽灵绘图 {orphans.Count} 条（按 signal_id）");
        foreach (var id in orphans)
        {
            await SendClearSignalDrawAsync(id, ct);
            UntrackDrawnSignal(id);
        }
    }

    private async Task SendClearAllDrawingsAsync(CancellationToken ct = default)
    {
        try
        {
            await _pipeServer.SendDrawCommandAsync(new { action = "clear_all" }, ct);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "clear_all 失败");
        }
        lock (_drawnLock) { _drawnSignalIds.Clear(); }
    }

    /// <summary>图表与托管账本强制一致：先清全部 ZL_ 对象，再仅重绘当前托管。</summary>
    private async Task SyncChartDrawingsToManagedAsync(CancellationToken ct = default)
    {
        await SendClearAllDrawingsAsync(ct);
        var managed = _positionManager.ActiveManagedStates();
        foreach (var state in managed)
            await RefreshManagedSignalDrawAsync(state);

        if (managed.Count == 0)
            Log("图表已同步：无托管信号，已清除全部烛龙标注");
        else
            Log($"图表已同步：重绘托管 {managed.Count} 条");
        UpdateSyncHealth();
    }

    private async Task SendClearSignalDrawAsync(string signalId, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(signalId))
            return;
        try
        {
            await _pipeServer.SendDrawCommandAsync(
                new { action = "clear_signal", signal_id = signalId }, ct);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "clear_signal 失败 {Signal}", signalId);
        }
    }

    private async Task SendDrawForSignalAsync(SignalModel signal, CancellationToken ct = default)
    {
        var ok = await _pipeServer.SendDrawCommandAsync(new
        {
            action = "draw_signal",
            signal_id = signal.SignalId,
            symbol = signal.Symbol,
            direction = signal.Direction,
            entry = signal.EntryPrice,
            sl = signal.StopLoss,
            tp = signal.TakeProfit,
            confidence = signal.Confidence,
            strategy = signal.Strategy,
            expiry_minutes = ResolveSignalExpiryMinutes(),
        }, ct);
        TrackDrawnSignal(signal.SignalId);
        if (!ok)
        {
            _logger.LogWarning("绘图管道未连接！信号 {SignalId} {Symbol} {Direction} entry={Entry:F2} 已排队等待",
                signal.SignalId, signal.Symbol, signal.Direction, signal.EntryPrice);
            Log($"⚠ 绘图管道断开！信号 {signal.Symbol} {signal.Direction} entry={signal.EntryPrice:F2} sl={signal.StopLoss:F2} tp={signal.TakeProfit:F2} — 已排队等待 MT5 重连");
        }
    }

	private string FormatPipeStatusText()
	{
		var dataStatus = State.PipeDataConnected ? "●" : "○";
		var drawStatus = State.PipeDrawConnected ? "●" : "○";
		var mode = State.PipeDataConnected
			? (State.PipeDrawConnected ? "管道就绪" : "推理正常·绘图待连")
			: (State.PipeDrawConnected ? "绘图已连·数据待连" : "等待连接");
		return $"数据{dataStatus} 绘图{drawStatus} {mode} | {State.PrimarySymbol} | 北京 {ChinaTime.Format(DateTimeOffset.UtcNow, "HH:mm:ss")}";
	}

    private async Task ReplayPendingDrawCommandsAsync()
    {
        if (!await _replayDrawGate.WaitAsync(0).ConfigureAwait(false))
            return;

        try
        {
            var expired = _pendingStore.MarkExpired(ResolveSignalExpiryMinutes());
            foreach (var sig in expired)
            {
                try { await _database.UpdateSignalStatusAsync(sig.SignalId, "expired", CancellationToken.None); } catch { }
                await SendClearSignalDrawAsync(sig.SignalId, CancellationToken.None);
                UntrackDrawnSignal(sig.SignalId);
                SignalStatusChanged?.Invoke(sig.SignalId, "expired", sig.CloseReason, null);
            }

            await ClearOrphanDrawingsAsync(CancellationToken.None);

            await SyncChartDrawingsToManagedAsync(CancellationToken.None);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "重放绘图命令失败");
        }
        finally
        {
            _replayDrawGate.Release();
        }
    }

    private bool IsTradingAgentEnabled() => _settings.TradingAgent?.Enabled == true;
    private bool IsMultiStrategyEnabled() => _settings.MultiStrategy?.Enabled == true;

    private string ResolveAgentConfigPath() =>
        AgentConfigSync.ResolveAgentConfigPath(_settings);

    /// <summary>价格涨跌幅 % → R 倍数（以初始 SL 距离为 1R）。</summary>
    private static double ProfitPctToR(ManagedPositionModel p)
    {
        if (p.EntryPrice > 0 && p.StopLoss > 0)
        {
            var slDistPct = Math.Abs(p.EntryPrice - p.StopLoss) / p.EntryPrice * 100.0;
            if (slDistPct > 1e-6)
                return p.ProfitPct / slDistPct;
        }
        return p.ProfitPct / 0.5;
    }

    private Dictionary<string, object> BuildAgentTickPayload(IReadOnlyList<string> symbols)
    {
        var ticks = new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        foreach (var sym in symbols)
        {
            var broker = _settings.ResolveBrokerSymbol(sym);
            var tick = _mt5.GetTickPrice(broker);
            if (tick is null || tick.Bid <= 0 || tick.Ask <= 0) continue;
            ticks[sym] = new Dictionary<string, object>
            {
                ["bid"] = tick.Bid,
                ["ask"] = tick.Ask,
            };
        }
        return ticks;
    }

    private List<Dictionary<string, object>> BuildAgentPositionPayload()
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        return _positionManager.ActiveManagedStates()
            .Select(s =>
            {
                var dirSign = s.Direction switch
                {
                    "buy" => 1.0,
                    "sell" => -1.0,
                    _ => 0.0,
                };
                return new Dictionary<string, object>
                {
                    ["symbol"] = s.Symbol,
                    ["direction"] = s.Direction,
                    ["direction_sign"] = dirSign,
                    ["entry"] = s.IsFilled ? s.EntryPrice : s.TargetEntry,
                    ["sl"] = s.TrailingActivated ? s.TrailingSl : s.StopLoss,
                    ["tp"] = s.TakeProfit,
                    ["profit_pct"] = s.IsFilled ? s.LastProfitPct : 0.0,
                    ["peak_profit_pct"] = s.IsFilled ? s.PeakProfitPct : 0.0,
                    ["hold_seconds"] = s.IsFilled
                        ? Math.Max(0, now - s.FilledAt)
                        : Math.Max(0, now - s.OpenTime),
                    ["is_filled"] = s.IsFilled,
                    ["signal_id"] = s.SignalId,
                    ["time_expired"] = s.TimeExpired,
                    ["max_hold_minutes"] = _settings.PositionManagement?.MaxHoldMinutes ?? 240,
                    ["min_hold_seconds_before_trailing"] =
                        _settings.PositionManagement?.MinHoldSecondsBeforeTrailing ?? 60,
                };
            })
            .ToList();
    }

    private int ResolveSignalExpiryMinutes() =>
        IsTradingAgentEnabled()
            ? AgentConfigSync.ResolveAgentSignalExpiryMinutes(_settings)
            : _settings.SignalFilters?.SignalExpiryMinutes ?? 240;

    private async Task NotifyAgentClosedTradeAsync(string symbol, double pnlR)
    {
        try
        {
            await _python.AgentRecordClosedTradeAsync(
                symbol, pnlR, ResolveAgentConfigPath(), _cts?.Token ?? CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "智能体平仓反馈失败 {Symbol}", symbol);
        }
    }

    private async Task NotifyAgentSignalEmittedAsync(string symbol)
    {
        try
        {
            await _python.AgentRecordSignalEmittedAsync(
                symbol, ResolveAgentConfigPath(), _cts?.Token ?? CancellationToken.None)
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "智能体日交易计数反馈失败 {Symbol}", symbol);
        }
    }

    private async Task TrySendDrawPayloadAsync(MultiStrategyTickResult r, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(r.DrawPayloadJson) || !_pipeServer.IsDrawConnected)
            return;
        try
        {
            using var doc = JsonDocument.Parse(r.DrawPayloadJson);
            await _pipeServer.SendDrawCommandAsync(doc.RootElement, ct).ConfigureAwait(false);
            if (doc.RootElement.TryGetProperty("signal_id", out var sidEl))
            {
                var sid = sidEl.GetString();
                if (!string.IsNullOrWhiteSpace(sid))
                    TrackDrawnSignal(sid);
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "draw_payload 发送失败 {Symbol}", r.Symbol);
        }
    }

    private void SyncPrimarySymbolState()
    {
        State.PrimarySymbol = _settings.Model?.PrimarySymbol?.Trim().ToUpperInvariant() ?? "XAUUSD";
    }

    private void PersistSettings()
    {
        try { _settings.Save(AppPaths.ConfigPath); }
        catch { }
    }

    private string FormatInferenceModeHint(ProductionModelGate.CheckResult prod)
    {
        var mode = _settings.Model?.InferAllReadySymbols == true ? "全品种" : "单品种";
        var primary = State.PrimarySymbol;
        return $"推理模式={mode} 主品种={primary} 就绪模型={string.Join(",", prod.ReadySymbols)}";
    }

    // ===== UI 绑定方法（SettingsPanel / MainViewModel 调用） =====
    public bool ConnectMt5()
    {
        if (!_mt5.Connected)
            _mt5.TryReconnect();
        State.Mt5Connected = _mt5.Connected;
        return State.Mt5Connected;
    }

    public bool GetMultiStrategyEnabled() =>
        _settings.MultiStrategy?.Enabled == true;

    public void SetMultiStrategyEnabled(bool value)
    {
        if (_settings.MultiStrategy != null)
            _settings.MultiStrategy.Enabled = value;
        if (value && _settings.TradingAgent?.Enabled == true)
        {
            _settings.TradingAgent.Enabled = false;
            Log("已自动关闭智能体模式（与多策略互斥）");
        }
        PersistSettings();
        Log($"多策略模式 → {(value ? "开启" : "关闭")}");
    }

    public bool GetTradingAgentEnabled() =>
        _settings.TradingAgent?.Enabled == true;

    public void SetTradingAgentEnabled(bool value)
    {
        if (_settings.TradingAgent != null)
            _settings.TradingAgent.Enabled = value;
        if (value && _settings.MultiStrategy?.Enabled == true)
        {
            _settings.MultiStrategy.Enabled = false;
            Log("已自动关闭多策略模式（与智能体互斥）");
        }
        PersistSettings();
        EnsureAgentConfigAligned();
        Log($"RL 智能体模式 → {(value ? "开启" : "关闭")}");
    }

    private void EnsureAgentConfigAligned()
    {
        var (_, changed) = AgentConfigSync.AlignWithAppSettings(_settings, _logger);
        if (changed && _settings.TradingAgent?.Enabled == true)
            Log("已自动同步 config_agent.json（智能体已开启）");
    }

    public void SetInferAllReadySymbols(bool value)
    {
        if (_settings.Model != null)
            _settings.Model.InferAllReadySymbols = value;
        PersistSettings();
        Log($"全品种推理 → {(value ? "开启" : "关闭")}");
    }
    // ===== 结束 =====

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        await _macro.DisposeAsync();
        await _pipeServer.DisposeAsync();
    }
}
