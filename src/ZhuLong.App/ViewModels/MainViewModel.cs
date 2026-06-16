using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Dispatching;
using Windows.ApplicationModel.DataTransfer;
using ZhuLong.App.Services;
using ZhuLong.App.Services.Membership;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;

namespace ZhuLong.App.ViewModels;

public sealed partial class MainViewModel : ObservableObject
{
    private const int MaxLogLines = 500;
    private readonly ZhuLongRuntimeService _runtime;
    private readonly AlertService _alerts;
    private readonly DispatcherQueueTimer? _healthTimer;
    private DispatcherQueue? _uiDispatcher;

    public ObservableCollection<SignalModel> ActiveSignals { get; } = new();
    public ObservableCollection<SignalModel> ClosedSignals { get; } = new();
    public ObservableCollection<ManagedPositionModel> Positions { get; } = new();
    public ObservableCollection<string> Logs { get; } = new();

    [ObservableProperty] private string _statusText = "未连接";
    [ObservableProperty] private bool _isRunning;
    [ObservableProperty] private bool _mt5Connected;
    [ObservableProperty] private string _licenseSummary = "授权检查中…";
    [ObservableProperty] private bool _modelsReady;
    [ObservableProperty] private string _mt5HealthText = "○ 未就绪";
    [ObservableProperty] private string _modelsHealthText = "○ 未就绪";
    [ObservableProperty] private string _barHealthText = "—";
    [ObservableProperty] private string _runningHealthText = "○ 未就绪";
    [ObservableProperty] private string _pipeStatus = "未启动";
    [ObservableProperty] private string _lastBarText = "—";
    [ObservableProperty] private string _macroBannerText = "宏观：载入中…";
    [ObservableProperty] private string _selectedPanel = "signals";
    [ObservableProperty] private string _strategyHealthText = "—";
    [ObservableProperty] private string _agentOpinionText = "智能体意见：—";
    [ObservableProperty] private string _syncHealthText = "托管 —";
    [ObservableProperty] private string? _modelDialogMessage;

    public string AppVersionLine => AppMetadata.FormatVersionLine();

    public MainViewModel(ZhuLongRuntimeService runtime, AlertService alerts)
    {
        _runtime = runtime;
        _alerts = alerts;
        _uiDispatcher = DispatcherQueue.GetForCurrentThread();
        _runtime.LogEmitted += msg => AppDispatch(() => PrependLog(msg));
        _alerts.AlertRaised += msg => AppDispatch(() => PrependLog(msg));
        _runtime.ModelsMissing += msg => AppDispatch(() => ModelDialogMessage = msg);
        _runtime.SignalCreated += sig => AppDispatch(() => UpsertActiveSignal(sig));
        _runtime.AgentOpinionUpdated += text => AppDispatch(() => AgentOpinionText = text);
        _runtime.SignalStatusChanged += (signalId, status, reason, pnlPercent) =>
            AppDispatch(() => ApplySignalStatusChange(signalId, status, reason, pnlPercent));
        _runtime.SignalsHydrated += () => AppDispatch(() => _ = ReloadSignalsFromDbAsync());
        _runtime.PositionUpdated += pos => AppDispatch(() => UpsertPosition(pos));

        MembershipHost.Instance.Refresh();
        LicenseSummary = MembershipHost.Instance.TierDisplayName;

        var dq = _uiDispatcher ?? DispatcherQueue.GetForCurrentThread();
        if (dq is not null)
        {
            _uiDispatcher ??= dq;
            _healthTimer = dq.CreateTimer();
            _healthTimer.Interval = TimeSpan.FromSeconds(1);
            _healthTimer.Tick += (_, _) => RefreshHealth();
            _healthTimer.Start();
        }
    }

    [RelayCommand]
    public async Task ConnectAsync()
    {
        try
        {
            await _runtime.InitializeAsync();
            Mt5Connected = _runtime.ConnectMt5();
            ModelsReady = _runtime.State.ModelsReady;
            Mt5HealthText = Mt5Connected ? "● 正常" : "○ 未就绪";

            if (Mt5Connected && !IsRunning)
            {
                _runtime.Start();
                IsRunning = true;
                RunningHealthText = "● 正常";
                StatusText = "MT5 已连接，管道已启动（请在图表加载 ZhuLongIndicator）";
            }
            else
            {
                StatusText = Mt5Connected
                    ? "MT5 已连接"
                    : "MT5 连接失败（请确认 MT5 已打开并登录；宏观/模型问题见日志）";
            }

            RefreshHealth();
        }
        catch (Exception ex)
        {
            StatusText = "初始化失败";
            AppDispatch(() => PrependLog(ex.Message));
        }
    }

    [RelayCommand]
    public void Start()
    {
        _runtime.Start();
        IsRunning = true;
        RunningHealthText = "● 正常";
        StatusText = "运行中";
        RefreshHealth();
    }

    [RelayCommand]
    public async Task StopAsync()
    {
        await _runtime.StopAsync();
        IsRunning = false;
        Mt5Connected = false;
        RunningHealthText = "○ 未就绪";
        Mt5HealthText = "○ 未就绪";
        StatusText = "已停止";
        RefreshHealth();
    }

    [RelayCommand]
    public async Task RefreshSignalsAsync() => await ReloadSignalsFromDbAsync();

    private async Task ReloadSignalsFromDbAsync()
    {
        try
        {
            var active = await _runtime.GetRecentSignalsAsync();
            var closed = await _runtime.GetRecentClosedSignalsAsync();
            AppDispatch(() =>
            {
                ActiveSignals.Clear();
                foreach (var s in active) ActiveSignals.Add(s);
                ClosedSignals.Clear();
                foreach (var s in closed) ClosedSignals.Add(s);
            });
        }
        catch (Exception ex)
        {
            AppDispatch(() => PrependLog("刷新信号失败: " + ex.Message));
        }
    }

    [RelayCommand]
    public void RefreshPositions()
    {
        Positions.Clear();
        foreach (var p in _runtime.GetPositionsForDisplay()) Positions.Add(p);
    }

    [RelayCommand]
    public void CopyComment(SignalModel? signal)
    {
        if (signal is null) return;
        var data = new DataPackage();
        data.SetText(signal.CommentHint);
        Clipboard.SetContent(data);
        AppDispatch(() => PrependLog("已复制 Comment: " + signal.CommentHint));
    }

    public AppSettings Settings => _runtime.Settings;

    partial void OnSelectedPanelChanged(string value)
    {
        if (value == "positions") RefreshPositionsCommand.Execute(null);
    }

    partial void OnIsRunningChanged(bool value) => RefreshHealth();

    private void RefreshHealth()
    {
        try
        {
            var st = _runtime.State;
            Mt5Connected = st.Mt5Connected;
            ModelsReady = st.ModelsReady;
            Mt5HealthText = st.Mt5Connected ? "● 正常" : "○ 未就绪";

            var prod = ProductionModelGate.Check(_runtime.Settings);
            if (_runtime.GetInferAllReadySymbols())
            {
                ModelsHealthText = prod.ReadySymbols.Count > 0
                    ? $"● 并行 {string.Join("/", prod.ReadySymbols)}"
                    : "○ 未就绪";
            }
            else
            {
                var primary = st.PrimarySymbol;
                var ready = prod.ReadySymbols.Contains(primary, StringComparer.OrdinalIgnoreCase);
                ModelsHealthText = ready ? $"● {primary}" : $"○ {primary} 未就绪";
            }

            var multiOn = _runtime.Settings.MultiStrategy?.Enabled != false;
            var agentOn = _runtime.GetTradingAgentEnabled();
            if (agentOn && st.IsRunning)
            {
                if (!string.IsNullOrEmpty(st.ActiveStrategy))
                    StrategyHealthText = $"{StrategyNames.DisplayMarketState(st.ActiveMarketState)} · RL智能体";
                else
                    StrategyHealthText = "● RL 智能体";
            }
            else if (multiOn && st.IsRunning && !string.IsNullOrEmpty(st.ActiveStrategy))
            {
                StrategyHealthText = $"{StrategyNames.DisplayMarketState(st.ActiveMarketState)} · {StrategyNames.Display(st.ActiveStrategy)}";
            }
            else if (multiOn && st.IsRunning)
            {
                StrategyHealthText = "● 多策略";
            }
            else if (multiOn)
            {
                StrategyHealthText = "多策略（未运行）";
            }
            else
            {
                StrategyHealthText = "单模型 V14";
            }

            RunningHealthText = st.IsRunning ? "● 正常" : "○ 未就绪";
            SyncHealthText = st.SyncHealthText;
            PipeStatus = !st.PipeDataConnected && !st.PipeDrawConnected
                ? "○ 未连接"
                : (string.IsNullOrEmpty(st.ActiveSymbol) ? st.PipeStatus : $"{st.ActiveSymbol} · {st.PipeStatus}");
            IsRunning = st.IsRunning;
            LastBarText = st.LatestM1BarTime.HasValue
                ? Mt5Time.FormatBar(st.LatestM1BarTime.Value, "HH:mm:ss")
                : (st.LastBarUtc?.ToLocalTime().ToString("HH:mm:ss") ?? "—");
            BarHealthText = EvaluateBarHealth(st.LatestM1BarTime);
            MembershipHost.Instance.Refresh();
            LicenseSummary = MembershipHost.Instance.TierDisplayName;
            UpdateMacroBanner();
        }
        catch (Exception ex)
        {
            AppDispatch(() => PrependLog("状态刷新异常: " + ex.Message));
        }
    }

    private void UpdateMacroBanner()
    {
        var evt = _runtime.GetNextHighImpactEvent();
        if (evt is null)
        {
            MacroBannerText = "宏观：未来暂无已载入的高影响事件";
            return;
        }

        var until = evt.EventTime - ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        if (until < TimeSpan.Zero)
            until = TimeSpan.Zero;

        MacroBannerText =
            $"宏观 · {evt.EventName}（{evt.Currency}）｜{evt.EventTime:yyyy-MM-dd HH:mm}｜倒计时 {FormatMacroCountdown(until)}";
    }

    private static string EvaluateBarHealth(DateTime? latestBarBeijing)
    {
        if (latestBarBeijing is null) return "—";
        var age = ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime - latestBarBeijing.Value;
        return age.TotalMinutes > 3 ? "⚠ 滞后" : "● 正常";
    }

    private static string FormatMacroCountdown(TimeSpan t)
    {
        if (t.TotalDays >= 1)
            return $"{(int)t.TotalDays}天{t.Hours:D2}时{t.Minutes:D2}分{t.Seconds:D2}秒";
        if (t.TotalHours >= 1)
            return $"{(int)t.TotalHours}时{t.Minutes:D2}分{t.Seconds:D2}秒";
        return $"{t.Minutes:D2}分{t.Seconds:D2}秒";
    }

    private static bool IsLiveSignalStatus(string status) =>
        status is "pending" or "active" or "awaiting_fill";

    private static bool IsRecordedCloseStatus(string status) => status switch
    {
        "stop_loss" or "take_profit" or "trailing_stop" or "trailing"
            or "profit_drawdown" or "time_stop" or "model_exit" or "agent_exit"
            or "external_close" => true,
        _ => false,
    };

    private void UpsertActiveSignal(SignalModel sig)
    {
        for (var i = 0; i < ActiveSignals.Count; i++)
        {
            if (ActiveSignals[i].SignalId == sig.SignalId)
            {
                ActiveSignals[i] = sig;
                return;
            }
        }
        ActiveSignals.Insert(0, sig);
    }

    private void ApplySignalStatusChange(string signalId, string status, string reason, double? pnlPercent)
    {
        SignalModel? found = null;
        for (var i = 0; i < ActiveSignals.Count; i++)
        {
            if (ActiveSignals[i].SignalId != signalId) continue;
            found = ActiveSignals[i];
            ActiveSignals.RemoveAt(i);
            break;
        }

        if (found is null && IsLiveSignalStatus(status))
        {
            found = new SignalModel { SignalId = signalId, Status = status, CloseReason = reason };
        }

        if (found is null) return;

        found.Status = status;
        found.CloseReason = reason;
        if (pnlPercent.HasValue)
            found.PnlPercent = pnlPercent;
        if (pnlPercent.HasValue || IsRecordedCloseStatus(status))
            found.CloseTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        if (IsLiveSignalStatus(status))
            ActiveSignals.Insert(0, found);
        else
        {
            ClosedSignals.Insert(0, found);
            while (ClosedSignals.Count > 20) ClosedSignals.RemoveAt(ClosedSignals.Count - 1);
        }
    }

    private void UpsertPosition(ManagedPositionModel pos)
    {
        for (var i = 0; i < Positions.Count; i++)
        {
            if (Positions[i].Ticket == pos.Ticket)
            {
                Positions[i] = pos;
                return;
            }
        }
        Positions.Insert(0, pos);
    }

    public void BindUiDispatcher(DispatcherQueue dispatcher) => _uiDispatcher = dispatcher;

    private void PrependLog(string msg)
    {
        Logs.Insert(0, msg);
        while (Logs.Count > MaxLogLines)
            Logs.RemoveAt(Logs.Count - 1);
    }

    private void AppDispatch(Action action)
    {
        if (_uiDispatcher is null)
            return;

        if (_uiDispatcher.HasThreadAccess)
            action();
        else
            _ = _uiDispatcher.TryEnqueue(() => action());
    }
}
