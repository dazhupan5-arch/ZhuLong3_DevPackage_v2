using Microsoft.Extensions.Logging;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;
using ZhuLong.Core.Services;

namespace ZhuLong.App.Services;

/// <summary>信号级持仓托管（G5/G8）— 直接跟踪 pending 信号，不依赖 MT5 真实订单。</summary>
public sealed class PositionManagerService
{
    private readonly ILogger<PositionManagerService> _logger;
    private readonly Mt5ApiWrapper _mt5;
    private readonly DatabaseService _db;
    private readonly PendingSignalStore _pendingStore;
    private readonly InferenceSnapshotStore _inference;
    private readonly MarketSnapshotStore _marketSnapshot;
    private readonly FeatureCacheService _featureCache;
    private readonly Dictionary<long, ManagedState> _managed = new();
    private readonly object _managedLock = new();   // P0-3: 线程锁保护 _managed
    private AppSettings _settings = new();
    private volatile bool _dataPipeConnected;

    /// <summary>MT5 数据管道连接状态；断连时禁止机械 SL/TP，避免陈旧 tick 误触发。</summary>
    public void SetDataPipeConnected(bool connected) => _dataPipeConnected = connected;

    public PositionManagerService(
        ILogger<PositionManagerService> logger,
        Mt5ApiWrapper mt5,
        DatabaseService db,
        PendingSignalStore pendingStore,
        InferenceSnapshotStore inference,
        MarketSnapshotStore marketSnapshot,
        FeatureCacheService featureCache,
        AlertService alerts)
    {
        _logger = logger;
        _mt5 = mt5;
        _db = db;
        _pendingStore = pendingStore;
        _inference = inference;
        _marketSnapshot = marketSnapshot;
        _featureCache = featureCache;
        _ = alerts;
    }

    public event Action<ManagedPositionModel>? PositionUpdated;
    public event Action<ManagedPositionModel>? PositionClosed;
    public event Action<string>? SignalDrawClearRequested;
    public event Action<ManagedState>? ChartRefreshRequested;
    public event Action<string>? LogEmitted;
    /// <summary>托管状态变迁（awaiting_fill / active），供面板与账本同步。</summary>
    public event Action<string, string>? ManagedStatusChanged;

    // P2-2: AI动态调整持仓事件
    public event Action<ManagedState, double?, double?, string>? AiSlTpUpdated;

    public int OpenManagedCount
    {
        get { lock (_managedLock) return _managed.Count; }
    }

    public bool IsManagingSignal(string signalId)
    {
        lock (_managedLock)
            return _managed.Values.Any(m => m.SignalId == signalId);
    }

    public bool HasManagedForSymbol(string symbol) =>
        HasWorkingIntentForSymbol(symbol) || HasFilledPositionForSymbol(symbol);

    public bool HasWorkingIntentForSymbol(string symbol)
    {
        lock (_managedLock)
            return _managed.Values.Any(m =>
                string.Equals(m.Symbol, symbol, StringComparison.OrdinalIgnoreCase) && !m.IsFilled);
    }

    public bool HasFilledPositionForSymbol(string symbol)
    {
        lock (_managedLock)
            return _managed.Values.Any(m =>
                string.Equals(m.Symbol, symbol, StringComparison.OrdinalIgnoreCase) && m.IsFilled);
    }

    public ManagedState? GetWorkingIntent(string symbol)
    {
        lock (_managedLock)
            return _managed.Values.FirstOrDefault(m =>
                string.Equals(m.Symbol, symbol, StringComparison.OrdinalIgnoreCase) && !m.IsFilled);
    }

    public ManagedState? GetManagedState(string signalId)
    {
        lock (_managedLock)
            return _managed.Values.FirstOrDefault(m => m.SignalId == signalId);
    }

    public IReadOnlyList<ManagedPositionModel> Snapshot()
    {
        lock (_managedLock)
            return _managed.Values.Select(m => m.ToModel(m.LastProfitPct, m.TrailingStateText)).ToList();
    }

    public void UpdateSettings(AppSettings settings) => _settings = settings;

    public async Task ScanAsync(CancellationToken ct)
    {
        await AdoptPendingSignalsAsync(ct);
        await UpdateManagedAsync(ct);
    }

    /// <summary>信号入队后立即尝试纳入虚拟托管（避免面板长时间 pending）。</summary>
    public Task AdoptPendingNowAsync(CancellationToken ct) => AdoptPendingSignalsAsync(ct);

    private async Task AdoptPendingSignalsAsync(CancellationToken ct)
    {
        var pending = _pendingStore.Snapshot()
            .Where(s => s.Status == "pending" && s.Direction is "buy" or "sell")
            .ToList();

        foreach (var sig in pending)
        {
            // ===== P2-1: 单信号约束 =====
            // 同一品种只能有一个活跃托管信号
            bool hasActive = false;
            lock (_managedLock)
            {
                hasActive = _managed.Values.Any(m =>
                    string.Equals(m.Symbol, sig.Symbol, StringComparison.OrdinalIgnoreCase));
            }
            if (hasActive)
            {
                _pendingStore.Remove(sig.SignalId);
                await _db.UpdateSignalStatusAsync(sig.SignalId, "rejected", ct);
                SignalDrawClearRequested?.Invoke(sig.SignalId);
                ManagedStatusChanged?.Invoke(sig.SignalId, "rejected");
                LogEmitted?.Invoke($"单信号约束：{sig.Symbol} 已有活跃持仓，拒绝信号 {sig.SignalId}");
                continue;
            }
            // ===== 结束 =====

            if (IsManagingSignal(sig.SignalId))
                continue;

            var brokerSym = _settings.ResolveBrokerSymbol(sig.Symbol);
            var ticket = VirtualTicket(sig.SignalId);
            var realTicket = _mt5.FindRealPositionTicket(sig.SignalId);
            MergeTickIntoSnapshot(brokerSym);

            var state = new ManagedState
            {
                Ticket = ticket,
                RealTicket = realTicket,
                SignalId = sig.SignalId,
                Symbol = sig.Symbol,
                Direction = sig.Direction,
                TargetEntry = sig.EntryPrice,
                EntryPrice = sig.EntryPrice,
                StopLoss = sig.StopLoss,
                TakeProfit = sig.TakeProfit,
                OpenTime = sig.CreatedAt > 0 ? sig.CreatedAt : DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                Volume = 1.0,
                BrokerSymbol = brokerSym,
                IsFilled = false,
            };

            var dbStatus = "awaiting_fill";
            if (TryMatchFill(state, out var fillPrice, out var fillSource))
            {
                state.IsFilled = true;
                state.FilledAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                state.EntryPrice = fillPrice;
                state.BestPrice = fillPrice;
                if (_featureCache.TryGetCurrentAtr(state.Symbol, out var atr))
                    state.AtrAtFill = atr;
                else if (sig.StopLoss > 0)
                    state.AtrAtFill = Math.Abs(fillPrice - sig.StopLoss) / 1.2;
                state.TrailingStateText = "已成交 · 移动止损未激活";
                dbStatus = "active";
            }
            else
            {
                state.TrailingStateText = BuildWorkingIntentText(state);
            }

            lock (_managedLock)
            {
                _managed[ticket] = state;
            }

            _pendingStore.Remove(sig.SignalId);
            await _db.UpdateSignalStatusAsync(sig.SignalId, dbStatus, ct);
            _logger.LogInformation(
                "信号托管开始 signal={Signal} {Symbol} {Dir} target={Target:F2} filled={Filled}",
                sig.SignalId, sig.Symbol, sig.Direction, sig.EntryPrice, state.IsFilled);
            LogEmitted?.Invoke(state.IsFilled
                ? $"限价成交({fillSource}) {sig.Symbol} {sig.Direction} @ {state.EntryPrice:F2} sl={sig.StopLoss:F2} tp={sig.TakeProfit:F2}"
                : $"挂单意图 {sig.Symbol} {sig.Direction} 目标≤{sig.EntryPrice:F2} sl={sig.StopLoss:F2} tp={sig.TakeProfit:F2}");
            ManagedStatusChanged?.Invoke(sig.SignalId, dbStatus);
            PositionUpdated?.Invoke(state.ToModel(state.IsFilled ? 0 : 0, state.TrailingStateText));
        }
    }

    private static long VirtualTicket(string signalId)
    {
        var hash = Math.Abs(signalId.GetHashCode(StringComparison.Ordinal));
        return hash == 0 ? -1 : -hash;
    }

    private long GetFillMaxWaitSeconds()
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        if (pm.EntryFillMaxWaitSeconds > 0)
            return pm.EntryFillMaxWaitSeconds;
        return (_settings.SignalFilters?.SignalExpiryMinutes ?? 240) * 60L;
    }

    private void MergeTickIntoSnapshot(string brokerSymbol)
    {
        if (!_mt5.Connected)
            return;
        var tick = _mt5.GetTickPrice(brokerSymbol);
        if (tick is null)
            return;
        if (!IsTickFresh(tick))
            return;
        var snap = _marketSnapshot.Get(brokerSymbol);
        if (snap?.BarClose > 0 && !IsPriceNearReference(tick.Bid, tick.Ask, snap.BarClose))
            return;
        _marketSnapshot.UpdateFromTick(brokerSymbol, tick.Bid, tick.Ask, tick.Time);
    }

    private static bool IsTickFresh(Mt5Tick tick)
    {
        if (tick.Time <= 0)
            return true;
        var ageSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - tick.Time;
        return ageSec <= 120;
    }

    /// <summary>成交价/上次有效价偏离超过阈值则视为脏 tick，禁止机械出场。</summary>
    private static bool IsPriceNearReference(double bid, double ask, double reference, double maxDevPct = 0.015)
    {
        if (reference <= 0)
            return true;
        if (bid > 0 && Math.Abs(bid - reference) / reference > maxDevPct)
            return false;
        if (ask > 0 && Math.Abs(ask - reference) / reference > maxDevPct)
            return false;
        return true;
    }

    private bool CanUseMechanicalExitPrices(ManagedState state, double bid, double ask, out string? blockReason)
    {
        blockReason = null;
        if (!_dataPipeConnected)
        {
            blockReason = "data_pipe_disconnected";
            return false;
        }

        if (bid <= 0 && ask <= 0)
        {
            blockReason = "no_tick";
            return false;
        }

        var reference = state.LastPrice > 0 ? state.LastPrice : state.EntryPrice;
        if (reference <= 0)
            return true;

        if (!IsPriceNearReference(bid, ask, reference))
        {
            blockReason = $"tick_deviation ref={reference:F2} bid={bid:F2} ask={ask:F2}";
            _logger.LogWarning(
                "跳过机械出场：{Symbol} signal={SignalId} {Reason}",
                state.Symbol, state.SignalId, blockReason);
            return false;
        }

        return true;
    }

    private MarketSnapshotStore.SymbolMarketSnapshot? GetSnapshotForState(ManagedState state) =>
        _marketSnapshot.Get(state.BrokerSymbol) ?? _marketSnapshot.Get(state.Symbol);

    private bool TryMatchFill(ManagedState state, out double fillPrice, out string source)
    {
        var snap = GetSnapshotForState(state);
        return IntentFillMatcher.TryMatchFill(state.Direction, state.TargetEntry, snap, out fillPrice, out source);
    }

    private static string BuildWorkingIntentText(ManagedState state)
    {
        if (state.Direction == "buy")
            return $"挂单意图 买入≤{state.TargetEntry:F2}";
        return $"挂单意图 卖出≥{state.TargetEntry:F2}";
    }

    /// <summary>M1 管道穿价检测（与 tick 轮询共用 MarketSnapshot）。</summary>
    public async Task ProcessBarPenetrationAsync(M1Bar bar, CancellationToken ct)
    {
        _marketSnapshot.UpdateFromBar(bar);

        List<KeyValuePair<long, ManagedState>> awaiting;
        lock (_managedLock)
        {
            awaiting = _managed
                .Where(kv => !kv.Value.IsFilled
                    && string.Equals(kv.Value.Symbol, bar.Symbol, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        foreach (var kv in awaiting)
        {
            if (!TryMatchFill(kv.Value, out var fillPrice, out var source))
                continue;

            await ConfirmFillAsync(kv.Value, fillPrice, ct, source);
        }
    }

    /// <summary>tick 级挂单撮合（不依赖移动止损开关）。</summary>
    public async Task ProcessWorkingIntentFillAsync(string brokerSymbol, double bid, double ask, CancellationToken ct)
    {
        if (bid > 0 || ask > 0)
            _marketSnapshot.UpdateFromTick(brokerSymbol, bid, ask);

        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock) { snapshot = _managed.ToList(); }

        foreach (var kv in snapshot)
        {
            var state = kv.Value;
            if (state.IsFilled)
                continue;
            if (!string.Equals(state.BrokerSymbol, brokerSymbol, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(state.Symbol, brokerSymbol, StringComparison.OrdinalIgnoreCase))
                continue;

            if (!await ProcessAwaitingFillAsync(kv.Key, state, ct))
                continue;
        }
    }

    public async Task<bool> UpdateWorkingIntentAsync(
        ManagedState state,
        MultiStrategySignalPayload payload,
        CancellationToken ct)
    {
        if (state.IsFilled)
            return false;

        var changed = false;
        if (payload.Entry > 0 && Math.Abs(payload.Entry - state.TargetEntry) > 0.01)
        {
            state.TargetEntry = payload.Entry;
            state.EntryPrice = payload.Entry;
            changed = true;
        }

        if (payload.Sl > 0 && Math.Abs(payload.Sl - state.StopLoss) > 0.01)
        {
            state.StopLoss = payload.Sl;
            changed = true;
        }

        if (payload.Tp > 0 && Math.Abs(payload.Tp - state.TakeProfit) > 0.01)
        {
            state.TakeProfit = payload.Tp;
            changed = true;
        }

        if (!changed)
            return false;

        await _db.UpdateSignalPlanAsync(state.SignalId, state.TargetEntry, state.StopLoss, state.TakeProfit, ct);
        state.TrailingStateText = BuildWorkingIntentText(state);
        ChartRefreshRequested?.Invoke(state);
        ManagedStatusChanged?.Invoke(state.SignalId, "awaiting_fill");
        PositionUpdated?.Invoke(state.ToModel(0, state.TrailingStateText));
        return true;
    }

    public Task RevokeWorkingIntentAsync(string signalId, string reason, CancellationToken ct)
    {
        long ticketKey = 0;
        ManagedState? state = null;
        lock (_managedLock)
        {
            foreach (var kv in _managed)
            {
                if (kv.Value.SignalId != signalId || kv.Value.IsFilled)
                    continue;
                ticketKey = kv.Key;
                state = kv.Value;
                break;
            }
        }

        if (state is null)
            return Task.CompletedTask;

        return CancelAwaitingFillAsync(ticketKey, state, reason, ct);
    }

    /// <returns>true 已成交可继续 SL/TP 管理；false 仍在等待成交。</returns>
    private async Task<bool> ProcessAwaitingFillAsync(long ticketKey, ManagedState state, CancellationToken ct)
    {
        if (state.IsFilled)
            return true;

        MergeTickIntoSnapshot(state.BrokerSymbol);
        var snap = GetSnapshotForState(state);

        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        if (now - state.OpenTime >= GetFillMaxWaitSeconds())
        {
            var closePrice = state.TargetEntry;
            if (snap is not null)
            {
                closePrice = state.Direction == "buy"
                    ? (snap.Bid > 0 ? snap.Bid : state.TargetEntry)
                    : (snap.Ask > 0 ? snap.Ask : state.TargetEntry);
            }
            await CloseVirtualAsync(ticketKey, state, closePrice, "entry_timeout", ct);
            LogEmitted?.Invoke($"挂单超时：{state.Symbol} signal={state.SignalId} 目标{state.TargetEntry:F2}");
            return false;
        }

        if (TryMatchFill(state, out var fillPrice, out var source))
        {
            await ConfirmFillAsync(state, fillPrice, ct, source);
            return true;
        }

        state.TrailingStateText = BuildWorkingIntentText(state);
        if (snap is not null)
            state.LastPrice = state.Direction == "buy" ? snap.Ask : snap.Bid;
        state.LastProfitPct = 0;
        PositionUpdated?.Invoke(state.ToModel(0, state.TrailingStateText));
        return false;
    }

    private async Task ConfirmFillAsync(ManagedState state, double fillPrice, CancellationToken ct, string source = "tick")
    {
        state.IsFilled = true;
        state.FilledAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        state.EntryPrice = fillPrice;
        state.BestPrice = fillPrice;
        if (_featureCache.TryGetCurrentAtr(state.Symbol, out var atr))
            state.AtrAtFill = atr;
        else if (state.StopLoss > 0)
            state.AtrAtFill = Math.Abs(fillPrice - state.StopLoss) / 1.2;
        state.PeakProfitPct = 0;
        state.LastProfitPct = 0;
        state.TrailingStateText = "已成交 · 移动止损未激活";
        await _db.UpdateSignalStatusAsync(state.SignalId, "active", ct);
        ManagedStatusChanged?.Invoke(state.SignalId, "active");
        LogEmitted?.Invoke($"限价成交({source}) {state.Symbol} {state.Direction} @ {fillPrice:F2}（目标{state.TargetEntry:F2}）");
        ChartRefreshRequested?.Invoke(state);
    }

    private async Task UpdateManagedAsync(CancellationToken ct)
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();

        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock) { snapshot = _managed.ToList(); }

        foreach (var kv in snapshot)
        {
            var state = kv.Value;
            MergeTickIntoSnapshot(state.BrokerSymbol);
            var snap = GetSnapshotForState(state);

            if (!state.IsFilled)
            {
                if (!await ProcessAwaitingFillAsync(kv.Key, state, ct))
                    continue;
                if (!state.IsFilled)
                    continue;
            }

            double bid = snap?.Bid ?? 0;
            double ask = snap?.Ask ?? 0;
            if ((bid <= 0 || ask <= 0) && _mt5.TryGetMarkPrice(state.BrokerSymbol, state.Direction, out var mark))
            {
                if (state.Direction == "buy")
                {
                    bid = mark;
                    if (ask <= 0) ask = mark;
                }
                else
                {
                    ask = mark;
                    if (bid <= 0) bid = mark;
                }
            }

            if (bid <= 0 && ask <= 0)
                continue;

            if (!CanUseMechanicalExitPrices(state, bid, ask, out _))
                continue;

            var price = state.Direction == "buy" ? bid : ask;
            state.LastPrice = price;
            var profit = ProfitPct(price, state);
            state.PeakProfitPct = Math.Max(state.PeakProfitPct, profit);
            state.LastProfitPct = profit;

            var effectiveSl = state.TrailingActivated ? state.TrailingSl : state.StopLoss;

            // ===== P0-2: 使用正确的 Bid/Ask 检查 SL/TP（仅成交后） =====
            double priceForSlCheck = state.Direction == "buy"
                ? bid
                : ask;
            double priceForTpCheck = state.Direction == "buy"
                ? ask
                : bid;
            // ===== 结束 =====

            if (state.Direction == "buy")
            {
                if (effectiveSl > 0 && priceForSlCheck <= effectiveSl)
                {
                    if (UseMechanicalExit(state.TrailingActivated ? "trailing_stop" : "stop_loss"))
                        await CloseVirtualAsync(kv.Key, state, priceForSlCheck, state.TrailingActivated ? "trailing_stop" : "stop_loss", ct);
                    continue;
                }
                if (state.TakeProfit > 0 && priceForTpCheck >= state.TakeProfit)
                {
                    if (UseMechanicalExit("take_profit"))
                        await CloseVirtualAsync(kv.Key, state, priceForTpCheck, "take_profit", ct);
                    continue;
                }
            }
            else
            {
                if (effectiveSl > 0 && priceForSlCheck >= effectiveSl)
                {
                    if (UseMechanicalExit(state.TrailingActivated ? "trailing_stop" : "stop_loss"))
                        await CloseVirtualAsync(kv.Key, state, priceForSlCheck, state.TrailingActivated ? "trailing_stop" : "stop_loss", ct);
                    continue;
                }
                if (state.TakeProfit > 0 && priceForTpCheck <= state.TakeProfit)
                {
                    if (UseMechanicalExit("take_profit"))
                        await CloseVirtualAsync(kv.Key, state, priceForTpCheck, "take_profit", ct);
                    continue;
                }
            }

            // 智能体托管时由 exit_assessment 统一出场；旧 V14 XGBoost 快照退出仅在非智能体模式启用
            var agentMode = _settings.TradingAgent?.Enabled == true;
            if (!agentMode && pm.UseModelExit && await TryModelExitAsync(kv.Key, state, price, ct))
                continue;

            // ===== 持仓时长：智能体模式仅标记到期，由 exit_assessment 决策平仓 =====
            var ageSec = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - state.FilledAt;
            if (agentMode)
            {
                if (pm.MaxHoldMinutes > 0 && ageSec >= pm.MaxHoldMinutes * 60L && !state.TimeExpired)
                {
                    state.TimeExpired = true;
                    var holdHint = profit > 0
                        ? "有浮盈，认知可顺势延长"
                        : "未盈利，认知应时间止损";
                    LogEmitted?.Invoke(
                        $"持仓已到期，等待智能体决策：{state.Symbol} signal={state.SignalId} profit={profit:F2}% hold={pm.MaxHoldMinutes}m ({holdHint})");
                }
                // 兜底：到期且仍无浮盈时机械时间止损（避免认知未触发而无限扛单）
                if (state.TimeExpired && profit <= 0 && ageSec >= pm.MaxHoldMinutes * 60L + 300)
                {
                    await CloseVirtualAsync(kv.Key, state, price, "time_stop", ct);
                    LogEmitted?.Invoke(
                        $"时间停止平仓（兜底）：{state.Symbol} signal={state.SignalId} 持仓{pm.MaxHoldMinutes}分钟未盈利");
                    continue;
                }
            }
            else if (UseMechanicalExit("time_stop") && ageSec >= pm.MaxHoldMinutes * 60L && profit <= 0)
            {
                await CloseVirtualAsync(kv.Key, state, price, "time_stop", ct);
                LogEmitted?.Invoke($"时间停止平仓：{state.Symbol} signal={state.SignalId} 持仓{pm.MaxHoldMinutes}分钟未盈利");
                continue;
            }
            else if (!agentMode && ageSec >= pm.MaxHoldMinutes * 60L && profit > 0 && !state.TimeExpired)
            {
                state.TimeExpired = true;
                LogEmitted?.Invoke($"持仓已到期但盈利，等待智能体决策：{state.Symbol} signal={state.SignalId} profit={profit:F2}%");
            }
            // ===== 结束 =====

            // 浮盈回撤平仓：峰值空间 + 成交后时间门槛，避免秒平
            var holdSinceFill = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - state.FilledAt;
            if (UseMechanicalExit("profit_drawdown")
                && holdSinceFill >= pm.MinHoldSecondsBeforeProfitDrawdown
                && state.PeakProfitPct >= pm.MinPeakProfitPctForDrawdown
                && profit < state.PeakProfitPct * (1 - pm.MaxDrawdownRatio))
            {
                await CloseVirtualAsync(kv.Key, state, price, "profit_drawdown", ct);
                continue;
            }

            if (UseMechanicalExit("trailing") && pm.TrailingStopEnabled)
                UpdateVirtualTrailingStop(state, pm, profit, price, ct);

            if (UseMechanicalExit("partial") && pm.PartialProfitEnabled)
                await CheckVirtualPartialProfitAsync(state, pm, profit, ct);

            if (state.TrailingStateText is "已成交 · 移动止损未激活" or "信号托管中 · 移动止损未激活")
                state.TrailingStateText = $"浮盈 {profit:F2}%";

            PositionUpdated?.Invoke(state.ToModel(profit, state.TrailingStateText));
        }
    }

    private async Task<bool> TryModelExitAsync(long ticketKey, ManagedState state, double price, CancellationToken ct)
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        if (!pm.UseModelExit) return false;
        if (!_inference.TryGet(state.Symbol, out var inf) || inf.Confidence < 0.55) return false;

        var opposite = (state.Direction == "buy" && inf.Direction < 0) ||
                       (state.Direction == "sell" && inf.Direction > 0);
        if (!opposite) return false;

        await CloseVirtualAsync(ticketKey, state, price, "model_exit", ct);
        LogEmitted?.Invoke($"模型出场 {state.Symbol} signal={state.SignalId} conf={inf.Confidence:F2}");
        return true;
    }

    private async Task CloseVirtualAsync(long ticketKey, ManagedState state, double closePrice, string reason, CancellationToken ct)
    {
        lock (_managedLock)
        {
            if (!_managed.ContainsKey(ticketKey)) return;  // 已平仓防重入
            _managed.Remove(ticketKey);
        }
        await RecordClosedTradeAsync(state, closePrice, reason, ct);
        await _db.UpdateSignalStatusAsync(state.SignalId, MapStatus(reason), ct);
        await LogEvent(state, "full_close", ct, reason: reason);
        SignalDrawClearRequested?.Invoke(state.SignalId);
        PositionClosed?.Invoke(state.ToModel(
            ProfitPct(closePrice, state),
            reason));
    }

    private static string MapStatus(string reason) => reason switch
    {
        "stop_loss" => "stop_loss",
        "take_profit" => "take_profit",
        "trailing_stop" => "trailing_stop",
        "trailing" => "trailing_stop",
        "profit_drawdown" => "profit_drawdown",
        "entry_timeout" => "expired",
        "time_stop" => "time_stop",
        "model_exit" => "model_exit",
        "agent_exit" => "agent_exit",
        _ => "normal_close",
    };

    private async Task RecordClosedTradeAsync(ManagedState state, double closePrice, string reason, CancellationToken ct)
    {
        var point = _mt5.SymbolPoint(state.BrokerSymbol);
        if (point <= 0) point = 0.01;
        double pnlPoints = state.Direction == "buy"
            ? (closePrice - state.EntryPrice) / point
            : (state.EntryPrice - closePrice) / point;

        var pnlPct = ProfitPct(closePrice, state);

        await _db.SaveTradeAsync(new TradeModel
        {
            SignalId = state.SignalId,
            OpenTime = state.IsFilled && state.FilledAt > 0 ? state.FilledAt : state.OpenTime,
            OpenPrice = state.IsFilled ? state.EntryPrice : state.TargetEntry,
            CloseTime = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            ClosePrice = closePrice,
            PnlPoints = pnlPoints,
            PnlPercent = pnlPct,
            IsWin = pnlPct > 0 ? 1 : 0,
            CloseReason = reason,
        }, ct);
        _logger.LogInformation("信号平仓 signal={Signal} pnl={Pnl:F2}% reason={Reason}", state.SignalId, pnlPct, reason);
        LogEmitted?.Invoke($"信号平仓 {state.Symbol} {state.SignalId} {pnlPct:F2}% ({reason})");
    }

    private static double ProfitPct(double price, ManagedState state)
    {
        if (state.EntryPrice <= 0) return 0;
        if (state.Direction == "buy") return (price - state.EntryPrice) / state.EntryPrice * 100;
        return (state.EntryPrice - price) / state.EntryPrice * 100;
    }

    private void UpdateVirtualTrailingStop(ManagedState state, AppSettings.PositionManagementSettings pm,
        double profitPct, double price, CancellationToken ct)
    {
        var agentMode = _settings.TradingAgent?.Enabled == true;
        TrailingStopEngine.TrailingResult result;

        if (pm.TrailingUseAtrMode)
        {
            var atr = state.AtrAtFill;
            if (atr <= 0)
                _featureCache.TryGetCurrentAtr(state.Symbol, out atr);
            if (atr <= 0 && state.EntryPrice > 0)
                atr = Math.Abs(state.EntryPrice - state.StopLoss) / 1.2;

            _featureCache.TryGetM5Bars(state.Symbol, out var m5Bars);
            var holdSec = (int)(DateTimeOffset.UtcNow.ToUnixTimeSeconds() - state.FilledAt);

            result = TrailingStopEngine.Evaluate(
                new TrailingStopEngine.TrailingContext
                {
                    Direction = state.Direction,
                    Entry = state.EntryPrice,
                    Price = price,
                    Atr = atr,
                    BestPrice = state.BestPrice,
                    TrailingSl = state.TrailingSl,
                    TrailingActivated = state.TrailingActivated,
                    LastTrailPrice = state.LastTrailPrice,
                    HoldSeconds = holdSec,
                    M5Bars = m5Bars,
                },
                pm,
                agentMode);
        }
        else
        {
            result = EvaluateLegacyPercentTrailing(state, pm, profitPct, price);
        }

        var wasActivated = state.TrailingActivated;
        var oldSl = state.TrailingSl;

        state.BestPrice = result.BestPrice;
        state.TrailingActivated = result.TrailingActivated;
        state.TrailingSl = result.TrailingSl;
        state.LastTrailPrice = result.LastTrailPrice;
        if (!string.IsNullOrEmpty(result.StateText))
            state.TrailingStateText = result.StateText;

        var activatedNow = !wasActivated && result.TrailingActivated;
        var slMoved = result.TrailingActivated && Math.Abs(result.TrailingSl - oldSl) > 0.01;

        if ((activatedNow || slMoved) && result.LogMessage != null)
            LogEmitted?.Invoke($"{result.LogMessage} signal={state.SignalId}");

        if (activatedNow || slMoved)
        {
            _ = _db.LogPositionEventAsync(
                state.SignalId, "move_sl",
                oldSl: activatedNow ? state.StopLoss : oldSl,
                newSl: result.TrailingSl, ct: ct);
            TryModifyRealPositionSlTp(state, result.TrailingSl, state.TakeProfit);
            ChartRefreshRequested?.Invoke(state);
        }
    }

    /// <summary>旧版百分比移动止损（TrailingUseAtrMode=false）。</summary>
    private static TrailingStopEngine.TrailingResult EvaluateLegacyPercentTrailing(
        ManagedState state, AppSettings.PositionManagementSettings pm, double profitPct, double price)
    {
        var dir = state.Direction == "buy" ? 1.0 : -1.0;
        var best = state.BestPrice > 0 ? state.BestPrice : state.EntryPrice;
        best = state.Direction == "buy" ? Math.Max(best, price) : Math.Min(best, price);

        if (profitPct >= pm.TrailingActivationPct && !state.TrailingActivated)
        {
            return new TrailingStopEngine.TrailingResult
            {
                BestPrice = best,
                TrailingSl = state.EntryPrice,
                TrailingActivated = true,
                LastTrailPrice = price,
                StateText = "移动止损已激活（保本）",
                LogMessage = $"移动止损激活 SL→保本 {state.EntryPrice:F2}",
            };
        }

        if (!state.TrailingActivated)
        {
            return new TrailingStopEngine.TrailingResult
            {
                BestPrice = best,
                TrailingSl = state.TrailingSl,
                TrailingActivated = false,
                LastTrailPrice = state.LastTrailPrice,
                StateText = state.TrailingStateText,
            };
        }

        var priceMove = (price - state.LastTrailPrice) * dir;
        var stepThreshold = pm.TrailingStepPct * state.EntryPrice / 100.0;
        if (priceMove < stepThreshold)
        {
            return new TrailingStopEngine.TrailingResult
            {
                BestPrice = best,
                TrailingSl = state.TrailingSl,
                TrailingActivated = true,
                LastTrailPrice = state.LastTrailPrice,
                StateText = state.TrailingStateText,
            };
        }

        var tighten = pm.TrailingStepPct * state.EntryPrice / 100.0 * pm.TrailingTightenFactor;
        var newSl = state.TrailingSl + tighten * dir;
        if (Math.Abs(newSl - state.TrailingSl) <= 0.01)
        {
            return new TrailingStopEngine.TrailingResult
            {
                BestPrice = best,
                TrailingSl = state.TrailingSl,
                TrailingActivated = true,
                LastTrailPrice = state.LastTrailPrice,
                StateText = state.TrailingStateText,
            };
        }

        return new TrailingStopEngine.TrailingResult
        {
            BestPrice = best,
            TrailingSl = newSl,
            TrailingActivated = true,
            LastTrailPrice = price,
            StateText = $"移动止损 SL={newSl:F2}",
            LogMessage = $"移动止损 SL {state.TrailingSl:F2} → {newSl:F2}",
        };
    }

    private void TryModifyRealPositionSlTp(ManagedState state, double sl, double tp)
    {
        if (state.RealTicket <= 0) return;
        Mt5OrderHelper.ModifySlTpWithRetry(_mt5, state.RealTicket, sl, tp, 3,
            msg => LogEmitted?.Invoke(msg));
    }

    private async Task CheckVirtualPartialProfitAsync(ManagedState state,
        AppSettings.PositionManagementSettings pm, double profitPct, CancellationToken ct)
    {
        if (state.PartialStep == 0 && profitPct >= pm.PartialTarget1Pct)
        {
            state.PartialStep = 1;
            await LogEvent(state, "partial_close", ct, volume: pm.PartialRatio1, reason: "target1");
            LogEmitted?.Invoke($"虚拟分批止盈 1 signal={state.SignalId} {profitPct:F2}%");
        }
        else if (state.PartialStep == 1 && profitPct >= pm.PartialTarget2Pct)
        {
            state.PartialStep = 2;
            await LogEvent(state, "partial_close", ct, volume: pm.PartialRatio2, reason: "target2");
            LogEmitted?.Invoke($"虚拟分批止盈 2 signal={state.SignalId} {profitPct:F2}%");
        }
    }

    private async Task LogEvent(ManagedState state, string eventType, CancellationToken ct,
        double? volume = null, string? reason = null)
    {
        await _db.LogPositionEventAsync(state.SignalId, eventType, volume: volume, ct: ct);
        _logger.LogInformation("持仓事件 {Type} signal={Signal} reason={Reason}", eventType, state.SignalId, reason);
    }

    // ========== 实时报价高频监控（由 ZhuLongRuntimeService 调用） ==========

    public IReadOnlyList<string> ActiveSymbols()
    {
        lock (_managedLock)
            return _managed.Values
                .Select(m => m.BrokerSymbol)
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
    }

    public IReadOnlyList<ManagedState> ActiveManagedStates()
    {
        lock (_managedLock)
            return _managed.Values.ToList();
    }

    /// <summary>从数据库恢复 active / awaiting_fill 信号到托管列表（开机/重启后重绘图表）。</summary>
    /// <returns>实际恢复条数（已超最大持仓时间的信号会写回 DB 为 time_stop，不恢复、不重绘）。</returns>
    public async Task<int> RestoreActiveSignalsAsync(IReadOnlyList<SignalModel> active, CancellationToken ct = default)
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var maxAgeSec = pm.MaxHoldMinutes * 60L;
        int restored = 0;

        foreach (var sig in active)
        {
            if (IsManagingSignal(sig.SignalId))
                continue;
            if (sig.Direction is not ("buy" or "sell"))
                continue;

            var openTime = sig.CreatedAt > 0 ? sig.CreatedAt : now;
            var isAwaitingFill = string.Equals(sig.Status, "awaiting_fill", StringComparison.OrdinalIgnoreCase);

            if (isAwaitingFill)
            {
                if (now - openTime >= GetFillMaxWaitSeconds())
                {
                    await _db.UpdateSignalStatusAsync(sig.SignalId, "expired", ct);
                    SignalDrawClearRequested?.Invoke(sig.SignalId);
                    _logger.LogInformation("跳过恢复超时限价信号 signal={Signal}", sig.SignalId);
                    LogEmitted?.Invoke($"跳过超时限价 {sig.Symbol} {sig.SignalId}（未成交已过期）");
                    continue;
                }
            }
            else if (maxAgeSec > 0 && now - openTime >= maxAgeSec)
            {
                await _db.UpdateSignalStatusAsync(sig.SignalId, "time_stop", ct);
                SignalDrawClearRequested?.Invoke(sig.SignalId);
                _logger.LogInformation(
                    "跳过恢复过期托管信号 signal={Signal} {Symbol}（持仓已超过 {MaxHold} 分钟）",
                    sig.SignalId, sig.Symbol, pm.MaxHoldMinutes);
                LogEmitted?.Invoke($"跳过过期托管 {sig.Symbol} {sig.SignalId}（已超过 {pm.MaxHoldMinutes} 分钟，不重绘）");
                continue;
            }

            // ===== P1-1: AI 评估 — 方向不匹配的信号不恢复（仅对已成交） =====
            if (!isAwaitingFill)
            {
                if (_inference.TryGet(sig.Symbol, out var inf) && inf.Direction != 0)
                {
                    bool signalBullish = sig.Direction == "buy";
                    bool agentBullish = inf.Direction > 0;
                    if (signalBullish != agentBullish)
                    {
                        await _db.UpdateSignalStatusAsync(sig.SignalId, "model_exit", ct);
                        SignalDrawClearRequested?.Invoke(sig.SignalId);
                        LogEmitted?.Invoke($"开机评估：智能体方向与旧信号 {sig.SignalId} 不一致，平仓不恢复");
                        continue;
                    }
                }
                if (_inference.TryGet(sig.Symbol, out var inf2) && inf2.Direction == 0)
                {
                    await _db.UpdateSignalStatusAsync(sig.SignalId, "normal_close", ct);
                    SignalDrawClearRequested?.Invoke(sig.SignalId);
                    LogEmitted?.Invoke($"开机评估：智能体看平，旧信号 {sig.SignalId} 平仓不恢复");
                    continue;
                }
            }
            // ===== 结束 =====

            var brokerSym = _settings.ResolveBrokerSymbol(sig.Symbol);
            var ticket = VirtualTicket(sig.SignalId);
            var realTicket = _mt5.FindRealPositionTicket(sig.SignalId);
            MergeTickIntoSnapshot(brokerSym);

            var state = new ManagedState
            {
                Ticket = ticket,
                RealTicket = realTicket,
                SignalId = sig.SignalId,
                Symbol = sig.Symbol,
                Direction = sig.Direction,
                TargetEntry = sig.EntryPrice,
                EntryPrice = sig.EntryPrice,
                StopLoss = sig.StopLoss,
                TakeProfit = sig.TakeProfit,
                OpenTime = openTime,
                Volume = 1.0,
                BrokerSymbol = brokerSym,
                IsFilled = !isAwaitingFill,
                FilledAt = isAwaitingFill ? 0 : openTime,
            };

            if (isAwaitingFill)
            {
                if (TryMatchFill(state, out var fillPrice, out var fillSource))
                {
                    state.IsFilled = true;
                    state.FilledAt = now;
                    state.EntryPrice = fillPrice;
                    state.TrailingStateText = "已成交 · 移动止损未激活";
                    await _db.UpdateSignalStatusAsync(sig.SignalId, "active", ct);
                    LogEmitted?.Invoke($"恢复后成交({fillSource}) {sig.Symbol} @ {fillPrice:F2}");
                }
                else
                {
                    state.TrailingStateText = BuildWorkingIntentText(state);
                }
            }
            else
            {
                state.TrailingStateText = "已成交 · 移动止损未激活";
            }

            lock (_managedLock)
            {
                _managed[ticket] = state;
            }
            restored++;
            _logger.LogInformation("恢复托管信号 signal={Signal} {Symbol} {Dir} filled={Filled}",
                sig.SignalId, sig.Symbol, sig.Direction, state.IsFilled);
        }

        return restored;
    }

    /// <summary>智能体 tick 评估：限价待成交动态调价 / 追价过高放弃。</summary>
    public async Task OptimizeEntryPricesAsync(string brokerSymbol, Mt5Tick tick, string? cognitionRegime, CancellationToken ct)
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        if (_settings.TradingAgent?.Enabled != true || !pm.AgentDrivenEntry)
            return;

        var regime = (cognitionRegime ?? "").Trim().ToLowerInvariant();
        var chaseMult = regime is "choppy" or "ranging" ? 0.25 : 0.35;

        List<KeyValuePair<long, ManagedState>> awaiting;
        lock (_managedLock)
        {
            awaiting = _managed
                .Where(kv => string.Equals(kv.Value.BrokerSymbol, brokerSymbol, StringComparison.OrdinalIgnoreCase)
                    && !kv.Value.IsFilled)
                .ToList();
        }

        foreach (var kv in awaiting)
        {
            var state = kv.Value;
            var atrEst = Math.Abs(state.TargetEntry - state.StopLoss) / 1.2;
            if (atrEst <= 0) atrEst = state.TargetEntry * 0.001;

            if (state.Direction == "buy")
            {
                var chase = tick.Ask - state.TargetEntry;
                if (chase > atrEst * chaseMult)
                {
                    await CancelAwaitingFillAsync(kv.Key, state, $"Ask={tick.Ask:F2} 追价过高>{state.TargetEntry:F2}", ct);
                    continue;
                }
                if (tick.Ask > 0 && tick.Ask < state.TargetEntry - 0.01)
                {
                    var better = Math.Min(state.TargetEntry, tick.Ask);
                    if (Math.Abs(better - state.TargetEntry) > 0.01)
                    {
                        state.TargetEntry = better;
                        state.EntryPrice = better;
                        await _db.UpdateSignalEntryAsync(state.SignalId, better, ct);
                        ChartRefreshRequested?.Invoke(state);
                    }
                }
            }
            else
            {
                var chase = state.TargetEntry - tick.Bid;
                if (chase > atrEst * chaseMult)
                {
                    await CancelAwaitingFillAsync(kv.Key, state, $"Bid={tick.Bid:F2} 追价过低<{state.TargetEntry:F2}", ct);
                    continue;
                }
                if (tick.Bid > state.TargetEntry + 0.01)
                {
                    var better = Math.Max(state.TargetEntry, tick.Bid);
                    if (Math.Abs(better - state.TargetEntry) > 0.01)
                    {
                        state.TargetEntry = better;
                        state.EntryPrice = better;
                        await _db.UpdateSignalEntryAsync(state.SignalId, better, ct);
                        ChartRefreshRequested?.Invoke(state);
                    }
                }
            }
        }
    }

    private async Task CancelAwaitingFillAsync(long ticketKey, ManagedState state, string reason, CancellationToken ct)
    {
        lock (_managedLock) { _managed.Remove(ticketKey); }
        await _db.UpdateSignalStatusAsync(state.SignalId, "intent_cancelled", ct);
        SignalDrawClearRequested?.Invoke(state.SignalId);
        ManagedStatusChanged?.Invoke(state.SignalId, "intent_cancelled");
        LogEmitted?.Invoke($"挂单意图撤销 {state.Symbol} {state.Direction} signal={state.SignalId}（{reason}）");
    }

    private bool UseMechanicalExit(string kind)
    {
        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        if (_settings.TradingAgent?.Enabled != true || !pm.AgentDrivenExit)
            return true;
        return kind switch
        {
            "stop_loss" => pm.AgentHardStopLoss,
            "trailing_stop" => pm.TrailingStopEnabled,
            "trailing" => pm.TrailingStopEnabled,
            _ => false,
        };
    }
    public async Task FastTrailingStopAsync(string brokerSymbol, double bid, double ask, CancellationToken ct)
    {
        await ProcessWorkingIntentFillAsync(brokerSymbol, bid, ask, ct);

        var pm = _settings.PositionManagement ?? new AppSettings.PositionManagementSettings();
        if (!pm.TrailingStopEnabled) return;

        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock) { snapshot = _managed.ToList(); }

        foreach (var kv in snapshot)
        {
            var state = kv.Value;
            if (!string.Equals(state.BrokerSymbol, brokerSymbol, StringComparison.OrdinalIgnoreCase))
                continue;

            if (!state.IsFilled)
                continue;

            if (!CanUseMechanicalExitPrices(state, bid, ask, out var blockReason))
            {
                if (blockReason == "data_pipe_disconnected")
                    LogEmitted?.Invoke($"数据管道断开，暂停机械 SL/TP：{state.Symbol} signal={state.SignalId}");
                continue;
            }

            // ===== P0-2: 使用正确的 Bid/Ask 计算盈亏和检查 SL/TP（仅成交后） =====
            double priceForProfit = state.Direction == "buy" ? bid : ask;
            double priceForSlCheck = state.Direction == "buy" ? bid : ask;
            double priceForTpCheck = state.Direction == "buy" ? ask : bid;

            var profit = ProfitPct(priceForProfit, state);
            state.PeakProfitPct = Math.Max(state.PeakProfitPct, profit);
            state.LastProfitPct = profit;
            state.LastPrice = priceForProfit;
            // ===== 结束 =====

            var effectiveSl = state.TrailingActivated ? state.TrailingSl : state.StopLoss;
            bool hitStop = state.Direction == "buy"
                ? (effectiveSl > 0 && priceForSlCheck <= effectiveSl)
                : (effectiveSl > 0 && priceForSlCheck >= effectiveSl);
            if (hitStop)
            {
                if (UseMechanicalExit(state.TrailingActivated ? "trailing_stop" : "stop_loss"))
                    await CloseVirtualAsync(kv.Key, state, priceForSlCheck, state.TrailingActivated ? "trailing_stop" : "stop_loss", ct);
                continue;
            }
            bool hitTp = state.Direction == "buy"
                ? (state.TakeProfit > 0 && priceForTpCheck >= state.TakeProfit)
                : (state.TakeProfit > 0 && priceForTpCheck <= state.TakeProfit);
            if (hitTp)
            {
                if (UseMechanicalExit("take_profit"))
                    await CloseVirtualAsync(kv.Key, state, priceForTpCheck, "take_profit", ct);
                continue;
            }

            if (UseMechanicalExit("trailing"))
                UpdateVirtualTrailingStop(state, pm, profit, priceForProfit, ct);
        }
    }

    /// <summary>保留旧版同步 FastTrailingStop 签名（委托给异步版本）。</summary>
    public void FastTrailingStop(string brokerSymbol, double price, CancellationToken ct)
    {
        // 不实现：旧签名已废弃，应使用 FastTrailingStopAsync(brokerSymbol, bid, ask, ct)
        LogEmitted?.Invoke("⚠ FastTrailingStop 旧签名被调用，请使用 FastTrailingStopAsync(brokerSymbol, bid, ask, ct)");
    }

    // ===== P2-2: AI 动态调整持仓 SL/TP =====
    public void ApplyAiPositionAdjustment(string signalId, double? newSl, double? newTp, string reason)
    {
        ManagedState? state;
        lock (_managedLock)
        {
            state = _managed.Values.FirstOrDefault(m => m.SignalId == signalId);
        }
        if (state is null) return;
        if (!state.IsFilled) return;

        bool changed = false;
        if (newSl.HasValue && Math.Abs(newSl.Value - state.StopLoss) > 0.01)
        {
            var candidate = newSl.Value;
            var current = state.TrailingActivated ? state.TrailingSl : state.StopLoss;
            var tightens = state.Direction == "buy"
                ? candidate > current
                : candidate < current;
            if (!tightens)
            {
                LogEmitted?.Invoke(
                    $"AI止损调整已忽略 {signalId}：{candidate:F2} 未收紧（当前 {current:F2}）");
            }
            else
            {
                if (state.TrailingActivated)
                    state.TrailingSl = candidate;
                else
                    state.StopLoss = candidate;
                TryModifyRealPositionSlTp(state, candidate, state.TakeProfit);
                LogEmitted?.Invoke($"AI调整止损 {signalId} → {candidate:F2} 理由：{reason}");
                changed = true;
            }
        }
        if (newTp.HasValue && Math.Abs(newTp.Value - state.TakeProfit) > 0.01)
        {
            var candidate = newTp.Value;
            var current = state.TakeProfit;
            var extends = state.Direction == "buy"
                ? candidate > current
                : candidate < current;
            if (!extends)
            {
                LogEmitted?.Invoke(
                    $"AI止盈调整已忽略 {signalId}：{candidate:F2} 未向有利方向扩展（当前 {current:F2}）");
            }
            else
            {
                state.TakeProfit = candidate;
                var slForMt5 = state.TrailingActivated ? state.TrailingSl : state.StopLoss;
                TryModifyRealPositionSlTp(state, slForMt5, candidate);
                LogEmitted?.Invoke($"AI调整止盈 {signalId} → {candidate:F2} 理由：{reason}");
                changed = true;
            }
        }

        if (changed)
        {
            ChartRefreshRequested?.Invoke(state);
            AiSlTpUpdated?.Invoke(state, newSl, newTp, reason);
        }
    }

    // ===== P0-3: MT5 实盘持仓同步 =====
    /// <summary>遍历所有托管信号，与 MT5 实际持仓强制对照。</summary>
    public async Task SyncRealPositionsAsync(CancellationToken ct)
    {
        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock) { snapshot = _managed.ToList(); }

        foreach (var kv in snapshot)
        {
            var state = kv.Value;
            if (!state.IsFilled || state.RealTicket <= 0) continue;

            var realPos = _mt5.GetPosition(state.RealTicket);
            if (realPos == null)
            {
                // MT5 持仓已不存在（手动平仓或外部触发）
                await CloseVirtualAsync(kv.Key, state, state.LastPrice > 0 ? state.LastPrice : state.EntryPrice, "external_close", ct);
                LogEmitted?.Invoke($"外部平仓检测：信号 {state.SignalId} 的 MT5 持仓已消失，同步平仓");
                continue;
            }

            // 检测外部 SL 修改
            if (realPos.Sl > 0)
            {
                if (Math.Abs(realPos.Sl - state.StopLoss) > 0.05 &&
                    Math.Abs(realPos.Sl - state.TrailingSl) > 0.05)
                {
                    LogEmitted?.Invoke($"外部 SL 修改检测 signal={state.SignalId} 虚拟SL={state.StopLoss:F2} 真实SL={realPos.Sl:F2}");
                }
            }
        }
    }

    public async Task CloseAllPositionsAsync(string reason, CancellationToken ct)
    {
        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock) { snapshot = _managed.ToList(); }

        foreach (var kv in snapshot)
        {
            await CloseVirtualAsync(kv.Key, kv.Value, kv.Value.LastPrice > 0 ? kv.Value.LastPrice : kv.Value.EntryPrice, reason, ct);
        }
    }

    /// <summary>智能体 exit_assessment 达阈值时平仓。</summary>
    public async Task TryAgentExitAsync(string symbol, double exitScore, string reason, CancellationToken ct, double threshold = 0.65)
    {
        if (exitScore < threshold) return;

        List<KeyValuePair<long, ManagedState>> snapshot;
        lock (_managedLock)
        {
            snapshot = _managed
                .Where(kv => string.Equals(kv.Value.Symbol, symbol, StringComparison.OrdinalIgnoreCase))
                .ToList();
        }

        foreach (var kv in snapshot)
        {
            var state = kv.Value;
            if (!state.IsFilled)
                continue;

            var price = state.LastPrice > 0 ? state.LastPrice : state.EntryPrice;
            await CloseVirtualAsync(kv.Key, state, price, "agent_exit", ct);
            LogEmitted?.Invoke($"智能体出场 {symbol} signal={state.SignalId} score={exitScore:F2} {reason}");
        }
    }

    public sealed class ManagedState
    {
        public long Ticket { get; init; }
        public long RealTicket { get; set; }
        public string SignalId { get; init; } = "";
        public string Symbol { get; init; } = "";
        public string BrokerSymbol { get; init; } = "";
        public string Direction { get; init; } = "";
        public double EntryPrice { get; set; }
        public double TargetEntry { get; set; }
        public double StopLoss { get; set; }
        public double TakeProfit { get; set; }
        public long OpenTime { get; init; }
        public long FilledAt { get; set; }
        public bool IsFilled { get; set; }
        public double Volume { get; init; }
        public double LastPrice { get; set; }
        public double PeakProfitPct { get; set; }
        public double LastProfitPct { get; set; }
        public int PartialStep { get; set; }
        public double TrailingSl { get; set; }
        public bool TrailingActivated { get; set; }
        public double LastTrailPrice { get; set; }
        public double BestPrice { get; set; }
        public double AtrAtFill { get; set; }
        public string TrailingStateText { get; set; } = "";
        // P0-1: 标记持仓是否已过期（已到时间但盈利）
        public bool TimeExpired { get; set; }

        public ManagedPositionModel ToModel(double profitPct, string trailing = "") => new()
        {
            Ticket = Ticket,
            SignalId = SignalId,
            Symbol = Symbol,
            Direction = Direction,
            EntryPrice = IsFilled ? EntryPrice : TargetEntry,
            Volume = Volume,
            ProfitPct = profitPct,
            TrailingState = trailing,
            IsManaged = true,
            IsFilled = IsFilled,
        };
    }
}
