using Microsoft.Extensions.Logging;
using Python.Runtime;
using ZhuLong.Core;
using ZhuLong.Core.Models;

namespace ZhuLong.App.Services;

/// <summary>MT5 API — Python.NET 调用 MetaTrader5 + mt5_ops。</summary>
public sealed class Mt5ApiWrapper : IDisposable
{
    private readonly ILogger<Mt5ApiWrapper> _logger;
    private readonly PythonGilExecutor _gil;
    private dynamic? _mt5;
    private dynamic? _mt5Ops;
    private bool _connected;
    private int _deviation = 20;

    public Mt5ApiWrapper(ILogger<Mt5ApiWrapper> logger, PythonGilExecutor gil)
    {
        _logger = logger;
        _gil = gil;
    }

    public bool Connected => _connected;

    public void SetDeviation(int deviation) => _deviation = deviation;

    public bool Connect()
    {
        StartupLog.Write("MT5 Connect 开始");
        try
        {
            return _gil.Run(() =>
            {
                try
                {
                    EnsurePythonPath();
                    _mt5 ??= Py.Import("MetaTrader5");
                    _mt5Ops ??= Py.Import("mt5_ops");
                }
                catch (PythonException ex) when (ex.Message.Contains("MetaTrader5", StringComparison.Ordinal))
                {
                    var hint = AppPaths.PythonDepsScriptHint;
                    _logger.LogError("MetaTrader5 包未安装，请运行 {Hint}", hint);
                    LogEmitted?.Invoke($"Python 已找到，但缺少 MetaTrader5 包。请运行: {hint}");
                    _connected = false;
                    return false;
                }

                var ok = (bool)_mt5.initialize();
                _connected = ok;
                if (!ok)
                {
                    _logger.LogError("mt5.initialize 失败（请确认 MT5 已打开并登录）");
                    LogEmitted?.Invoke("mt5.initialize 失败：请先打开 MetaTrader 5 并登录账户");
                }
                else
                {
                    _logger.LogInformation("MT5 已连接");
                    StartupLog.Write("MT5 Connect 成功");
                }
                return ok;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "MT5 连接异常");
            StartupLog.Write("MT5 Connect 异常: " + ex.Message);
            _connected = false;
            return false;
        }
    }

    /// <summary>断线后自动重连（R5.1）。</summary>
    public bool TryReconnect()
    {
        if (_connected) return true;
        _logger.LogInformation("尝试重连 MT5…");
        LogEmitted?.Invoke("MT5 断开，正在重连…");
        return Connect();
    }

    public event Action<string>? LogEmitted;

    public void Disconnect()
    {
        if (!_connected || _mt5 is null) return;
        _gil.Run(() => _mt5.shutdown());
        _connected = false;
    }

    public int GetPositionCountSafe()
    {
        if (!_connected || _mt5 is null) return 0;
        if (_gil.TryRun(() => GetPositionsCore().Count, TimeSpan.FromSeconds(3), out var count))
            return count;
        _logger.LogWarning("MT5 持仓查询超时，按 0 处理");
        LogEmitted?.Invoke("MT5 持仓查询超时，风控按 0 持仓继续");
        return 0;
    }

    public IReadOnlyList<Mt5Position> GetPositions()
    {
        if (!_connected || _mt5 is null) return Array.Empty<Mt5Position>();
        return _gil.Run(() => GetPositionsCore());
    }

    /// <summary>带超时，避免 UI 线程在 GIL 争用时卡死。</summary>
    public IReadOnlyList<Mt5Position> GetPositionsSafe(TimeSpan timeout = default)
    {
        if (timeout == default) timeout = TimeSpan.FromSeconds(3);
        if (!_connected || _mt5 is null) return Array.Empty<Mt5Position>();
        if (_gil.TryRun(() => GetPositionsCore(), timeout, out var list))
            return list;
        _logger.LogWarning("MT5 持仓列表查询超时");
        LogEmitted?.Invoke("MT5 持仓查询超时（列表可能不完整）");
        return Array.Empty<Mt5Position>();
    }

    private IReadOnlyList<Mt5Position> GetPositionsCore()
    {
        using (Py.GIL())
        {
            dynamic positions = _mt5!.positions_get();
            if (positions == null) return Array.Empty<Mt5Position>();
            var list = new List<Mt5Position>();
            foreach (PyObject item in positions)
            {
                list.Add(new Mt5Position
                {
                    Ticket = item.GetAttr("ticket").As<long>(),
                    Symbol = item.GetAttr("symbol").As<string>(),
                    Type = item.GetAttr("type").As<int>(),
                    Volume = item.GetAttr("volume").As<double>(),
                    PriceOpen = item.GetAttr("price_open").As<double>(),
                    PriceCurrent = item.GetAttr("price_current").As<double>(),
                    Sl = item.GetAttr("sl").As<double>(),
                    Tp = item.GetAttr("tp").As<double>(),
                    Comment = item.GetAttr("comment").As<string>() ?? "",
                    Time = item.GetAttr("time").As<long>(),
                });
            }
            return list;
        }
    }

    public double SymbolPoint(string brokerSymbol)
    {
        if (!_connected || _mt5Ops is null) return 0.01;
        return _gil.Run(() => (double)_mt5Ops.symbol_point(brokerSymbol));
    }

    /// <summary>信号托管用报价：buy 用 bid，sell 用 ask。</summary>
    public bool TryGetMarkPrice(string brokerSymbol, string direction, out double price)
    {
        price = 0;
        if (!_connected || _mt5 is null) return false;
        if (!_gil.TryRun(() =>
            {
                dynamic tick = _mt5!.symbol_info_tick(brokerSymbol);
                if (tick is null) return 0.0;
                var bid = (double)tick.bid;
                var ask = (double)tick.ask;
                if (bid <= 0 || ask <= 0) return 0.0;
                return direction == "buy" ? bid : ask;
            }, TimeSpan.FromSeconds(2), out price))
            return false;
        return price > 0;
    }

    /// <summary>获取完整 Tick 报价（含买卖价和时间戳），供高频监控使用。</summary>
    public Mt5Tick? GetTickPrice(string brokerSymbol)
    {
        if (!_connected || _mt5 is null) return null;
        try
        {
            return _gil.Run(() =>
            {
                dynamic tick = _mt5!.symbol_info_tick(brokerSymbol);
                if (tick is null) return null;
                return new Mt5Tick
                {
                    Bid = (double)tick.bid,
                    Ask = (double)tick.ask,
                    Time = (long)tick.time,
                };
            });
        }
        catch { return null; }
    }

    public bool ModifySlTp(long ticket, double sl, double tp)
    {
        if (!_connected || _mt5Ops is null) return false;
        return _gil.Run(() => (bool)_mt5Ops.modify_sl_tp(ticket, sl, tp, _deviation));
    }

    /// <summary>通过 Comment 字段查找实盘 MT5 持仓票据号。</summary>
    public long FindRealPositionTicket(string signalId)
    {
        if (!_connected || _mt5 is null) return 0;
        try
        {
            using (Py.GIL())
            {
                dynamic positions = _mt5.positions_get();
                if (positions is null) return 0;
                foreach (PyObject item in positions)
                {
                    var comment = item.GetAttr("comment").As<string>() ?? "";
                    if (comment.Contains(signalId, StringComparison.Ordinal))
                        return item.GetAttr("ticket").As<long>();
                }
            }
        }
        catch { /* ignore */ }
        return 0;
    }

    /// <summary>按 ticket 获取 MT5 实盘持仓（P0-3 同步用）。</summary>
    public Mt5Position? GetPosition(long ticket)
    {
        if (!_connected || _mt5 is null || ticket <= 0) return null;
        try
        {
            using (Py.GIL())
            {
                dynamic positions = _mt5.positions_get();
                if (positions is null) return null;
                foreach (PyObject item in positions)
                {
                    var t = item.GetAttr("ticket").As<long>();
                    if (t == ticket)
                    {
                        return new Mt5Position
                        {
                            Ticket = t,
                            Symbol = item.GetAttr("symbol").As<string>() ?? "",
                            Type = item.GetAttr("type").As<int>(),
                            Volume = item.GetAttr("volume").As<double>(),
                            PriceOpen = item.GetAttr("price_open").As<double>(),
                            PriceCurrent = item.GetAttr("price_current").As<double>(),
                            Sl = item.GetAttr("sl").As<double>(),
                            Tp = item.GetAttr("tp").As<double>(),
                            Comment = item.GetAttr("comment").As<string>() ?? "",
                            Time = item.GetAttr("time").As<long>(),
                        };
                    }
                }
            }
        }
        catch { /* ignore */ }
        return null;
    }

    public bool ClosePartial(long ticket, double volume)
    {
        if (!_connected || _mt5Ops is null) return false;
        return _gil.Run(() => (bool)_mt5Ops.close_partial(ticket, volume, _deviation));
    }

    public bool CloseFull(long ticket)
    {
        if (!_connected || _mt5Ops is null) return false;
        return _gil.Run(() => (bool)_mt5Ops.close_full(ticket, _deviation));
    }

    /// <summary>经 MT5 API 拉取历史 M1（烛龙启动/管道重连时预热，不依赖指标重推）。</summary>
    public IReadOnlyList<M1Bar> FetchM1History(string standardSymbol, string brokerSymbol, int count = 1000)
    {
        if (!_connected || _mt5Ops is null) return Array.Empty<M1Bar>();
        try
        {
            if (!_gil.TryRun(() => FetchM1HistoryCore(brokerSymbol, count, standardSymbol),
                    TimeSpan.FromSeconds(15), out var bars, default))
            {
                _logger.LogWarning("fetch_m1_history 超时 {Broker}", brokerSymbol);
                return Array.Empty<M1Bar>();
            }
            return bars;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "fetch_m1_history 失败 {Broker}", brokerSymbol);
            return Array.Empty<M1Bar>();
        }
    }

    private IReadOnlyList<M1Bar> FetchM1HistoryCore(string brokerSymbol, int count, string standardSymbol)
    {
        dynamic rows = _mt5Ops!.fetch_m1_history(brokerSymbol, count);
        if (rows is null) return Array.Empty<M1Bar>();

        var list = new List<M1Bar>();
        foreach (dynamic row in rows)
        {
            var unix = (long)row["time"];
            list.Add(new M1Bar
            {
                Symbol = standardSymbol,
                Time = Mt5Time.FromUnixUtcSeconds(unix),
                Open = (double)row["open"],
                High = (double)row["high"],
                Low = (double)row["low"],
                Close = (double)row["close"],
                Volume = (double)(int)row["volume"],
            });
        }
        return list;
    }

    private static void EnsurePythonPath()
    {
        dynamic sys = Py.Import("sys");
        foreach (var p in new[] { AppPaths.PythonEngineDir, AppPaths.InstallDir })
        {
            if (Directory.Exists(p))
                sys.path.append(p);
        }
        var root = AppPaths.FindDevRoot();
        if (root is not null)
            sys.path.append(root);
    }

    public void Dispose() => Disconnect();
}

public sealed class Mt5Tick
{
    public double Bid { get; init; }
    public double Ask { get; init; }
    public long Time { get; init; }
}

public sealed class Mt5Position
{
    public long Ticket { get; init; }
    public string Symbol { get; init; } = "";
    public int Type { get; init; }
    public const int TypeBuy = 0;
    public const int TypeSell = 1;
    public double Volume { get; init; }
    public double PriceOpen { get; init; }
    public double PriceCurrent { get; init; }
    public double Sl { get; init; }
    public double Tp { get; init; }
    public string Comment { get; init; } = "";
    public long Time { get; init; }
}
