using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Data;
using ZhuLong.Core.Macro;

namespace ZhuLong.Core.Services;

/// <summary>
/// 宏观日历服务：启动/定时拉取 REST API → SQLite，合成 8 维特征。
/// </summary>
public sealed class MacroCalendarService : IAsyncDisposable
{
    private readonly MacroCalendarFetcher _fetcher;
    private readonly IDbContextFactory<ZhuLongDbContext> _dbFactory;
    private readonly ILogger<MacroCalendarService> _logger;
    private readonly object _lock = new();
    private List<MacroEventRecord> _cache = [];
    private FredSnapshot? _fred;
    private SentimentSnapshot? _sentiment;
    private AppSettings.MacroSettings _settings = new();
    private CancellationTokenSource? _refreshCts;
    private Task? _refreshLoop;

    public MacroCalendarService(
        MacroCalendarFetcher fetcher,
        IDbContextFactory<ZhuLongDbContext> dbFactory,
        ILogger<MacroCalendarService> logger)
    {
        _fetcher = fetcher;
        _dbFactory = dbFactory;
        _logger = logger;
    }

    public DateTime? LastRefreshUtc { get; private set; }
    public string LastSource { get; private set; } = "none";

    public void Configure(AppSettings settings)
    {
        _settings = settings.Macro ?? new AppSettings.MacroSettings();
        ReloadOfflineJson();
    }

    public void ReloadOfflineJson()
    {
        var macro = _settings;
        var fredPath = ResolvePath(macro.Fred?.JsonPath, AppPaths.FredLatestPath);
        var sentPath = ResolvePath(macro.Sentiment?.JsonPath, AppPaths.SentimentPath);
        lock (_lock)
        {
            _fred = MacroFeatureBuilder.LoadFred(fredPath);
            _sentiment = MacroFeatureBuilder.LoadSentiment(sentPath);
        }
        _logger.LogInformation("宏观离线 JSON 已加载 fred={FredOk} sentiment={SentOk}",
            _fred is not null, _sentiment is not null);
    }

    public async Task InitializeAsync(CancellationToken ct = default)
    {
        MacroBundledDataSync.SyncMacroEventsCsvFromInstall(_logger);
        await EnsureMacroTableAsync(ct);
        await RefreshCalendarAsync(ct);
        StartDailyRefreshLoop();
    }

    public void StartDailyRefreshLoop()
    {
        _refreshCts?.Cancel();
        _refreshCts = new CancellationTokenSource();
        var hours = _settings.ReloadIntervalHours > 0 ? _settings.ReloadIntervalHours : 24;
        _refreshLoop = Task.Run(async () =>
        {
            while (!_refreshCts.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(TimeSpan.FromHours(hours), _refreshCts.Token);
                    ReloadOfflineJson();
                    await RefreshCalendarAsync(_refreshCts.Token);
                }
                catch (OperationCanceledException) { break; }
                catch (Exception ex) { _logger.LogWarning(ex, "宏观定时刷新异常"); }
            }
        });
    }

    public async Task RefreshCalendarAsync(CancellationToken ct = default)
    {
        await EnsureMacroTableAsync(ct);
        if (!_settings.Enabled) return;
        var cal = _settings.Calendar ?? new AppSettings.MacroCalendarSettings();
        var lookahead = cal.LookaheadHours > 0 ? cal.LookaheadHours : 168;
        var from = DateTime.UtcNow.AddHours(-24);
        var to = DateTime.UtcNow.AddHours(lookahead);

        var fetched = await _fetcher.FetchAsync(_settings, from, to, ct);
        if (fetched.Count == 0)
        {
            await LoadFromDbAsync(ct);
            return;
        }

        LastSource = fetched[0].Source;
        await UpsertEventsAsync(fetched, ct);
        lock (_lock) _cache = fetched.OrderBy(e => e.EventTime).ToList();
        LastRefreshUtc = DateTime.UtcNow;
        _logger.LogInformation("宏观日历已刷新 {Count} 条 source={Source}", fetched.Count, LastSource);
    }

    public IReadOnlyList<MacroEventRecord> GetUpcomingEvents(double hoursWindow = 48)
    {
        var now = ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        var end = now.AddHours(hoursWindow);
        lock (_lock)
            return _cache.Where(e => e.EventTime >= now && e.EventTime <= end).ToList();
    }

    public float[] GetFeatures(DateTime? now = null)
    {
        if (!_settings.Enabled) return new float[MacroFeatureBuilder.FeatureDim];
        List<MacroEventRecord> events;
        FredSnapshot? fred;
        SentimentSnapshot? sentiment;
        lock (_lock)
        {
            events = _cache.ToList();
            fred = _fred;
            sentiment = _sentiment;
        }
        return MacroFeatureBuilder.Build(events, fred, sentiment, now);
    }

    public double? GetNextEventHours(DateTime? now = null)
    {
        var next = GetNextHighImpactEvent(now);
        if (next is null) return null;
        now ??= ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        return (next.EventTime - now.Value).TotalHours;
    }

    public MacroEventRecord? GetNextHighImpactEvent(DateTime? now = null)
    {
        now ??= ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        lock (_lock)
        {
            return _cache
                .Where(e => e.EventTime > now && MacroImpactHelper.IsHighImpact(e.Impact))
                .OrderBy(e => e.EventTime)
                .FirstOrDefault();
        }
    }

    public bool IsSilenceWindow(DateTime? now = null)
    {
        if (!_settings.ForceSilence) return false;
        now ??= ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        var beforeH = _settings.SilenceBeforeMinutes / 60.0;
        var afterH = _settings.SilenceAfterMinutes / 60.0;
        var keywords = _settings.ForceSilenceEvents ?? [];

        lock (_lock)
        {
            foreach (var e in _cache)
            {
                if (!keywords.Any(k => e.EventName.Contains(k, StringComparison.OrdinalIgnoreCase)))
                    continue;
                var delta = (e.EventTime - now.Value).TotalHours;
                if (delta >= -afterH && delta <= beforeH) return true;
            }
        }
        return false;
    }

    private async Task EnsureMacroTableAsync(CancellationToken ct)
    {
        await using var db = await _dbFactory.CreateDbContextAsync(ct);
        await MacroEventsSchemaMigrator.EnsureReadyAsync(db, _logger, ct);
    }

    private async Task UpsertEventsAsync(IReadOnlyList<MacroEventRecord> events, CancellationToken ct)
    {
        var fetchedAt = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        await using var db = await _dbFactory.CreateDbContextAsync(ct);
        // 窗口内全量替换，避免重复
        var minUnix = events.Min(e => new DateTimeOffset(e.EventTime).ToUnixTimeSeconds());
        var maxUnix = events.Max(e => new DateTimeOffset(e.EventTime).ToUnixTimeSeconds());
        await db.MacroEvents
            .Where(e => e.EventTimeUnix >= minUnix && e.EventTimeUnix <= maxUnix)
            .ExecuteDeleteAsync(ct);

        foreach (var e in events)
        {
            db.MacroEvents.Add(new MacroEventEntity
            {
                EventTimeUnix = ToBeijingUnix(e.EventTime),
                EventName = e.EventName,
                Impact = e.Impact,
                Currency = e.Currency,
                Source = e.Source,
                FetchedAtUnix = fetchedAt,
                ExternalId = e.ExternalId,
            });
        }
        await db.SaveChangesAsync(ct);
    }

    private async Task LoadFromDbAsync(CancellationToken ct)
    {
        await using var db = await _dbFactory.CreateDbContextAsync(ct);
        var nowUnix = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var window = (_settings.Calendar?.LookaheadHours ?? 168) * 3600L;
        var rows = await db.MacroEvents
            .Where(e => e.EventTimeUnix >= nowUnix - 86400 && e.EventTimeUnix <= nowUnix + window)
            .OrderBy(e => e.EventTimeUnix)
            .ToListAsync(ct);

        lock (_lock)
        {
            _cache = rows.Select(r => new MacroEventRecord(
                ChinaTime.ToBeijing(DateTimeOffset.FromUnixTimeSeconds(r.EventTimeUnix)).DateTime,
                r.EventName,
                r.Impact,
                r.Currency,
                r.Source,
                r.ExternalId)).ToList();
        }
        if (_cache.Count > 0)
            LastSource = "sqlite";
        _logger.LogInformation("从 SQLite 加载宏观事件 {Count} 条", _cache.Count);
    }

    private static string ResolvePath(string? configured, string defaultPath) =>
        string.IsNullOrWhiteSpace(configured)
            ? defaultPath
            : Path.IsPathRooted(configured)
                ? configured
                : Path.Combine(AppPaths.WritableDataDir, Path.GetFileName(configured.Replace('/', Path.DirectorySeparatorChar)));

    private static long ToBeijingUnix(DateTime beijingWallClock)
    {
        var offset = ChinaTime.Zone.GetUtcOffset(
            DateTime.SpecifyKind(beijingWallClock, DateTimeKind.Unspecified));
        return new DateTimeOffset(beijingWallClock, offset).ToUnixTimeSeconds();
    }

    /// <summary>距下一高影响事件 &lt;= hours 时写一条运行日志（每事件仅提醒一次）。</summary>
    public void TryLogUpcomingReminder(Action<string> log, double hours = 12.0)
    {
        var evt = GetNextHighImpactEvent();
        if (evt is null) return;
        var until = evt.EventTime - ChinaTime.ToBeijing(DateTimeOffset.UtcNow).DateTime;
        if (until.TotalHours > hours || until.TotalHours < 0) return;
        var key = $"{evt.EventName}|{evt.EventTime:yyyyMMddHHmm}";
        lock (_lock)
        {
            if (_remindedKeys.Contains(key)) return;
            _remindedKeys.Add(key);
        }
        log($"宏观重要事件提醒：{evt.EventName}（{evt.Currency}）北京时间 {evt.EventTime:yyyy-MM-dd HH:mm}，约 {FormatHours(until.TotalHours)} 后发布；静默窗口内将暂停新开仓信号");
    }

    private readonly HashSet<string> _remindedKeys = new(StringComparer.Ordinal);

    private static string FormatHours(double h)
    {
        if (h < 1) return $"{(int)(h * 60)} 分钟";
        if (h < 24) return $"{h:F1} 小时";
        return $"{h / 24:F1} 天";
    }

    public async ValueTask DisposeAsync()
    {
        _refreshCts?.Cancel();
        if (_refreshLoop is not null)
            try { await _refreshLoop; } catch { /* ignore */ }
        _refreshCts?.Dispose();
    }
}
