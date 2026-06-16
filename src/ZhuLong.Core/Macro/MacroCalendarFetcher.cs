using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;

namespace ZhuLong.Core.Macro;

/// <summary>经济日历 REST 拉取：Finnhub → FMP → Investing 备用 → CSV 合并。</summary>
public sealed class MacroCalendarFetcher
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(30) };

    private readonly IApiSecrets _secrets;
    private readonly ILogger<MacroCalendarFetcher> _logger;

    public MacroCalendarFetcher(IApiSecrets secrets, ILogger<MacroCalendarFetcher> logger)
    {
        _secrets = secrets;
        _logger = logger;
    }

    public async Task<IReadOnlyList<MacroEventRecord>> FetchAsync(
        AppSettings.MacroSettings settings,
        DateTime fromUtc,
        DateTime toUtc,
        CancellationToken ct = default)
    {
        var cal = settings.Calendar ?? new AppSettings.MacroCalendarSettings();
        var provider = (cal.Provider ?? "auto").ToLowerInvariant();

        if (provider is "finnhub" or "auto")
        {
            var events = await TryFinnhubAsync(fromUtc, toUtc, ct);
            if (events.Count > 0)
                return MergeWithCsv(events, fromUtc, toUtc);
        }

        if (provider is "fmp" or "auto")
        {
            var events = await TryFmpAsync(fromUtc, toUtc, ct);
            if (events.Count > 0)
                return MergeWithCsv(events, fromUtc, toUtc);
        }

        if (cal.FallbackHtmlEnabled)
        {
            var events = await TryInvestingFallbackAsync(fromUtc, toUtc, ct);
            if (events.Count > 0)
                return MergeWithCsv(events, fromUtc, toUtc);
        }

        var csv = LoadCsvFallback();
        if (csv.Count > 0)
        {
            _logger.LogWarning("经济日历 API 不可用，已回退 macro_events.csv ({Count} 条)", csv.Count);
            return FilterWindow(csv, fromUtc, toUtc);
        }

        _logger.LogError("经济日历全部来源失败");
        return Array.Empty<MacroEventRecord>();
    }

    private async Task<List<MacroEventRecord>> TryFinnhubAsync(DateTime fromUtc, DateTime toUtc, CancellationToken ct)
    {
        var key = _secrets.ResolveFinnhubApiKey();
        if (string.IsNullOrWhiteSpace(key)) return [];

        var url =
            $"https://finnhub.io/api/v1/calendar/economic?from={fromUtc:yyyy-MM-dd}&to={toUtc:yyyy-MM-dd}&token={Uri.EscapeDataString(key)}";
        try
        {
            using var resp = await Http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
            {
                _logger.LogWarning("Finnhub 日历 HTTP {Code}", (int)resp.StatusCode);
                return [];
            }

            await using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
            if (!doc.RootElement.TryGetProperty("economicCalendar", out var arr) ||
                arr.ValueKind != JsonValueKind.Array)
                return [];

            var list = new List<MacroEventRecord>();
            foreach (var item in arr.EnumerateArray())
            {
                var name = item.TryGetProperty("event", out var ev) ? ev.GetString() ?? "" : "";
                if (string.IsNullOrWhiteSpace(name)) continue;
                var impact = ParseImpactField(item, name);
                var country = item.TryGetProperty("country", out var c) ? c.GetString() ?? "USD" : "USD";
                var time = ParseFinnhubEventTime(item);
                if (time is null) continue;
                list.Add(new MacroEventRecord(time.Value, name, impact, MapCountryToCurrency(country), "finnhub"));
            }

            _logger.LogInformation("Finnhub 日历拉取 {Count} 条（UTC→本地）", list.Count);
            return list;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Finnhub 日历异常");
            return [];
        }
    }

    private async Task<List<MacroEventRecord>> TryFmpAsync(DateTime fromUtc, DateTime toUtc, CancellationToken ct)
    {
        var key = _secrets.ResolveFmpApiKey();
        if (string.IsNullOrWhiteSpace(key)) return [];

        var url =
            $"https://financialmodelingprep.com/api/v3/economic_calendar?from={fromUtc:yyyy-MM-dd}&to={toUtc:yyyy-MM-dd}&apikey={Uri.EscapeDataString(key)}";
        try
        {
            using var resp = await Http.GetAsync(url, ct);
            if (!resp.IsSuccessStatusCode)
            {
                _logger.LogWarning("FMP 日历 HTTP {Code}", (int)resp.StatusCode);
                return [];
            }

            await using var stream = await resp.Content.ReadAsStreamAsync(ct);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: ct);
            if (doc.RootElement.ValueKind != JsonValueKind.Array) return [];

            var list = new List<MacroEventRecord>();
            foreach (var item in doc.RootElement.EnumerateArray())
            {
                var name = item.TryGetProperty("event", out var ev) ? ev.GetString() ?? "" : "";
                if (string.IsNullOrWhiteSpace(name)) continue;
                var impact = ParseImpactField(item, name);
                var country = item.TryGetProperty("country", out var c) ? c.GetString() ?? "US" : "US";
                if (!item.TryGetProperty("date", out var dtEl)) continue;
                var time = MacroEventTime.ParseApiUtc(dtEl.GetString());
                if (time is null) continue;
                list.Add(new MacroEventRecord(time.Value, name, impact, MapCountryToCurrency(country), "fmp"));
            }

            _logger.LogInformation("FMP 日历拉取 {Count} 条（UTC→本地）", list.Count);
            return list;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "FMP 日历异常");
            return [];
        }
    }

    private async Task<List<MacroEventRecord>> TryInvestingFallbackAsync(
        DateTime fromUtc, DateTime toUtc, CancellationToken ct)
    {
        var url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData";
        try
        {
            using var req = new HttpRequestMessage(HttpMethod.Post, url);
            req.Headers.TryAddWithoutValidation("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36");
            req.Headers.TryAddWithoutValidation("X-Requested-With", "XMLHttpRequest");
            req.Headers.Referrer = new Uri("https://www.investing.com/economic-calendar/");
            var body = $"dateFrom={fromUtc:yyyy-MM-dd}&dateTo={toUtc:yyyy-MM-dd}&country=5&timeZone=8&timeFilter=timeRemain&currentTab=custom&limit_from=0";
            req.Content = new StringContent(body, System.Text.Encoding.UTF8, "application/x-www-form-urlencoded");

            using var resp = await Http.SendAsync(req, ct);
            if (!resp.IsSuccessStatusCode)
            {
                _logger.LogWarning("Investing 备用 HTTP {Code}", (int)resp.StatusCode);
                return [];
            }

            return ParseInvestingHtml(await resp.Content.ReadAsStringAsync(ct), fromUtc, toUtc);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Investing 备用日历异常");
            return [];
        }
    }

    private List<MacroEventRecord> ParseInvestingHtml(string html, DateTime fromUtc, DateTime toUtc)
    {
        var list = new List<MacroEventRecord>();
        var pattern = new Regex(
            @"data-event-datetime=""([^""]+)""[^>]*>.*?class=""[^""]*event[^""]*""[^>]*>([^<]+)",
            RegexOptions.Singleline | RegexOptions.IgnoreCase);
        var fromLocal = ChinaTime.ToBeijing(new DateTimeOffset(fromUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;
        var toLocal = ChinaTime.ToBeijing(new DateTimeOffset(toUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;
        foreach (Match m in pattern.Matches(html))
        {
            var rawTime = m.Groups[1].Value.Replace('/', '-');
            var dt = MacroEventTime.ParseLocalCsv(rawTime);
            if (dt is null || dt < fromLocal || dt > toLocal) continue;
            var name = System.Net.WebUtility.HtmlDecode(m.Groups[2].Value.Trim());
            if (string.IsNullOrWhiteSpace(name)) continue;
            var impact = MacroImpactHelper.GuessImpactFromName(name);
            list.Add(new MacroEventRecord(dt.Value, name, impact, "USD", "investing"));
        }

        _logger.LogInformation("Investing 备用解析 {Count} 条", list.Count);
        return list;
    }

    internal static List<MacroEventRecord> LoadCsvFallback()
    {
        var path = AppPaths.MacroEventsPath;
        if (!File.Exists(path)) return [];
        var list = new List<MacroEventRecord>();
        foreach (var line in File.ReadAllLines(path).Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var parts = line.Split(',');
            if (parts.Length < 4) continue;
            var dt = MacroEventTime.ParseLocalCsv(parts[0]);
            if (dt is null) continue;
            var impact = MacroImpactHelper.Normalize(parts[2].Trim(), parts[1].Trim());
            list.Add(new MacroEventRecord(dt.Value, parts[1].Trim(), impact, parts[3].Trim(), "csv"));
        }
        return list;
    }

    public static List<MacroEventRecord> MergeWithCsv(
        IReadOnlyList<MacroEventRecord> api,
        DateTime fromUtc,
        DateTime toUtc,
        IReadOnlyList<MacroEventRecord>? csvOverride = null)
    {
        var csv = csvOverride ?? LoadCsvFallback();
        if (csv.Count == 0) return api.OrderBy(e => e.EventTime).ToList();

        var merged = api.ToList();
        var fromLocal = ChinaTime.ToBeijing(new DateTimeOffset(fromUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;
        var toLocal = ChinaTime.ToBeijing(new DateTimeOffset(toUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;

        foreach (var c in csv)
        {
            if (c.EventTime < fromLocal || c.EventTime > toLocal) continue;
            if (!MacroImpactHelper.IsTier1Event(c.EventName)) continue;

            merged.RemoveAll(e => MacroImpactHelper.SameEventFamily(e.EventName, c.EventName));
            merged.Add(c);
        }

        return merged.OrderBy(e => e.EventTime).ToList();
    }

    private static List<MacroEventRecord> FilterWindow(
        IReadOnlyList<MacroEventRecord> events,
        DateTime fromUtc,
        DateTime toUtc)
    {
        var fromLocal = ChinaTime.ToBeijing(new DateTimeOffset(fromUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;
        var toLocal = ChinaTime.ToBeijing(new DateTimeOffset(toUtc.ToUniversalTime(), TimeSpan.Zero)).DateTime;
        return events.Where(e => e.EventTime >= fromLocal && e.EventTime <= toLocal)
            .OrderBy(e => e.EventTime)
            .ToList();
    }

    private static DateTime? ParseFinnhubEventTime(JsonElement item)
    {
        if (item.TryGetProperty("time", out var t))
        {
            var parsed = MacroEventTime.ParseApiUtc(t.GetString());
            if (parsed is not null) return parsed;
        }
        if (item.TryGetProperty("date", out var d))
            return MacroEventTime.ParseApiUtc(d.GetString());
        return null;
    }

    private static string ParseImpactField(JsonElement item, string eventName)
    {
        if (item.TryGetProperty("impact", out var im))
        {
            if (im.ValueKind == JsonValueKind.Number && im.TryGetInt32(out var n))
                return MacroImpactHelper.Normalize(n.ToString(), eventName);
            if (im.ValueKind == JsonValueKind.String)
                return MacroImpactHelper.Normalize(im.GetString(), eventName);
        }
        return MacroImpactHelper.Normalize("medium", eventName);
    }

    private static string MapCountryToCurrency(string country) => country.ToUpperInvariant() switch
    {
        "US" or "USA" => "USD",
        "EU" or "EZ" or "DE" => "EUR",
        "GB" or "UK" => "GBP",
        "JP" => "JPY",
        "CN" => "CNY",
        _ => country.Length == 3 ? country : "USD",
    };
}
