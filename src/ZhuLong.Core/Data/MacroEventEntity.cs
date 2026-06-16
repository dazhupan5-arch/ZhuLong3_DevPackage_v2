namespace ZhuLong.Core.Data;

/// <summary>经济日历事件 — 来自 Finnhub/FMP/备用源，存 SQLite。</summary>
public sealed class MacroEventEntity
{
    public long Id { get; set; }
    public long EventTimeUnix { get; set; }
    public string EventName { get; set; } = "";
    public string Impact { get; set; } = "medium";
    public string Currency { get; set; } = "USD";
    public string Source { get; set; } = "";
    public long FetchedAtUnix { get; set; }
    public string? ExternalId { get; set; }
}
