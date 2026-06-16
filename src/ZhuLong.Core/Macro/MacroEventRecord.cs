namespace ZhuLong.Core.Macro;

public sealed record MacroEventRecord(
    DateTime EventTime,
    string EventName,
    string Impact,
    string Currency,
    string Source = "",
    string? ExternalId = null);
