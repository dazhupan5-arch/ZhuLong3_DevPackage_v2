using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

/// <summary>各品种最近一次推理结果（模型辅助出场）。</summary>
public sealed class InferenceSnapshotStore
{
    private readonly object _lock = new();
    private readonly Dictionary<string, InferenceResult> _last = new(StringComparer.OrdinalIgnoreCase);

    public void Set(string symbol, InferenceResult result)
    {
        lock (_lock) _last[symbol] = result;
    }

    public bool TryGet(string symbol, out InferenceResult result)
    {
        lock (_lock) return _last.TryGetValue(symbol, out result!);
    }

    public bool IsEmpty
    {
        get { lock (_lock) return _last.Count == 0; }
    }
}
