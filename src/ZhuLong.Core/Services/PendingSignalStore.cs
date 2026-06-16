using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

/// <summary>线程安全 pending 信号队列（G10）。</summary>
public sealed class PendingSignalStore
{
    private readonly object _lock = new();
    private readonly List<SignalModel> _items = [];

    public int Count
    {
        get { lock (_lock) return _items.Count; }
    }

    public void Add(SignalModel signal)
    {
        lock (_lock)
        {
            if (_items.Any(s => s.SignalId == signal.SignalId))
                return;
            _items.Add(signal);
        }
    }

    public bool Contains(string signalId)
    {
        lock (_lock) return _items.Any(s => s.SignalId == signalId);
    }

    public bool Remove(string signalId)
    {
        lock (_lock)
        {
            var idx = _items.FindIndex(s => s.SignalId == signalId);
            if (idx < 0) return false;
            _items.RemoveAt(idx);
            return true;
        }
    }

    /// <summary>将过期信号标记为 "expired"（保留供人工查看，每日清理时才删除）。</summary>
    public IReadOnlyList<SignalModel> MarkExpired(int expiryMinutes)
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var maxAge = expiryMinutes * 60L;
        lock (_lock)
        {
            var expired = _items.Where(s => s.Status == "pending" && now - s.CreatedAt > maxAge).ToList();
            foreach (var e in expired)
            {
                e.Status = "expired";
                e.CloseReason = $"超过 {expiryMinutes / 60} 小时未成交";
            }
            return expired;
        }
    }

    public bool UpdateStatus(string signalId, string status, string reason)
    {
        lock (_lock)
        {
            var idx = _items.FindIndex(s => s.SignalId == signalId);
            if (idx < 0) return false;
            var item = _items[idx];
            item.Status = status;
            item.CloseReason = reason;
            return true;
        }
    }

    public List<SignalModel> Snapshot()
    {
        lock (_lock) return _items.ToList();
    }

    /// <summary>移除所有非 pending 且超过 maxAgeDays 天的已完成信号。</summary>
    public List<SignalModel> RemoveStaleClosed(int maxAgeDays)
    {
        var cutoff = DateTimeOffset.UtcNow.ToUnixTimeSeconds() - maxAgeDays * 86400L;
        lock (_lock)
        {
            var stale = _items.Where(s => s.Status != "pending" && s.CreatedAt < cutoff).ToList();
            foreach (var e in stale)
                _items.RemoveAll(x => x.SignalId == e.SignalId);
            return stale;
        }
    }

    public object Lock => _lock;
}
