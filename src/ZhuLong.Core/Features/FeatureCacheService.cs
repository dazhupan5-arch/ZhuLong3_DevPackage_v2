using ZhuLong.Core;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Features;

/// <summary>M1 缓冲 → M5 合成 → 序列特征（对齐 Python feature_engine）。</summary>
public sealed class FeatureCacheService
{
    private readonly Dictionary<string, List<M1Bar>> _m1 = new();
    private readonly Dictionary<string, List<OhlcBar>> _m5 = new();
    private readonly Dictionary<string, DateTime> _lastM5Bucket = new();
    private readonly Dictionary<string, List<M1Bar>> _historyPending = new();
    private readonly object _lock = new();

    public event Action<string, OhlcBar>? M5BarCompleted;

    public int GetM1Count(string symbol)
    {
        lock (_lock)
            return _m1.TryGetValue(symbol, out var list) ? list.Count : 0;
    }

    public int GetM5Count(string symbol)
    {
        lock (_lock)
            return _m5.TryGetValue(symbol, out var m5) ? m5.Count : 0;
    }

    public bool TryGetLatestM1Time(string symbol, out DateTime time)
    {
        lock (_lock)
        {
            if (_m1.TryGetValue(symbol, out var list) && list.Count > 0)
            {
                time = list[^1].Time;
                return true;
            }
        }
        time = default;
        return false;
    }

    public IReadOnlyList<string> GetTrackedSymbols()
    {
        lock (_lock)
            return _m1.Keys.ToList();
    }

    /// <summary>broker 符号历史合并到标准符号（如 XAUUSDm → XAUUSD）。</summary>
    public void RekeySymbol(string from, string to)
    {
        if (string.Equals(from, to, StringComparison.OrdinalIgnoreCase))
            return;

        lock (_lock)
        {
            if (_m1.TryGetValue(from, out var m1From))
            {
                if (!_m1.TryGetValue(to, out var m1To))
                    _m1[to] = m1From;
                else
                {
                    m1To.AddRange(m1From);
                    _m1[to] = m1To
                        .GroupBy(b => b.Time)
                        .Select(g => g.Last())
                        .OrderBy(b => b.Time)
                        .ToList();
                }
                _m1.Remove(from);
            }

            _m5.Remove(from);
            _lastM5Bucket.Remove(from);
            _historyPending.Remove(from);
            if (_m1.ContainsKey(to))
                RebuildM5(to, fireCompleted: false);
        }
    }

    /// <summary>MT5 启动时分批推送的历史 M1；<paramref name="final"/>=true 时合并并重建 M5。</summary>
    public void AppendHistoryChunk(string symbol, IReadOnlyList<M1Bar> bars, bool final)
    {
        if (bars.Count == 0 && !final) return;

        lock (_lock)
        {
            if (!_historyPending.TryGetValue(symbol, out var acc))
            {
                acc = new List<M1Bar>();
                _historyPending[symbol] = acc;
            }
            acc.AddRange(bars);

            if (!final) return;

            var ordered = acc
                .GroupBy(b => b.Time)
                .Select(g => g.Last())
                .OrderBy(b => b.Time)
                .ToList();
            _historyPending.Remove(symbol);

            if (_m1.TryGetValue(symbol, out var existing) && existing.Count > 0)
            {
                existing.AddRange(ordered);
                ordered = existing
                    .GroupBy(b => b.Time)
                    .Select(g => g.Last())
                    .OrderBy(b => b.Time)
                    .ToList();
            }

            if (ordered.Count > 5000)
                ordered = ordered[^5000..];

            _m1[symbol] = ordered;
            RebuildM5(symbol, fireCompleted: false);
        }
    }

    public void Ingest(M1Bar bar)
    {
        lock (_lock)
        {
            if (!_m1.TryGetValue(bar.Symbol, out var list))
            {
                list = new List<M1Bar>();
                _m1[bar.Symbol] = list;
            }

            var idx = list.FindIndex(b => b.Time == bar.Time);
            if (idx >= 0)
                list[idx] = bar;
            else
                list.Add(bar);

            if (list.Count > 5000) list.RemoveRange(0, list.Count - 5000);

            RebuildM5(bar.Symbol, fireCompleted: true);
        }
    }

    private void RebuildM5(string symbol, bool fireCompleted)
    {
        if (!_m1.TryGetValue(symbol, out var list)) return;

        var prevCount = _m5.TryGetValue(symbol, out var prevM5) ? prevM5.Count : 0;
        var m5 = ResampleM5(list);
        _m5[symbol] = m5;

        if (!fireCompleted || m5.Count <= prevCount || m5.Count == 0) return;

        // 新 M5 桶刚创建时，上一根柱子才是刚刚完成的 ★修复★
        var completed = m5.Count > 1 ? m5[^2] : m5[^1];
        var bucket = completed.Time;
        if (!_lastM5Bucket.TryGetValue(symbol, out var last) || last != bucket)
        {
            _lastM5Bucket[symbol] = bucket;
            M5BarCompleted?.Invoke(symbol, completed);
        }
    }

    public bool TryGetSequence(string symbol, int seqLen, out float[,] seq, out float[] hourly, out double atrPct)
    {
        seq = new float[0, 0];
        hourly = Array.Empty<float>();
        atrPct = 0;
        lock (_lock)
        {
            if (!_m5.TryGetValue(symbol, out var m5) || m5.Count < seqLen + 20) return false;
            var feats = FeatureCalculator.ComputeM5Features(m5);
            var mtf = FeatureMtfCalculator.Compute(m5);
            if (feats.GetLength(0) < seqLen) return false;
            var rows = feats.GetLength(0);
            var baseSeq = new float[seqLen, feats.GetLength(1)];
            var mtfSeq = new float[seqLen, FeatureMtfCalculator.MtfDim];
            for (var i = 0; i < seqLen; i++)
            {
                var src = rows - seqLen + i;
                for (var j = 0; j < feats.GetLength(1); j++)
                    baseSeq[i, j] = feats[src, j];
                for (var j = 0; j < FeatureMtfCalculator.MtfDim; j++)
                    mtfSeq[i, j] = mtf[src, j];
            }
            seq = FeaturePad.ToModelDim(baseSeq, mtfSeq);
            hourly = FeatureCalculator.ComputeHourlyBackground(m5);
            atrPct = FeatureCalculator.CurrentAtrPct(m5);
            return true;
        }
    }

    public bool TryGetLatestClose(string symbol, out double close)
    {
        close = 0;
        lock (_lock)
        {
            if (!_m5.TryGetValue(symbol, out var m5) || m5.Count == 0) return false;
            close = m5[^1].Close;
            return true;
        }
    }

    public bool TryGetM5Bars(string symbol, out IReadOnlyList<OhlcBar> bars)
    {
        lock (_lock)
        {
            if (_m5.TryGetValue(symbol, out var m5) && m5.Count > 0)
            {
                bars = m5;
                return true;
            }
        }
        bars = Array.Empty<OhlcBar>();
        return false;
    }

    public bool TryGetCurrentAtr(string symbol, out double atr, int period = 14)
    {
        atr = 0;
        lock (_lock)
        {
            if (!_m5.TryGetValue(symbol, out var m5) || m5.Count == 0) return false;
            atr = FeatureCalculator.CurrentAtr(m5, period);
            return atr > 0;
        }
    }

    /// <summary>导出 M5 OHLCV 供 Python V14/智能体特征（避免 Python 侧再调 MT5）。</summary>
    public bool TryExportM5Bars(string symbol, out (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[] bars)
    {
        bars = [];
        lock (_lock)
        {
            if (!_m5.TryGetValue(symbol, out var m5) || m5.Count == 0) return false;
            bars = ExportM5Array(m5);
            return bars.Length > 0;
        }
    }

    /// <summary>智能体推理：仅导出已闭合 M5（去掉形成中最后一根），并返回决策 bar 时间戳。</summary>
    public bool TryExportAgentM5Bars(string symbol, out (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[] bars, out long decisionBarUnix)
    {
        bars = [];
        decisionBarUnix = 0;
        lock (_lock)
        {
            if (!_m5.TryGetValue(symbol, out var m5) || m5.Count == 0) return false;
            var export = m5.Count >= 2 ? m5.Take(m5.Count - 1).ToList() : m5;
            if (export.Count == 0) return false;
            bars = ExportM5Array(export);
            if (bars.Length == 0) return false;
            decisionBarUnix = bars[^1].TimeUnix;
            return true;
        }
    }

    private static (long TimeUnix, double Open, double High, double Low, double Close, double Volume)[] ExportM5Array(
        IReadOnlyList<OhlcBar> m5)
    {
        return m5.Select(b =>
        {
            var utc = b.Time.Kind == DateTimeKind.Unspecified
                ? TimeZoneInfo.ConvertTimeToUtc(b.Time, ChinaTime.Zone)
                : b.Time.ToUniversalTime();
            return (
                new DateTimeOffset(utc).ToUnixTimeSeconds(),
                b.Open,
                b.High,
                b.Low,
                b.Close,
                b.Volume);
        }).ToArray();
    }

    private static List<OhlcBar> ResampleM5(List<M1Bar> m1)
    {
        if (m1.Count == 0) return [];
        var ordered = m1.OrderBy(b => b.Time).ToList();
        var buckets = new Dictionary<DateTime, OhlcBar>();
        foreach (var b in ordered)
        {
            var bucket = new DateTime(b.Time.Year, b.Time.Month, b.Time.Day, b.Time.Hour, b.Time.Minute / 5 * 5, 0);
            if (!buckets.TryGetValue(bucket, out var o))
            {
                buckets[bucket] = new OhlcBar
                {
                    Time = bucket, Open = b.Open, High = b.High, Low = b.Low, Close = b.Close, Volume = b.Volume,
                };
            }
            else
            {
                buckets[bucket] = o with
                {
                    High = Math.Max(o.High, b.High),
                    Low = Math.Min(o.Low, b.Low),
                    Close = b.Close,
                    Volume = o.Volume + b.Volume,
                };
            }
        }
        return buckets.Values.OrderBy(x => x.Time).ToList();
    }
}
