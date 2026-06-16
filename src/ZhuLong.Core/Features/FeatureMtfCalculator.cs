using ZhuLong.Core.Models;

namespace ZhuLong.Core.Features;

/// <summary>M15/H1 多周期趋势特征（对齐 Python compute_mtf_trend_features）。</summary>
public static class FeatureMtfCalculator
{
    public const int MtfDim = 6;

    public static float[,] Compute(IReadOnlyList<OhlcBar> m5)
    {
        var n = m5.Count;
        var f = new float[n, MtfDim];
        if (n < 10) return f;

        var m15 = Resample(m5, 15);
        var h1 = Resample(m5, 60);
        if (m15.Count < 10 || h1.Count < 10) return f;

        var m15Closes = m15.Select(b => b.Close).ToArray();
        var m15Ema = Ema(m15Closes, 20);
        var m15Bias = m15.Select((b, i) => (b.Close - m15Ema[i]) / Math.Max(m15Ema[i], 1e-9)).ToArray();
        var m15Trend = new double[m15.Count];
        for (var i = 3; i < m15.Count; i++)
            m15Trend[i] = Math.Sign(m15Closes[i] - m15Closes[i - 3]);

        var h1Closes = h1.Select(b => b.Close).ToArray();
        var h1Rsi = Rsi(h1Closes, 14).Select(v => v / 100.0).ToArray();
        var ema12 = Ema(h1Closes, 12);
        var ema26 = Ema(h1Closes, 26);
        var h1MacdSign = ema12.Zip(ema26, (a, b) => (double)Math.Sign(a - b)).ToArray();
        var h1Ema60 = Ema(h1Closes, 60);
        var h1Bias = h1.Select((b, i) => (b.Close - h1Ema60[i]) / Math.Max(h1Ema60[i], 1e-9)).ToArray();
        var h1Atr = Atr(h1, 14);
        var h1AtrPct = h1.Select((b, i) => h1Atr[i] / Math.Max(b.Close, 1e-9)).ToArray();

        for (var i = 0; i < n; i++)
        {
            var t = m5[i].Time;
            f[i, 0] = (float)Align(m15, m15Bias, t);
            f[i, 1] = (float)Align(m15, m15Trend, t);
            f[i, 2] = (float)Align(h1, h1Rsi, t);
            f[i, 3] = (float)Align(h1, h1MacdSign, t);
            f[i, 4] = (float)Align(h1, h1Bias, t);
            f[i, 5] = (float)Align(h1, h1AtrPct, t);
        }

        return f;
    }

    private static double Align(IReadOnlyList<OhlcBar> bars, IReadOnlyList<double> values, DateTime t)
    {
        var idx = bars.ToList().FindLastIndex(b => b.Time <= t);
        return idx >= 0 ? values[idx] : 0;
    }

    private static List<OhlcBar> Resample(IReadOnlyList<OhlcBar> m5, int minutes)
    {
        var buckets = new Dictionary<DateTime, OhlcBar>();
        foreach (var b in m5)
        {
            var bucket = new DateTime(b.Time.Year, b.Time.Month, b.Time.Day, b.Time.Hour,
                b.Time.Minute / minutes * minutes, 0);
            if (!buckets.TryGetValue(bucket, out var o))
                buckets[bucket] = b with { Time = bucket };
            else
                buckets[bucket] = o with
                {
                    High = Math.Max(o.High, b.High),
                    Low = Math.Min(o.Low, b.Low),
                    Close = b.Close,
                    Volume = o.Volume + b.Volume,
                };
        }
        return buckets.Values.OrderBy(x => x.Time).ToList();
    }

    private static double[] Ema(IReadOnlyList<double> values, int period)
    {
        var result = new double[values.Count];
        if (values.Count == 0) return result;
        var k = 2.0 / (period + 1);
        result[0] = values[0];
        for (var i = 1; i < values.Count; i++)
            result[i] = values[i] * k + result[i - 1] * (1 - k);
        return result;
    }

    private static double[] Rsi(IReadOnlyList<double> closes, int period)
    {
        var result = new double[closes.Count];
        for (var i = 0; i < closes.Count; i++)
        {
            if (i < period) { result[i] = 50; continue; }
            double gain = 0, loss = 0;
            for (var j = i - period + 1; j <= i; j++)
            {
                var d = closes[j] - closes[j - 1];
                if (d > 0) gain += d;
                else loss -= d;
            }
            gain /= period;
            loss /= period;
            result[i] = loss < 1e-12 ? 100 : 100 - 100 / (1 + gain / loss);
        }
        return result;
    }

    private static double[] Atr(IReadOnlyList<OhlcBar> bars, int period)
    {
        var tr = new double[bars.Count];
        for (var i = 0; i < bars.Count; i++)
        {
            var prev = i > 0 ? bars[i - 1].Close : bars[i].Close;
            tr[i] = Math.Max(bars[i].High - bars[i].Low,
                Math.Max(Math.Abs(bars[i].High - prev), Math.Abs(bars[i].Low - prev)));
        }
        return Ema(tr, period);
    }
}
