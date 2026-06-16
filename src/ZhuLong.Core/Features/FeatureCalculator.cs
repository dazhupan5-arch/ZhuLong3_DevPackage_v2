using ZhuLong.Core.Models;

namespace ZhuLong.Core.Features;

public static class FeatureCalculator
{
    public const int FeatureDim = 22;

    public static float[,] ComputeM5Features(IReadOnlyList<OhlcBar> m5)
    {
        var n = m5.Count;
        var f = new float[n, FeatureDim];
        var closes = m5.Select(b => b.Close).ToArray();
        var volumes = m5.Select(b => b.Volume).ToArray();
        var ema30 = Ema(closes, 30);
        var ema60 = Ema(closes, 60);
        var ema12 = Ema(closes, 12);
        var ema26 = Ema(closes, 26);
        var atr = Atr(m5, 14);
        var rsi = Rsi(closes, 14);
        var macdLine = ema12.Zip(ema26, (a, b) => a - b).ToArray();
        var macdSignal = Ema(macdLine, 9);
        var volMa5 = RollingMean(volumes, 5);

        for (var i = 0; i < n; i++)
        {
            var b = m5[i];
            var o = b.Open;
            var hl = Math.Max(b.High - b.Low, 1e-9);
            f[i, 0] = (float)((b.Close - o) / Math.Max(o, 1e-9));
            f[i, 1] = (float)((b.High - b.Low) / Math.Max(o, 1e-9));
            f[i, 2] = (float)((b.Close - b.Low) / hl);
            f[i, 3] = volMa5[i] > 1e-9 ? (float)(volumes[i] / volMa5[i]) : 1f;
            f[i, 4] = (float)(rsi[i] / 100.0);
            f[i, 5] = (float)(macdLine[i] / Math.Max(b.Close, 1e-9));
            f[i, 6] = (float)(macdSignal[i] / Math.Max(b.Close, 1e-9));
            f[i, 7] = (float)((macdLine[i] - macdSignal[i]) / Math.Max(b.Close, 1e-9));
            f[i, 8] = (float)(atr[i] / Math.Max(b.Close, 1e-9));
            var upper = ema30[i] + 3 * atr[i];
            var lower = ema30[i] - 3 * atr[i];
            var width = Math.Max(upper - lower, 1e-9);
            f[i, 9] = (float)(width / Math.Max(b.Close, 1e-9));
            f[i, 10] = (float)((b.Close - lower) / width);
            f[i, 11] = b.Close > upper ? 1f : 0f;
            f[i, 12] = b.Close < lower ? 1f : 0f;
            f[i, 13] = (float)((ema30[i] - ema60[i]) / Math.Max(ema60[i], 1e-9));
            f[i, 14] = (float)((b.Close - ema30[i]) / Math.Max(ema30[i], 1e-9));
            f[i, 15] = (float)((b.Close - ema60[i]) / Math.Max(ema60[i], 1e-9));
            f[i, 16] = i >= 5
                ? (float)((ema30[i] - ema30[i - 5]) / Math.Max(ema30[i - 5], 1e-9))
                : 0f;
            if (i > 0)
            {
                var prev = Math.Sign(ema30[i - 1] - ema60[i - 1]);
                var curr = Math.Sign(ema30[i] - ema60[i]);
                f[i, 17] = (float)Math.Clamp(curr - prev, -1, 1);
            }
            f[i, 18] = (float)Math.Sin(2 * Math.PI * b.Time.Hour / 24.0);
            f[i, 19] = (float)Math.Cos(2 * Math.PI * b.Time.Hour / 24.0);
            f[i, 20] = (float)Math.Sin(2 * Math.PI * (int)b.Time.DayOfWeek / 7.0);
            f[i, 21] = (float)Math.Cos(2 * Math.PI * (int)b.Time.DayOfWeek / 7.0);
        }
        return f;
    }

    public static float[] ComputeHourlyBackground(IReadOnlyList<OhlcBar> m5)
    {
        if (m5.Count < 60) return new float[10];
        var h1 = ResampleH1(m5);
        if (h1.Count < 5) return new float[10];
        var closes = h1.Select(x => x.Close).ToArray();
        var ema30 = Ema(closes, 30);
        var ema60 = Ema(closes, 60);
        var atr = Atr(h1, 14);
        var last = h1[^1];
        var diff = closes.Zip(closes.Skip(1).Append(closes[^1]), (a, b) => b - a).ToArray();
        var pctStd = RollingStd(PctChange(closes), 4);
        var upCount = RollingCountPositive(diff, 4);
        var downCount = RollingCountNegative(diff, 4);
        var upBar = closes.Zip(closes.Prepend(closes[0]), (c, p) => c > p ? 1.0 : 0.0).ToArray();
        var li = h1.Count - 1;
        return
        [
            (float)((last.Close - ema30[li]) / Math.Max(ema30[li], 1e-9)),
            (float)((last.Close - ema60[li]) / Math.Max(ema60[li], 1e-9)),
            (float)(3 * atr[li] / Math.Max(last.Close, 1e-9)),
            (float)pctStd[li],
            (float)upCount[li],
            (float)downCount[li],
            (float)upBar[li],
            0, 0, 0,
        ];
    }

    public static double CurrentAtrPct(IReadOnlyList<OhlcBar> m5) =>
        CurrentAtrPct(m5, 14);

    public static double CurrentAtrPct(IReadOnlyList<OhlcBar> m5, int period)
    {
        if (m5.Count == 0) return 0;
        var atr = Atr(m5, period);
        return atr[^1] / Math.Max(m5[^1].Close, 1e-9) * 100.0;
    }

    /// <summary>最新 ATR 绝对价格（非百分比）。</summary>
    public static double CurrentAtr(IReadOnlyList<OhlcBar> m5, int period = 14)
    {
        if (m5.Count == 0) return 0;
        var atr = Atr(m5, period);
        return atr[^1];
    }

    private static double[] PctChange(IReadOnlyList<double> values)
    {
        var r = new double[values.Count];
        for (var i = 1; i < values.Count; i++)
            r[i] = (values[i] - values[i - 1]) / Math.Max(values[i - 1], 1e-9);
        return r;
    }

    private static double[] RollingMean(IReadOnlyList<double> values, int window)
    {
        var r = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            var start = Math.Max(0, i - window + 1);
            var slice = values.Skip(start).Take(i - start + 1);
            r[i] = slice.Average();
        }
        return r;
    }

    private static double[] RollingStd(IReadOnlyList<double> values, int window)
    {
        var r = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            var start = Math.Max(0, i - window + 1);
            var slice = values.Skip(start).Take(i - start + 1).ToArray();
            if (slice.Length < 2) { r[i] = 0; continue; }
            var mean = slice.Average();
            r[i] = Math.Sqrt(slice.Select(v => (v - mean) * (v - mean)).Average());
        }
        return r;
    }

    private static double[] RollingCountPositive(IReadOnlyList<double> values, int window)
    {
        var r = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            var start = Math.Max(0, i - window + 1);
            r[i] = values.Skip(start).Take(i - start + 1).Count(v => v > 0);
        }
        return r;
    }

    private static double[] RollingCountNegative(IReadOnlyList<double> values, int window)
    {
        var r = new double[values.Count];
        for (var i = 0; i < values.Count; i++)
        {
            var start = Math.Max(0, i - window + 1);
            r[i] = values.Skip(start).Take(i - start + 1).Count(v => v < 0);
        }
        return r;
    }

    private static List<OhlcBar> ResampleH1(IReadOnlyList<OhlcBar> m5)
    {
        var buckets = new Dictionary<DateTime, OhlcBar>();
        foreach (var b in m5)
        {
            var bucket = new DateTime(b.Time.Year, b.Time.Month, b.Time.Day, b.Time.Hour, 0, 0);
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
