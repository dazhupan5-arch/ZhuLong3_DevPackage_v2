using ZhuLong.Core.Models;

namespace ZhuLong.Core.Features;

/// <summary>M5 波段高低点，用于移动止损结构约束。</summary>
public static class SwingStructureHelper
{
    /// <summary>最近 lookback 内最高的 pivot 低点（上升趋势中的有效支撑）。</summary>
    public static double RecentSwingLow(IReadOnlyList<OhlcBar> bars, int lookback, int pivotBars = 2)
    {
        if (bars.Count < pivotBars * 2 + 1) return 0;

        var start = Math.Max(0, bars.Count - lookback);
        var end = bars.Count - pivotBars;
        double best = 0;

        for (var i = start + pivotBars; i < end; i++)
        {
            var low = bars[i].Low;
            if (!IsPivotLow(bars, i, pivotBars)) continue;
            if (low > best)
                best = low;
        }

        if (best > 0) return best;

        // 无 pivot 时用 lookback 内最低 low（排除最后一根未闭合柱）
        var sliceEnd = Math.Max(start, bars.Count - 1);
        if (sliceEnd <= start) return 0;
        return bars.Skip(start).Take(sliceEnd - start).Min(b => b.Low);
    }

    /// <summary>最近 lookback 内最低的 pivot 高点（下降趋势中的有效阻力）。</summary>
    public static double RecentSwingHigh(IReadOnlyList<OhlcBar> bars, int lookback, int pivotBars = 2)
    {
        if (bars.Count < pivotBars * 2 + 1) return 0;

        var start = Math.Max(0, bars.Count - lookback);
        var end = bars.Count - pivotBars;
        double best = 0;

        for (var i = start + pivotBars; i < end; i++)
        {
            var high = bars[i].High;
            if (!IsPivotHigh(bars, i, pivotBars)) continue;
            if (best <= 0 || high < best)
                best = high;
        }

        if (best > 0) return best;

        var sliceEnd = Math.Max(start, bars.Count - 1);
        if (sliceEnd <= start) return 0;
        return bars.Skip(start).Take(sliceEnd - start).Max(b => b.High);
    }

    private static bool IsPivotLow(IReadOnlyList<OhlcBar> bars, int i, int pivotBars)
    {
        var low = bars[i].Low;
        for (var j = 1; j <= pivotBars; j++)
        {
            if (bars[i - j].Low <= low || bars[i + j].Low <= low)
                return false;
        }
        return true;
    }

    private static bool IsPivotHigh(IReadOnlyList<OhlcBar> bars, int i, int pivotBars)
    {
        var high = bars[i].High;
        for (var j = 1; j <= pivotBars; j++)
        {
            if (bars[i - j].High >= high || bars[i + j].High >= high)
                return false;
        }
        return true;
    }
}
