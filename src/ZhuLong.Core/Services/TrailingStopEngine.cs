using ZhuLong.Core.Configuration;
using ZhuLong.Core.Features;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

/// <summary>
/// ATR 移动止损（对齐 trade_sim.py）+ M5 结构约束。
/// 浮盈 ≥ breakeven×ATR → 保本；≥ tighten×ATR → SL 跟 best ± step×ATR；结构限制 SL 上推/下压幅度。
/// </summary>
public static class TrailingStopEngine
{
    public sealed class TrailingContext
    {
        public required string Direction { get; init; }
        public double Entry { get; init; }
        public double Price { get; init; }
        public double Atr { get; init; }
        public double BestPrice { get; init; }
        public double TrailingSl { get; init; }
        public bool TrailingActivated { get; init; }
        public double LastTrailPrice { get; init; }
        public int HoldSeconds { get; init; }
        public IReadOnlyList<OhlcBar>? M5Bars { get; init; }
    }

    public sealed class TrailingResult
    {
        public double BestPrice { get; init; }
        public double TrailingSl { get; init; }
        public bool TrailingActivated { get; init; }
        public double LastTrailPrice { get; init; }
        public string StateText { get; init; } = "";
        public string? LogMessage { get; init; }
        public bool StructureCapped { get; init; }
    }

    public static TrailingResult Evaluate(
        TrailingContext ctx,
        AppSettings.PositionManagementSettings pm,
        bool agentMode)
    {
        var atr = ctx.Atr > 0 ? ctx.Atr : ctx.Entry * 0.001;
        var isBuy = ctx.Direction == "buy";

        var best = ctx.BestPrice > 0 ? ctx.BestPrice : ctx.Entry;
        best = isBuy ? Math.Max(best, ctx.Price) : Math.Min(best, ctx.Price);

        var breakevenMult = pm.TrailingBreakevenAtrMult;
        var tightenMult = pm.TrailingTightenAtrMult;
        var stepMult = pm.TrailingStepAtrMult;

        if (agentMode && pm.AgentDrivenExit && pm.AgentTrailingWidenFactor > 1.0)
        {
            breakevenMult *= pm.AgentTrailingWidenFactor;
            tightenMult *= pm.AgentTrailingWidenFactor;
        }

        var favAtr = isBuy
            ? (best - ctx.Entry) / atr
            : (ctx.Entry - best) / atr;

        var trailingSl = ctx.TrailingSl;
        var activated = ctx.TrailingActivated;
        var lastTrail = ctx.LastTrailPrice;

        if (!activated)
        {
            if (ctx.HoldSeconds < pm.MinHoldSecondsBeforeTrailing)
            {
                return Idle(best, trailingSl, activated, lastTrail,
                    "已成交 · 移动止损未激活");
            }

            if (favAtr < breakevenMult)
            {
                return Idle(best, trailingSl, activated, lastTrail,
                    $"已成交 · 浮盈 {FavorPct(isBuy, ctx.Entry, ctx.Price):F2}%（待 {breakevenMult:F1}×ATR）");
            }

            trailingSl = ctx.Entry;
            activated = true;
            lastTrail = ctx.Price;

            return new TrailingResult
            {
                BestPrice = best,
                TrailingSl = trailingSl,
                TrailingActivated = true,
                LastTrailPrice = lastTrail,
                StateText = $"移动止损已激活（保本 {trailingSl:F2}）",
                LogMessage = $"移动止损激活 SL→保本 {trailingSl:F2} fav={favAtr:F2}ATR",
            };
        }

        if (favAtr < tightenMult)
        {
            return Idle(best, trailingSl, activated, lastTrail,
                $"移动止损保本 fav={favAtr:F2}/{tightenMult:F1}×ATR");
        }

        var candidate = isBuy
            ? best - stepMult * atr
            : best + stepMult * atr;

        var structureCapped = false;
        if (pm.TrailingUseStructureConstraint && ctx.M5Bars is { Count: >= 5 })
        {
            var buffer = pm.TrailingStructureBufferAtrMult * atr;
            if (isBuy)
            {
                var swingLow = SwingStructureHelper.RecentSwingLow(
                    ctx.M5Bars, pm.TrailingSwingLookbackBars);
                if (swingLow > 0)
                {
                    var maxSl = swingLow - buffer;
                    if (candidate > maxSl)
                    {
                        candidate = maxSl;
                        structureCapped = true;
                    }
                }
            }
            else
            {
                var swingHigh = SwingStructureHelper.RecentSwingHigh(
                    ctx.M5Bars, pm.TrailingSwingLookbackBars);
                if (swingHigh > 0)
                {
                    var minSl = swingHigh + buffer;
                    if (candidate < minSl)
                    {
                        candidate = minSl;
                        structureCapped = true;
                    }
                }
            }
        }

        // 多单：SL 只升不降；空单：SL 只降不升。且不能穿过 entry 放宽到亏损侧（除非尚未保本）
        var improved = isBuy
            ? candidate > trailingSl + 0.01
            : (trailingSl <= 0 || candidate < trailingSl - 0.01);

        if (!improved)
        {
            var suffix = structureCapped ? " · 结构限制" : "";
            return Idle(best, trailingSl, activated, lastTrail,
                $"移动止损 SL={trailingSl:F2}{suffix}");
        }

        // 保本后不允许多单 SL 低于 entry
        if (isBuy && candidate < ctx.Entry)
            candidate = ctx.Entry;
        if (!isBuy && candidate > ctx.Entry)
            candidate = ctx.Entry;

        var old = trailingSl;
        trailingSl = candidate;
        lastTrail = ctx.Price;

        var capNote = structureCapped ? " 结构约束" : "";
        return new TrailingResult
        {
            BestPrice = best,
            TrailingSl = trailingSl,
            TrailingActivated = true,
            LastTrailPrice = lastTrail,
            StructureCapped = structureCapped,
            StateText = $"移动止损 SL={trailingSl:F2}{capNote}",
            LogMessage = $"移动止损 SL {old:F2} → {trailingSl:F2} fav={favAtr:F2}ATR{capNote}",
        };
    }

    private static TrailingResult Idle(
        double best, double trailingSl, bool activated, double lastTrail, string text) =>
        new()
        {
            BestPrice = best,
            TrailingSl = trailingSl,
            TrailingActivated = activated,
            LastTrailPrice = lastTrail,
            StateText = text,
        };

    private static double FavorPct(bool isBuy, double entry, double price)
    {
        if (entry <= 0) return 0;
        return isBuy
            ? (price - entry) / entry * 100.0
            : (entry - price) / entry * 100.0;
    }
}
