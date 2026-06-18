using System.Text.Json;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Models;

namespace ZhuLong.Core.Services;

public sealed class SignalGeneratorService
{
    private readonly Dictionary<(string Symbol, string Direction), DateTime> _cooldown = new();
    public string? LastRejectReason { get; private set; }

    public SignalModel? TryGenerate(
        AppSettings settings,
        string symbol,
        InferenceResult inference,
        double atrPct,
        double closePrice)
    {
        LastRejectReason = null;
        var sf = settings.SignalFilters ?? new AppSettings.SignalFilterSettings();
        var sg = settings.SignalGeometry ?? new AppSettings.SignalGeometrySettings();

        if (inference.Direction == 0) { LastRejectReason = "方向=观望"; return null; }
        var dir = inference.Direction == 1 ? "buy" : "sell";
        if (inference.Confidence < sf.ProbThreshold)
        {
            LastRejectReason = $"置信度 {inference.Confidence:F2} < {sf.ProbThreshold:F2}";
            return null;
        }
        if (settings.Model?.UseXgbExpectedReturn == true && sf.MinExpectedReturn > 0 &&
            inference.ExpectedReturn < sf.MinExpectedReturn)
        {
            LastRejectReason = $"预期收益 {inference.ExpectedReturn:F2} < {sf.MinExpectedReturn:F2}";
            return null;
        }
        if (atrPct < sf.MinVolatilityAtr || atrPct > sf.MaxVolatilityAtr)
        {
            LastRejectReason = $"波动率 ATR% {atrPct:F2} 超出 [{sf.MinVolatilityAtr},{sf.MaxVolatilityAtr}]";
            return null;
        }

        var risk = atrPct * 1.2;
        if (sf.MinRiskReward > 0 && risk > 0 && inference.ExpectedReturn / risk < sf.MinRiskReward)
        {
            LastRejectReason = $"盈亏比不足 ({inference.ExpectedReturn / risk:F2} < {sf.MinRiskReward:F2})";
            return null;
        }

        var offsetPct = inference.EntryOffset * 100;
        var marketEntry = Math.Abs(inference.EntryOffset) < 1e-6;
        if (!marketEntry)
        {
            if (inference.Direction == 1)
            {
                if (offsetPct < sf.EntryOffsetBuyMin || offsetPct > sf.EntryOffsetBuyMax)
                {
                    LastRejectReason = $"entry_offset {offsetPct:F2} 不在做多区间";
                    return null;
                }
            }
            else if (offsetPct < sf.EntryOffsetSellMin || offsetPct > sf.EntryOffsetSellMax)
            {
                LastRejectReason = $"entry_offset {offsetPct:F2} 不在做空区间";
                return null;
            }
        }

        var key = (symbol, dir);
        if (_cooldown.TryGetValue(key, out var last) &&
            DateTime.UtcNow - last < TimeSpan.FromMinutes(sf.CooldownMinutes))
        {
            LastRejectReason = $"冷却中（{sf.CooldownMinutes} 分钟）";
            return null;
        }

        var entry = marketEntry ? closePrice : closePrice * (1 + inference.EntryOffset);
        var atrAbs = closePrice * atrPct / 100.0;
        var slMultLong = sg.InitialStopLossAtrMult;
        var slMult = dir == "sell" && sg.ShortStopLossAtrMult > 0 ? sg.ShortStopLossAtrMult : slMultLong;
        var tpMult = sg.InitialTakeProfitAtrMult;
        double sl, tp;
        if (dir == "buy")
        {
            sl = entry - atrAbs * slMult;
            tp = entry + atrAbs * tpMult;
        }
        else
        {
            sl = entry + atrAbs * slMult;
            tp = entry - atrAbs * tpMult;
        }

        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var signalId = $"{DateTime.UtcNow:yyyyMMdd_HHmm}_{symbol}_{dir}";
        return BuildSignal(settings, symbol, dir, entry, sl, tp, inference.Confidence, inference.ExpectedReturn,
            signalId, "ai_model", sf);
    }

    public SignalModel? TryGenerateFromStrategySignal(
        AppSettings settings,
        MultiStrategySignalPayload payload,
        string? attributionJson = null)
    {
        LastRejectReason = null;
        var sf = settings.SignalFilters ?? new AppSettings.SignalFilterSettings();
        var dir = payload.Direction;
        if (dir is not ("buy" or "sell"))
        {
            LastRejectReason = payload.RejectReason ?? "方向=观望";
            return null;
        }

        if (string.Equals(payload.Strategy, "ai_model", StringComparison.OrdinalIgnoreCase))
        {
            if (payload.Confidence < sf.ProbThreshold)
            {
                LastRejectReason = $"置信度 {payload.Confidence:F2} < {sf.ProbThreshold:F2}";
                return null;
            }
        }
        else if (payload.Confidence < 0.5)
        {
            LastRejectReason = $"置信度 {payload.Confidence:F2} 过低";
            return null;
        }

        var key = (payload.Symbol, dir);
        if (_cooldown.TryGetValue(key, out var last) &&
            DateTime.UtcNow - last < TimeSpan.FromMinutes(sf.CooldownMinutes))
        {
            LastRejectReason = $"冷却中（{sf.CooldownMinutes} 分钟）";
            return null;
        }

        var signalId = string.IsNullOrWhiteSpace(payload.SignalId)
            ? $"{DateTime.UtcNow:yyyyMMdd_HHmm}_{payload.Symbol}_{dir}"
            : payload.SignalId;

        return BuildSignal(settings, payload.Symbol, dir, payload.Entry, payload.Sl, payload.Tp,
            payload.Confidence, 0, signalId, payload.Strategy, sf, attributionJson);
    }

    private SignalModel BuildSignal(
        AppSettings settings,
        string symbol,
        string dir,
        double entry,
        double sl,
        double tp,
        double confidence,
        double expectedReturn,
        string signalId,
        string strategy,
        AppSettings.SignalFilterSettings sf,
        string? attributionJson = null)
    {
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var magic = Math.Abs(signalId.GetHashCode()) & 0xFFFF;
        if (magic == 0) magic = 1;
        var prefix = settings.Mt5?.CommentPrefix ?? "ZhuLong";
        var comment = $"{prefix}_{strategy}_{signalId}";

        _cooldown[(symbol, dir)] = DateTime.UtcNow;

        return new SignalModel
        {
            SignalId = signalId,
            Timestamp = now,
            Symbol = symbol,
            Direction = dir,
            EntryPrice = entry,
            StopLoss = sl,
            TakeProfit = tp,
            Confidence = confidence,
            ExpectedReturn = expectedReturn,
            MagicNumber = magic,
            CommentHint = comment,
            Strategy = strategy,
            Status = "pending",
            ParamsSnapshot = JsonSerializer.Serialize(new { sf.ProbThreshold, strategy }),
            AttributionJson = attributionJson,
            CreatedAt = now,
        };
    }
}
