using System.Text.Json;
using System.Text.Json.Serialization;

namespace ZhuLong.Core.Configuration;

public sealed class AppSettings
{
    public AppInfo? App { get; set; }
    public ModelSettings? Model { get; set; }
    public SignalFilterSettings? SignalFilters { get; set; }
    public SignalGeometrySettings? SignalGeometry { get; set; }
    public PositionManagementSettings? PositionManagement { get; set; }
    public MacroSettings? Macro { get; set; }
    public AtrChannelSettings? AtrChannel { get; set; }
    public Dictionary<string, string> SymbolMapping { get; set; } = new();
    public Mt5Settings? Mt5 { get; set; }
    public PipeSettings? Pipes { get; set; }
    public LoggingSettings? Logging { get; set; }
    public RiskGuardSettings? RiskGuard { get; set; }
    public MultiStrategySettings? MultiStrategy { get; set; }
    public TradingAgentSettings? TradingAgent { get; set; }

    public sealed class TradingAgentSettings
    {
        public bool Enabled { get; set; } = true;
        public string ConfigPath { get; set; } = "config/config_agent.json";
    }

    public sealed class MultiStrategySettings
    {
        public bool Enabled { get; set; } = true;
        public string ConfigPath { get; set; } = "config/config_multi_strategy.json";
    }

    public sealed class AppInfo
    {
        public string Name { get; set; } = "烛龙 ZhuLong";
        public string Version { get; set; } = "3.0.0";
    }

    public sealed class ModelSettings
    {
        public int SeqLen { get; set; } = 60;
        public int FeatureDim5Min { get; set; } = 30;
        public bool UseXgbExpectedReturn { get; set; }
        public int HistoricalAvgGainWindow { get; set; } = 100;
        public string[] DefaultSymbols { get; set; } = ["XAUUSD", "USOIL"];
        public string PrimarySymbol { get; set; } = "XAUUSD";
        public bool InferAllReadySymbols { get; set; }
    }

    public sealed class SignalFilterSettings
    {
        public double ProbThreshold { get; set; } = 0.60;
        public double MinExpectedReturn { get; set; } = 0.15;
        public double MinRiskReward { get; set; } = 1.5;
        public double EntryOffsetBuyMin { get; set; } = -0.30;
        public double EntryOffsetBuyMax { get; set; } = -0.05;
        public double EntryOffsetSellMin { get; set; } = 0.05;
        public double EntryOffsetSellMax { get; set; } = 0.30;
        public int CooldownMinutes { get; set; } = 30;
        public double MinVolatilityAtr { get; set; } = 0.2;
        public double MaxVolatilityAtr { get; set; } = 1.0;
        public int SignalExpiryMinutes { get; set; } = 240;
    }

    public sealed class SignalGeometrySettings
    {
        public double InitialStopLossAtrMult { get; set; } = 1.2;
        public double ShortStopLossAtrMult { get; set; } = 1.0;
        public double InitialTakeProfitAtrMult { get; set; } = 2.0;
    }

    public sealed class PositionManagementSettings
    {
        public bool TrailingStopEnabled { get; set; } = true;
        /// <summary>旧版百分比激活（TrailingUseAtrMode=false 时使用）。</summary>
        public double TrailingActivationPct { get; set; } = 0.25;
        public double TrailingStepPct { get; set; } = 0.10;
        public double TrailingTightenFactor { get; set; } = 0.8;
        /// <summary>使用 ATR 移动止损（对齐 trade_sim.py）；false 则回退百分比模式。</summary>
        public bool TrailingUseAtrMode { get; set; } = true;
        /// <summary>浮盈 ≥ 此倍数×ATR 激活保本。</summary>
        public double TrailingBreakevenAtrMult { get; set; } = 1.0;
        /// <summary>浮盈 ≥ 此倍数×ATR 才开始收紧 SL。</summary>
        public double TrailingTightenAtrMult { get; set; } = 1.5;
        /// <summary>收紧时 SL = best ∓ 此倍数×ATR。</summary>
        public double TrailingStepAtrMult { get; set; } = 0.5;
        /// <summary>结构支撑/阻力下方的 ATR buffer。</summary>
        public double TrailingStructureBufferAtrMult { get; set; } = 0.3;
        /// <summary>M5 回看根数（swing 检测）。</summary>
        public int TrailingSwingLookbackBars { get; set; } = 24;
        /// <summary>收紧 SL 时应用 M5 结构约束。</summary>
        public bool TrailingUseStructureConstraint { get; set; } = true;
        /// <summary>智能体模式下 ATR 阈值放宽倍数（出场由 exit_assessment 主导）。</summary>
        public double AgentTrailingWidenFactor { get; set; } = 1.5;
        /// <summary>开仓后至少等待此秒数才允许激活移动止损（与入场价优化窗口对齐）。</summary>
        public int MinHoldSecondsBeforeTrailing { get; set; } = 60;
        /// <summary>浮盈峰值低于此百分比时不触发 MaxDrawdownRatio 回撤平仓。</summary>
        public double MinPeakProfitPctForDrawdown { get; set; } = 0.35;
        /// <summary>成交后至少持仓此秒数，才允许浮盈回撤保护生效。</summary>
        public int MinHoldSecondsBeforeProfitDrawdown { get; set; } = 180;
        /// <summary>限价待成交最长等待（秒）；0=沿用 signal_expiry_minutes。</summary>
        public int EntryFillMaxWaitSeconds { get; set; }
        public bool PartialProfitEnabled { get; set; } = true;
        public double PartialTarget1Pct { get; set; } = 0.25;
        public double PartialRatio1 { get; set; } = 0.5;
        public double PartialTarget2Pct { get; set; } = 0.40;
        public double PartialRatio2 { get; set; } = 0.5;
        public int MaxHoldMinutes { get; set; } = 240;
        public double MaxDrawdownRatio { get; set; } = 0.4;
        public bool UseModelExit { get; set; } = false;
        /// <summary>智能体模式：出场由 exit_assessment 决策，机械 trailing/TP/回撤/时间止损仅作兜底。</summary>
        public bool AgentDrivenExit { get; set; } = true;
        /// <summary>智能体模式：入场价由 cognition tick 评估，OptimizeEntryPrices 动态调价/放弃追高。</summary>
        public bool AgentDrivenEntry { get; set; } = true;
        /// <summary>智能体模式仍保留硬止损（风控底线）。</summary>
        public bool AgentHardStopLoss { get; set; } = true;
        public double MatchPriceTolerancePoints { get; set; } = 5;
        public int MatchTimeWindowSeconds { get; set; } = 60;
        public int OrderRetryMax { get; set; } = 3;
    }

    public sealed class MacroSettings
    {
        public bool Enabled { get; set; } = true;
        public bool ForceSilence { get; set; }
        public string[] ForceSilenceEvents { get; set; } = [];
        public int SilenceBeforeMinutes { get; set; } = 30;
        public int SilenceAfterMinutes { get; set; } = 15;
        public int ReloadIntervalHours { get; set; } = 24;
        public MacroCalendarSettings? Calendar { get; set; }
        public MacroFredSettings? Fred { get; set; }
        public MacroSentimentSettings? Sentiment { get; set; }
    }

    public sealed class MacroCalendarSettings
    {
        public string Provider { get; set; } = "auto";
        public string? FinnhubApiKey { get; set; }
        public string? FmpApiKey { get; set; }
        public bool FetchOnStartup { get; set; } = true;
        public int DailyRefreshHourUtc { get; set; } = 0;
        public int LookaheadHours { get; set; } = 168;
        public bool FallbackHtmlEnabled { get; set; } = true;
    }

    public sealed class MacroFredSettings
    {
        public string? ApiKey { get; set; }
        public string[] Series { get; set; } = ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10YIE"];
        public string JsonPath { get; set; } = "data/fred_latest.json";
    }

    public sealed class MacroSentimentSettings
    {
        public string Provider { get; set; } = "api2d";
        public string? ApiKey { get; set; }
        public string? BaseUrl { get; set; }
        public string? Model { get; set; }
        public string JsonPath { get; set; } = "data/sentiment.json";
    }

    public sealed class AtrChannelSettings
    {
        public int Period { get; set; } = 14;
        public double Multiplier { get; set; } = 3.0;
        public int EmaFast { get; set; } = 30;
        public int EmaSlow { get; set; } = 60;
    }

    public sealed class Mt5Settings
    {
        public int Deviation { get; set; } = 20;
        public string CommentPrefix { get; set; } = "ZhuLong";
    }

    public sealed class PipeSettings
    {
        public string DataPipe { get; set; } = @"\\.\pipe\ZhuLong_Data";
        public string DrawingPipe { get; set; } = @"\\.\pipe\ZhuLong_Drawing";
    }

    public sealed class LoggingSettings
    {
        public string Level { get; set; } = "INFO";
    }

    public sealed class RiskGuardSettings
    {
        public bool Enabled { get; set; } = true;
        public double MaxDailyLossPct { get; set; } = 3.0;
        public int MaxConcurrentPositions { get; set; } = 4;
        public int MaxPendingSignals { get; set; } = 10;
        public int SymbolCooldownMinutes { get; set; } = 30;
    }

    public static AppSettings Load(string path)
    {
        var text = File.ReadAllText(path);
        if (text.Length > 0 && text[0] == '\uFEFF')
            text = text[1..];
        return JsonSerializer.Deserialize<AppSettings>(text, JsonOptions()) ?? new AppSettings();
    }

    public static AppSettings LoadOrCreate(string path)
    {
        if (!File.Exists(path))
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);

            var install = Path.Combine(AppPaths.InstallDir, "config.json");
            if (File.Exists(install))
                File.Copy(install, path);
            else
                new AppSettings().Save(path);
        }

        return Load(path);
    }

    private static JsonSerializerOptions JsonOptions() => new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        WriteIndented = true,
    };

    public void Save(string path)
    {
        var json = JsonSerializer.Serialize(this, JsonOptions());
        File.WriteAllText(path, json, new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
    }

    public string ResolveBrokerSymbol(string standard) =>
        SymbolMapping.TryGetValue(standard, out var mapped) ? mapped : standard;

    public string ResolveStandardSymbol(string brokerOrRaw)
    {
        if (string.IsNullOrWhiteSpace(brokerOrRaw))
            return brokerOrRaw;

        var raw = brokerOrRaw.Trim();
        var defaults = Model?.DefaultSymbols ?? ["XAUUSD"];

        foreach (var std in defaults)
        {
            if (string.Equals(std, raw, StringComparison.OrdinalIgnoreCase))
                return std;
        }

        foreach (var (std, broker) in SymbolMapping)
        {
            if (string.Equals(broker, raw, StringComparison.OrdinalIgnoreCase))
                return std;
            if (string.Equals(std, raw, StringComparison.OrdinalIgnoreCase))
                return std;
        }

        var upper = raw.ToUpperInvariant();
        if (IsGoldBrokerAlias(upper))
            return "XAUUSD";
        if (IsOilBrokerAlias(upper))
            return "USOIL";

        return raw;
    }

    private static bool IsGoldBrokerAlias(string upper) =>
        upper.Contains("XAU", StringComparison.Ordinal) || upper.Contains("GOLD", StringComparison.Ordinal);

    private static bool IsOilBrokerAlias(string upper) =>
        upper.Contains("OIL", StringComparison.Ordinal)
        || upper.Contains("XTI", StringComparison.Ordinal)
        || upper.Contains("WTI", StringComparison.Ordinal);
}
