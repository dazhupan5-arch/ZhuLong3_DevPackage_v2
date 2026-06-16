using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml.Controls;
using ZhuLong.App.Services;
using ZhuLong.Core;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Services;

namespace ZhuLong.App.Views;

public sealed partial class ParamsPanel : UserControl
{
    private readonly ZhuLongRuntimeService _runtime;
    private string _configPath = AppPaths.ConfigPath;

    public ParamsPanel()
    {
        InitializeComponent();
        _runtime = App.Services.GetRequiredService<ZhuLongRuntimeService>();
    }

    public void LoadFromDisk()
    {
        _configPath = AppPaths.ConfigPath;
        var s = AppSettings.LoadOrCreate(_configPath);
        s.SignalFilters ??= new AppSettings.SignalFilterSettings();
        s.SignalGeometry ??= new AppSettings.SignalGeometrySettings();
        s.PositionManagement ??= new AppSettings.PositionManagementSettings();
        s.AtrChannel ??= new AppSettings.AtrChannelSettings();
        s.Macro ??= new AppSettings.MacroSettings();
        s.Mt5 ??= new AppSettings.Mt5Settings();
        s.RiskGuard ??= new AppSettings.RiskGuardSettings();

        var f = s.SignalFilters;
        SetNum(NumProbThreshold, f.ProbThreshold);
        SetNum(NumMinExpectedReturn, f.MinExpectedReturn);
        SetNum(NumMinRiskReward, f.MinRiskReward);
        SetNum(NumCooldownMinutes, f.CooldownMinutes);
        SetNum(NumSignalExpiry, f.SignalExpiryMinutes);
        SetNum(NumMinVolAtr, f.MinVolatilityAtr);
        SetNum(NumMaxVolAtr, f.MaxVolatilityAtr);

        var g = s.SignalGeometry;
        SetNum(NumSlAtrMult, g.InitialStopLossAtrMult);
        SetNum(NumTpAtrMult, g.InitialTakeProfitAtrMult);

        var p = s.PositionManagement;
        SetNum(NumTrailAct, p.TrailingActivationPct);
        SetNum(NumTrailStep, p.TrailingStepPct);
        SetNum(NumPartial1Pct, p.PartialTarget1Pct);
        SetNum(NumPartial1Ratio, p.PartialRatio1);
        SetNum(NumPartial2Pct, p.PartialTarget2Pct);
        SetNum(NumPartial2Ratio, p.PartialRatio2);
        SetNum(NumMaxHold, p.MaxHoldMinutes);
        SetNum(NumMaxDd, p.MaxDrawdownRatio);
        SetNum(NumMatchTol, p.MatchPriceTolerancePoints);
        SetNum(NumOrderRetry, p.OrderRetryMax);
        ChkModelExit.IsChecked = p.UseModelExit;

        var a = s.AtrChannel;
        SetNum(NumAtrPeriod, a.Period);
        SetNum(NumAtrMult, a.Multiplier);
        SetNum(NumEmaFast, a.EmaFast);
        SetNum(NumEmaSlow, a.EmaSlow);

        SetNum(NumSilenceBefore, s.Macro.SilenceBeforeMinutes);
        SetNum(NumSilenceAfter, s.Macro.SilenceAfterMinutes);
        ChkMacroEnabled.IsChecked = s.Macro.Enabled;
        ChkForceSilence.IsChecked = s.Macro.ForceSilence;
        SetNum(NumDeviation, s.Mt5.Deviation);

        var rg = s.RiskGuard;
        ChkRiskEnabled.IsChecked = rg.Enabled;
        SetNum(NumMaxDailyLoss, rg.MaxDailyLossPct);
        SetNum(NumMaxPositions, rg.MaxConcurrentPositions);
        SetNum(NumMaxPending, rg.MaxPendingSignals);
        SetNum(NumSymbolCooldown, rg.SymbolCooldownMinutes);

        TxtSymbolMapping.Text = FormatSymbolMapping(s.SymbolMapping);
        TxtParamsMessage.Text = $"已载入 {_configPath}";
    }

    private void OnReloadClick(object sender, RoutedEventArgs e) => LoadFromDisk();

    private void OnSaveClick(object sender, RoutedEventArgs e)
    {
        try
        {
            var s = AppSettings.LoadOrCreate(_configPath);
            s.SignalFilters ??= new AppSettings.SignalFilterSettings();
            s.SignalGeometry ??= new AppSettings.SignalGeometrySettings();
            s.PositionManagement ??= new AppSettings.PositionManagementSettings();
            s.AtrChannel ??= new AppSettings.AtrChannelSettings();
            s.Macro ??= new AppSettings.MacroSettings();
            s.Mt5 ??= new AppSettings.Mt5Settings();
            s.RiskGuard ??= new AppSettings.RiskGuardSettings();

            var f = s.SignalFilters;
            f.ProbThreshold = GetNum(NumProbThreshold, f.ProbThreshold);
            f.MinExpectedReturn = GetNum(NumMinExpectedReturn, f.MinExpectedReturn);
            f.MinRiskReward = GetNum(NumMinRiskReward, f.MinRiskReward);
            f.CooldownMinutes = (int)GetNum(NumCooldownMinutes, f.CooldownMinutes);
            f.SignalExpiryMinutes = (int)GetNum(NumSignalExpiry, f.SignalExpiryMinutes);
            f.MinVolatilityAtr = GetNum(NumMinVolAtr, f.MinVolatilityAtr);
            f.MaxVolatilityAtr = GetNum(NumMaxVolAtr, f.MaxVolatilityAtr);

            var g = s.SignalGeometry;
            g.InitialStopLossAtrMult = GetNum(NumSlAtrMult, g.InitialStopLossAtrMult);
            g.InitialTakeProfitAtrMult = GetNum(NumTpAtrMult, g.InitialTakeProfitAtrMult);

            var p = s.PositionManagement;
            p.TrailingActivationPct = GetNum(NumTrailAct, p.TrailingActivationPct);
            p.TrailingStepPct = GetNum(NumTrailStep, p.TrailingStepPct);
            p.PartialTarget1Pct = GetNum(NumPartial1Pct, p.PartialTarget1Pct);
            p.PartialRatio1 = GetNum(NumPartial1Ratio, p.PartialRatio1);
            p.PartialTarget2Pct = GetNum(NumPartial2Pct, p.PartialTarget2Pct);
            p.PartialRatio2 = GetNum(NumPartial2Ratio, p.PartialRatio2);
            p.MaxHoldMinutes = (int)GetNum(NumMaxHold, p.MaxHoldMinutes);
            p.MaxDrawdownRatio = GetNum(NumMaxDd, p.MaxDrawdownRatio);
            p.MatchPriceTolerancePoints = GetNum(NumMatchTol, p.MatchPriceTolerancePoints);
            p.OrderRetryMax = (int)GetNum(NumOrderRetry, p.OrderRetryMax);
            p.UseModelExit = ChkModelExit.IsChecked == true;

            var a = s.AtrChannel;
            a.Period = (int)GetNum(NumAtrPeriod, a.Period);
            a.Multiplier = GetNum(NumAtrMult, a.Multiplier);
            a.EmaFast = (int)GetNum(NumEmaFast, a.EmaFast);
            a.EmaSlow = (int)GetNum(NumEmaSlow, a.EmaSlow);

            s.Macro.SilenceBeforeMinutes = (int)GetNum(NumSilenceBefore, s.Macro.SilenceBeforeMinutes);
            s.Macro.SilenceAfterMinutes = (int)GetNum(NumSilenceAfter, s.Macro.SilenceAfterMinutes);
            s.Macro.Enabled = ChkMacroEnabled.IsChecked == true;
            s.Macro.ForceSilence = ChkForceSilence.IsChecked == true;
            s.Mt5.Deviation = (int)GetNum(NumDeviation, s.Mt5.Deviation);

            s.SymbolMapping = ParseSymbolMapping(TxtSymbolMapping.Text);

            var rg = s.RiskGuard;
            rg.Enabled = ChkRiskEnabled.IsChecked == true;
            rg.MaxDailyLossPct = GetNum(NumMaxDailyLoss, rg.MaxDailyLossPct);
            rg.MaxConcurrentPositions = (int)GetNum(NumMaxPositions, rg.MaxConcurrentPositions);
            rg.MaxPendingSignals = (int)GetNum(NumMaxPending, rg.MaxPendingSignals);
            rg.SymbolCooldownMinutes = (int)GetNum(NumSymbolCooldown, rg.SymbolCooldownMinutes);

            var errors = ConfigValidator.Validate(s);
            if (errors.Count > 0)
            {
                TxtParamsMessage.Text = "校验失败：" + string.Join("; ", errors);
                return;
            }

            s.Save(_configPath);
            _runtime.ReloadSettingsFromDisk();
            TxtParamsMessage.Text = "已保存并热更新（信号过滤 / 持仓 / 宏观 / MT5 滑点）";
        }
        catch (Exception ex)
        {
            TxtParamsMessage.Text = "保存失败：" + ex.Message;
        }
    }

    private static void SetNum(NumberBox box, double value)
    {
        box.Value = double.IsNaN(value) ? 0 : value;
    }

    private static double GetNum(NumberBox box, double fallback) =>
        double.IsNaN(box.Value) ? fallback : box.Value;

    private static string FormatSymbolMapping(Dictionary<string, string> map)
    {
        if (map.Count == 0) return "";
        return string.Join(", ", map.Select(kv => $"{kv.Key}={kv.Value}"));
    }

    private static Dictionary<string, string> ParseSymbolMapping(string text)
    {
        var d = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var part in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            var kv = part.Split('=', 2, StringSplitOptions.TrimEntries);
            if (kv.Length == 2 && !string.IsNullOrWhiteSpace(kv[0]))
                d[kv[0]] = kv[1];
        }
        return d;
    }
}
