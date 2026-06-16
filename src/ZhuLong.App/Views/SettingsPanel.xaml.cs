using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml.Controls;
using Windows.ApplicationModel.DataTransfer;
using Windows.UI;
using ZhuLong.App.Services;
using ZhuLong.App.Services.Membership;
using ZhuLong.Core.Configuration;
using ZhuLong.Core.Services;

namespace ZhuLong.App.Views;

public sealed partial class SettingsPanel : UserControl
{
    private readonly UserSecretsStore _secrets;
    private readonly MacroOfflineRefreshService _macroRefresh;
    private readonly PythonEnvironmentCoordinator _envRepair;
    private readonly ZhuLongRuntimeService _runtime;
    private readonly DispatcherQueue? _dispatcher;

    private static readonly Color SelectedBtnBg = Color.FromArgb(255, 45, 90, 140);
    private static readonly Color DefaultBtnBg = Color.FromArgb(255, 40, 40, 40);

    public SettingsPanel()
    {
        InitializeComponent();
        _secrets = (UserSecretsStore)App.Services.GetRequiredService<IApiSecrets>();
        _macroRefresh = App.Services.GetRequiredService<MacroOfflineRefreshService>();
        _envRepair = App.Services.GetRequiredService<PythonEnvironmentCoordinator>();
        _runtime = App.Services.GetRequiredService<ZhuLongRuntimeService>();
        _dispatcher = DispatcherQueue.GetForCurrentThread();
    }

    public void RefreshUi()
    {
        MembershipHost.Instance.Refresh();
        var m = MembershipHost.Instance;
        TxtLicenseStatus.Text = m.StatusSummary;
        TxtDeviceId.Text = "设备标识（前 8 位）：" + m.HardwareIdShort;
        BindExistingKeys();
        RefreshModelUi();
    }

    private void RefreshModelUi()
    {
        var prod = ProductionModelGate.Check(_runtime.Settings);
        var lines = new List<string>();
        foreach (var sym in _runtime.GetConfiguredSymbols())
        {
            var ready = prod.ReadySymbols.Contains(sym, StringComparer.OrdinalIgnoreCase);
            var label = sym switch
            {
                "XAUUSD" => "XAUUSD 黄金",
                "USOIL" => "USOIL 原油",
                _ => sym,
            };
            lines.Add($"{(ready ? "●" : "○")} {label} — {(ready ? "已就绪" : "未就绪")}");
        }

        if (prod.PendingReasons.Count > 0)
            lines.Add(string.Join("；", prod.PendingReasons));

        TxtModelStatus.Text = lines.Count > 0 ? string.Join(Environment.NewLine, lines) : "—";

        var primary = _runtime.GetPrimarySymbol();
        var inferAll = _runtime.GetInferAllReadySymbols();
        ChkInferAll.IsChecked = inferAll;
        ChkMultiStrategy.IsChecked = _runtime.GetMultiStrategyEnabled();
        ChkTradingAgent.IsChecked = _runtime.GetTradingAgentEnabled();
        HighlightPrimaryButton(primary);

        var modeHint = ChkTradingAgent.IsChecked == true
            ? " · RL 智能体已启用"
            : ChkMultiStrategy.IsChecked == true
                ? " · 多策略引擎已启用（状态机自动切换）"
                : " · 单模型 V14 模式";
        TxtPrimaryHint.Text = inferAll
            ? prod.ReadySymbols.Count > 0
                ? $"当前：并行推理 {string.Join(", ", prod.ReadySymbols)}{modeHint}"
                : $"当前：并行模式（尚无已验收品种）{modeHint}"
            : ChkTradingAgent.IsChecked == true
                ? $"当前：RL 智能体 → {primary}{modeHint}"
            : prod.ReadySymbols.Contains(primary, StringComparer.OrdinalIgnoreCase)
                ? $"当前：单品种推理 → {primary}{modeHint}"
                : $"当前：单品种 → {primary}（未就绪，请查看上方状态）{modeHint}";

        RefreshBacktestSummary();
    }

    private void RefreshBacktestSummary()
    {
        var blocks = ModelManifestService.ReadInstalled(_runtime.GetConfiguredSymbols())
            .Select(m => m.FormatBlock())
            .ToList();
        TxtBacktestSummary.Text = blocks.Count > 0
            ? string.Join(Environment.NewLine + Environment.NewLine, blocks)
            : "未找到模型 manifest（请确认已安装正式 models 目录）";
    }

    private void HighlightPrimaryButton(string primary)
    {
        SetButtonSelected(BtnPickXau, string.Equals(primary, "XAUUSD", StringComparison.OrdinalIgnoreCase));
        SetButtonSelected(BtnPickOil, string.Equals(primary, "USOIL", StringComparison.OrdinalIgnoreCase));
    }

    private static void SetButtonSelected(Button btn, bool selected)
    {
        btn.Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(selected ? SelectedBtnBg : DefaultBtnBg);
    }

    private void OnPickXauClick(object sender, RoutedEventArgs e)
    {
        _runtime.SetPrimarySymbol("XAUUSD");
        RefreshModelUi();
        TxtSettingsMessage.Text = "已切换推理品种 → XAUUSD";
    }

    private void OnPickOilClick(object sender, RoutedEventArgs e)
    {
        _runtime.SetPrimarySymbol("USOIL");
        RefreshModelUi();
        TxtSettingsMessage.Text = "已切换推理品种 → USOIL";
    }

    private void OnInferAllClick(object sender, RoutedEventArgs e)
    {
        _runtime.SetInferAllReadySymbols(ChkInferAll.IsChecked == true);
        RefreshModelUi();
        TxtSettingsMessage.Text = ChkInferAll.IsChecked == true
            ? "已启用多品种并行推理"
            : $"已切换为单品种推理 → {_runtime.GetPrimarySymbol()}";
    }

    private void OnMultiStrategyClick(object sender, RoutedEventArgs e)
    {
        _runtime.SetMultiStrategyEnabled(ChkMultiStrategy.IsChecked == true);
        RefreshModelUi();
        TxtSettingsMessage.Text = ChkMultiStrategy.IsChecked == true
            ? "已启用多策略自动切换"
            : "已切换为单模型 V14 推理";
    }

    private void OnTradingAgentClick(object sender, RoutedEventArgs e)
    {
        _runtime.SetTradingAgentEnabled(ChkTradingAgent.IsChecked == true);
        RefreshModelUi();
        TxtSettingsMessage.Text = ChkTradingAgent.IsChecked == true
            ? "已启用 RL 交易智能体"
            : "已关闭 RL 智能体";
    }

    private void BindExistingKeys()
    {
        if (!string.IsNullOrEmpty(_secrets.ResolveFredApiKey())) PwdFred.Password = _secrets.ResolveFredApiKey()!;
        if (!string.IsNullOrEmpty(_secrets.ResolveFinnhubApiKey())) PwdFinnhub.Password = _secrets.ResolveFinnhubApiKey()!;
        if (!string.IsNullOrEmpty(_secrets.ResolveFmpApiKey())) PwdFmp.Password = _secrets.ResolveFmpApiKey()!;
        if (!string.IsNullOrEmpty(_secrets.ResolveLlmApiKey())) PwdLlm.Password = _secrets.ResolveLlmApiKey()!;
    }

    private void OnActivateClick(object sender, RoutedEventArgs e)
    {
        MembershipHost.Instance.Refresh();
        if (MembershipHost.Instance.TryApplyActivationCode(TxtActivationCode.Text, out var err))
        {
            TxtSettingsMessage.Text = "激活成功";
            RefreshUi();
        }
        else
        {
            TxtSettingsMessage.Text = err ?? "激活失败";
        }
    }

    private void OnCopyDeviceClick(object sender, RoutedEventArgs e)
    {
        MembershipHost.Instance.Refresh();
        var dp = new DataPackage();
        dp.SetText(MembershipHost.Instance.DiagnosticClipboardText);
        Clipboard.SetContent(dp);
        TxtSettingsMessage.Text = "已复制设备信息到剪贴板";
    }

    private void OnSaveKeysClick(object sender, RoutedEventArgs e)
    {
        SaveKeysFromUi();
        TxtSettingsMessage.Text = "API 密钥已保存到本机";
    }

    private void OnEnvCheckClick(object sender, RoutedEventArgs e)
    {
        TxtEnvLog.Text = string.Empty;
        AppendEnvLog("开始自检…");
        try
        {
            _envRepair.RunSelfCheck(AppendEnvLog);
            TxtSettingsMessage.Text = "环境自检完成（见下方日志）";
        }
        catch (Exception ex)
        {
            AppendEnvLog("自检异常: " + ex.Message);
            TxtSettingsMessage.Text = ex.Message;
        }
    }

    private async void OnEnvRepairClick(object sender, RoutedEventArgs e)
    {
        BtnEnvRepair.IsEnabled = false;
        BtnEnvCheck.IsEnabled = false;
        TxtEnvLog.Text = string.Empty;
        var started = DateTimeOffset.Now;
        TxtSettingsMessage.Text = "正在修复 Python 环境（pip 会逐行滚动，请勿关闭窗口）…";
        var timer = _dispatcher?.CreateTimer();
        if (timer is not null)
        {
            timer.Interval = TimeSpan.FromSeconds(15);
            timer.Tick += (_, _) =>
            {
                var min = (int)(DateTimeOffset.Now - started).TotalMinutes;
                TxtSettingsMessage.Text = $"正在修复… 已运行 {min} 分钟（日志在下方滚动）";
            };
            timer.Start();
        }

        try
        {
            var ok = await _envRepair.RunOneClickRepairAsync(AppendEnvLog);
            TxtSettingsMessage.Text = ok
                ? "✓ 环境修复成功：请回主界面点「连接 MT5」→「开始运行」→「保存并刷新宏观」"
                : "✗ 环境修复未完成，请查看下方日志（含 [×] 行）";
        }
        catch (Exception ex)
        {
            AppendEnvLog("修复异常: " + ex.Message);
            TxtSettingsMessage.Text = "✗ 修复异常: " + ex.Message;
        }
        finally
        {
            timer?.Stop();
            BtnEnvRepair.IsEnabled = true;
            BtnEnvCheck.IsEnabled = true;
        }
    }

    private async void OnRefreshMacroClick(object sender, RoutedEventArgs e)
    {
        SaveKeysFromUi();
        TxtSettingsMessage.Text = "正在刷新宏观数据…";
        BtnRefreshMacro.IsEnabled = false;
        try
        {
            var (ok, msg) = await _macroRefresh.RefreshAllAsync();
            TxtSettingsMessage.Text = ok ? msg : "部分失败：" + msg;
        }
        catch (Exception ex)
        {
            TxtSettingsMessage.Text = ex.Message;
        }
        finally
        {
            BtnRefreshMacro.IsEnabled = true;
        }
    }

    private void AppendEnvLog(string line)
    {
        if (_dispatcher is null || _dispatcher.HasThreadAccess)
        {
            TxtEnvLog.Text += line + Environment.NewLine;
            return;
        }

        _ = _dispatcher.TryEnqueue(() => TxtEnvLog.Text += line + Environment.NewLine);
    }

    private void SaveKeysFromUi()
    {
        _secrets.SetFredApiKey(NullIfEmpty(PwdFred.Password));
        _secrets.SetFinnhubApiKey(NullIfEmpty(PwdFinnhub.Password));
        _secrets.SetFmpApiKey(NullIfEmpty(PwdFmp.Password));
        _secrets.SetLlmApiKey(NullIfEmpty(PwdLlm.Password));
    }

    private static string? NullIfEmpty(string s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();
}
