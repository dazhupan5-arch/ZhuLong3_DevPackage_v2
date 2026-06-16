using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using ScottPlot.WinUI;
using ZhuLong.Core.Services;

namespace ZhuLong.App.Views;

public sealed partial class AnalysisPanel : UserControl
{
    private readonly AttributionService _attribution;
    private readonly DispatcherTimer _refreshTimer = new();
    private WinUIPlot? _pnlPlot;
    private bool _refreshTimerStarted;

    public AnalysisPanel()
    {
        _attribution = App.Services.GetRequiredService<AttributionService>();
        InitializeComponent();
    }

    private void EnsurePlotControl()
    {
        if (_pnlPlot is not null)
            return;

        _pnlPlot = new WinUIPlot();
        PnlPlotHost.Child = _pnlPlot;
    }

    private void EnsureRefreshTimer()
    {
        if (_refreshTimerStarted)
            return;

        _refreshTimerStarted = true;
        _refreshTimer.Interval = TimeSpan.FromSeconds(15);
        _refreshTimer.Tick += async (_, _) => await RefreshAsync();
        _refreshTimer.Start();
        Unloaded += (_, _) => _refreshTimer.Stop();
    }

    private async void OnRefreshClick(object sender, RoutedEventArgs e) => await RefreshAsync();

    public async Task RefreshAsync()
    {
        EnsurePlotControl();
        EnsureRefreshTimer();

        try
        {
            var summary = await _attribution.LoadSummaryAsync(30);
            TotalTradesText.Text = summary.TotalTrades.ToString();
            WinRateText.Text = $"{summary.WinRate * 100:F1}%";
            AvgPnlText.Text = $"{summary.AvgPnlPct:F2}%";
            ProfitFactorText.Text = summary.ProfitFactor >= 99 ? "∞" : $"{summary.ProfitFactor:F2}";
            BinList.ItemsSource = summary.ConfidenceBins
                .Select(b => $"{b.BinLabel}  n={b.Count}  胜率={b.WinRateText}  均盈亏={b.AvgPnlText}")
                .ToList();
            TradeList.ItemsSource = summary.RecentTrades
                .Select(t => $"{t.OpenTimeText}  {t.SignalId}  {t.PnlText}  {t.CloseReason}")
                .ToList();
            DrawCumulativeCurve(summary.CumulativePnlPct);
        }
        catch (Exception ex)
        {
            TotalTradesText.Text = "—";
            WinRateText.Text = "—";
            AvgPnlText.Text = "—";
            ProfitFactorText.Text = "—";
            BinList.ItemsSource = new[] { $"加载失败: {ex.Message}" };
            TradeList.ItemsSource = null;
            if (_pnlPlot is not null)
            {
                _pnlPlot.Plot.Clear();
                _pnlPlot.Refresh();
            }
        }
    }

    private void DrawCumulativeCurve(double[] values)
    {
        if (_pnlPlot is null)
            return;

        _pnlPlot.Plot.Clear();
        if (values.Length < 2)
        {
            _pnlPlot.Refresh();
            return;
        }

        var xs = Enumerable.Range(0, values.Length).Select(i => (double)i).ToArray();
        _pnlPlot.Plot.Add.Scatter(xs, values);
        _pnlPlot.Plot.Axes.Bottom.Label.Text = "交易序号";
        _pnlPlot.Plot.Axes.Left.Label.Text = "累计盈亏 %";
        _pnlPlot.Refresh();
    }
}
