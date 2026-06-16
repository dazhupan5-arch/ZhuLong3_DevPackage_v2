//+------------------------------------------------------------------+
//|                                            ZhuLongIndicator.mq5 |
//|                                      烛龙 ZhuLong · Stephen.Pan |
//|  推送 M1 K 线 → \\.\pipe\ZhuLong_Data                           |
//|  接收绘图指令 ← \\.\pipe\ZhuLong_Drawing                         |
//|  独立绘制：ATR 通道（EMA30 中轴 ± 1/2/3×ATR）+ EMA30/EMA60      |
//+------------------------------------------------------------------+
#property copyright "Stephen.Pan · 烛龙 ZhuLong"
#property link      "https://zhulong.local"
#property version   "1.18"
#property indicator_chart_window
#property indicator_buffers 8
#property indicator_plots   8

// ATR 通道上轨（1倍、2倍、3倍）
#property indicator_label1  "ATR Upper 1x"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrLightSkyBlue
#property indicator_width1  1
#property indicator_style1  STYLE_SOLID

#property indicator_label2  "ATR Upper 2x"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue
#property indicator_width2  1
#property indicator_style2  STYLE_SOLID

#property indicator_label3  "ATR Upper 3x"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrBlue
#property indicator_width3  1
#property indicator_style3  STYLE_SOLID

// ATR 通道下轨（1倍、2倍、3倍）
#property indicator_label4  "ATR Lower 1x"
#property indicator_type4   DRAW_LINE
#property indicator_color4  clrLightCoral
#property indicator_width4  1
#property indicator_style4  STYLE_SOLID

#property indicator_label5  "ATR Lower 2x"
#property indicator_type5   DRAW_LINE
#property indicator_color5  clrSalmon
#property indicator_width5  1
#property indicator_style5  STYLE_SOLID

#property indicator_label6  "ATR Lower 3x"
#property indicator_type6   DRAW_LINE
#property indicator_color6  clrRed
#property indicator_width6  1
#property indicator_style6  STYLE_SOLID

// EMA 线
#property indicator_label7  "EMA30"
#property indicator_type7   DRAW_LINE
#property indicator_color7  clrGold
#property indicator_width7  2
#property indicator_style7  STYLE_SOLID

#property indicator_label8  "EMA60"
#property indicator_type8   DRAW_LINE
#property indicator_color8  clrOrange
#property indicator_width8  1
#property indicator_style8  STYLE_DASH

//--- 与 config.json atr_channel 默认值保持一致 ---
input int      InpATRPeriod       = 14;     // ATR 周期
input int      InpEMAFast           = 30;     // 快 EMA 周期
input int      InpEMASlow           = 60;     // 慢 EMA 周期
input bool     InpShowATRChannel    = true;   // 显示 ATR 通道
input bool     InpShowEMALines      = true;   // 显示 EMA 线
input bool     InpEnablePipe        = true;   // 启用命名管道
input int      InpHistoryM1Bars     = 1000;   // 启动时推送历史 M1 根数(0=不推送)
input int      InpHistoryBatchSize  = 50;     // 历史 M1 每批根数（过大易管道拥塞）
input int      InpMaxCalcBars       = 100000; // 最大计算K线数(0=全历史, M1建议≤100000)

input group    "训练数据导出"
input bool     InpExportM5OnInit    = false;  // 启动时导出 M5 CSV
input int      InpExportM5Bars      = 100000; // 导出 M5 根数（尽量多）
input bool     InpExportM5Common    = true;   // 写入 Terminal\Common\Files（便于脚本拉取）

#import "ZhuLongMt5Pipe.dll"
   int  ZhuLongPipeConnectV1(string pipeLogicalName, int mode, uint connectTimeoutMs);
   int  ZhuLongPipeWriteLineV1(int handleId, string line);
   int  ZhuLongPipeReadLineV1(int handleId, uchar &buffer[], int capacity);
   int  ZhuLongPipePollV1(int handleId);
   void ZhuLongPipeDisconnectV1(int handleId);
#import

#define ZL_PIPE_WRITE 1
#define ZL_PIPE_READ  2

//--- 指标缓冲区 ---
double ATR_Upper1[];
double ATR_Upper2[];
double ATR_Upper3[];
double ATR_Lower1[];
double ATR_Lower2[];
double ATR_Lower3[];
double EMA30_Buffer[];
double EMA60_Buffer[];

//--- 内置指标句柄（终端 C++ 计算，避免 MQL5 全历史手算卡死）---
int            hEmaFast         = INVALID_HANDLE;
int            hEmaSlow         = INVALID_HANDLE;
int            hAtr             = INVALID_HANDLE;
string         g_handleSymbol   = "";
ENUM_TIMEFRAMES g_handlePeriod = PERIOD_CURRENT;

//--- 命名管道（ZhuLongMt5Pipe.dll）---
int      hDataPipe            = 0;
int      hDrawPipe            = 0;
datetime lastBarTime            = 0;
int    g_trailIdx               = 0;        // 移动止损轨迹线计数器
datetime lastDataPipeFailTime   = 0;
datetime lastDataHeartbeatTime  = 0;
datetime lastDrawPipeFailTime   = 0;
bool     pipeWarnOnce           = false;
bool     g_indicatorReady       = false;
bool     g_exportM5Pending      = false;
bool     g_historySent          = false;
bool     g_historyPending       = false;
int      g_historyNextEndIdx    = -1;
int      g_timerTicks           = 0;

#define ZL_PIPE_CONNECT_MS       2000  // 数据管道 connect 超时（WaitNamedPipe 预算）
#define ZL_PIPE_DRAW_CONNECT_MS  2000  // 绘图管道：短超时 + 下轮 OnTimer 再试，避免卡 chart
#define ZL_PIPE_RETRY_SEC        2
#define ZL_PIPE_DRAW_RETRY_SEC   2     // 绘图管道重试间隔
#define ZL_PIPE_WRITE_ATTEMPTS   5     // 实时/心跳写入：约 100ms 预算
#define ZL_PIPE_HISTORY_ATTEMPTS 8     // 历史批次：失败下 tick 续传
#define ZL_TIMER_SEC             2
#define ZL_DRAW_POLL_EVERY       1     // 每轮定时器都读绘图管道
#define ZL_WARM_GV_KEY           "ZL_WarmReconnect"
#define ZL_HISTORY_DONE_GV_KEY   "ZL_ColdHistoryDone"

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, ATR_Upper1, INDICATOR_DATA);
   SetIndexBuffer(1, ATR_Upper2, INDICATOR_DATA);
   SetIndexBuffer(2, ATR_Upper3, INDICATOR_DATA);
   SetIndexBuffer(3, ATR_Lower1, INDICATOR_DATA);
   SetIndexBuffer(4, ATR_Lower2, INDICATOR_DATA);
   SetIndexBuffer(5, ATR_Lower3, INDICATOR_DATA);
   SetIndexBuffer(6, EMA30_Buffer, INDICATOR_DATA);
   SetIndexBuffer(7, EMA60_Buffer, INDICATOR_DATA);

   PlotIndexSetInteger(0, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(0, PLOT_LINE_COLOR, clrLightSkyBlue);
   PlotIndexSetInteger(0, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(0, PLOT_LABEL, "ATR Upper 1x");

   PlotIndexSetInteger(1, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(1, PLOT_LINE_COLOR, clrDodgerBlue);
   PlotIndexSetInteger(1, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(1, PLOT_LABEL, "ATR Upper 2x");

   PlotIndexSetInteger(2, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(2, PLOT_LINE_COLOR, clrBlue);
   PlotIndexSetInteger(2, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(2, PLOT_LABEL, "ATR Upper 3x");

   PlotIndexSetInteger(3, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(3, PLOT_LINE_COLOR, clrLightCoral);
   PlotIndexSetInteger(3, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(3, PLOT_LABEL, "ATR Lower 1x");

   PlotIndexSetInteger(4, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(4, PLOT_LINE_COLOR, clrSalmon);
   PlotIndexSetInteger(4, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(4, PLOT_LABEL, "ATR Lower 2x");

   PlotIndexSetInteger(5, PLOT_DRAW_TYPE, InpShowATRChannel ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(5, PLOT_LINE_COLOR, clrRed);
   PlotIndexSetInteger(5, PLOT_LINE_WIDTH, 1);
   PlotIndexSetString(5, PLOT_LABEL, "ATR Lower 3x");

   PlotIndexSetInteger(6, PLOT_DRAW_TYPE, InpShowEMALines ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(6, PLOT_LINE_COLOR, clrGold);
   PlotIndexSetInteger(6, PLOT_LINE_WIDTH, 2);
   PlotIndexSetInteger(6, PLOT_LINE_STYLE, STYLE_SOLID);
   PlotIndexSetString(6, PLOT_LABEL, "EMA" + IntegerToString(InpEMAFast));

   PlotIndexSetInteger(7, PLOT_DRAW_TYPE, InpShowEMALines ? DRAW_LINE : DRAW_NONE);
   PlotIndexSetInteger(7, PLOT_LINE_COLOR, clrOrange);
   PlotIndexSetInteger(7, PLOT_LINE_WIDTH, 1);
   PlotIndexSetInteger(7, PLOT_LINE_STYLE, STYLE_DASH);
   PlotIndexSetString(7, PLOT_LABEL, "EMA" + IntegerToString(InpEMASlow));

   IndicatorSetString(INDICATOR_SHORTNAME, "ZhuLong v1.18");
   g_indicatorReady = false;
   g_exportM5Pending = InpExportM5OnInit;
   g_historySent = false;
   g_historyPending = (InpEnablePipe && InpHistoryM1Bars > 0 && !IsWarmChartChangePending());

   if(!EnsureCoreHandles())
      return(INIT_FAILED);

   if(InpEnablePipe)
     {
      if(g_historyPending)
         Print("烛龙 ZhuLong 指标已启动 — 冷连接，将推送历史 M1 ", InpHistoryM1Bars, " 根");
      else
         Print("烛龙 ZhuLong 指标已启动 — 热重连（切换周期），跳过历史，立即恢复实时 M1");
     }

   EventSetTimer(ZL_TIMER_SEC);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   ReleaseCoreHandles();
   if(hDataPipe > 0) { ZhuLongPipeDisconnectV1(hDataPipe); hDataPipe = 0; }
   ReleaseDrawPipe();
   if(reason == REASON_CHARTCHANGE)
      GlobalVariableSet(ZL_WARM_GV_KEY, 1.0);
  }

//+------------------------------------------------------------------+
void ReleaseCoreHandles()
  {
   if(hEmaFast != INVALID_HANDLE) { IndicatorRelease(hEmaFast); hEmaFast = INVALID_HANDLE; }
   if(hEmaSlow != INVALID_HANDLE) { IndicatorRelease(hEmaSlow); hEmaSlow = INVALID_HANDLE; }
   if(hAtr != INVALID_HANDLE)     { IndicatorRelease(hAtr);     hAtr     = INVALID_HANDLE; }
   g_handleSymbol = "";
   g_handlePeriod = PERIOD_CURRENT;
  }

//+------------------------------------------------------------------+
bool EnsureCoreHandles()
  {
   if(hEmaFast != INVALID_HANDLE && g_handleSymbol == _Symbol && g_handlePeriod == _Period)
      return true;

   ReleaseCoreHandles();
   hEmaFast = iMA(_Symbol, _Period, InpEMAFast, 0, MODE_EMA, PRICE_CLOSE);
   hEmaSlow = iMA(_Symbol, _Period, InpEMASlow, 0, MODE_EMA, PRICE_CLOSE);
   hAtr     = iATR(_Symbol, _Period, InpATRPeriod);
   if(hEmaFast == INVALID_HANDLE || hEmaSlow == INVALID_HANDLE || hAtr == INVALID_HANDLE)
     {
      Print("创建 iMA/iATR 句柄失败 err=", GetLastError());
      return false;
     }
   g_handleSymbol = _Symbol;
   g_handlePeriod = _Period;
   lastBarTime = 0;
   return true;
  }

//+------------------------------------------------------------------+
void SetAtrChannelValues(const int i, const double atr, const double emaMid)
  {
   ATR_Upper1[i] = emaMid + 1.0 * atr;
   ATR_Upper2[i] = emaMid + 2.0 * atr;
   ATR_Upper3[i] = emaMid + 3.0 * atr;
   ATR_Lower1[i] = emaMid - 1.0 * atr;
   ATR_Lower2[i] = emaMid - 2.0 * atr;
   ATR_Lower3[i] = emaMid - 3.0 * atr;
  }

//+------------------------------------------------------------------+
void ClearAtrChannelValues(const int i)
  {
   ATR_Upper1[i] = EMPTY_VALUE;
   ATR_Upper2[i] = EMPTY_VALUE;
   ATR_Upper3[i] = EMPTY_VALUE;
   ATR_Lower1[i] = EMPTY_VALUE;
   ATR_Lower2[i] = EMPTY_VALUE;
   ATR_Lower3[i] = EMPTY_VALUE;
  }

//+------------------------------------------------------------------+
int UpdateIndicatorBuffers(const int rates_total, const int prev_calculated)
  {
   if(!EnsureCoreHandles())
      return 0;

   const int minBars = MathMax(InpATRPeriod, MathMax(InpEMAFast, InpEMASlow));
   if(rates_total < minBars)
      return 0;

   if(BarsCalculated(hEmaFast) < minBars || BarsCalculated(hAtr) < minBars)
      return 0;

   // 直接 CopyBuffer 到指标缓冲区（index 0 = 最旧 K 线），勿经中间数组倒序
   if(CopyBuffer(hEmaFast, 0, 0, rates_total, EMA30_Buffer) <= 0)
     {
      Print("CopyBuffer EMA", InpEMAFast, " 失败 err=", GetLastError());
      return prev_calculated;
     }
   if(CopyBuffer(hEmaSlow, 0, 0, rates_total, EMA60_Buffer) <= 0)
     {
      Print("CopyBuffer EMA", InpEMASlow, " 失败 err=", GetLastError());
      return prev_calculated;
     }

   double atrVals[];
   ArrayResize(atrVals, rates_total);
   ArraySetAsSeries(atrVals, false);
   if(CopyBuffer(hAtr, 0, 0, rates_total, atrVals) <= 0)
     {
      Print("CopyBuffer ATR 失败 err=", GetLastError());
      return prev_calculated;
     }

   int calcFrom = 0;
   if(InpMaxCalcBars > 0 && rates_total > InpMaxCalcBars)
      calcFrom = rates_total - InpMaxCalcBars;

   int loopFrom = (prev_calculated > 0) ? prev_calculated - 1 : 0;
   if(loopFrom < calcFrom)
      loopFrom = calcFrom;

   for(int i = 0; i < calcFrom; i++)
     {
      ClearAtrChannelValues(i);
      EMA30_Buffer[i] = EMPTY_VALUE;
      EMA60_Buffer[i] = EMPTY_VALUE;
     }

   for(int i = loopFrom; i < rates_total; i++)
     {
      if(i < InpATRPeriod - 1)
        {
         ClearAtrChannelValues(i);
         if(!InpShowEMALines)
           {
            EMA30_Buffer[i] = EMPTY_VALUE;
            EMA60_Buffer[i] = EMPTY_VALUE;
           }
         continue;
        }

      double atr = atrVals[i];
      if(atr <= 0.0 || atr == EMPTY_VALUE)
        {
         ClearAtrChannelValues(i);
         continue;
        }

      SetAtrChannelValues(i, atr, EMA30_Buffer[i]);

      if(!InpShowEMALines)
        {
         EMA30_Buffer[i] = EMPTY_VALUE;
         EMA60_Buffer[i] = EMPTY_VALUE;
        }
      if(!InpShowATRChannel)
         ClearAtrChannelValues(i);
     }

   return rates_total;
  }

//+------------------------------------------------------------------+
void ExportM5HistoryCsv()
  {
   const int want = MathMax(InpExportM5Bars, 100);
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, PERIOD_M5, 0, want, rates);
   if(copied <= 0)
     {
      Print("M5 导出失败 CopyRates=0 symbol=", _Symbol, " err=", GetLastError());
      return;
     }

   const string folder = "ZhuLong";
   const string fname  = folder + "\\" + _Symbol + "_M5.csv";
   const int flags = FILE_WRITE | FILE_CSV | FILE_ANSI | (InpExportM5Common ? FILE_COMMON : 0);

   if(InpExportM5Common)
      FolderCreate(folder, FILE_COMMON);

   int fh = FileOpen(fname, flags, ',');
   if(fh == INVALID_HANDLE)
     {
      Print("M5 导出失败 FileOpen err=", GetLastError(), " path=", fname);
      return;
     }

   FileWrite(fh, "time", "open", "high", "low", "close", "volume");
   for(int i = 0; i < copied; i++)
     {
      MqlDateTime dt;
      TimeToStruct(rates[i].time, dt);
      string ts = StringFormat("%04d-%02d-%02d %02d:%02d:%02d",
                               dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
      FileWrite(fh,
                ts,
                DoubleToString(rates[i].open, _Digits),
                DoubleToString(rates[i].high, _Digits),
                DoubleToString(rates[i].low, _Digits),
                DoubleToString(rates[i].close, _Digits),
                IntegerToString(rates[i].tick_volume));
     }
   FileClose(fh);

   string where = InpExportM5Common ? "Terminal\\Common\\Files\\" : "MQL5\\Files\\";
   Print("M5 导出完成 ", copied, " 根 → ", where, fname,
         " （用于 train.py --m5-csv）");
  }

//+------------------------------------------------------------------+
bool PipeRetryCooldownActive(const datetime lastFail, const int cooldownSec = ZL_PIPE_RETRY_SEC)
  {
   if(lastFail == 0)
      return false;
   return (TimeLocal() - lastFail) < cooldownSec;
  }

//+------------------------------------------------------------------+
bool IsWarmChartChangePending()
  {
   return GlobalVariableCheck(ZL_WARM_GV_KEY) && GlobalVariableGet(ZL_WARM_GV_KEY) > 0.0;
  }

//+------------------------------------------------------------------+
bool ConsumeWarmChartChangeFlag()
  {
   if(!IsWarmChartChangePending())
      return false;
   GlobalVariableDel(ZL_WARM_GV_KEY);
   return true;
  }

//+------------------------------------------------------------------+
bool ShouldSkipHistoryOnReconnect()
  {
   return IsWarmChartChangePending()
       || (GlobalVariableCheck(ZL_HISTORY_DONE_GV_KEY) && GlobalVariableGet(ZL_HISTORY_DONE_GV_KEY) > 0.0);
  }

//+------------------------------------------------------------------+
void ReleaseDrawPipe()
  {
   if(hDrawPipe > 0)
     {
      ZhuLongPipeDisconnectV1(hDrawPipe);
      hDrawPipe = 0;
     }
  }

//+------------------------------------------------------------------+
bool IsDrawPipeAlive()
  {
   if(hDrawPipe <= 0)
      return false;
   int poll = ZhuLongPipePollV1(hDrawPipe);
   if(poll >= 0)
      return true;
   Print("ZhuLong_Drawing 管道已失效 poll=", poll, "，将重连");
   ReleaseDrawPipe();
   lastDrawPipeFailTime = TimeLocal();
   return false;
  }

//+------------------------------------------------------------------+
void InvalidateDataPipe(const string reason)
  {
   if(hDataPipe <= 0)
      return;
   Print("ZhuLong_Data 管道将重连：", reason);
   ZhuLongPipeDisconnectV1(hDataPipe);
   hDataPipe = 0;
   lastBarTime = 0;
   lastDataPipeFailTime = TimeLocal();
   // 烛龙重启时两条管道同时失效；数据写失败时绘图 handle 亦不可信
   ReleaseDrawPipe();
   lastDrawPipeFailTime = 0;
   if(InpHistoryM1Bars <= 0)
      return;
   if(StringFind(reason, "心跳") >= 0)
     {
      GlobalVariableDel(ZL_HISTORY_DONE_GV_KEY);
      g_historySent = false;
      g_historyPending = true;
      g_historyNextEndIdx = -1;
      return;
     }
   if(!ShouldSkipHistoryOnReconnect())
     {
      g_historySent = false;
      g_historyPending = true;
      g_historyNextEndIdx = -1;
     }
  }

//+------------------------------------------------------------------+
bool EnsureDataPipe()
  {
   // 注意：ZhuLongPipePollV1 仅适用于读管道；写管道调用恒返回 -1，不可用于存活检测
   if(hDataPipe > 0)
      return true;
   if(PipeRetryCooldownActive(lastDataPipeFailTime))
      return false;

   hDataPipe = ZhuLongPipeConnectV1("ZhuLong_Data", ZL_PIPE_WRITE, ZL_PIPE_CONNECT_MS);
   if(hDataPipe <= 0)
     {
      lastDataPipeFailTime = TimeLocal();
      if(!pipeWarnOnce)
        {
         pipeWarnOnce = true;
         Print("无法连接 ZhuLong_Data 管道 — 请先运行 ZhuLong.exe 并点「连接 MT5」；确认 MQL5\\Libraries\\ZhuLongMt5Pipe.dll 与允许 DLL 导入");
        }
      return false;
     }
   Print("已连接 ZhuLong_Data 管道 handle=", hDataPipe);
   pipeWarnOnce = false;
   lastDataPipeFailTime = 0;
   lastBarTime = 0;

   const bool warm = ConsumeWarmChartChangeFlag() || ShouldSkipHistoryOnReconnect();
   if(warm && InpHistoryM1Bars > 0)
     {
      g_historySent = true;
      g_historyPending = false;
      g_historyNextEndIdx = -1;
      Print("管道热重连 — 跳过历史 M1，立即恢复实时推送");
      if(!SendWarmSessionNotice())
         Print("热重连 session 通知失败（烛龙仍可通过 API 补数）");
     }
   else if(InpHistoryM1Bars > 0)
     {
      g_historySent = false;
      g_historyPending = true;
      g_historyNextEndIdx = -1;
      Print("管道冷重连 — 将推送历史 M1 ", InpHistoryM1Bars, " 根");
     }
   return true;
  }

//+------------------------------------------------------------------+
bool EnsureDrawPipe()
  {
   if(hDrawPipe > 0)
      return IsDrawPipeAlive();
   if(PipeRetryCooldownActive(lastDrawPipeFailTime, ZL_PIPE_DRAW_RETRY_SEC))
      return false;

   hDrawPipe = ZhuLongPipeConnectV1("ZhuLong_Drawing", ZL_PIPE_READ, ZL_PIPE_DRAW_CONNECT_MS);
   if(hDrawPipe <= 0)
     {
      lastDrawPipeFailTime = TimeLocal();
      static int drawFailCount = 0;
      drawFailCount++;
      if(drawFailCount <= 3 || drawFailCount % 20 == 0)
         Print("无法连接 ZhuLong_Drawing 管道（第", drawFailCount, "次）— 请确认 ZhuLong.exe 已运行并点「连接 MT5」");
      return false;
     }
   Print("已连接 ZhuLong_Drawing 管道 handle=", hDrawPipe);
   lastDrawPipeFailTime = 0;
   return true;
  }

//+------------------------------------------------------------------+
int ServerOffsetSec()
  {
   return (int)(TimeTradeServer() - TimeGMT());
  }

long BarTimeUnixUtc(const datetime t)
  {
   // 与 Python mt5_ops 一致：去掉服务器时区，得到真 UTC Unix
   return (long)t - ServerOffsetSec();
  }

//+------------------------------------------------------------------+
bool WriteDataPipeLineWithRetry(const string json, const int maxAttempts = ZL_PIPE_WRITE_ATTEMPTS)
  {
   if(hDataPipe <= 0)
      return false;
   for(int attempt = 0; attempt < maxAttempts; attempt++)
     {
      if(ZhuLongPipeWriteLineV1(hDataPipe, json) == 1)
         return true;
      Sleep(20);
     }
   return false;
  }

//+------------------------------------------------------------------+
bool SendWarmSessionNotice()
  {
   string json = StringFormat(
      "{\"type\":\"session\",\"warm\":true,\"symbol\":\"%s\",\"chart_id\":%I64d}",
      _Symbol, ChartID());
   return WriteDataPipeLineWithRetry(json, ZL_PIPE_WRITE_ATTEMPTS);
  }

//+------------------------------------------------------------------+
string FormatBarJsonObject(const MqlRates &r)
  {
   return StringFormat(
      "{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%s}",
      BarTimeUnixUtc(r.time),
      r.open, r.high, r.low, r.close,
      IntegerToString(r.tick_volume)
   );
  }

//+------------------------------------------------------------------+
bool SendHistoryM1Bars()
  {
   if(g_historySent || !g_historyPending || InpHistoryM1Bars <= 0)
      return true;
   if(!EnsureDataPipe())
      return false;

   const int want = MathMin(InpHistoryM1Bars, 5000);
   const int batch = MathMax(InpHistoryBatchSize, 10);

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(_Symbol, PERIOD_M1, 1, want, rates);
   if(copied <= 0)
     {
      Print("历史 M1 推送失败 CopyRates=0 symbol=", _Symbol, " err=", GetLastError());
      return false;
     }

   if(g_historyNextEndIdx < 0)
      g_historyNextEndIdx = copied - 1;

   if(g_historyNextEndIdx < 0)
     {
      g_historySent = true;
      g_historyPending = false;
      return true;
     }

   int endIdx = g_historyNextEndIdx;
   int startIdx = MathMax(0, endIdx - batch + 1);
   bool isFinal = (startIdx == 0);

   string body = "";
   for(int i = endIdx; i >= startIdx; i--)
     {
      if(StringLen(body) > 0)
         body += ",";
      body += FormatBarJsonObject(rates[i]);
     }

   string json = StringFormat(
      "{\"type\":\"m1_history\",\"symbol\":\"%s\",\"final\":%s,\"bars\":[%s]}",
      _Symbol,
      isFinal ? "true" : "false",
      body
   );

   if(!WriteDataPipeLineWithRetry(json, ZL_PIPE_HISTORY_ATTEMPTS))
     {
      Print("历史 M1 批次写入失败 endIdx=", endIdx, "，下轮重试");
      InvalidateDataPipe("历史 M1 写入失败");
      return false;
     }

   g_historyNextEndIdx = startIdx - 1;
   if(isFinal)
     {
      lastBarTime = rates[0].time;
      g_historyNextEndIdx = -1;
      g_historySent = true;
      g_historyPending = false;
      GlobalVariableSet(ZL_HISTORY_DONE_GV_KEY, 1.0);
      Print("历史 M1 已推送 ", copied, " 根 symbol=", _Symbol, " → 烛龙3 可立即推理");
     }
   return true;
  }

//+------------------------------------------------------------------+
void SendBar()
  {
   if(!InpEnablePipe)
      return;
   if(!EnsureDataPipe())
      return;

   datetime t = iTime(_Symbol, PERIOD_M1, 1);
   if(t == 0 || t == lastBarTime)
      return;
   lastBarTime = t;

   double o = iOpen(_Symbol, PERIOD_M1, 1);
   double h = iHigh(_Symbol, PERIOD_M1, 1);
   double l = iLow(_Symbol, PERIOD_M1, 1);
   double c = iClose(_Symbol, PERIOD_M1, 1);
   long   v = iVolume(_Symbol, PERIOD_M1, 1);

   string json = StringFormat(
      "{\"type\":\"bar\",\"symbol\":\"%s\",\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"volume\":%s}",
      _Symbol,
      BarTimeUnixUtc(t),
      o, h, l, c,
      IntegerToString(v)
   );

   if(!WriteDataPipeLineWithRetry(json, ZL_PIPE_WRITE_ATTEMPTS))
      InvalidateDataPipe("M1 写入失败");
  }

//+------------------------------------------------------------------+
void PollDrawingCommands()
  {
   if(!InpEnablePipe)
      return;
   if(!EnsureDrawPipe())
      return;

   uchar buf[];
   ArrayResize(buf, 16384);
   ArrayInitialize(buf, 0);

   // 循环读取直到管道无数据——一条连接内可能有多次写入
   int pollCount = 0;
   while(pollCount < 8)
   {
      int rc = ZhuLongPipeReadLineV1(hDrawPipe, buf, 16384);
      if(rc == 1)
      {
         string line = CharArrayToString(buf, 0, WHOLE_ARRAY, CP_UTF8);
         if(StringLen(line) >= 5)
         {
            ProcessDrawCommand(line);
            pollCount++;
            ArrayInitialize(buf, 0);
            continue;
         }
         pollCount++;
         continue;
      }
      else if(rc < 0)
      {
         if(ZhuLongPipePollV1(hDrawPipe) < 0)
           {
            Print("ZhuLong_Drawing 读失败 rc=", rc, "，将重连");
            ReleaseDrawPipe();
           }
         break;
      }
      // rc == 0: 暂无数据，停止轮询
      break;
   }
  }

//+------------------------------------------------------------------+
void ProcessDrawCommand(string json)
  {
   string action = JsonGet(json, "action");
   if(action == "ping")
      return;
   if(action == "draw_signal")
     {
      string sym  = JsonGet(json, "symbol");
      string sid  = JsonGet(json, "signal_id");
      string dir  = JsonGet(json, "direction");

      if(StringLen(sym) > 0 && !SymbolMatchesChart(sym))
        {
         Print("⚠ 烛龙绘图跳过：符号不匹配 payload=", sym, " chart=", _Symbol, " signal_id=", sid,
               "（请检查 config.json symbol_mapping）");
         return;
        }

      double entry = StringToDouble(JsonGet(json, "entry"));
      double sl    = StringToDouble(JsonGet(json, "sl"));
      double tp    = StringToDouble(JsonGet(json, "tp"));
      string conf  = JsonGet(json, "confidence");
      string expiry = JsonGet(json, "expiry_minutes");
      string label = "ZL conf=" + conf + " exp=" + expiry + "m";
      DrawSignalObjects(sid, dir, entry, sl, tp, label);
     }
   else if(action == "clear_signal")
      ClearSignalObjects(JsonGet(json, "signal_id"));
   else if(action == "clear_all")
      ClearAllZhuLongObjects();
   else if(StringLen(action) > 0)
     {
      Print("⚠ 烛龙绘图：未知 action=", action, " json=", StringSubstr(json, 0, 200));
     }
  }

//+------------------------------------------------------------------+
bool SymbolMatchesChart(string payloadSym)
  {
   if(payloadSym == _Symbol)
      return true;
   string a = payloadSym;
   string b = _Symbol;
   StringToUpper(a);
   StringToUpper(b);
   if(a == b)
      return true;

   // 后缀容错：剥离常见经纪商后缀后再比较（. / m / # / pro / ecn / raw / std 等）
   if(StripBrokerSuffix(a) == StripBrokerSuffix(b))
      return true;

   // 原油经纪商别名：USOIL / XTIUSD / CL-OIL
   if((StringFind(a, "OIL") >= 0 || StringFind(a, "XTI") >= 0 || StringFind(a, "WTI") >= 0) &&
      (StringFind(b, "OIL") >= 0 || StringFind(b, "XTI") >= 0 || StringFind(b, "WTI") >= 0))
      return true;
   if((StringFind(a, "XAU") >= 0 || StringFind(a, "GOLD") >= 0) &&
      (StringFind(b, "XAU") >= 0 || StringFind(b, "GOLD") >= 0))
      return true;
   return false;
  }

//+------------------------------------------------------------------+
//| 剥离经纪商常见后缀（XAUUSD.pro → XAUUSD / XAUUSDm → XAUUSD 等）  |
//+------------------------------------------------------------------+
string StripBrokerSuffix(string symbol)
  {
   // 常见后缀模式（从长到短匹配）
   string suffixes[] = {".PRO", ".ECN", ".RAW", ".STD", ".MICRO", ".MINI",
                         "PRO", "ECN", "RAW", "STD"};
   for(int i = 0; i < ArraySize(suffixes); i++)
     {
      int pos = StringFind(symbol, suffixes[i]);
      if(pos >= 2)
        {
         string stripped = StringSubstr(symbol, 0, pos);
         // 如果剥离后末尾还有残留符号（. / # / m / -），继续剥离
         int last = StringLen(stripped) - 1;
         while(last >= 2)
           {
            ushort ch = StringGetCharacter(stripped, last);
            if(ch == '.' || ch == '#' || ch == '-' || ch == 'm' || ch == 'M')
              {
               stripped = StringSubstr(stripped, 0, last);
               last = StringLen(stripped) - 1;
              }
            else break;
           }
         return stripped;
        }
     }

   // 无已知后缀模式则剥离末尾的 . / # / m / -
   int len = StringLen(symbol);
   while(len > 2)
     {
      ushort ch = StringGetCharacter(symbol, len - 1);
      if(ch == '.' || ch == '#' || ch == '-' || ch == 'm' || ch == 'M')
        {
         symbol = StringSubstr(symbol, 0, len - 1);
         len = StringLen(symbol);
        }
      else break;
     }
   return symbol;
  }

//+------------------------------------------------------------------+
string JsonGet(string json, string key)
  {
   string pat = "\"" + key + "\":";
   int p = StringFind(json, pat);
   if(p < 0) return "";
   p += StringLen(pat);
   if(StringGetCharacter(json, p) == '\"')
     {
      p++;
      int q = StringFind(json, "\"", p);
      return StringSubstr(json, p, q - p);
     }
   int q = StringFind(json, ",", p);
   if(q < 0) q = StringFind(json, "}", p);
   return StringSubstr(json, p, q - p);
  }

//+------------------------------------------------------------------+
bool IsLongDirection(string dir)
  {
   StringTrimLeft(dir);
   StringTrimRight(dir);
   StringToLower(dir);
   return (dir == "buy" || dir == "long");
  }

//+------------------------------------------------------------------+
//| 箭头放在 K 线外侧，避免遮挡实体                                   |
//+------------------------------------------------------------------+
double SignalArrowPrice(const bool isLong, const double entry)
  {
   double low0  = iLow(_Symbol, _Period, 0);
   double high0 = iHigh(_Symbol, _Period, 0);
   if(low0 <= 0.0 || high0 <= 0.0)
     {
      double gap = MathMax(_Point * 150, entry * 0.0015);
      return isLong ? entry - gap : entry + gap;
     }
   double barRange = MathMax(high0 - low0, _Point * 20);
   double gap = MathMax(barRange * 1.5, _Point * 150);
   return isLong ? (low0 - gap) : (high0 + gap);
  }

//+------------------------------------------------------------------+
void ClearAllZhuLongObjects()
  {
   int n = ObjectsDeleteAll(0, "ZL_");
   if(n > 0)
      Print("烛龙清除全部图表对象 deleted=", n);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
void ClearSignalObjects(string sid)
  {
   StringTrimLeft(sid);
   StringTrimRight(sid);
   if(StringLen(sid) < 1)
      return;

   string prefix = "ZL_" + sid + "_";
   ObjectDelete(0, prefix + "arrow");
   ObjectDelete(0, prefix + "entry");
   ObjectDelete(0, prefix + "sl");
   ObjectDelete(0, prefix + "tp");
   ObjectDelete(0, prefix + "lbl");
   int n = ObjectsDeleteAll(0, "ZL_" + sid);
   if(n > 0)
      Print("烛龙清除图表对象 signal_id=", sid, " deleted=", n);
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
void DrawSignalObjects(string sid, string dir, double entry, double sl, double tp, string labelExtra)
  {
   if(StringLen(sid) < 2)
      return;

   datetime t = iTime(_Symbol, _Period, 0);
   if(t <= 0)
      t = TimeCurrent();

   if(entry <= 0.0)
      entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   bool isLong = IsLongDirection(dir);
   double arrowPrice = SignalArrowPrice(isLong, entry);

   string prefix = "ZL_" + sid + "_";
   color clr = isLong ? clrLime : clrRed;

   // 记录旧的 SL 价格（用于移动止损轨迹线；OBJ_HLINE 用 OBJPROP_PRICE）
   double oldSl = 0.0;
   if(ObjectFind(0, prefix + "sl") >= 0)
      ObjectGetDouble(0, prefix + "sl", OBJPROP_PRICE, 0, oldSl);

   ObjectDelete(0, prefix + "arrow");
   ObjectDelete(0, prefix + "sl");
   ObjectDelete(0, prefix + "tp");
   ObjectDelete(0, prefix + "lbl");
   ObjectDelete(0, prefix + "entry");

   ObjectCreate(0, prefix + "arrow", OBJ_ARROW, 0, t, arrowPrice);
   ObjectSetInteger(0, prefix + "arrow", OBJPROP_COLOR, clr);
   ObjectSetInteger(0, prefix + "arrow", OBJPROP_ARROWCODE, isLong ? 233 : 234);
   ObjectSetInteger(0, prefix + "arrow", OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, prefix + "arrow", OBJPROP_ANCHOR, isLong ? ANCHOR_TOP : ANCHOR_BOTTOM);

   ObjectCreate(0, prefix + "entry", OBJ_HLINE, 0, 0, entry);
   ObjectSetInteger(0, prefix + "entry", OBJPROP_COLOR, clrGold);
   ObjectSetInteger(0, prefix + "entry", OBJPROP_STYLE, STYLE_DASH);
   ObjectSetString(0, prefix + "entry", OBJPROP_TEXT, "Entry");

   if(sl > 0.0)
     {
      ObjectCreate(0, prefix + "sl", OBJ_HLINE, 0, 0, sl);
      ObjectSetInteger(0, prefix + "sl", OBJPROP_COLOR, clrRed);
      ObjectSetInteger(0, prefix + "sl", OBJPROP_WIDTH, 2);
      ObjectSetString(0, prefix + "sl", OBJPROP_TEXT, "SL " + DoubleToString(sl, _Digits));

      // 移动止损轨迹线：从上一个 SL 位置连线到当前 SL
      if(oldSl > 0.0 && MathAbs(oldSl - sl) > _Point * 10)
        {
         string trailPrefix = prefix + "trail_";
         g_trailIdx++;
         if(g_trailIdx > 9999)
            g_trailIdx = 0;
         ObjectCreate(0, trailPrefix + IntegerToString(g_trailIdx), OBJ_TREND, 0, t, oldSl, t, sl);
         ObjectSetInteger(0, trailPrefix + IntegerToString(g_trailIdx), OBJPROP_COLOR, clrCoral);
         ObjectSetInteger(0, trailPrefix + IntegerToString(g_trailIdx), OBJPROP_WIDTH, 1);
         ObjectSetInteger(0, trailPrefix + IntegerToString(g_trailIdx), OBJPROP_STYLE, STYLE_DOT);
         ObjectSetInteger(0, trailPrefix + IntegerToString(g_trailIdx), OBJPROP_RAY_RIGHT, false);
         ObjectSetInteger(0, trailPrefix + IntegerToString(g_trailIdx), OBJPROP_BACK, true);
        }
     }

   if(tp > 0.0)
     {
      ObjectCreate(0, prefix + "tp", OBJ_HLINE, 0, 0, tp);
      ObjectSetInteger(0, prefix + "tp", OBJPROP_COLOR, clrGreen);
      ObjectSetString(0, prefix + "tp", OBJPROP_TEXT, "TP");
     }

   if(StringLen(labelExtra) > 0)
     {
      ObjectCreate(0, prefix + "lbl", OBJ_TEXT, 0, t, arrowPrice);
      ObjectSetString(0, prefix + "lbl", OBJPROP_TEXT, labelExtra);
      ObjectSetInteger(0, prefix + "lbl", OBJPROP_COLOR, clr);
      ObjectSetInteger(0, prefix + "lbl", OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, prefix + "lbl", OBJPROP_ANCHOR, isLong ? ANCHOR_UPPER : ANCHOR_LOWER);
     }

   string dirLabel = isLong ? "buy" : "sell";
   Print("烛龙绘图 ", dirLabel, " ", _Symbol, " entry=", DoubleToString(entry, _Digits),
         " sl=", DoubleToString(sl, _Digits), " tp=", DoubleToString(tp, _Digits));
   ChartRedraw(0);
  }

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   int done = UpdateIndicatorBuffers(rates_total, prev_calculated);
   if(done >= rates_total)
      g_indicatorReady = true;
   return done;
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   // 无论指标是否就绪，始终轮询绘图管道（信号绘制不依赖 ATR/EMA 缓冲区）
   PollDrawingCommands();

   if(!g_indicatorReady)
     {
      // 指标缓冲区未就绪时，仅维持管道通信，不推送数据
      static int notReadyLogTicks = 0;
      if(++notReadyLogTicks % 30 == 1)  // 每 60 秒输出一次状态
         Print("烛龙指标缓冲区未就绪（等待 ≥60 根 K 线），管道通信正常轮询中");
      return;
     }

   if(g_exportM5Pending)
     {
      g_exportM5Pending = false;
      ExportM5HistoryCsv();
     }

   SendBar();

   if(g_historyPending && !g_historySent)
      SendHistoryM1Bars();

   // 写管道无法用 Poll 检测存活；定期心跳探测烛龙是否重启
   if(hDataPipe > 0 && TimeLocal() - lastDataHeartbeatTime >= 20)
     {
      lastDataHeartbeatTime = TimeLocal();
      if(!WriteDataPipeLineWithRetry("{\"type\":\"heartbeat\"}", ZL_PIPE_WRITE_ATTEMPTS))
         InvalidateDataPipe("心跳写入失败（烛龙可能已重启）");
     }

  }
//+------------------------------------------------------------------+
