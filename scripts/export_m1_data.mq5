//+------------------------------------------------------------------+
//| 烛龙 ZhuLong · M1数据导出脚本                                    |
//| 导出指定日期范围的M1 K线到CSV文件，供回放测试使用                |
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs

input string InpSymbol = "XAUUSD";         // 品种
input datetime InpFromTime = D'2026.06.08 00:00'; // 起始时间 (UTC)
input datetime InpToTime = D'2026.06.10 23:59';   // 结束时间 (UTC)
input string InpFileName = "M1_export.csv";       // 导出文件名

void OnStart()
{
    if (InpFromTime >= InpToTime)
    {
        Print("错误：起始时间必须早于结束时间");
        return;
    }

    // 用 CopyRates 获取指定时间范围内的 M1 数据
    MqlRates rates[];
    int obtained = CopyRates(InpSymbol, PERIOD_M1, InpFromTime, InpToTime, rates);
    if (obtained <= 0)
    {
        Print("错误：无法获取历史数据 (", GetLastError(), ")");
        Print("请确认：1) MT5 已连接服务器 2) 图表窗口已打开 3) 历史数据已下载");
        return;
    }

    // 写入 CSV 文件到 Terminal Data 目录
    string folder = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files";
    string path = folder + "\\" + InpFileName;

    int handle = FileOpen(path, FILE_WRITE|FILE_CSV|FILE_ANSI, ",", CP_UTF8);
    if (handle == INVALID_HANDLE)
    {
        Print("错误：无法创建文件 ", path, " (", GetLastError(), ")");
        return;
    }

    // 写入 CSV 头
    FileWrite(handle, "time_unix", "time_str", "open", "high", "low", "close", "volume");

    int written = 0;
    for (int i = 0; i < obtained; i++)
    {
        // 时间戳统一为 Unix 秒（UTC）
        long time_unix = (long)rates[i].time;
        string time_str = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);

        FileWrite(handle,
            time_unix,
            time_str,
            DoubleToString(rates[i].open, 5),
            DoubleToString(rates[i].high, 5),
            DoubleToString(rates[i].low, 5),
            DoubleToString(rates[i].close, 5),
            rates[i].tick_volume
        );
        written++;
    }

    FileClose(handle);

    // 也尝试写入 Common 目录（便于 Python 读取）
    string commonPath = TerminalInfoString(TERMINAL_COMMONDATA_PATH) + "\\Files\\Common";
    if (DirectoryExists(commonPath))
    {
        FileCopy(path, 0, commonPath + "\\" + InpFileName, FILE_REWRITE);
    }

    Print("导出完成：", written, " 根 M1 K线 → ", path);
    Print("时间范围：", TimeToString(InpFromTime), " ~ ", TimeToString(InpToTime));
    Print("品种：", InpSymbol);
    Print("使用说明：将 CSV 文件复制到 simulation/data/ 目录，运行 replay_simulation.py");
}
//+------------------------------------------------------------------+
