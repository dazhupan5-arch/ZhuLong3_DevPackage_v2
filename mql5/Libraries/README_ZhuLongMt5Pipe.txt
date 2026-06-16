ZhuLongMt5Pipe.dll（x64）
=====================

烛龙三代 MT5 指标通过本 DLL 连接 ZhuLong.exe 命名管道（MQL5 原生 FileOpen 无法可靠打开 \\.\pipe\）。

生成：
  .\scripts\build-zhulong-mt5-pipe.ps1

部署到 MT5 终端：
  <MT5数据目录>\MQL5\Libraries\ZhuLongMt5Pipe.dll

终端须勾选「允许 DLL 导入」。仅支持 64 位 MetaTrader 5。

管道名（与 config.json 一致）：
  ZhuLong_Data     — 指标 → EXE（M1 K 线 JSON）
  ZhuLong_Drawing  — EXE → 指标（绘图指令 JSON）
