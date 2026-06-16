# v12 回测评估（别名，指向 run_v12_evaluation.ps1）
param([string]$Symbol = 'XAUUSD')
& (Join-Path $PSScriptRoot 'run_v12_evaluation.ps1') -Symbol $Symbol
