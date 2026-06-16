# 烛龙 v9 优化（复用 v8 特征，重训 XGB 分类 + v9 回测规则）
param([string]$Symbol = 'XAUUSD', [switch]$Quick)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$logDir = "data/training/reports/v9/$Symbol"
New-Item -ItemType Directory -Force -Path $logDir, "models/$Symbol/v9" | Out-Null

Write-Host "== v9: train XGB classifier + ensemble + backtest ==" -ForegroundColor Cyan
$args = @('-3', 'scripts/train_v9.py', '--symbol', $Symbol)
if ($Quick) { $args += '--quick' }
py @args
$exit = $LASTEXITCODE

py -3 scripts/write_acceptance_v9.py --symbol $Symbol
exit $exit
