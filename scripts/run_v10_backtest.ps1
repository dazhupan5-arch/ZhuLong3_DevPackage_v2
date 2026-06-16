# v10 双向回测（复用 v9 模型）
param([string]$Symbol = 'XAUUSD')

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

New-Item -ItemType Directory -Force -Path "data/training/reports/v10/$Symbol" | Out-Null

Write-Host "== v10 val tune + test1 both ==" -ForegroundColor Cyan
py -3 scripts/backtest_v10.py --symbol $Symbol --split val --mode both
py -3 scripts/backtest_v10.py --symbol $Symbol --split test1 --mode both
py -3 scripts/backtest_v10.py --symbol $Symbol --split test1 --mode long
py -3 scripts/backtest_v10.py --symbol $Symbol --split test1 --mode short
