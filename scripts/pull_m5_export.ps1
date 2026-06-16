# 从 MT5 Common\Files\ZhuLong 拉取指标导出的 M5 CSV → data/training/
param(
    [string]$Symbol = 'XAUUSD',
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
if (-not $OutDir) { $OutDir = Join-Path $root 'data\training' }

$srcName = "${Symbol}_M5.csv"
$candidates = @(
    (Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files\ZhuLong\$srcName")
)

Get-ChildItem (Join-Path $env:APPDATA 'MetaQuotes\Terminal') -Directory -EA SilentlyContinue |
    Where-Object { $_.Name -match '^[0-9A-F]{32}$' } |
    ForEach-Object {
        $candidates += Join-Path $_.FullName "MQL5\Files\ZhuLong\$srcName"
    }

$src = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $src) {
    Write-Host "未找到 $srcName" -ForegroundColor Red
    Write-Host '请在 MT5 中：编译 ZhuLongIndicator → 挂到图表 → 勾选「启动时导出 M5 CSV」→ 重新加载指标' -ForegroundColor Yellow
    Write-Host "期望路径: $($candidates[0])" -ForegroundColor Gray
    exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$dst = Join-Path $OutDir $srcName
Copy-Item -Force $src $dst
$lines = (Get-Content $dst | Measure-Object -Line).Lines - 1
Write-Host "OK: $dst ($lines M5 bars from $src)" -ForegroundColor Green
Write-Host "训练: py -3 train.py --symbol $Symbol --m5-csv $dst"
