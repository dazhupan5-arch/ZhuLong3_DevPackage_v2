# 拉取 MT5 导出的 M5 CSV 并训练模型
param(
    [string]$Symbol = 'XAUUSD',
    [int]$SeqLen = 60
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host '== deploy indicator (mq5) ==' -ForegroundColor Cyan
& (Join-Path $root 'scripts\deploy-mt5-indicator.ps1')

Write-Host '== pull M5 CSV from MT5 ==' -ForegroundColor Cyan
& (Join-Path $root 'scripts\pull_m5_export.ps1') -Symbol $Symbol
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$csv = Join-Path $root "data\training\${Symbol}_M5.csv"
Write-Host '== train.py (system Python) ==' -ForegroundColor Cyan
& py -3 (Join-Path $root 'train.py') --symbol $Symbol --m5-csv $csv --seq-len $SeqLen
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== copy models to publish dirs ==' -ForegroundColor Cyan
foreach ($dst in @('models', 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\models')) {
    $target = Join-Path $root "$dst\$Symbol"
    if (Test-Path (Join-Path $root "models\$Symbol")) {
        New-Item -ItemType Directory -Force -Path $target | Out-Null
        Copy-Item -Recurse -Force (Join-Path $root "models\$Symbol\*") $target
    }
}

Write-Host '训练完成。请重启 ZhuLong 或重新连接 MT5 以加载新模型。' -ForegroundColor Green
