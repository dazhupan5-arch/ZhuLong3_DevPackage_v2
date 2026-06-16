# 烛龙双品种实盘服务（XAUUSD v12 + USOIL v1）
#Requires -Version 5.1
param(
    [switch]$Once,
    [switch]$DeployOnly,
    [string]$ConfigXau = 'config/config_xau_v14.json',
    [string]$ConfigOil = 'config/config_oil_v14.json',
    [string]$BrokerOil = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if ($BrokerOil) {
    Write-Host "== 更新原油 broker_symbol -> $BrokerOil ==" -ForegroundColor Cyan
    $oilCfg = Get-Content $ConfigOil -Raw | ConvertFrom-Json
    $oilCfg.broker_symbol = $BrokerOil
    $oilCfg | ConvertTo-Json -Depth 6 | Set-Content $ConfigOil -Encoding UTF8
}

Write-Host '== deploy dual models ==' -ForegroundColor Cyan
py -3 scripts/deploy_dual_production.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== MT5 indicator ==' -ForegroundColor Cyan
$indicatorDst = Join-Path $root 'indicators\ZhuLongIndicator.mq5'
if (Test-Path $indicatorDst) {
    Write-Host "指标文件: $indicatorDst" -ForegroundColor Gray
    Write-Host '请复制到 MT5 数据目录 MQL5/Indicators/ 并编译' -ForegroundColor Yellow
}
$pipeDll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (Test-Path $pipeDll) {
    try { & (Join-Path $root 'scripts\deploy-mt5-indicator.ps1') } catch { }
}

if ($DeployOnly) {
    Write-Host 'Deploy only, done.' -ForegroundColor Green
    exit 0
}

Write-Host '== start dual realtime_signal.py ==' -ForegroundColor Green
Write-Host 'Ensure: MT5 logged in + ZhuLongIndicator on XAUUSD M1 AND oil symbol M1' -ForegroundColor Yellow
Write-Host 'Oil broker_symbol in config_oil_v1.json must match MT5 Market Watch' -ForegroundColor Yellow

if ($Once) {
    py -3 scripts/realtime_signal.py --once --config-xau $ConfigXau --config-oil $ConfigOil
} else {
    py -3 scripts/realtime_signal.py --config-xau $ConfigXau --config-oil $ConfigOil
}
