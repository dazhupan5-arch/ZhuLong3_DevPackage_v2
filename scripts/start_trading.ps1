# ZhuLong v12 Python realtime trading service
#Requires -Version 5.1
param(
    [switch]$Once,
    [switch]$DeployOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

Write-Host '== deploy v12 model ==' -ForegroundColor Cyan
py -3 scripts/deploy_v12_production.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== MT5 indicator ==' -ForegroundColor Cyan
$pipeDll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
if (Test-Path $pipeDll) {
    try { & (Join-Path $root 'scripts\deploy-mt5-indicator.ps1') } catch { }
}

if ($DeployOnly) {
    Write-Host 'Deploy only, done.' -ForegroundColor Green
    exit 0
}

Write-Host '== start realtime_signal.py ==' -ForegroundColor Green
Write-Host 'Ensure: MT5 demo logged in + ZhuLongIndicator on XAUUSD M1' -ForegroundColor Yellow
if ($Once) {
    py -3 scripts/realtime_signal.py --once
} else {
    py -3 scripts/realtime_signal.py
}
