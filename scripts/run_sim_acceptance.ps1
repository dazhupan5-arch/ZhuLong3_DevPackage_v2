# ZhuLong 基础设施验收（无正式模型阶段：不要求 signals>0）
#Requires -Version 5.1
param(
    [int] $WaitMinutes = 3,
    [switch] $RestartApp,
    [switch] $RequireProductionModels
)

$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$appData = Join-Path $env:APPDATA 'ZhuLong'
$configUser = Join-Path $appData 'config.json'
$logDir = Join-Path $appData 'logs'
$bin = Join-Path $root 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64'
$exe = Join-Path $bin 'ZhuLong.exe'

function Write-Step([string]$Msg) { Write-Host "`n== $Msg ==" -ForegroundColor Cyan }
function Write-Ok([string]$Msg) { Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn2([string]$Msg) { Write-Host "[!!] $Msg" -ForegroundColor Yellow }
function Write-Fail([string]$Msg) { Write-Host "[FAIL] $Msg" -ForegroundColor Red }

$report = [ordered]@{
    StartedAt = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    AuditPass = $false
    ProductionModels = $false
    InferencePaused = $false
    Mt5Connected = $false
    M1Live = $false
    SignalGenerated = $false
    Signals = 0
    Trades = 0
    Notes = @()
}

Write-Step '1/7 audit_live'
& (Join-Path $root 'scripts\audit_live.ps1')
$report.AuditPass = ($LASTEXITCODE -eq 0) -or $?

Write-Step '2/7 production model gate'
$modelJson = & powershell -NoProfile -File (Join-Path $root 'scripts\check_production_models.ps1') 2>&1 | Select-Object -Last 1
try {
    $modelCheck = $modelJson | ConvertFrom-Json
    $report.ProductionModels = [bool]$modelCheck.ok
    if ($modelCheck.ok) {
        Write-Ok ("production models {0}/{1}" -f $modelCheck.ready, $modelCheck.total)
    } else {
        Write-Warn2 ("no production models: {0}" -f ($modelCheck.pending -join ', '))
        $report.Notes += '正式模型未部署 — 本阶段验收不要求 signals>0'
    }
} catch {
    Write-Warn2 ("model check: {0}" -f $modelJson)
}

Write-Step '3/7 MT5 probe'
$mt5Json = & py -3 (Join-Path $root 'scripts\sim_mt5_probe.py') 2>&1 | Select-Object -Last 1
try {
    $mt5 = $mt5Json | ConvertFrom-Json
    if ($mt5.ok) {
        $report.Mt5Connected = $true
        Write-Ok ("MT5 login={0} server={1} bid={2}" -f $mt5.login, $mt5.server, $mt5.bid)
    } else {
        Write-Fail ("MT5: {0}" -f $mt5.error)
    }
} catch {
    Write-Fail ("MT5 parse: {0}" -f $mt5Json)
}

if (Test-Path $configUser) {
    Write-Step '3b merge risk_guard (UTF-8 no BOM)'
    $installCfg = Join-Path $root 'config.json'
    $raw = Get-Content $configUser -Raw -Encoding UTF8 | ConvertFrom-Json
    if (-not $raw.risk_guard -and (Test-Path $installCfg)) {
        $tpl = Get-Content $installCfg -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($tpl.risk_guard) { $raw | Add-Member risk_guard $tpl.risk_guard }
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($configUser, ($raw | ConvertTo-Json -Depth 20), $utf8NoBom)
        Write-Ok 'risk_guard merged'
    }
}

if ($RestartApp) {
    Write-Step '3c restart ZhuLong'
    Get-Process ZhuLong -EA SilentlyContinue | ForEach-Object {
        try { Stop-Process -Id $_.Id -Force -EA Stop } catch { }
    }
    Start-Sleep -Seconds 2
    if (Test-Path $exe) {
        foreach ($item in @('config.json', 'models', 'data', 'zhulong', 'ZhuLong.PythonEngine')) {
            $src = Join-Path $root $item
            $dst = Join-Path $bin $item
            if (Test-Path $src) { Copy-Item -Recurse -Force $src $dst }
        }
        Start-Process $exe -ArgumentList '--sim-connect' -WorkingDirectory $bin
        Start-Sleep -Seconds 15
        Write-Ok 'Started with --sim-connect'
    }
}

Write-Step ("4/7 wait up to {0} min for M1 + inference-paused log" -f $WaitMinutes)
$deadline = (Get-Date).AddMinutes($WaitMinutes)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 15
    $todayLog = Get-ChildItem (Join-Path $logDir 'log*.txt') -EA SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $todayLog) { continue }
    $tail = Get-Content $todayLog.FullName -Tail 50 -EA SilentlyContinue
    if ($tail -match 'M1 XAUUSD') { $report.M1Live = $true }
    if ($tail -match '\[inference-paused\]') { $report.InferencePaused = $true }
    $hit = $tail | Where-Object { $_ -match 'M1 XAUUSD' -or $_ -match '\[inference-paused\]' } | Select-Object -Last 1
    if ($hit) { Write-Host ("  {0}" -f $hit.Trim()) -ForegroundColor Gray }
    if ($report.M1Live -and ($report.InferencePaused -or $report.ProductionModels)) { break }
}

Write-Step '5/7 dotnet test'
Push-Location $root
dotnet test tests\ZhuLong.Core.Tests\ZhuLong.Core.Tests.csproj -c Release --no-restore 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Write-Ok 'Core tests PASS' } else { Write-Fail 'Core tests FAIL' }
Pop-Location

Write-Step '6/7 SQLite'
$dbJson = & py -3 (Join-Path $root 'scripts\sim_db_status.py') 2>&1 | Select-Object -Last 1
try {
    $db = $dbJson | ConvertFrom-Json
    if ($db.ok) {
        $report.Signals = [int]$db.signals
        $report.Trades = [int]$db.trades
        Write-Ok ("signals={0} trades={1}" -f $db.signals, $db.trades)
    }
} catch {
    Write-Warn2 ("DB: {0}" -f $dbJson)
}

Write-Step '7/7 signal gate (only if production models)'
if ($RequireProductionModels -and $report.ProductionModels) {
    if ($report.Signals -gt 0) { $report.SignalGenerated = $true; Write-Ok 'signals in DB' }
    else { Write-Fail 'production models present but no signals' }
} else {
    Write-Ok 'Skipped signal requirement (no-model phase)'
}

Write-Step 'summary'
$report.EndedAt = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$report | Format-List
$score = @(
    $report.AuditPass,
    $report.Mt5Connected,
    $report.M1Live,
    ($report.InferencePaused -or $report.ProductionModels),
    ($LASTEXITCODE -eq 0)
) | Where-Object { $_ }
Write-Host ("Infra score: {0}/5 (signals not required until models ready)" -f $score.Count) -ForegroundColor Yellow
