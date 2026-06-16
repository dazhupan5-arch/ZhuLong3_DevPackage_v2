# ZhuLong V1.0 dual-closure audit (engineering + live)
#Requires -Version 5.1
param(
    [string]$InstallDir = '',
    [switch]$StrictSignals
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

function Resolve-InstallDir {
    param([string]$Hint)
    if ($Hint -and (Test-Path (Join-Path $Hint 'ZhuLong.exe'))) { return (Resolve-Path $Hint).Path }
    $proc = Get-Process ZhuLong -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($proc -and $proc.Path -and (Test-Path $proc.Path)) {
        return (Split-Path $proc.Path -Parent)
    }
    foreach ($p in @(
            (Join-Path ${env:ProgramFiles} 'ZhuLong'),
            (Join-Path $repo 'publish\win-x64'),
            (Join-Path $repo 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64')
        )) {
        if (Test-Path (Join-Path $p 'ZhuLong.exe')) { return (Resolve-Path $p).Path }
    }
    return $null
}

function Test-Gate {
    param([string]$Id, [string]$Name, [bool]$Ok, [string]$Detail = '')
    $mark = if ($Ok) { 'PASS' } else { 'FAIL' }
    $color = if ($Ok) { 'Green' } else { 'Red' }
    Write-Host "[$mark] $Id $Name" -ForegroundColor $color
    if ($Detail) { Write-Host "      $Detail" -ForegroundColor Gray }
    [pscustomobject]@{ Id = $Id; Name = $Name; Pass = $Ok; Detail = $Detail }
}

$install = Resolve-InstallDir -Hint $InstallDir
$appData = Join-Path $env:APPDATA 'ZhuLong'
$logDir = Join-Path $appData 'logs'
$dbPath = Join-Path $appData 'trading.db'
$results = @()

Write-Host '=== ZhuLong dual-closure audit ===' -ForegroundColor Cyan
Write-Host "Install: $(if ($install) { $install } else { 'not found' })" -ForegroundColor Gray
Write-Host ''

Write-Host '-- Leap 1: engineering --' -ForegroundColor Cyan
Push-Location $repo
dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 -v q 2>&1 | Out-Null
$results += Test-Gate 'L1-1' 'dotnet build' ($LASTEXITCODE -eq 0) ''
dotnet test tests/ZhuLong.Core.Tests/ZhuLong.Core.Tests.csproj -c Release -v q 2>&1 | Out-Null
$results += Test-Gate 'L1-2' 'dotnet test' ($LASTEXITCODE -eq 0) 'Core integration included'
Pop-Location

Write-Host ''
Write-Host '-- Leap 2: install/runtime --' -ForegroundColor Cyan
$results += Test-Gate 'L2-1a' 'ZhuLong.exe' ($null -ne $install) $install

$v12Files = @(
    'models\XAUUSD\manifest.json',
    'models\XAUUSD\xgb_triple.json',
    'models\XAUUSD\v12_meta.pkl',
    'models\XAUUSD\feature_columns.json',
    'models\XAUUSD\imf_vmd.parquet',
    'zhulong\training\v12\backtest.py',
    'install_python_deps.ps1',
    'resolve_system_python.ps1'
)
$missing = @()
if ($install) {
    foreach ($f in $v12Files) {
        if (-not (Test-Path (Join-Path $install $f))) { $missing += $f }
    }
}
$results += Test-Gate 'L2-1b' 'v12 install bundle' ($missing.Count -eq 0) $(if ($missing.Count) { $missing -join ', ' } else { 'complete' })

$manifestOk = $false
$manifestDetail = 'no manifest'
if ($install -and (Test-Path (Join-Path $install 'models\XAUUSD\manifest.json'))) {
    try {
        $m = Get-Content (Join-Path $install 'models\XAUUSD\manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
        $manifestOk = ($m.kind -eq 'production') -and $m.acceptance_passed -and ($m.classifier_mode -eq 'triple_xgb')
        $manifestDetail = "kind=$($m.kind) accepted=$($m.acceptance_passed) mode=$($m.classifier_mode)"
    } catch { $manifestDetail = $_.Exception.Message }
}
$results += Test-Gate 'L2-1c' 'XAUUSD production manifest' $manifestOk $manifestDetail

& (Join-Path $repo 'scripts\resolve_system_python.ps1') -Quiet | Out-Null
$pyOk = $LASTEXITCODE -eq 0
$mt5PkgOk = $false
if ($pyOk) {
    & py -3 -c "import MetaTrader5" 2>$null
    $mt5PkgOk = ($LASTEXITCODE -eq 0)
}
$results += Test-Gate 'L2-1d' 'Python and MetaTrader5 pkg' ($pyOk -and $mt5PkgOk) $(if ($pyOk) { $env:PYTHONNET_PYDLL } else { 'install Python 3.10+' })

$importOk = $false
$importDetail = 'skipped'
if ($install -and $pyOk) {
    $pyCode = @"
import sys
sys.path.insert(0, r'$repo')
sys.path.insert(0, r'$install')
from zhulong.v12_live import validate_v12_artifacts
print('ok' if validate_v12_artifacts('XAUUSD') else 'bad')
"@
    $out = & py -3 -c $pyCode 2>&1
    $importOk = ($LASTEXITCODE -eq 0) -and ($out -match 'ok')
    $importDetail = ($out | Out-String).Trim()
}
$results += Test-Gate 'L2-1e' 'v12 import chain' $importOk $importDetail

$macroCsvInstall = $false
$macroCsvApp = Test-Path (Join-Path $appData 'data\macro\macro_daily.csv')
if ($install) { $macroCsvInstall = Test-Path (Join-Path $install 'data\macro\macro_daily.csv') }
$macroOk = $macroCsvInstall -or $macroCsvApp
$macroDetail = "install=$macroCsvInstall appdata=$macroCsvApp"
$results += Test-Gate 'L2-1f' 'macro_daily.csv available' $macroOk $macroDetail

Write-Host ''
Write-Host '-- Live: MT5 / pipe / DB --' -ForegroundColor Cyan
$proc = Get-Process ZhuLong -ErrorAction SilentlyContinue | Select-Object -First 1
$results += Test-Gate 'L2-2a' 'GUI running' ($null -ne $proc) $(if ($proc) { "PID=$($proc.Id)" } else { 'start ZhuLong.exe' })

$mt5Roots = Get-ChildItem (Join-Path $env:APPDATA 'MetaQuotes\Terminal') -Directory -EA SilentlyContinue |
    Where-Object { $_.Name -match '^[0-9A-F]{32}$' }
$mt5Deployed = 0
foreach ($t in $mt5Roots) {
    $dll = Join-Path $t.FullName 'MQL5\Libraries\ZhuLongMt5Pipe.dll'
    $mq5 = Join-Path $t.FullName 'MQL5\Indicators\ZhuLongIndicator.mq5'
    if ((Test-Path $dll) -and (Test-Path $mq5)) { $mt5Deployed++ }
}
$results += Test-Gate 'L2-2b' 'MT5 indicator and DLL' ($mt5Deployed -gt 0) "$mt5Deployed terminal(s)"

$todayLog = Get-ChildItem (Join-Path $logDir 'log*.txt') -EA SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$pipeOk = $false
$pipeDetail = 'no log'
$inferOk = $false
$inferDetail = 'no recent inference log'
if ($todayLog) {
    $logJson = & py -3 (Join-Path $repo 'scripts\sim_log_audit.py') 2>&1 | Select-Object -Last 1
    try {
        $la = $logJson | ConvertFrom-Json
        if ($la.ok) {
            $pipeOk = [bool]$la.m1 -and [bool]$la.pipe
            $pipeDetail = if ($pipeOk) { "$($la.log): M1 and pipe OK" } else { "$($la.log): check MT5 indicator" }
            if ($la.infer_fail_last) {
                $inferOk = $false
                $inferDetail = $la.infer_fail_last
            } elseif ($la.infer_ok) {
                $inferOk = $true
                $inferDetail = $la.infer_ok_last
            } elseif ($la.infer_started) {
                $inferDetail = 'inference started but not completed yet'
            }
        }
    } catch {
        $pipeDetail = $logJson
    }
}
$results += Test-Gate 'L2-2c' 'M1 data stream' $pipeOk $pipeDetail

$macroTableOk = $false
$macroCount = 0
if (Test-Path $dbPath) {
    $dbJson = & py -3 (Join-Path $repo 'scripts\sim_db_status.py') 2>&1 | Select-Object -Last 1
    try {
        $db = $dbJson | ConvertFrom-Json
        if ($db.ok -and $null -ne $db.macro_events) {
            $macroCount = [int]$db.macro_events
            $macroTableOk = $macroCount -ge 0
        }
    } catch { }
}
$results += Test-Gate 'L2-2d' 'SQLite macro_events' $macroTableOk "rows=$macroCount"

$sigCount = 0
if (Test-Path $dbPath) {
    $dbJson = & py -3 (Join-Path $repo 'scripts\sim_db_status.py') 2>&1 | Select-Object -Last 1
    try {
        $db = $dbJson | ConvertFrom-Json
        if ($db.ok) { $sigCount = [int]$db.signals }
    } catch { }
}
$results += Test-Gate 'L2-3a' 'v12 inference log' $inferOk $inferDetail
$sigGateOk = ($sigCount -gt 0) -or (-not $StrictSignals)
$sigDetail = "signals=$sigCount"
if (-not $StrictSignals -and $sigCount -eq 0) { $sigDetail += ' (non-strict: no-trade/filter OK)' }
$results += Test-Gate 'L2-3b' 'signals in DB' $sigGateOk $sigDetail

Write-Host ''
Write-Host '=== Manual closure (human sign-off) ===' -ForegroundColor Cyan
Write-Host '  L2-3c  MT5 chart arrows match WinUI signal list'
Write-Host '  L2-4   Order Comment ZhuLong_signal_id then position matched'
Write-Host '  L2-5   Trailing SL in position_events'
Write-Host '  L2-6   Sim account 3 trading days (docs/ACCEPTANCE.md)'
Write-Host ''

$fail = @($results | Where-Object { -not $_.Pass }).Count
$pass = @($results | Where-Object { $_.Pass }).Count
$engPass = @($results | Where-Object { $_.Pass -and $_.Id -like 'L1-*' }).Count
$engTotal = @($results | Where-Object { $_.Id -like 'L1-*' }).Count
$livePass = @($results | Where-Object { $_.Pass -and $_.Id -like 'L2-*' }).Count
$liveTotal = @($results | Where-Object { $_.Id -like 'L2-*' }).Count

Write-Host "Engineering closure: $engPass / $engTotal"
Write-Host "Live auto closure:   $livePass / $liveTotal"
Write-Host "Total automated:     $pass PASS / $fail FAIL / $($results.Count) checks"
if ($fail -gt 0) { exit 1 }
exit 0
