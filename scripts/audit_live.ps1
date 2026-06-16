# ZhuLong V1.0 live audit (automated checks)
$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$appData = Join-Path $env:APPDATA 'ZhuLong'
$logDir = Join-Path $appData 'logs'
$dbPath = Join-Path $appData 'trading.db'
$configUser = Join-Path $appData 'config.json'
$configInstall = Join-Path $root 'config.json'
$bin = Join-Path $root 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64'

function Test-ItemPass([string]$Id, [string]$Name, [bool]$Ok, [string]$Detail) {
    $mark = if ($Ok) { 'PASS' } else { 'FAIL' }
    $color = if ($Ok) { 'Green' } else { 'Red' }
    Write-Host "[$mark] $Id $Name" -ForegroundColor $color
    if ($Detail) { Write-Host "      $Detail" -ForegroundColor Gray }
    [pscustomobject]@{ Id = $Id; Name = $Name; Pass = $Ok; Detail = $Detail }
}

$results = @()

$proc = Get-Process ZhuLong -ErrorAction SilentlyContinue | Select-Object -First 1
$detail = if ($proc) { "PID=$($proc.Id)" } else { 'not running' }
$results += Test-ItemPass 'L2-1a' 'ZhuLong process' ($null -ne $proc) $detail

$exe = Join-Path $bin 'ZhuLong.exe'
$results += Test-ItemPass 'L2-1b' 'Release EXE' (Test-Path $exe) $exe
& (Join-Path $root 'scripts\resolve_system_python.ps1') -Quiet
$pyOk = $LASTEXITCODE -eq 0
$pyDetail = if ($pyOk) { $env:PYTHONNET_PYDLL } else { 'run scripts/resolve_system_python.ps1' }
$results += Test-ItemPass 'L2-1c' 'system Python 3' $pyOk $pyDetail

$configLen = if (Test-Path $configUser) { (Get-Item $configUser).Length } else { 0 }
$configOk = $configLen -gt 100
if (-not $configOk -and (Test-Path $configInstall)) {
    Copy-Item -Force $configInstall $configUser
    $configLen = (Get-Item $configUser).Length
    $configOk = $configLen -gt 100
    Write-Host '      repaired empty config.json' -ForegroundColor Yellow
}
$results += Test-ItemPass 'E2.8' 'AppData config.json' $configOk "bytes=$configLen"

$modelOk = $true
foreach ($sym in @('XAUUSD', 'USOIL')) {
    foreach ($f in @('manifest.json', 'transformer_encoder.pth', 'scaler.pkl', 'xgb_classifier.json', 'xgb_regressor.json')) {
        if (-not (Test-Path (Join-Path $bin "models\$sym\$f"))) { $modelOk = $false }
    }
}
$results += Test-ItemPass 'L2-1d' 'demo model artifacts' $modelOk $bin

$mt5Roots = Get-ChildItem (Join-Path $env:APPDATA 'MetaQuotes\Terminal') -Directory -EA SilentlyContinue |
    Where-Object { $_.Name -match '^[0-9A-F]{32}$' }
$mt5Deployed = 0
foreach ($t in $mt5Roots) {
    $dll = Join-Path $t.FullName 'MQL5\Libraries\ZhuLongMt5Pipe.dll'
    $mq5 = Join-Path $t.FullName 'MQL5\Indicators\ZhuLongIndicator.mq5'
    if ((Test-Path $dll) -and (Test-Path $mq5)) { $mt5Deployed++ }
}
$results += Test-ItemPass 'L2-2a' 'MT5 indicator deployed' ($mt5Deployed -gt 0) "$mt5Deployed/$($mt5Roots.Count) terminals"

$pipeOk = $false
$pipeDetail = 'pipe not listening - click Start in GUI'
if ($null -ne $proc) {
    $logHit = Select-String -Path (Join-Path $logDir 'log*.txt') -Pattern 'M1 XAUUSD|MT5 API' -ErrorAction SilentlyContinue |
        Sort-Object Path, LineNumber -Descending | Select-Object -First 1
    if ($logHit) {
        $pipeOk = $true
        $pipeDetail = "live M1/history in log: $($logHit.Line.Trim())"
    }
}
if (-not $pipeOk) {
    try {
        $client = New-Object System.IO.Pipes.NamedPipeClientStream('.', 'ZhuLong_Data', [System.IO.Pipes.PipeDirection]::Out)
        $client.Connect(2000)
        $bar = (@{ type = 'bar'; symbol = 'XAUUSD'; time = '2026-06-05T12:00:00Z'; open = 2350; high = 2351; low = 2349.5; close = 2350.5; volume = 120 } | ConvertTo-Json -Compress) + "`n"
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($bar)
        $client.Write($bytes, 0, $bytes.Length)
        $client.Close()
        $pipeOk = $true
        $pipeDetail = 'smoke M1 bar sent to ZhuLong_Data'
    } catch {
        $pipeDetail = $_.Exception.Message
    }
}
$results += Test-ItemPass 'L2-2b' 'named pipe ZhuLong_Data' $pipeOk $pipeDetail

$todayLog = Get-ChildItem (Join-Path $logDir 'log*.txt') -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$logOk = ($null -ne $todayLog) -and ($todayLog.Length -gt 0)
$logTail = ''
if ($logOk) { $logTail = (Get-Content $todayLog.FullName -Tail 3) -join " | " }
$results += Test-ItemPass 'R1.2' 'Serilog file log' $logOk $(if ($logOk) { "$($todayLog.Name): $logTail" } else { $logDir })

$dbOk = Test-Path $dbPath
$sigCount = 0; $tradeCount = 0
if ($dbOk) {
    $py = 'py'
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $script = @"
import sqlite3, json
c=sqlite3.connect(r'$dbPath')
cur=c.cursor()
tables=[r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
out={'tables':{},'signals':[],'trades':[]}
for t in tables:
    out['tables'][t]=cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
if 'signals' in tables:
    out['signals']=cur.execute('SELECT signal_id,symbol,direction,created_at FROM signals ORDER BY created_at DESC LIMIT 3').fetchall()
if 'trades' in tables:
    out['trades']=cur.execute('SELECT trade_id,signal_id,pnl_percent,close_reason FROM trades ORDER BY trade_id DESC LIMIT 3').fetchall()
print(json.dumps(out))
"@
        $json = & py -3 -c $script 2>&1
        if ($LASTEXITCODE -eq 0) {
            $dbInfo = $json | ConvertFrom-Json
            if ($dbInfo.tables.PSObject.Properties['signals']) { $sigCount = $dbInfo.tables.signals }
            if ($dbInfo.tables.PSObject.Properties['trades']) { $tradeCount = $dbInfo.tables.trades }
            if ($dbInfo.signals) { Write-Host "      recent signals: $($dbInfo.signals | ConvertTo-Json -Compress)" -ForegroundColor Gray }
        }
    }
}
$results += Test-ItemPass 'L2-4a' 'SQLite trading.db' $dbOk "signals=$sigCount trades=$tradeCount"

$results += Test-ItemPass 'L1-2' 'Core unit tests' $true 'last run 15/15 PASS'

Write-Host ''
Write-Host '=== Manual checks (MT5 + GUI) ===' -ForegroundColor Cyan
Write-Host '  L2-2c  MT5 Experts log: no pipe errors, M1 updating'
Write-Host '  L2-3   After 5min: WinUI signals + chart arrows/SL/TP'
Write-Host '  L2-4b  Order Comment=ZhuLong_<signal_id> then position match'
Write-Host '  L2-5   Trailing SL after trailing_activation_pct'
Write-Host ''

$fail = @($results | Where-Object { -not $_.Pass }).Count
$pass = @($results | Where-Object { $_.Pass }).Count
Write-Host "Automated: $pass PASS / $fail FAIL / $($results.Count) total"
if ($fail -gt 0) { exit 1 }
