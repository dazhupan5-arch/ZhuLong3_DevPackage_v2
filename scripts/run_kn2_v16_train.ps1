# KN2 V16 长期方案：prepare(断点) -> train
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("kn2_v16_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# 备份 legacy KN2
$bk = Join-Path (Get-Location) "models\backups\kn2_legacy_20260614"
New-Item -ItemType Directory -Force -Path $bk | Out-Null
foreach ($f in @("kn2_trader.pth", "kn2_trader.meta.json")) {
    $src = Join-Path (Get-Location) "models\$f"
    if (Test-Path $src) { Copy-Item -Force $src (Join-Path $bk $f) }
}

Write-Host "=== KN2 V16 TRAIN ===" -ForegroundColor Cyan
Write-Host "Log: $log"

$existing = Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'train_kn2_v16\.py' }
if ($existing) {
    $p = $existing | Select-Object -First 1
    Write-Warning "KN2 V16 train already running (PID $($p.ProcessId)); skip duplicate start."
    exit 0
}

function Log-Run([string]$Label, [string[]]$PyArgs) {
    Write-Host "`n--- $Label ---" -ForegroundColor Yellow
    & py -3 -u @PyArgs 2>&1 | ForEach-Object {
        Write-Host $_
        try { Add-Content -Path $log -Value $_ -Encoding utf8 } catch { }
    }
    return $LASTEXITCODE
}

$rc = Log-Run "prepare_kn2_v16_data" @("scripts/prepare_kn2_v16_data.py", "--checkpoint-every", "50000")
if ($rc -ne 0) { exit $rc }

$rc = Log-Run "train_kn2_v16" @("scripts/train_kn2_v16.py", "--mode", "fast", "--epochs", "120", "--patience", "25")
exit $rc
