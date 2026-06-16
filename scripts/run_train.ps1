# Train on system Python (visible console output)
param(
    [string]$Symbol = 'XAUUSD',
    [string]$Csv = '',
    [int]$SeqLen = 60
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

& (Join-Path $root 'scripts\resolve_system_python.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $Csv) {
    $Csv = Join-Path $root "data\training\${Symbol}_M5.csv"
}
if (-not (Test-Path $Csv)) {
    Write-Host "Missing CSV: $Csv" -ForegroundColor Red
    Write-Host 'Export first: py -3 scripts/export_m5_mt5.py --symbol XAUUSD' -ForegroundColor Yellow
    exit 1
}

$log = Join-Path $root "data\training\train_${Symbol}.log"
Write-Host "Training log: $log" -ForegroundColor Cyan
Write-Host 'Process name: python.exe (visible in Task Manager)' -ForegroundColor Gray

& py -3 train.py --symbol $Symbol --m5-csv $Csv --seq-len $SeqLen 2>&1 | Tee-Object -FilePath $log
exit $LASTEXITCODE
