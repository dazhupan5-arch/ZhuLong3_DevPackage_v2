# 从桌面 XAUUSD5.csv / XTIUSD5.csv 增量补齐到 data/training/lgb/
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$desk = [Environment]::GetFolderPath('Desktop')

$pairs = @(
    @{ Source = Join-Path $desk 'XAUUSD5.csv'; Target = 'data/training/lgb/XAUUSD/XAUUSD_M5.csv' },
    @{ Source = Join-Path $desk 'XTIUSD5.csv'; Target = 'data/training/lgb/USOIL/USOIL_M5.csv' }
)

Set-Location $root
foreach ($p in $pairs) {
    if (-not (Test-Path $p.Source)) {
        Write-Host "跳过（桌面无文件）: $($p.Source)" -ForegroundColor Yellow
        continue
    }
    Write-Host "== $($p.Source) -> $($p.Target) ==" -ForegroundColor Cyan
    $args = @('scripts/merge_m5_supplement.py', '--source', $p.Source, '--target', $p.Target)
    if ($DryRun) { $args += '--dry-run' }
    & py -3 @args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host 'Done. Re-train: delete data/training/v13/*/features.parquet then regenerate labels.' -ForegroundColor Green
