# Horizon V16 达标管线：恢复 backup → 调参试训 → ONNX → 验收（Horizon 段）
# 若 Horizon training+OOS 通过，再手动跑 PPO+全验收
param(
    [string]$BackupTag = "v16_near_threshold_20260616",
    [string]$Trials = ""
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("horizon_pass_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Log($msg) {
    Write-Host $msg
    try { Add-Content -Path $log -Value $msg -Encoding utf8 } catch { }
}

Log "=== Horizon V16 PASS pipeline ==="
Log "Backup: models/backups/$BackupTag"

& "$PSScriptRoot\restore_v16_backup.ps1" -Tag $BackupTag 2>&1 | ForEach-Object { Log $_ }

$tuneArgs = @("scripts/tune_horizon_v16.py")
if ($Trials) { $tuneArgs += @("--trials", $Trials) }

Log "--- tune_horizon_v16 ---"
& py -3 -u @tuneArgs 2>&1 | ForEach-Object { Log $_ }
$tuneRc = $LASTEXITCODE
if ($tuneRc -eq 1) {
    Log "Tune failed (no trials)."
    exit 1
}

Log "--- export ONNX ---"
& py -3 scripts/convert_knowledge_net_to_onnx.py --model models/horizon_v16.pth --out models/horizon_v16.onnx --no-benchmark 2>&1 |
    ForEach-Object { Log $_ }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Log "--- acceptance (training + onnx + oos) ---"
& py -3 -u scripts/accept_horizon_v16.py 2>&1 | ForEach-Object { Log $_ }
$accRc = $LASTEXITCODE

if ($accRc -eq 0) {
    Log "Horizon acceptance PASSED — run PPO retrain next:"
    Log "  py -3 scripts/run_v16_pipeline.py --skip-prepare --skip-enrich --skip-train --skip-onnx --jobs 1"
} else {
    Log "Horizon acceptance FAILED (exit $accRc). Best tune model kept in models/horizon_v16.pth"
}

exit $accRc
