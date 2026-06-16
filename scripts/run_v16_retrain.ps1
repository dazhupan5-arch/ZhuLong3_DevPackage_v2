# V16 重训：跳过 prep，调参 Horizon -> ONNX -> PPO -> 验收
# 备份：models/backups/v16_near_threshold_20260616
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("v16_retrain_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$status = @{
    updated_at = (Get-Date).ToUniversalTime().ToString("o")
    stage = "retrain_running"
    backup = "models/backups/v16_near_threshold_20260616"
    pipeline_running = $true
    retrain_params = @{
        class_weights = "3.2,0.8,3.2"
        smote_ratio = 0.7
        patience = 25
        lr = 0.0004
        epochs = 120
    }
    note = "Horizon retrain1 -> ONNX -> PPO -> acceptance"
}
$status | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $logDir "v16_status.json") -Encoding utf8

Write-Host "=== V16 RETRAIN started ===" -ForegroundColor Cyan
Write-Host "Log: $log"
Write-Host "Backup fallback: models/backups/v16_near_threshold_20260616"

& py -3 -u scripts/run_v16_pipeline.py --skip-prepare --skip-enrich --jobs 1 --retrain *>&1 |
    ForEach-Object {
        Write-Host $_
        try { Add-Content -Path $log -Value $_ -Encoding utf8 } catch { }
    }

$rc = $LASTEXITCODE
$status.updated_at = (Get-Date).ToUniversalTime().ToString("o")
$status.pipeline_running = $false
$status.stage = if ($rc -eq 0) { "retrain_acceptance_passed" } else { "retrain_failed" }
$status.exit_code = $rc
$status | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $logDir "v16_status.json") -Encoding utf8

exit $rc
