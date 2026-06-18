# Horizon V16 本地训练管线（不含 KN2 — KN2 在 GPU 机 train_kn2_v16_remote.ps1）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$logDir = Join-Path (Get-Location) "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("horizon_only_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Log($msg) {
    Write-Host $msg
    Add-Content -Path $log -Value $msg -Encoding utf8
}

Log "=== Horizon V16 only (KN2 on GPU machine) ==="

Log "--- refine (clean data) ---"
py -3 -u scripts/refine_horizon_v16.py --trials flat_boost,smote_055,longer_ce 2>&1 | ForEach-Object { Log $_ }

Log "--- val F1 calibration ---"
py -3 scripts/calibrate_horizon_v16.py --temporal-val --apply 2>&1 | ForEach-Object { Log $_ }

Log "--- export ONNX ---"
py -3 scripts/convert_knowledge_net_to_onnx.py --model models/horizon_v16.pth --out models/horizon_v16.onnx --no-benchmark 2>&1 | ForEach-Object { Log $_ }

Log "--- acceptance ---"
py -3 -u scripts/accept_horizon_v16.py --apply 2>&1 | ForEach-Object { Log $_ }
exit $LASTEXITCODE
