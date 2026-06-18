# USOIL V16：Horizon NPZ 就绪后 → enrich → location 标签 → git push
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$hz = "data/clean/training_horizon_v16_usoil.npz"
$loc = "data/clean/training_horizon_v16_usoil_location.npz"

if (-not (Test-Path $hz)) {
    Write-Error "缺少 $hz — 请先完成 prepare_horizon_v16_data.py"
}

Write-Host "=== enrich OHLCV ===" -ForegroundColor Cyan
py -3 scripts/enrich_horizon_v16_npz.py --symbol USOIL --npz $hz
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== location 标签 P4 ===" -ForegroundColor Cyan
py -3 scripts/prepare_horizon_v16_location_labels.py --npz $hz --out $loc
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== git add + commit + lfs push ===" -ForegroundColor Cyan
git add $hz $loc data/clean/cleaning_report.json scripts/train_usoil_v16_gpu_remote.ps1
git status --short
$msg = "Add USOIL V16 horizon training NPZ (location labels) for GPU training."
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "无新变更可提交"
} else {
    git commit -m $msg
    git lfs push origin main
    git push origin main
    Write-Host "已 push — 训练机执行:" -ForegroundColor Green
    Write-Host "  git pull && git lfs pull" -ForegroundColor Green
    Write-Host "  powershell -File scripts/train_usoil_v16_gpu_remote.ps1 -InstallDeps" -ForegroundColor Green
}
