# V16 一键全自动：prep(断点续跑) -> Horizon -> ONNX -> PPO -> 验收 passed 才可部署
# 用法: .\scripts\run_v16_full_auto.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

& "$PSScriptRoot\run_prepare_v16_resilient.ps1"
exit $LASTEXITCODE
