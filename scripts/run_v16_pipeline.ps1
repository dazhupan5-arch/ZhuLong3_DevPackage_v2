# V16 全管线（prep 完成后自动训 Horizon / PPO / 验收）
# 推荐: .\scripts\run_v16_full_auto.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

& "$PSScriptRoot\run_prepare_v16_resilient.ps1"
exit $LASTEXITCODE
