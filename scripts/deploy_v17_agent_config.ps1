# 将仓库 V17 Agent 配置 merge 到 AppData（及可选安装目录）
param(
    [string]$AppDataDir = "",
    [string]$InstallDir = "",
    [double]$DirectionMinScore = 0.0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($AppDataDir)) {
    $AppDataDir = Join-Path $env:APPDATA "ZhuLong"
}
. (Join-Path $PSScriptRoot "Merge-V17AgentConfig.ps1")

$cfgPath = Join-Path $AppDataDir "config_agent.json"
Merge-V17AgentConfig -TargetPath $cfgPath -DirectionMinScore $DirectionMinScore

if (-not [string]::IsNullOrWhiteSpace($InstallDir) -and (Test-Path $InstallDir)) {
    $installCfg = Join-Path $InstallDir "config\config_agent.json"
    $installDirParent = Split-Path $installCfg -Parent
    if (-not (Test-Path $installDirParent)) {
        New-Item -ItemType Directory -Force -Path $installDirParent | Out-Null
    }
    Merge-V17AgentConfig -TargetPath $installCfg -DirectionMinScore $DirectionMinScore
    Write-Host "Install config updated: $installCfg" -ForegroundColor Green
}

Write-Host "Done. Restart ZhuLong after model deploy." -ForegroundColor Yellow
