#Requires -Version 5.1
<#
.SYNOPSIS
  部署 KN1 V14 蒸馏模型到烛龙生产目录，并切换为 KN1 模式（关闭 KN2）。
#>
param(
    [string] $InstallDir = 'C:\Program Files\ZhuLong',
    [switch] $AlsoPublish
)

$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
$DevModels = Join-Path $Repo 'models'

$knFiles = @(
    'knowledge_net.onnx',
    'knowledge_net.meta.json',
    'knowledge_scaler.pkl',
    'knowledge_net.pth'
)

$codeFiles = @(
    'zhulong\agent\knowledge_net.py',
    'zhulong\agent\trading_agent.py',
    'zhulong\engine\agent_engine.py',
    'ZhuLong.PythonEngine\inference_cli.py'
)

function Deploy-ToTarget([string]$Target) {
    if (-not (Test-Path $Target)) {
        Write-Host "SKIP missing target: $Target" -ForegroundColor Yellow
        return
    }
    $modelsDst = Join-Path $Target 'models'
    New-Item -ItemType Directory -Force -Path $modelsDst | Out-Null
    foreach ($f in $knFiles) {
        $src = Join-Path $DevModels $f
        if (-not (Test-Path $src)) { throw "Missing dev model: $src" }
        Copy-Item -Force $src (Join-Path $modelsDst $f)
        Write-Host "  model -> $Target\models\$f" -ForegroundColor Cyan
    }
    foreach ($rel in $codeFiles) {
        $src = Join-Path $Repo $rel
        if (-not (Test-Path $src)) { continue }
        $dst = Join-Path $Target $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Copy-Item -Force $src $dst
        Write-Host "  code  -> $Target\$rel" -ForegroundColor DarkCyan
    }
    $cfgSrc = Join-Path $Repo 'config\config_agent.json'
    $cfgDst = Join-Path $Target 'config\config_agent.json'
    if (Test-Path $cfgSrc) {
        Copy-Item -Force $cfgSrc $cfgDst
        Write-Host "  config -> $cfgDst" -ForegroundColor Cyan
    }
}

Write-Host "== Deploy KN1 to $InstallDir ==" -ForegroundColor Green
try {
    Deploy-ToTarget $InstallDir
} catch {
    Write-Host "Install dir deploy failed (need admin?): $_" -ForegroundColor Red
}

if ($AlsoPublish) {
    $pub = Join-Path $Repo 'publish\win-x64'
    Write-Host "== Deploy KN1 to publish ==" -ForegroundColor Green
    Deploy-ToTarget $pub
}

Write-Host "DONE deploy_kn1_production" -ForegroundColor Green
