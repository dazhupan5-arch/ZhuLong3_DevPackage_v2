# 部署 Horizon V16 模型与 Python 补丁；config 走 Merge-V16AgentConfig（禁止精简模板覆写）
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [switch]$SkipPreDeployGate
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$appData = Join-Path $env:APPDATA "ZhuLong"
New-Item -ItemType Directory -Force -Path $appData | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $appData "models\XAUUSD\v16") | Out-Null

$devRoot = Get-Location
$metaPath = Join-Path $devRoot "models\horizon_v16.meta.json"
if (-not (Test-Path $metaPath)) { throw "Missing models\horizon_v16.meta.json — train + accept first" }
if (-not $SkipPreDeployGate) {
    Write-Host "=== Pre-deploy gate ===" -ForegroundColor Cyan
    py -3 scripts/pre_deploy_v16_gate.py
    if ($LASTEXITCODE -ne 0) { throw "pre_deploy_v16_gate FAILED — 禁止部署未验收模型（开发中可加 -SkipPreDeployGate）" }
} else {
    Write-Warning "SkipPreDeployGate: 跳过部署门禁（仅模型/代码同步，待新模型验收后再正式部署）"
}

$meta = Get-Content $metaPath -Raw -Encoding UTF8 | ConvertFrom-Json
$cal = $meta.calibration
$minConf = if ($cal.min_confidence) { [double]$cal.min_confidence } else { 0.48 }

Write-Host "=== Deploy Horizon V16 (flat_boost + calibration) ===" -ForegroundColor Cyan
Write-Host "macro_f1=$($meta.macro_f1) trial=$($meta.trial) min_conf=$minConf"

$modelFiles = @(
    "models\horizon_v16.onnx",
    "models\horizon_v16_scaler.pkl",
    "models\horizon_v16.meta.json",
    "models\horizon_v16.pth",
    "models\rl_agent_xau.zip",
    "models\XAUUSD\v16\rl_meta.json",
    "data\agent_state_scaler_xauusd.json"
)

foreach ($rel in $modelFiles) {
    $src = Join-Path $devRoot $rel
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    $dst = Join-Path $appData $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "AppData OK $rel"
}

$pyPatches = @(
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\execution_composer.py",
    "zhulong\agent\structure_service.py",
    "zhulong\agent\tick_brief.py"
)

$cfgPath = Join-Path $appData "config_agent.json"
. (Join-Path $PSScriptRoot "Merge-V16AgentConfig.ps1")
Merge-V16AgentConfig -TargetPath $cfgPath -HorizonMinConfidence $minConf

$deployMeta = @{
    deployed_at = (Get-Date).ToUniversalTime().ToString("o")
    tag = "horizon_v16_flat_boost_calibrated"
    trial = $meta.trial
    macro_f1 = $meta.macro_f1
    calibration = $cal
    kn2_status = "awaiting_gpu_acceptance"
}
$deployMeta | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $appData "models\XAUUSD\v16\deploy_status.json") -Encoding utf8

if (Test-Path $InstallDir) {
    $canInstall = $true
    try {
        $testDir = Join-Path $InstallDir "models\_write_test"
        New-Item -ItemType Directory -Force -Path $testDir -ErrorAction Stop | Out-Null
        Remove-Item -Recurse -Force $testDir -ErrorAction SilentlyContinue
    } catch {
        $canInstall = $false
        Write-Warning "No write access to $InstallDir — run deploy_v16_models_admin.ps1 as Admin for Program Files"
    }
    if ($canInstall) {
        foreach ($rel in $modelFiles) {
            $src = Join-Path $appData $rel
            if (-not (Test-Path $src)) { continue }
            $dst = Join-Path $InstallDir $rel
            $dir = Split-Path $dst -Parent
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            Copy-Item -Force $src $dst
            Write-Host "Install OK $rel"
        }
        foreach ($rel in $pyPatches) {
            $src = Join-Path $devRoot $rel
            if (-not (Test-Path $src)) { continue }
            $dst = Join-Path $InstallDir $rel
            $dir = Split-Path $dst -Parent
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            Copy-Item -Force $src $dst
            Write-Host "PATCH $rel"
        }
    }
}

Write-Host "`nHorizon V16 deployed to production (AppData)." -ForegroundColor Green
Write-Host "Restart ZhuLong.exe to load."
Write-Host "Config: merged from repo (execution_composer + trading_env intact). KN2 mode unchanged unless deploy_kn2_v16_when_ready.ps1 runs."
