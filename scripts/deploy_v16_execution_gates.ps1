# 部署 V16 执行门控 + 结构位置过滤 + UI 日志字段（Python → AppData）
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$appData = Join-Path $env:APPDATA "ZhuLong"
New-Item -ItemType Directory -Force -Path $appData | Out-Null

$devRoot = Get-Location
Write-Host "=== Deploy V16 execution gates + structure location filter ===" -ForegroundColor Cyan

$pyFiles = @(
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\trader_mind.py",
    "zhulong\agent\execution_composer.py",
    "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\knowledge_net.py",
    "zhulong\agent\knowledge_net_kn2.py",
    "zhulong\agent\tick_brief.py",
    "zhulong\agent\cognition.py",
    "zhulong\agent\structure_service.py",
    "zhulong\engine\agent_engine.py"
)

foreach ($rel in $pyFiles) {
    $src = Join-Path $devRoot $rel
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    foreach ($base in @($appData, $InstallDir)) {
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) {
            try { New-Item -ItemType Directory -Force -Path $dir | Out-Null } catch { continue }
        }
        try {
            Copy-Item -Force $src $dst -ErrorAction Stop
        } catch {
            if ($base -eq $InstallDir) {
                Write-Warning "skip Program Files (need admin): $rel"
            } else {
                throw
            }
        }
    }
    Write-Host "OK $rel"
}

# inference_cli → AppData（子进程勿从 Program Files 路径启动，易挂起/超时）
$cliRel = "ZhuLong.PythonEngine\inference_cli.py"
$cliSrc = Join-Path $devRoot $cliRel
if (Test-Path $cliSrc) {
    $cliDst = Join-Path $appData $cliRel
    $cliDir = Split-Path $cliDst -Parent
    if (-not (Test-Path $cliDir)) { New-Item -ItemType Directory -Force -Path $cliDir | Out-Null }
    Copy-Item -Force $cliSrc $cliDst
    Write-Host "OK $cliRel (AppData hotfix for subprocess)"
}

$cfgPath = Join-Path $appData "config_agent.json"
. (Join-Path $PSScriptRoot "Merge-V16AgentConfig.ps1")
Merge-V16AgentConfig -TargetPath $cfgPath

Write-Host @"

Python 热更新完成。UI 日志变更需重新编译安装 ZhuLong.App：
  dotnet build src\ZhuLong.App\ZhuLong.App.csproj -c Release

重启烛龙后验证：
  日志前缀 [V16·Horizon]
  震荡高位追多应被 structure_gate 拦截

"@ -ForegroundColor Green
