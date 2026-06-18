# 管理员部署：V16 模型 + Python 补丁 → Program Files（实机子进程只读安装目录）
#Requires -RunAsAdministrator
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [string]$DevRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $DevRoot) {
    $DevRoot = Split-Path $PSScriptRoot -Parent
}
$appData = Join-Path $env:APPDATA "ZhuLong"

Write-Host "=== V16 Admin Deploy ===" -ForegroundColor Cyan
Write-Host "Install: $InstallDir"
Write-Host "AppData: $appData"
Write-Host "Dev:     $DevRoot"

$modelFiles = @(
    "models\horizon_v16.onnx",
    "models\horizon_v16_scaler.pkl",
    "models\horizon_v16.meta.json",
    "models\horizon_v16.pth",
    "models\kn2_trader_v16.pth",
    "models\kn2_trader_v16.meta.json",
    "models\rl_agent_xau.zip",
    "models\XAUUSD\v16\rl_meta.json",
    "data\agent_state_scaler_xauusd.json"
)

foreach ($rel in $modelFiles) {
    $src = Join-Path $appData $rel
    if (-not (Test-Path $src)) { $src = Join-Path $DevRoot $rel }
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    $dst = Join-Path $InstallDir $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "  OK $rel"
}

$pyPatches = @(
    @{ Src = "zhulong\agent\rl_agent.py"; Dst = "zhulong\agent\rl_agent.py" },
    @{ Src = "zhulong\agent\horizon_predictor.py"; Dst = "zhulong\agent\horizon_predictor.py" },
    @{ Src = "zhulong\agent\trading_agent.py"; Dst = "zhulong\agent\trading_agent.py" },
    @{ Src = "zhulong\agent\tick_brief.py"; Dst = "zhulong\agent\tick_brief.py" },
    @{ Src = "zhulong\agent\structure_service.py"; Dst = "zhulong\agent\structure_service.py" },
    @{ Src = "zhulong\agent\trader_mind.py"; Dst = "zhulong\agent\trader_mind.py" },
    @{ Src = "zhulong\agent\execution_composer.py"; Dst = "zhulong\agent\execution_composer.py" },
    @{ Src = "zhulong\agent\kn2_location_labels.py"; Dst = "zhulong\agent\kn2_location_labels.py" },
    @{ Src = "zhulong\agent\knowledge_net_kn2.py"; Dst = "zhulong\agent\knowledge_net_kn2.py" },
    @{ Src = "zhulong\agent\knowledge_net.py"; Dst = "zhulong\agent\knowledge_net.py" },
    @{ Src = "zhulong\agent\cognition.py"; Dst = "zhulong\agent\cognition.py" },
    @{ Src = "zhulong\utils\paths.py"; Dst = "zhulong\utils\paths.py" },
    @{ Src = "ZhuLong.PythonEngine\inference_cli.py"; Dst = "ZhuLong.PythonEngine\inference_cli.py" }
)

foreach ($p in $pyPatches) {
    $src = Join-Path $DevRoot $p.Src
    if (-not (Test-Path $src)) { Write-Warning "skip patch $($p.Src)"; continue }
    foreach ($base in @($appData, $InstallDir)) {
        $dst = Join-Path $base $p.Dst
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        try {
            Copy-Item -Force $src $dst -ErrorAction Stop
            Write-Host "  OK $($p.Dst)" -ForegroundColor $(if ($base -eq $appData) { "Yellow" } else { "Green" })
        } catch {
            if ($base -eq $InstallDir) { Write-Warning "  skip InstallDir $($p.Dst)" }
        }
    }
}

$cliSrc = Join-Path $DevRoot "ZhuLong.PythonEngine\inference_cli.py"
if (Test-Path $cliSrc) {
    $cliDst = Join-Path $appData "ZhuLong.PythonEngine\inference_cli.py"
    $cliDir = Split-Path $cliDst -Parent
    if (-not (Test-Path $cliDir)) { New-Item -ItemType Directory -Force -Path $cliDir | Out-Null }
    Copy-Item -Force $cliSrc $cliDst
    Write-Host "  OK AppData ZhuLong.PythonEngine\inference_cli.py" -ForegroundColor Yellow
}

Write-Host "`nDone. Restart ZhuLong.exe" -ForegroundColor Green
