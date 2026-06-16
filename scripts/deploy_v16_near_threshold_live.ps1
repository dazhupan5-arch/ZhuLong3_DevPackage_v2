# 部署临近门槛 V16 备份到实机（试点：v16 架构 + 略降 RL 门控便于观察信号）
param(
    [string]$Tag = "v16_near_threshold_20260616",
    [string]$InstallDir = "C:\Program Files\ZhuLong"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

& "$PSScriptRoot\restore_v16_backup.ps1" -Tag $Tag

$appData = Join-Path $env:APPDATA "ZhuLong"
New-Item -ItemType Directory -Force -Path $appData | Out-Null

$agentCfg = @{
    enabled = $true
    use_rl = $true
    primary_symbol = "XAUUSD"
    signal_expiry_minutes = 240
    state_file = "data/agent_state.json"
    fallback_strategy = "none"
    architecture = @{
        version = "v16"
        horizon_predictor = @{
            horizon_bars = 12
            gain_threshold = 0.002
            min_direction_confidence = 0.42
            model_path = "models/horizon_v16.onnx"
            scaler_path = "models/horizon_v16_scaler.pkl"
            model_id = "horizon_v16"
        }
    }
    trader_mind = @{
        max_consecutive_losses = 6
        sl_atr_mult = 1.2
        tp_atr_mult = 2.0
        min_confidence = 0.42
    }
    kn2 = @{
        enabled = $false
        shadow_mode = $true
        model_path = "models/kn2_trader_v16.pth"
        min_confidence = 0.48
    }
    cognition = @{
        enabled = $true
        symbol = "XAUUSD"
        direction_threshold = 0.42
        base_confidence_threshold = 0.48
    }
    rl_inference = @{
        action_threshold = 0.52
        min_confidence_for_trade = 0.52
        max_daily_trades = 5
    }
    knowledge_net = @{
        model_path = "models/horizon_v16.onnx"
        scaler_path = "models/horizon_v16_scaler.pkl"
    }
    rl = @{
        model_path_xau = "models/rl_agent_xau"
    }
    symbols = @{
        XAUUSD = @{
            enabled = $true
            broker_symbol = "XAUUSD"
            state_scaler_path = "data/agent_state_scaler_xauusd.json"
            state_file = "data/agent_state_xauusd.json"
            rl = @{ model_path = "models/rl_agent_xau" }
        }
    }
}

$cfgPath = Join-Path $appData "config_agent.json"
$agentCfg | ConvertTo-Json -Depth 6 | Set-Content -Path $cfgPath -Encoding utf8
Write-Host "Wrote $cfgPath"

$modelFiles = @(
    "models\horizon_v16.onnx",
    "models\horizon_v16_scaler.pkl",
    "models\horizon_v16.meta.json",
    "models\rl_agent_xau.zip",
    "models\XAUUSD\v16\rl_meta.json",
    "data\agent_state_scaler_xauusd.json"
)

if (Test-Path $InstallDir) {
    $canInstall = $true
    try {
        $testDir = Join-Path $InstallDir "models\_write_test"
        New-Item -ItemType Directory -Force -Path $testDir -ErrorAction Stop | Out-Null
        Remove-Item -Recurse -Force $testDir -ErrorAction SilentlyContinue
    } catch {
        $canInstall = $false
        Write-Warning "No write access to $InstallDir — skipping Program Files copy"
    }
    if ($canInstall) {
        foreach ($rel in $modelFiles) {
            $src = Join-Path (Get-Location) $rel
            $dst = Join-Path $InstallDir $rel
            if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
            $dir = Split-Path $dst -Parent
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
            Copy-Item -Force $src $dst
            Write-Host "Installed $rel"
        }
        $devTa = Join-Path (Get-Location) "zhulong\agent\trading_agent.py"
        $instTa = Join-Path $InstallDir "zhulong\agent\trading_agent.py"
        if ((Test-Path $devTa) -and (Test-Path (Split-Path $instTa -Parent))) {
            Copy-Item -Force $devTa $instTa
            Write-Host "Updated zhulong/agent/trading_agent.py"
        }
    }
}

$appModels = Join-Path $appData "models"
New-Item -ItemType Directory -Force -Path (Join-Path $appModels "XAUUSD\v16") | Out-Null
foreach ($rel in $modelFiles) {
    $src = Join-Path (Get-Location) $rel
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path $appData $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
}
Write-Host "AppData deploy: $appData"

Write-Host "`nV16 near-threshold pilot deployed. Restart ZhuLong to load." -ForegroundColor Green
Write-Host "Note: acceptance passed=false — pilot only until KN2 V16 ready."
