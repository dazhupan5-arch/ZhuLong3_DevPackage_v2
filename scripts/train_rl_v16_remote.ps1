<#
.SYNOPSIS
  V16 PPO 跨机训练（Execution Parity + Horizon V16 状态，与实盘 Agent 一致）

.DESCRIPTION
  1. 检查 Python / PyTorch / stable-baselines3 / gymnasium
  2. 可选 -InstallDeps 安装依赖
  3. 校验 data/clean/training_horizon_v16*.npz（含 OHLCV）与 models/horizon_v16.onnx
  4. 以 execution_parity 环境训练 PPO → models/rl_agent_xau.zip
  5. 输出 rl meta 与拷回清单

  说明：SB3 MLP 策略在多数 GPU 上仍走 CPU 更快；本脚本在 GPU 训练机长跑，
  如需强制 CUDA 加 -Sb3Device cuda（config_training.yaml device.rl 可被覆盖）。

.PARAMETER InstallDeps
  首次 GPU/训练机运行加此开关

.PARAMETER Timesteps
  PPO 总步数，默认 config_training.yaml rl.total_timesteps (800000)

.PARAMETER Quick
  冒烟 5000 步

.EXAMPLE
  git pull; git lfs pull
  powershell -ExecutionPolicy Bypass -File scripts/train_rl_v16_remote.ps1 -InstallDeps
  powershell -ExecutionPolicy Bypass -File scripts/train_rl_v16_remote.ps1 -Timesteps 800000

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/train_rl_v16_remote.ps1 -Symbol USOIL
#>
[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [ValidateSet("XAUUSD", "USOIL")]
    [string]$Symbol = "XAUUSD",
    [int]$Timesteps = 0,
    [switch]$Quick,
    [switch]$SkipDataCheck,
    [switch]$SkipEnrich,
    [ValidateSet("", "auto", "cuda", "cpu")]
    [string]$Sb3Device = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

Write-Step "环境"
py -3 --version

if ($InstallDeps) {
    py -3 -m pip install -U pip
    py -3 -m pip install -r requirements.txt
    py -3 -m pip install stable-baselines3 gymnasium tensorboard
    py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}

py -3 -c @"
import importlib.util
for pkg in ('torch', 'numpy', 'pandas', 'gymnasium', 'stable_baselines3'):
    if importlib.util.find_spec(pkg) is None:
        raise SystemExit(f'missing package: {pkg}')
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
"@

$sym = $Symbol.ToUpper()
if ($sym -eq "USOIL") {
    $npzRel = "data\clean\training_horizon_v16_usoil.npz"
    $onnxRel = "models\horizon_v16_usoil.onnx"
    $rlOut = "models\rl_agent_oil.zip"
    $rlMeta = "models\XAUUSD\v16\rl_meta_oil.json"
} else {
    $npzRel = "data\clean\training_horizon_v16.npz"
    $onnxRel = "models\horizon_v16.onnx"
    $rlOut = "models\rl_agent_xau.zip"
    $rlMeta = "models\XAUUSD\v16\rl_meta.json"
}

$npz = Join-Path $Root $npzRel
$onnx = Join-Path $Root $onnxRel
$scaler = Join-Path $Root "models\horizon_v16_scaler.pkl"
if ($sym -eq "USOIL") {
    $scaler = Join-Path $Root "models\horizon_v16_usoil_scaler.pkl"
}

if (-not $SkipDataCheck) {
    Write-Step "数据与 Horizon 模型 (git lfs)"
    if (-not (Test-Path $npz)) {
        Write-Host @"

缺少 Horizon NPZ: $npzRel

训练机：
  git pull
  git lfs pull

开发机生成：
  py -3 scripts/prepare_horizon_v16_data.py --symbol $sym
  py -3 scripts/enrich_horizon_v16_npz.py --symbol $sym

"@ -ForegroundColor Yellow
        exit 1
    }
    $sizeMb = [math]::Round((Get-Item $npz).Length / 1MB, 1)
    Write-Host "npz: $npzRel (${sizeMb} MB)" -ForegroundColor Green

    $hasOpen = py -3 -c "import numpy as np; d=np.load(r'$npz', allow_pickle=True); print('open' in d.files)"
    if ($hasOpen -notmatch "True") {
        if ($SkipEnrich) {
            Write-Host "NPZ 缺少 OHLCV，请运行 enrich_horizon_v16_npz.py" -ForegroundColor Red
            exit 1
        }
        Write-Step "补齐 OHLCV (enrich_horizon_v16_npz.py)"
        py -3 scripts/enrich_horizon_v16_npz.py --symbol $sym
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (-not (Test-Path $onnx)) {
        Write-Host "缺少 $onnxRel — 请先训练 Horizon 并 export ONNX，或 git lfs pull" -ForegroundColor Red
        exit 1
    }
    if (-not (Test-Path $scaler)) {
        Write-Host "缺少 scaler — git lfs pull models/" -ForegroundColor Red
        exit 1
    }
    Write-Host "Horizon ONNX OK: $onnxRel" -ForegroundColor Green
}

Write-Step "Execution Parity 栈校验"
py -3 -c @"
from pathlib import Path
from zhulong.agent.execution_composer import ExecutionComposer, limit_fill_on_bar
from zhulong.agent.trading_env import TradingEnv
print('execution_composer OK')
print('limit_fill', limit_fill_on_bar('short', 2005.0, 2010.0, 1990.0, 2000.0))
"@

$logDir = Join-Path $Root "logs\rl"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("rl_v16_execution_parity_" + $sym + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

Write-Step "V16 PPO 训练 (execution_parity, symbol=$sym)"
$trainArgs = @(
    "-u", "scripts/train_rl_agent.py",
    "--v16",
    "--symbol", $sym
)
if ($Quick) {
    $trainArgs += "--quick"
} elseif ($Timesteps -gt 0) {
    $trainArgs += @("--timesteps", $Timesteps)
}
if ($Sb3Device) {
    $trainArgs += @("--device", $Sb3Device)
}

Write-Host "Command: py -3 $($trainArgs -join ' ')" -ForegroundColor DarkGray
Write-Host "Log: $log" -ForegroundColor DarkGray

py -3 @trainArgs 2>&1 | Tee-Object -FilePath $log

if ($LASTEXITCODE -ne 0) {
    Write-Host "PPO 训练失败，见 $log" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Step "完成 — 拷回开发机"
$artifacts = @(
    $rlOut,
    "data\agent_state_scaler_$($sym.ToLower()).json",
    "logs\rl_$sym.json"
)
if ($sym -eq "XAUUSD") {
    $artifacts += @("models\XAUUSD\v16\rl_meta.json")
} else {
    $artifacts += @("models\USOIL\v16\rl_meta.json")
}
foreach ($rel in $artifacts) {
    $full = Join-Path $Root $rel
    if (Test-Path $full) {
        $fi = Get-Item $full
        Write-Host ("  OK  {0}  ({1:N0} bytes)" -f $rel, $fi.Length) -ForegroundColor Green
    } else {
        Write-Host ("  SKIP  {0}" -f $rel) -ForegroundColor Yellow
    }
}

Write-Host @"

开发机部署（模型拷回后）:
  copy models\rl_agent_*.zip → 安装目录 models\
  copy data\agent_state_scaler_*.json → 安装目录 data\
  powershell -File scripts/deploy_v16_full_stack.ps1

"@ -ForegroundColor Cyan
