<#
.SYNOPSIS
  V16 全栈无泄露重训：Horizon → KN2 → RL（契约 v16_no_leak_1）

.DESCRIPTION
  严格时间切分（train <= 2024-12-31，OOS val = 2025），禁止随机 val。
  顺序：
    1. 契约审计（代码守卫）
    2. 清洗 CSV + Horizon NPZ + 位置标签
    3. Horizon temporal 训练 + 校准 + ONNX
    4. KN2 NPZ（校验 Horizon meta）+ 位置标签 + KN2 训练
    5. RL PPO（execution_parity，train<=2024 / eval 2025）
    6. 契约审计（产物）

  GPU 机推荐：
    git pull; git lfs pull
    powershell -ExecutionPolicy Bypass -File scripts/retrain_v16_no_leak.ps1 -InstallDeps

.PARAMETER InstallDeps
  安装 requirements + CUDA torch + stable-baselines3

.PARAMETER SkipClean
  跳过 CSV/Horizon NPZ 重建（已有 clean 数据时）

.PARAMETER SkipHorizon
  跳过 Horizon 训练（KN2/RL 仍会继续）

.PARAMETER Quick
  RL 冒烟 5000 步

.PARAMETER Timesteps
  RL 总步数（默认 config_training.yaml rl.total_timesteps）

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/retrain_v16_no_leak.ps1 -InstallDeps
#>
[CmdletBinding()]
param(
    [switch]$InstallDeps,
    [switch]$SkipClean,
    [switch]$SkipHorizon,
    [switch]$SkipKn2,
    [switch]$SkipRl,
    [switch]$Quick,
    [int]$Timesteps = 0,
    [ValidateSet("XAUUSD", "USOIL")]
    [string]$Symbol = "XAUUSD"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

function Write-Step([string]$Msg) {
    Write-Host "`n=== $Msg ===" -ForegroundColor Cyan
}

$TrainEnd = "2024-12-31"
$ValYear = 2025
$HorizonNpz = "data/clean/training_horizon_v16.npz"
$HorizonLocNpz = "data/clean/training_horizon_v16_location.npz"
$Kn2LocNpz = "data/clean/kn2_training_v16_location.npz"
$LogDir = Join-Path $Root "logs/training/retrain_no_leak"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"

Write-Step "契约审计（训练前）"
py -3 scripts/audit_training_no_leak.py --pre
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($InstallDeps) {
    Write-Step "依赖"
    py -3 -m pip install -U pip
    py -3 -m pip install -r requirements.txt
    py -3 -m pip install stable-baselines3 gymnasium tensorboard
    py -3 -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}

py -3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if (-not $SkipClean) {
    Write-Step "清洗 + Horizon NPZ（无 KN2，待 Horizon 训完再打 embed）"
    py -3 scripts/clean_training_data_v16.py `
        --symbol $Symbol `
        --full-rebuild `
        --skip-kn2 `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "01_clean_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "Horizon 位置标签 NPZ"
    py -3 scripts/prepare_horizon_v16_location_labels.py `
        --npz $HorizonNpz `
        --out $HorizonLocNpz `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "02_horizon_location_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipHorizon) {
    Write-Step "Horizon V16 训练（temporal-val，禁止随机 val）"
    py -3 -u scripts/train_horizon_v16.py `
        --npz $HorizonLocNpz `
        --label-mode location `
        --temporal-val `
        --train-end $TrainEnd `
        --epochs 80 `
        --patience 15 `
        --class-weights 2.4,1.1,2.4 `
        --smote-ratio 0.52 `
        --lr 0.00025 `
        --log-suffix no_leak `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "03_horizon_train_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "Horizon 校准（temporal-val）"
    py -3 scripts/calibrate_horizon_v16.py `
        --temporal-val `
        --train-end $TrainEnd `
        --apply `
        --npz $HorizonLocNpz `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "04_horizon_cal_$Ts.log")

    Write-Step "导出 Horizon ONNX"
    py -3 scripts/convert_knowledge_net_to_onnx.py `
        --model models/horizon_v16.pth `
        --out models/horizon_v16.onnx `
        --no-benchmark
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "Horizon 门禁（分类 F1>0.5, long/short P/R>=80%，不含 RL）"
    py -3 scripts/accept_horizon_v16.py --horizon-only
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Horizon 分类验收不合格 exit=$LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if (-not $SkipKn2) {
    Write-Step "KN2 NPZ（Horizon temporal meta 校验）"
    py -3 scripts/prepare_kn2_v16_data.py `
        --npz $HorizonLocNpz `
        --out $Kn2LocNpz `
        --horizon-onnx models/horizon_v16.onnx `
        --horizon-scaler models/horizon_v16_scaler.pkl `
        --train-end $TrainEnd `
        --rebuild `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "05_kn2_npz_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "KN2 位置标签"
    py -3 scripts/prepare_kn2_v16_location_labels.py `
        --npz $Kn2LocNpz `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "06_kn2_location_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "KN2 V16 训练（train<=$TrainEnd, val=$ValYear OOS）"
    py -3 -u scripts/train_kn2_v16.py `
        --npz $Kn2LocNpz `
        --label-mode location `
        --mode fast `
        --batch-size 48 `
        --epochs 120 `
        --patience 25 `
        --train-end $TrainEnd `
        --val-year $ValYear `
        2>&1 | Tee-Object -FilePath (Join-Path $LogDir "07_kn2_train_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    py -3 scripts/accept_kn2_v16.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "KN2 验收不合格 exit=$LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if (-not $SkipRl) {
    Write-Step "RL NPZ 补齐 OHLCV（location NPZ）"
    py -3 scripts/enrich_horizon_v16_npz.py --symbol $Symbol --npz $HorizonLocNpz
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Step "RL PPO（execution_parity, temporal train/eval）"
    $rlArgs = @("-u", "scripts/train_rl_agent.py", "--v16", "--symbol", $Symbol)
    if ($Quick) { $rlArgs += "--quick" }
    elseif ($Timesteps -gt 0) { $rlArgs += @("--timesteps", $Timesteps) }
    py -3 @rlArgs 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "08_rl_train_$Ts.log")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Step "全栈验收（Horizon+OOS+RL+Agent，通过后写入 meta/config）"
py -3 scripts/accept_horizon_v16.py --apply
if ($LASTEXITCODE -ne 0) {
    Write-Host "全栈验收不合格 exit=$LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Step "契约审计（训练后，未通过则中断）"
py -3 scripts/audit_training_no_leak.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "契约审计未通过 exit=$LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Step "部署门禁（开发机拷回模型后须再跑）"
py -3 scripts/pre_deploy_v16_gate.py --require-kn2-live
if ($LASTEXITCODE -ne 0) {
    Write-Host "部署门禁未通过（KN2 LIVE 需 acceptance_report）" -ForegroundColor Yellow
}

Write-Step "完成 — 部署清单"
@(
    "models/horizon_v16.pth",
    "models/horizon_v16.onnx",
    "models/horizon_v16_scaler.pkl",
    "models/horizon_v16.meta.json",
    "models/kn2_trader_v16.pth",
    "models/kn2_trader_v16.meta.json",
    "models/rl_agent_xau.zip",
    "data/agent_state_scaler_xauusd.json"
) | ForEach-Object {
    $f = Join-Path $Root $_
    if (Test-Path $f) {
        Write-Host ("  OK  {0}" -f $_) -ForegroundColor Green
    } else {
        Write-Host ("  --  {0}" -f $_) -ForegroundColor Yellow
    }
}

Write-Host @"

开发机部署：
  copy models/* → 安装目录 models/
  powershell -File scripts/pack-installer.ps1 -ForcePack

"@ -ForegroundColor Cyan
