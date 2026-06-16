#Requires -Version 5.1
# 黄金 + 原油 RL 智能体完整训练（顺序执行，日志写入 logs/training/）
param(
    [string] $LogDir = "logs/training",
    [switch] $SkipXauPrepare,
    [switch] $SkipXauKnowledge,
    [switch] $SkipXauRl,
    [switch] $SkipXauBacktest,
    [switch] $SkipOilPrepare
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$log = Join-Path $LogDir ("full_xau_usoil_{0:yyyyMMdd_HHmmss}.log" -f (Get-Date))
"LOG=$log" | Out-File -Encoding utf8 (Join-Path $LogDir 'current_run.txt')

function Write-Log([string]$Msg) {
    $line = "[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $Msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding utf8
}

function Invoke-TrainStep([string]$Name, [string[]]$PyArgs) {
    Write-Log "== $Name =="
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & py -3 @PyArgs 2>&1
    $ErrorActionPreference = $prevEap
    $output | ForEach-Object { Write-Log $_ }
    if ($LASTEXITCODE -ne 0) {
        Write-Log "FAILED: $Name (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Write-Log "START full agent training at $root"
Write-Log "Bootstrap PyTorch DLL"
& py -3 -c "import torch; print('torch', torch.__version__)" 2>&1 | ForEach-Object { Write-Log $_ }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipXauPrepare) {
    Invoke-TrainStep 'XAU prepare' @(
        'scripts/prepare_training_data.py', '--symbol', 'XAUUSD',
        '--start', '2016-01-01', '--end', '2025-12-31'
    )
} else {
    Write-Log "SKIP XAU prepare (existing npz)"
}
if (-not $SkipXauKnowledge) {
    Invoke-TrainStep 'XAU knowledge' @('scripts/train_knowledge_net.py', '--symbol', 'XAUUSD')
} else {
    Write-Log "SKIP XAU knowledge (existing model)"
}
if (-not $SkipXauRl) {
    Invoke-TrainStep 'XAU rl' @('scripts/train_rl_agent.py', '--symbol', 'XAUUSD')
} else {
    Write-Log "SKIP XAU rl"
}
if (-not $SkipXauBacktest) {
    Invoke-TrainStep 'XAU backtest' @('scripts/backtest_rl.py', '--symbol', 'XAUUSD')
} else {
    Write-Log "SKIP XAU backtest"
}

if (-not $SkipOilPrepare) {
    Invoke-TrainStep 'USOIL prepare' @(
        'scripts/prepare_training_data.py', '--symbol', 'USOIL',
        '--start', '2016-01-01', '--end', '2025-12-31'
    )
} else {
    Write-Log "SKIP USOIL prepare (existing npz)"
}
Invoke-TrainStep 'USOIL knowledge' @('scripts/train_knowledge_net.py', '--symbol', 'USOIL')
Invoke-TrainStep 'USOIL rl' @('scripts/train_rl_agent.py', '--symbol', 'USOIL')
Invoke-TrainStep 'USOIL backtest' @('scripts/backtest_rl.py', '--symbol', 'USOIL')

Write-Log 'ALL DONE'
