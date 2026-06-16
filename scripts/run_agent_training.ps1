#Requires -Version 5.1
<#
.SYNOPSIS
  烛龙 RL 智能体完整训练流水线（XAUUSD / USOIL）
.EXAMPLE
  .\scripts\run_agent_training.ps1 -Symbol XAUUSD
  .\scripts\run_agent_training.ps1 -Symbol USOIL -Quick
#>
param(
    [ValidateSet('XAUUSD', 'USOIL')]
    [string] $Symbol = 'XAUUSD',
    [switch] $SkipPrepare,
    [switch] $SkipKnowledge,
    [switch] $SkipRl,
    [switch] $SkipBacktest,
    [switch] $Quick
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path

function Invoke-Step([string]$Name, [string[]]$PyArgs, [int[]]$AllowExit = @()) {
    Write-Host "== $Name ==" -ForegroundColor Cyan
    & py -3 @PyArgs
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -notin $AllowExit) {
        throw "$Name failed (exit $LASTEXITCODE)"
    }
    if ($LASTEXITCODE -in $AllowExit) {
        Write-Host "WARN: $Name exit $LASTEXITCODE (continuing pipeline)" -ForegroundColor Yellow
    }
}

if (-not $SkipPrepare) {
    Invoke-Step -Name 'prepare_knowledge_data' -PyArgs @(
        'scripts/prepare_knowledge_data.py',
        '--symbol', $Symbol
    )
}

if (-not $SkipKnowledge) {
    $knScript = if ($Symbol -eq 'XAUUSD') { 'scripts/train_knowledge_net_v14.py' } else { 'scripts/train_knowledge_net.py' }
    Invoke-Step -Name 'train_knowledge_net' -PyArgs @(
        $knScript,
        '--symbol', $Symbol
    ) -AllowExit 2
    Invoke-Step -Name 'convert_knowledge_net_onnx' -PyArgs @(
        'scripts/convert_knowledge_net_to_onnx.py',
        '--symbol', $Symbol,
        '--no-benchmark'
    )
}

if (-not $SkipRl) {
    $rlArgs = @('scripts/train_rl_agent.py', '--symbol', $Symbol)
    if ($Quick) {
        $rlArgs += '--quick'
    } else {
        $cfgPath = Join-Path $root 'config_training.yaml'
        if (Test-Path $cfgPath) {
            $m = Select-String -Path $cfgPath -Pattern '^\s*total_timesteps:\s*(\d+)' | Select-Object -First 1
            if ($m -and $m.Matches.Groups[1].Value) {
                $rlArgs += '--timesteps', $m.Matches.Groups[1].Value
            }
        }
    }
    Invoke-Step -Name 'train_rl_agent' -PyArgs $rlArgs
}

if (-not $SkipBacktest) {
    Invoke-Step -Name 'backtest_rl' -PyArgs @(
        'scripts/backtest_rl.py',
        '--symbol', $Symbol
    ) -AllowExit 2
}

Write-Host "DONE: $Symbol training pipeline" -ForegroundColor Green
Write-Host "Logs: $root\logs\training and $root\logs\rl" -ForegroundColor Gray
