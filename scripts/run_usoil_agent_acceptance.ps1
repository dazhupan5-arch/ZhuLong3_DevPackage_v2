#Requires -Version 5.1
<#
.SYNOPSIS
  原油 RL 智能体严格验收流水线：任一步 FAIL 即中止，不启用智能体。
#>
param(
    [switch] $SkipPrepare
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path

function Invoke-Strict([string]$Name, [string[]]$PyArgs) {
    Write-Host "== $Name ==" -ForegroundColor Cyan
    & py -3 @PyArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAIL: $Name (exit $LASTEXITCODE) — 流水线中止，智能体保持禁用" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host 'USOIL 智能体严格验收流水线（验收通过前 config_agent.enabled=false）' -ForegroundColor Yellow

if (-not $SkipPrepare) {
    Invoke-Strict -Name 'prepare_knowledge_data (USOIL V14+labels)' -PyArgs @(
        'scripts/prepare_knowledge_data.py', '--symbol', 'USOIL'
    )
}

Invoke-Strict -Name 'train_knowledge_net' -PyArgs @(
    'scripts/train_knowledge_net.py', '--symbol', 'USOIL'
)

Invoke-Strict -Name 'convert_knowledge_net_onnx' -PyArgs @(
    'scripts/convert_knowledge_net_to_onnx.py', '--symbol', 'USOIL', '--no-benchmark'
)

$timesteps = 500000
$cfgPath = Join-Path $root 'config_training.yaml'
if (Test-Path $cfgPath) {
    $m = Select-String -Path $cfgPath -Pattern '^\s*total_timesteps:\s*(\d+)' | Select-Object -First 1
    if ($m) { $timesteps = [int]$m.Matches.Groups[1].Value }
}

Invoke-Strict -Name 'train_rl_agent' -PyArgs @(
    'scripts/train_rl_agent.py', '--symbol', 'USOIL', '--timesteps', "$timesteps"
)

Invoke-Strict -Name 'backtest_rl' -PyArgs @(
    'scripts/backtest_rl.py', '--symbol', 'USOIL', '--year', '2025'
)

# 验收通过 → 写入摘要并提示启用
$summary = @{
    symbol             = 'USOIL'
    acceptance_passed  = $true
    knowledge_net      = 'models/knowledge_net_oil.onnx'
    rl_agent           = 'models/rl_agent_oil.zip'
    passed_at          = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
}
$summaryPath = Join-Path $root 'models\USOIL\agent_acceptance_summary.json'
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host ''
Write-Host '=== 全部验收 PASS ===' -ForegroundColor Green
Write-Host "报告: $summaryPath"
Write-Host '请手动将 config/config_agent.json 中 enabled 设为 true 后启用智能体' -ForegroundColor Yellow
