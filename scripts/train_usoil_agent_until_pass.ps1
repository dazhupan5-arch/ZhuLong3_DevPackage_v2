#Requires -Version 5.1
<#
.SYNOPSIS
  清理不合格产物 → 循环训练直至 KnowledgeNet + PPO 回测全部 PASS，方可装机。
#>
param(
    [int] $MaxKnAttempts = 5,
    [int] $MaxRlAttempts = 3
)

$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$root = (Get-Location).Path

function Remove-OilAgentArtifacts {
    Write-Host '== 清理不合格 USOIL 智能体产物 ==' -ForegroundColor Yellow
    $patterns = @(
        'models\knowledge_net_oil.pth',
        'models\knowledge_net_oil.onnx',
        'models\knowledge_net_oil.meta.json',
        'models\knowledge_scaler_oil.pkl',
        'models\rl_agent_oil.zip',
        'models\USOIL\agent_acceptance_summary.json',
        'logs\training\backtest_USOIL_2025.json',
        'logs\training\knowledge_USOIL.log',
        'logs\training\rl_metrics_USOIL.jsonl',
        'logs\training\rl_USOIL.json'
    )
    foreach ($rel in $patterns) {
        $p = Join-Path $root $rel
        if (Test-Path $p) {
            Remove-Item -Force $p
            Write-Host "  removed $rel"
        }
    }
    $rlDir = Join-Path $root 'logs\rl\usoil'
    if (Test-Path $rlDir) {
        Remove-Item -Recurse -Force $rlDir
        Write-Host '  removed logs/rl/usoil/'
    }
    # SB3 best_model 残留
    Get-ChildItem (Join-Path $root 'models') -Filter 'best_model.zip' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Invoke-Step([string]$Name, [string[]]$PyArgs) {
    Write-Host "== $Name ==" -ForegroundColor Cyan
    & py -3 @PyArgs
    return $LASTEXITCODE
}

# 确保智能体禁用
$agentCfg = Join-Path $root 'config\config_agent.json'
if (Test-Path $agentCfg) {
    $json = Get-Content $agentCfg -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($json.enabled -eq $true) {
        $json.enabled = $false
        $json | ConvertTo-Json -Depth 20 | Set-Content $agentCfg -Encoding UTF8
        Write-Host 'config_agent.json: enabled=false' -ForegroundColor Yellow
    }
}

Remove-OilAgentArtifacts

# --- 1. 准备 V14 训练数据（一次性） ---
$npz = Join-Path $root 'data\oil_training_data.npz'
if (-not (Test-Path $npz)) {
    $ec = Invoke-Step 'prepare_knowledge_data' @('scripts/prepare_knowledge_data.py', '--symbol', 'USOIL')
    if ($ec -ne 0) { throw "prepare_knowledge_data failed ($ec)" }
} else {
    Write-Host '== 已有 oil_training_data.npz，跳过 prepare（如需重建请删除该文件）==' -ForegroundColor Gray
}

# --- 2. KnowledgeNet 循环直到 PASS ---
$knPass = $false
for ($i = 1; $i -le $MaxKnAttempts; $i++) {
    Write-Host "`n--- KnowledgeNet 尝试 $i / $MaxKnAttempts ---" -ForegroundColor Magenta
    if ($i -gt 1) {
        Remove-OilAgentArtifacts
        if ($i -eq 2 -and -not (Test-Path (Join-Path $root 'data\training\v14\USOIL\features.parquet'))) {
            Write-Host '重建 V14 特征数据...' -ForegroundColor Cyan
            $ec = Invoke-Step 'prepare_knowledge_data (rebuild)' @('scripts/prepare_knowledge_data.py', '--symbol', 'USOIL')
            if ($ec -ne 0) { continue }
        }
    }
    $ec = Invoke-Step 'train_knowledge_net' @('scripts/train_knowledge_net.py', '--symbol', 'USOIL')
    if ($ec -eq 0) {
        $knPass = $true
        break
    }
    Write-Host "KnowledgeNet 未达标 (exit $ec)，重试..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}

if (-not $knPass) {
    Write-Host 'KnowledgeNet 在最大尝试次数内未通过验收，中止。' -ForegroundColor Red
    exit 1
}

$ec = Invoke-Step 'convert_knowledge_net_onnx' @(
    'scripts/convert_knowledge_net_to_onnx.py', '--symbol', 'USOIL', '--no-benchmark'
)
if ($ec -ne 0) { exit $ec }

# --- 3. PPO + 回测循环直到 PASS ---
$timesteps = 500000
$cfgPath = Join-Path $root 'config_training.yaml'
if (Test-Path $cfgPath) {
    $m = Select-String -Path $cfgPath -Pattern '^\s*total_timesteps:\s*(\d+)' | Select-Object -First 1
    if ($m) { $timesteps = [int]$m.Matches.Groups[1].Value }
}

$rlPass = $false
for ($j = 1; $j -le $MaxRlAttempts; $j++) {
    Write-Host "`n--- PPO+回测 尝试 $j / $MaxRlAttempts ---" -ForegroundColor Magenta
    if ($j -gt 1) {
        Remove-Item -Force (Join-Path $root 'models\rl_agent_oil.zip') -ErrorAction SilentlyContinue
    }
    $steps = [int]($timesteps * (1 + ($j - 1) * 0.25))
    $ec = Invoke-Step 'train_rl_agent' @(
        'scripts/train_rl_agent.py', '--symbol', 'USOIL', '--timesteps', "$steps"
    )
    if ($ec -ne 0) { continue }

    $ec = Invoke-Step 'backtest_rl' @('scripts/backtest_rl.py', '--symbol', 'USOIL', '--year', '2025')
    if ($ec -eq 0) {
        $rlPass = $true
        break
    }
    Write-Host "PPO 回测未达标 (exit $ec)，增加 timesteps 重训..." -ForegroundColor Yellow
}

if (-not $rlPass) {
    Write-Host 'PPO 回测在最大尝试次数内未通过验收，中止。' -ForegroundColor Red
    exit 1
}

# --- 4. 写入验收摘要，启用智能体 ---
$summary = @{
    symbol            = 'USOIL'
    acceptance_passed = $true
    knowledge_net     = 'models/knowledge_net_oil.onnx'
    rl_agent          = 'models/rl_agent_oil.zip'
    passed_at         = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    note              = 'KnowledgeNet + PPO 回测均已 PASS，可装机'
}
$summaryPath = Join-Path $root 'models\USOIL\agent_acceptance_summary.json'
New-Item -ItemType Directory -Force -Path (Split-Path $summaryPath) | Out-Null
$summary | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryPath -Encoding UTF8

$json = Get-Content $agentCfg -Raw -Encoding UTF8 | ConvertFrom-Json
$json.enabled = $true
$json | ConvertTo-Json -Depth 20 | Set-Content $agentCfg -Encoding UTF8

Write-Host ''
Write-Host '=== 全部验收 PASS — 已启用 config_agent.enabled=true，可装机 ===' -ForegroundColor Green
Write-Host "摘要: $summaryPath"
exit 0
