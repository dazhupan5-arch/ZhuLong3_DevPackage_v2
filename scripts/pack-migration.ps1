# 烛龙3 工程移机包（GPU 训练用，不含安装包/publish/python_runtime）
#Requires -Version 5.1
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$RepoRoot = (Get-Location).Path
$outDir = Join-Path $RepoRoot 'output'
$stamp = Get-Date -Format 'yyyyMMdd'
$zipName = "ZhuLong3_Migration_$stamp.zip"
$zipPath = Join-Path $outDir $zipName
$stage = Join-Path $env:TEMP "ZhuLong3_Migration_$stamp"

if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$excludeDirs = @(
    'publish', 'output', 'python_runtime', 'test_install', '_publish_m1fix',
    '.pytest_cache', 'logs', 'legacy', 'node_modules', '__pycache__', '.git'
)
$excludeFiles = @('*.log', '*.pyc')

Write-Host "== Staging ZhuLong_3 -> $stage ==" -ForegroundColor Cyan
$robocopyArgs = @($RepoRoot, $stage, '/E', '/NFL', '/NDL', '/NJH', '/NJS', '/XD') + $excludeDirs
& robocopy @robocopyArgs | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed exit $LASTEXITCODE" }
$global:LASTEXITCODE = 0

# README for GPU machine
$readme = @"
# 烛龙3 移机包 (GPU 训练)

## 环境
```powershell
py -3 -m pip install -r requirements.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## V14 重训（目标: ≥3单/天, WR≥68%）
```powershell
py -3 scripts/train_v14.py --symbol XAUUSD --gain 0.001 --horizon 12
py -3 scripts/deploy_v14_production.py
```

## RL 智能体
```powershell
py -3 scripts/prepare_rl_data.py --csv data/training/lgb/XAUUSD/XAUUSD_M5.csv --train-knowledge
py -3 scripts/train_rl_agent.py --symbol XAUUSD --timesteps 2000000
```

## 已有模型
- models/XAUUSD/ — V14 黄金 (400 trees, L=0.70 S=0.70)
- models/USOIL/v14/ — V14 原油 (400 trees, L=0.50 S=0.54)
- models/knowledge_net.pth, models/rl_agent_xau.zip

## 数据
- data/training/lgb/ — M5 CSV
- data/training/v14/ — 68维特征缓存
- data/rl_features_xau.npz — RL 训练包

配置: config_training.yaml, config/config_agent.json (use_rl=true)
"@
Set-Content -Path (Join-Path $stage 'MIGRATION_README.md') -Value $readme -Encoding UTF8

Write-Host "== Compressing -> $zipPath ==" -ForegroundColor Cyan
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path "$stage\*" -DestinationPath $zipPath -CompressionLevel Optimal

Remove-Item -Recurse -Force $stage
$mb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "OK: $zipPath ($mb MB)" -ForegroundColor Green
