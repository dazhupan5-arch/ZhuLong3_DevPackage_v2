# 导出烛龙3 审计系统包 → 桌面（DeepSeek / ChatGPT）
#Requires -Version 5.1
param(
    [string]$OutputRoot = '',
    [int]$LogTailLines = 3000,
    [switch]$IncludeModelBinaries,
    [ValidateSet('DeepSeek', 'ChatGPT')]
    [string]$Audience = 'DeepSeek'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$desktop = [Environment]::GetFolderPath('Desktop')
$outName = "ZhuLong3_${Audience}_Audit_$stamp"
$outDir = if ($OutputRoot) { Join-Path $OutputRoot $outName } else { Join-Path $desktop $outName }
$appData = Join-Path $env:APPDATA 'ZhuLong'

function Write-Utf8($path, $text) {
    $dir = Split-Path $path -Parent
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.File]::WriteAllText($path, $text, [System.Text.UTF8Encoding]::new($false))
}

function Copy-IfExists($src, $dst) {
    if (Test-Path $src) {
        $d = Split-Path $dst -Parent
        if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
        Copy-Item -Force $src $dst
        return $true
    }
    return $false
}

function Get-FileInventory($base, $patterns) {
    $items = @()
    foreach ($pat in $patterns) {
        Get-ChildItem -Path $base -Recurse -File -Filter $pat -ErrorAction SilentlyContinue | ForEach-Object {
            $items += [pscustomobject]@{
                Path = $_.FullName.Substring($base.Length).TrimStart('\')
                SizeBytes = $_.Length
                Modified = $_.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')
            }
        }
    }
    return $items
}

function Export-LogTail($src, $dst, $lines) {
    if (-not (Test-Path $src)) { return $false }
    $tail = Get-Content $src -Tail $lines -ErrorAction SilentlyContinue
    Write-Utf8 $dst ($tail -join "`n")
    return $true
}

function Redact-Secrets($dir) {
    Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.Name -match 'api_key|secret|password|token') {
            Write-Utf8 $_.FullName '[REDACTED - not included in audit package]'
        }
    }
}

Write-Host "== ZhuLong3 $Audience Audit Export ==" -ForegroundColor Cyan
Write-Host "Output: $outDir"

if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# --- 项目树（深度 4，排除大目录）---
$skip = @('python_runtime', '.git', 'node_modules', '__pycache__', '.pytest_cache', 'publish', 'test_install', 'legacy', 'output')
$treeLines = @("ZhuLong_3 Project Tree (depth<=4, excludes: $($skip -join ', '))", "")
function Walk-Tree($path, $prefix, $depth) {
    if ($depth -gt 4) { return }
    Get-ChildItem $path -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
        if ($skip -contains $_.Name) {
            $script:treeLines += "$prefix$($_.Name)/ [skipped]"
            return
        }
        if ($_.PSIsContainer) {
            $script:treeLines += "$prefix$($_.Name)/"
            Walk-Tree $_.FullName ($prefix + '  ') ($depth + 1)
        } else {
            $sz = if ($_.Length -gt 1MB) { '{0:N1}MB' -f ($_.Length/1MB) } else { '{0:N0}B' -f $_.Length }
            $script:treeLines += "$prefix$($_.Name) ($sz)"
        }
    }
}
Walk-Tree $root '' 0
Write-Utf8 (Join-Path $outDir '02_PROJECT_TREE.txt') ($treeLines -join "`n")

# --- Git 信息 ---
$gitInfo = @()
if (Test-Path (Join-Path $root '.git')) {
    Push-Location $root
    try {
        $branch = git rev-parse --abbrev-ref HEAD 2>$null
        if ($LASTEXITCODE -eq 0) { $gitInfo += "branch: $branch" } else { $gitInfo += 'branch: (no commits)' }
        $commit = git rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0) { $gitInfo += "commit: $commit" }
        $st = git status -sb 2>$null
        if ($st) { $gitInfo += 'status:'; $gitInfo += $st }
    } catch { $gitInfo += "git error: $_" } finally { Pop-Location }
} else { $gitInfo += 'not a git repo' }
Write-Utf8 (Join-Path $outDir '03_GIT_STATUS.txt') ($gitInfo -join "`n")

# --- 系统清单 JSON ---
$pyVer = try { (& py -3 -c 'import sys; print(sys.version)') 2>$null } catch { 'unknown' }
$manifest = [ordered]@{
    package_name = "ZhuLong3_${Audience}_Audit"
    audience = $Audience
    generated_at = (Get-Date).ToUniversalTime().ToString('o')
    project_root = $root
    python_version = $pyVer
    symbols = @('XAUUSD', 'USOIL')
    pipelines = @{
        xau = 'v12 (v11 triple XGB + asymmetric rules)'
        oil = 'v1 (triple XGB + EIA filter)'
    }
    key_paths = @{
        configs = @('config/config_xau_v12.json', 'config/config_oil_v1.json', 'config_v12.json', 'config.json')
        inference = @('scripts/realtime_signal.py', 'zhulong/inference/v12.py', 'zhulong/inference/oil_v1.py')
        training = @('zhulong/training/v11', 'zhulong/training/v12', 'zhulong/training/oil_v1')
        mql5 = @('mql5/ZhuLongIndicator.mq5', 'indicators/ZhuLongIndicator.mq5')
    }
    appdata = $appData
}
$manifest.models = @{}
foreach ($sym in @('XAUUSD', 'USOIL')) {
    $mDir = Join-Path $root "models\$sym"
    if (Test-Path $mDir) {
        $files = Get-ChildItem $mDir -Recurse -File | ForEach-Object {
            @{ path = $_.FullName.Substring($root.Length).TrimStart('\'); size = $_.Length; modified = $_.LastWriteTime.ToString('o') }
        }
        $manifest.models[$sym] = $files
    }
}
Write-Utf8 (Join-Path $outDir '01_SYSTEM_MANIFEST.json') ($manifest | ConvertTo-Json -Depth 8)

# --- 配置 ---
$configDst = Join-Path $outDir 'config'
New-Item -ItemType Directory -Force -Path $configDst | Out-Null
@(
    'config.json', 'config_v12.json',
    'config\config_xau_v12.json', 'config\config_oil_v1.json', 'config\config.schema.json'
) | ForEach-Object { Copy-IfExists (Join-Path $root $_) (Join-Path $outDir $_) | Out-Null }
if (Test-Path (Join-Path $appData 'config.json')) {
    Copy-Item -Force (Join-Path $appData 'config.json') (Join-Path $configDst 'config_user_appdata.json')
}

# --- 文档 ---
$docsDst = Join-Path $outDir 'docs'
New-Item -ItemType Directory -Force -Path $docsDst | Out-Null
Copy-Item -Force (Join-Path $root 'docs\*') $docsDst -Recurse -ErrorAction SilentlyContinue

# --- 模型元数据（manifest / feature_columns / config，可选二进制）---
$modelsDst = Join-Path $outDir 'models'
foreach ($sym in @('XAUUSD', 'USOIL')) {
    $src = Join-Path $root "models\$sym"
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path $modelsDst $sym
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem $src -Recurse -File | Where-Object {
        $_.Extension -match '\.(json|pkl|md|yaml|txt)$' -or $_.Name -eq 'manifest.json'
    } | ForEach-Object {
        $rel = $_.FullName.Substring($src.Length).TrimStart('\')
        $target = Join-Path $dst $rel
        $td = Split-Path $target -Parent
        if (-not (Test-Path $td)) { New-Item -ItemType Directory -Force -Path $td | Out-Null }
        Copy-Item -Force $_.FullName $target
    }
    if ($IncludeModelBinaries) {
        Get-ChildItem $src -Recurse -File | Where-Object { $_.Extension -in '.json' -and $_.Length -gt 100000 } | ForEach-Object {
            Copy-Item -Force $_.FullName (Join-Path $dst $_.Name)
        }
        Get-ChildItem $src -Recurse -Include 'xgb_triple*.json','imf_vmd.parquet' -File -ErrorAction SilentlyContinue | ForEach-Object {
            $rel = $_.FullName.Substring($src.Length).TrimStart('\')
            $target = Join-Path $dst $rel
            $td = Split-Path $target -Parent
            if (-not (Test-Path $td)) { New-Item -ItemType Directory -Force -Path $td | Out-Null }
            Copy-Item -Force $_.FullName $target
        }
    }
}

# --- 训练报告 ---
$repDst = Join-Path $outDir 'training_reports'
@(
    'data\training\reports\v11\XAUUSD',
    'data\training\reports\v12\XAUUSD',
    'data\training\reports\oil_v1\USOIL'
) | ForEach-Object {
    $src = Join-Path $root $_
    if (Test-Path $src) {
        $name = ($_ -replace '[\\/]', '_')
        Copy-Item -Recurse -Force $src (Join-Path $repDst $name)
    }
}

# --- 运行数据（AppData，脱敏）---
$rtDst = Join-Path $outDir 'runtime'
New-Item -ItemType Directory -Force -Path $rtDst | Out-Null

# 宏观/情绪（无密钥）
if (Test-Path (Join-Path $appData 'data')) {
    $macroDst = Join-Path $rtDst 'appdata_data'
    New-Item -ItemType Directory -Force -Path $macroDst | Out-Null
    Copy-Item -Force (Join-Path $appData 'data\*') $macroDst -Recurse -ErrorAction SilentlyContinue
}

# 实时状态
@(
    'data\v12_realtime_state.json',
    'data\realtime_state_xau.json',
    'data\realtime_state_oil.json',
    'data\train_state.json',
    'logs\trading.log'
) | ForEach-Object {
    Copy-IfExists (Join-Path $root $_) (Join-Path $rtDst ("project_" + ($_ -replace '[\\/]', '_'))) | Out-Null
}

# 日志尾部
$logDst = Join-Path $rtDst 'logs'
New-Item -ItemType Directory -Force -Path $logDst | Out-Null
Export-LogTail (Join-Path $root 'logs\trading.log') (Join-Path $logDst 'project_trading_log_tail.txt') $LogTailLines | Out-Null
if (Test-Path (Join-Path $appData 'logs')) {
    Get-ChildItem (Join-Path $appData 'logs') -File | ForEach-Object {
        $dstName = "appdata_$($_.BaseName)_tail.txt"
        Export-LogTail $_.FullName (Join-Path $logDst $dstName) $LogTailLines | Out-Null
        # 记录完整大小
        Add-Content (Join-Path $logDst '_log_inventory.txt') "$($_.Name) full_size=$($_.Length) exported_tail=$LogTailLines"
    }
}

# SQLite 摘要（不含敏感）
$dbPath = Join-Path $appData 'trading.db'
if (Test-Path $dbPath) {
    $dbPy = Join-Path $outDir '_export_db_summary.py'
    @"
import json, sqlite3, pathlib, sys
db = pathlib.Path(sys.argv[1])
conn = sqlite3.connect(db)
cur = conn.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
out = {'db_path': str(db), 'db_size_bytes': db.stat().st_size, 'tables': {}}
for t in tables:
    out['tables'][t] = {'count': cur.execute('SELECT COUNT(*) FROM [' + t + ']').fetchone()[0]}
    cols = [r[1] for r in cur.execute('PRAGMA table_info([' + t + '])')]
    out['tables'][t]['columns'] = cols
    try:
        rows = cur.execute('SELECT * FROM [' + t + '] ORDER BY rowid DESC LIMIT 20').fetchall()
        out['tables'][t]['sample_last_20'] = [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        out['tables'][t]['sample_error'] = str(e)
conn.close()
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
"@ | Set-Content -Path $dbPy -Encoding UTF8
    $dbJson = & py -3 $dbPy $dbPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Utf8 (Join-Path $rtDst 'trading_db_summary.json') ($dbJson -join "`n")
    }
    Remove-Item -Force $dbPy -ErrorAction SilentlyContinue
}

# MT5 部署状态
$mt5Lines = @('MT5 Terminal Deployment Check', '')
$mt5Roots = Get-ChildItem (Join-Path $env:APPDATA 'MetaQuotes\Terminal') -Directory -EA SilentlyContinue |
    Where-Object { $_.Name -match '^[0-9A-F]{32}$' }
foreach ($t in $mt5Roots) {
    $dll = Join-Path $t.FullName 'MQL5\Libraries\ZhuLongMt5Pipe.dll'
    $mq5 = Join-Path $t.FullName 'MQL5\Indicators\ZhuLongIndicator.mq5'
    $ex5 = Join-Path $t.FullName 'MQL5\Indicators\ZhuLongIndicator.ex5'
    $mt5Lines += "terminal: $($t.Name)"
    $mt5Lines += "  DLL: $(Test-Path $dll) $(if(Test-Path $dll){(Get-Item $dll).Length}else{'missing'})"
    $mt5Lines += "  MQ5: $(Test-Path $mq5)"
    $mt5Lines += "  EX5 compiled: $(Test-Path $ex5) $(if(Test-Path $ex5){(Get-Item $ex5).LastWriteTime}else{'not compiled'})"
}
Write-Utf8 (Join-Path $rtDst 'mt5_deployment_status.txt') ($mt5Lines -join "`n")

# --- 源码索引（关键模块清单）---
$keyPatterns = @('*.py', '*.ps1', '*.cs', '*.mq5', '*.json', '*.md')
$inv = Get-FileInventory $root $keyPatterns | Where-Object {
    $_.Path -notmatch 'python_runtime|test_install|publish|legacy|__pycache__|\.pytest_cache'
}
$inv | Export-Csv (Join-Path $outDir '04_SOURCE_INVENTORY.csv') -NoTypeInformation -Encoding UTF8

# --- 依赖 ---
Copy-IfExists (Join-Path $root 'requirements.txt') (Join-Path $outDir 'requirements.txt') | Out-Null
Copy-IfExists (Join-Path $root 'requirements-inference.txt') (Join-Path $outDir 'requirements-inference.txt') | Out-Null

# --- 关键源码快照（推理/训练核心，不含大二进制）---
$srcSnap = Join-Path $outDir 'source_snapshot'
$keyFiles = @(
    'scripts/realtime_signal.py', 'scripts/mt5_bridge.py', 'scripts/deploy_dual_production.py',
    'scripts/start_trading_dual.ps1', 'scripts/export_deepseek_audit.ps1',
    'zhulong/inference/v12.py', 'zhulong/inference/oil_v1.py',
    'zhulong/live_v8_features.py', 'zhulong/live_oil_features.py',
    'zhulong/training/v11/train.py', 'zhulong/training/v12/backtest.py',
    'zhulong/training/oil_v1/train.py', 'zhulong/training/oil_v1/backtest.py',
    'zhulong/training/oil_v1/features.py', 'zhulong/training/lgb/acceptance.py',
    'mql5/ZhuLongIndicator.mq5', 'indicators/ZhuLongIndicator.mq5'
)
foreach ($f in $keyFiles) {
    $s = Join-Path $root $f
    if (Test-Path $s) {
        $d = Join-Path $srcSnap $f
        $dd = Split-Path $d -Parent
        if (-not (Test-Path $dd)) { New-Item -ItemType Directory -Force -Path $dd | Out-Null }
        Copy-Item -Force $s $d
    }
}

# --- 审计 README ---
$auditFocus = if ($Audience -eq 'ChatGPT') {
@"

## ChatGPT 全量审计提示词（推荐）

``````
你正在对烛龙3（ZhuLong_3）双品种量化交易系统做全量架构与运行时审计。导出包已按目录编号组织。

请按以下步骤输出结构化报告：

### A. 系统概览
- 阅读 00_AUDIT_README.md、01_SYSTEM_MANIFEST.json、docs/ 下架构文档
- 归纳 XAUUSD v12 与 USOIL v1 双管线目标、实时推理与 MT5 集成方式

### B. 运行时状态
- runtime/ — AppData 宏观数据、日志尾部、trading.db 摘要、MT5 部署状态
- 对照 training_reports/ 验收指标与 runtime/logs/ 实盘日志

### C. 代码与特征
- source_snapshot/ 与 04_SOURCE_INVENTORY.csv 中的推理/训练核心
- 特征对齐：live_v8_features vs v12 训练；live_oil_features vs oil_v1 训练

### D. 模型与配置
- config/ 与 models/ 中 manifest、阈值、broker_symbol 映射
- 标签泄漏与样本外泛化风险

### E. 风险与债务
- 双品种配置漂移、MT5 EX5 编译状态、密钥与部署缺口

### F. 优先级建议
- P0 风控与特征/标签一致性
- P1 实盘与回测指标对齐
- P2 文档与配置可维护性
``````

## 审计关注点

1. 特征对齐：训练 vs 实盘 live_v8_features / live_oil_features
2. 标签泄漏：v11/oil_v1 标签生成是否仅用历史窗口
3. 样本外验收：training_reports 中 test1 指标 vs 实盘日志
4. 符号映射：config_oil_v1.json broker_symbol 与 MT5
5. 管道/指标：mt5_deployment_status.txt 中 EX5 是否已编译
6. 密钥：本包已排除 secrets/ 下 API Key
"@
} else {
@"

## 审计关注点（建议 DeepSeek 检查）

1. 特征对齐：训练 vs 实盘 live_v8_features / live_oil_features
2. 标签泄漏：v11/oil_v1 标签生成是否仅用历史窗口
3. 样本外验收：training_reports 中 test1 指标 vs 实盘日志
4. 符号映射：config_oil_v1.json broker_symbol 与 MT5
5. 管道/指标：mt5_deployment_status.txt 中 EX5 是否已编译
6. 密钥：本包已排除 secrets/ 下 API Key
"@
}

$readme = @"
# 烛龙 ZhuLong_3 — $Audience 审计系统包

生成时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
项目路径: $root

## 包结构

| 目录/文件 | 内容 |
|-----------|------|
| 00_AUDIT_README.md | 本说明 |
| 01_SYSTEM_MANIFEST.json | 系统清单、模型文件列表 |
| 02_PROJECT_TREE.txt | 项目目录树 |
| 03_GIT_STATUS.txt | Git 分支/提交状态 |
| 04_SOURCE_INVENTORY.csv | 源码文件清单 |
| config/ | 部署与训练配置 |
| docs/ | 全部项目文档 |
| models/ | 模型 manifest、feature_columns、config（$(if($IncludeModelBinaries){'含'}else{'不含'})大型 xgb/parquet 二进制） |
| training_reports/ | XAU v11/v12、USOIL v1 验收报告 |
| runtime/ | AppData 宏观数据、日志尾部、DB 摘要、MT5 部署状态 |
| source_snapshot/ | 推理/训练/部署核心源码 |
| requirements*.txt | Python 依赖 |

## 双品种架构摘要

- **XAUUSD v12**: v11 三分类 XGBoost + 不对称阈值(0.84/0.88)、H1 做空趋势过滤
- **USOIL v1**: 121 维特征、动态 ATR 标签、EIA 屏蔽、极端趋势过滤
- **实时服务**: scripts/realtime_signal.py（双品种轮询 MT5 M5）
- **MT5 桥接**: 命名管道 ZhuLong_Data / ZhuLong_Drawing + ZhuLongIndicator.mq5
$auditFocus

## 运行数据说明

- 完整 AppData 日志仅导出最后 $LogTailLines 行（见 runtime/logs/）
- trading.db 仅导出表结构与最近 20 条样本
- 如需完整模型权重请加 -IncludeModelBinaries 重新导出

## 重新导出

``````powershell
cd $root
.\scripts\export_deepseek_audit.ps1 -Audience $Audience
.\scripts\export_deepseek_audit.ps1 -Audience $Audience -IncludeModelBinaries
``````
"@
Write-Utf8 (Join-Path $outDir '00_AUDIT_README.md') $readme

# --- 打 ZIP ---
$zipPath = "$outDir.zip"
if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path $outDir -DestinationPath $zipPath -CompressionLevel Optimal

$zipSize = (Get-Item $zipPath).Length
Write-Host "OK: $zipPath ($([math]::Round($zipSize/1MB, 2)) MB)" -ForegroundColor Green
Write-Host "Folder: $outDir" -ForegroundColor Green
