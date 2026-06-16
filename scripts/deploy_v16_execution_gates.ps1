# 部署 V16 执行门控 + 结构位置过滤 + UI 日志字段（Python → AppData）
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$appData = Join-Path $env:APPDATA "ZhuLong"
New-Item -ItemType Directory -Force -Path $appData | Out-Null

$devRoot = Get-Location
Write-Host "=== Deploy V16 execution gates + structure location filter ===" -ForegroundColor Cyan

$pyFiles = @(
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\trader_mind.py",
    "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\knowledge_net.py"
)

foreach ($rel in $pyFiles) {
    $src = Join-Path $devRoot $rel
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    foreach ($base in @($appData, $InstallDir)) {
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) {
            try { New-Item -ItemType Directory -Force -Path $dir | Out-Null } catch { continue }
        }
        try {
            Copy-Item -Force $src $dst -ErrorAction Stop
        } catch {
            if ($base -eq $InstallDir) {
                Write-Warning "skip Program Files (need admin): $rel"
            } else {
                throw
            }
        }
    }
    Write-Host "OK $rel"
}

$cfgPath = Join-Path $appData "config_agent.json"
$cfgSrc = Join-Path $devRoot "config\config_agent.json"
if (Test-Path $cfgSrc) {
    $newCfg = Get-Content $cfgSrc -Raw -Encoding UTF8 | ConvertFrom-Json
    if (Test-Path $cfgPath) {
        $old = Get-Content $cfgPath -Raw -Encoding UTF8 | ConvertFrom-Json
        foreach ($prop in @("execution_gates", "kn2", "trader_mind", "rl_inference", "architecture")) {
            if ($newCfg.$prop) { $old | Add-Member -NotePropertyName $prop -NotePropertyValue $newCfg.$prop -Force }
        }
        $old | ConvertTo-Json -Depth 20 | Set-Content $cfgPath -Encoding UTF8
        Write-Host "Merged config_agent.json → AppData"
    } else {
        Copy-Item -Force $cfgSrc $cfgPath
        Write-Host "Copied config_agent.json → AppData"
    }
}

Write-Host @"

Python 热更新完成。UI 日志变更需重新编译安装 ZhuLong.App：
  dotnet build src\ZhuLong.App\ZhuLong.App.csproj -c Release

重启烛龙后验证：
  日志前缀 [V16·Horizon]
  震荡高位追多应被 structure_gate 拦截

"@ -ForegroundColor Green
