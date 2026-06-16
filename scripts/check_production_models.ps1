# 检查正式模型是否已部署（非 demo、acceptance_passed=true）
#Requires -Version 5.1
$root = Split-Path $PSScriptRoot -Parent
$modelsRoot = Join-Path $root 'models'
$configPath = Join-Path $env:APPDATA 'ZhuLong\config.json'
if (-not (Test-Path $configPath)) { $configPath = Join-Path $root 'config.json' }

$symbols = @('XAUUSD', 'USOIL')
if (Test-Path $configPath) {
    try {
        $cfg = Get-Content $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($cfg.model.default_symbols) { $symbols = @($cfg.model.default_symbols) }
    } catch { }
}

$ready = 0
$pending = @()
foreach ($sym in $symbols) {
    $manifest = Join-Path $modelsRoot "$sym\manifest.json"
    if (-not (Test-Path $manifest)) {
        $pending += "$sym missing manifest"
        continue
    }
    $m = Get-Content $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($m.kind -eq 'demo') { $pending += "$sym demo only"; continue }
    if (-not $m.acceptance_passed) { $pending += "$sym not accepted"; continue }
    $ready++
}

$result = @{ ok = ($ready -eq $symbols.Count); ready = $ready; total = $symbols.Count; pending = $pending }
$result | ConvertTo-Json -Compress
if (-not $result.ok) { exit 1 }
