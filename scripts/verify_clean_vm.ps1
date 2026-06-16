# 干净机 / 新环境验收（不跑模型推理）
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$fail = 0

function Check($name, [scriptblock]$fn) {
    try {
        & $fn
        Write-Host "[PASS] $name" -ForegroundColor Green
    } catch {
        Write-Host "[FAIL] $name — $($_.Exception.Message)" -ForegroundColor Red
        $script:fail++
    }
}

Check "Python 3.10+" {
    $v = & py -3 -c "import sys; print(sys.version_info[:2])" 2>$null
    if (-not $v) { throw "py -3 不可用" }
    $parts = $v.Trim('()').Split(',')
    if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 10)) {
        throw "需要 Python 3.10+，当前 $v"
    }
}

Check "MetaTrader5 包" {
    & py -3 -c "import MetaTrader5" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "pip install MetaTrader5" }
}

Check "config.json" {
    $cfg = Join-Path $root "config.json"
    if (-not (Test-Path $cfg)) { throw "缺少 config.json" }
}

Check "config.schema.json" {
    $schema = Join-Path $root "config\config.schema.json"
    if (-not (Test-Path $schema)) { throw "缺少 config.schema.json" }
}

Check "feature_schema.json" {
    $fs = Join-Path $root "zhulong\feature_schema.json"
    if (-not (Test-Path $fs)) { throw "缺少 feature_schema.json" }
}

Check "dotnet build" {
    Push-Location $root
    dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 -v q
    if ($LASTEXITCODE -ne 0) { throw "编译失败" }
    Pop-Location
}

Check "dotnet test" {
    Push-Location $root
    dotnet test tests/ZhuLong.Core.Tests/ZhuLong.Core.Tests.csproj -c Release -v q
    if ($LASTEXITCODE -ne 0) { throw "单元测试失败" }
    Pop-Location
}

Check "pytest" {
    Push-Location $root
    & py -3 -m pytest tests/test_config_validator.py tests/test_feature_golden.py tests/test_labels.py -q
    if ($LASTEXITCODE -ne 0) { throw "pytest 失败" }
    Pop-Location
}

Check "ZhuLongIndicator.mq5" {
    $mq5 = Join-Path $root "mql5\ZhuLongIndicator.mq5"
    if (-not (Test-Path $mq5)) { throw "缺少指标源文件" }
}

Write-Host ""
if ($fail -eq 0) {
    Write-Host "干净机验收: $($fail) 失败 — 全部通过" -ForegroundColor Cyan
    exit 0
}
Write-Host "干净机验收: $fail 项失败" -ForegroundColor Red
exit 1
