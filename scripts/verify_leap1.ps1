# Leap 1 工程闭合一键验收（L1-1 ~ L1-5）
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "=== Leap 1 工程验收 ===" -ForegroundColor Cyan

Write-Host "[1/5] 演示模型四件套..."
py -3 scripts/create_demo_models.py
if ($LASTEXITCODE -ne 0) { throw "create_demo_models 失败" }

Write-Host "[2/5] dotnet build Release x64..."
dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64
if ($LASTEXITCODE -ne 0) { throw "dotnet build 失败" }

Write-Host "[3/5] dotnet test (含 L1-4/L1-5 集成测试)..."
dotnet test tests/ZhuLong.Core.Tests -c Release
if ($LASTEXITCODE -ne 0) { throw "dotnet test 失败" }

Write-Host "[4/5] Python validate_models..."
py -3 scripts/validate_models.py
if ($LASTEXITCODE -ne 0) { throw "validate_models 失败" }

Write-Host "[5/5] 模型文件清单..."
$required = @(
    "transformer_encoder.pth",
    "scaler.pkl",
    "xgb_classifier.json",
    "xgb_regressor.json",
    "manifest.json"
)
foreach ($sym in @("XAUUSD", "USOIL")) {
    foreach ($f in $required) {
        $p = Join-Path "models\$sym" $f
        if (-not (Test-Path $p)) { throw "缺少 $p" }
    }
}

Write-Host ""
Write-Host "Leap 1 工程验收通过 (L1-1~L1-4 自动化)." -ForegroundColor Green
Write-Host "L1-5 实机 smoke: 启动 ZhuLong.exe -> 开始运行 -> python scripts/smoke_pipe.py"
