# 烛龙发布：WinUI 3 + Python.NET
Set-Location $PSScriptRoot\..

Write-Host "[1/5] Logo 与 App 图标..."
py -3 scripts/generate_app_icons.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/5] 演示模型..."
py -3 scripts/create_demo_models.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/5] dotnet publish WinUI..."
dotnet publish src/ZhuLong.App/ZhuLong.App.csproj `
  -c Release -r win-x64 -p:Platform=x64 `
  --self-contained -o publish/win-x64
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/5] 复制运行时资源..."
$pub = "publish/win-x64"
foreach ($item in @("config.json", "ZhuLong.PythonEngine", "models", "data", "zhulong", "mql5", "assets")) {
    $src = Join-Path (Get-Location) $item
    $dst = Join-Path $pub (Split-Path $item -Leaf)
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
    Copy-Item -Recurse -Force $src $dst
}
$pyDst = Join-Path $pub "python_runtime"
if (Test-Path $pyDst) { Remove-Item -Recurse -Force $pyDst }
Write-Host "  不打包 python_runtime（目标机需自行安装 Python 3 + pip install -r requirements.txt）" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$pub\indicators" | Out-Null
Copy-Item -Force mql5\ZhuLongIndicator.mq5 "$pub\indicators\"

Write-Host "[5/5] 完成。执行: iscc installer\build_installer.iss"
Write-Host "输出目录: $pub"
