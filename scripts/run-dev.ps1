# 本地运行（不打包）：clean + build + 启动（使用本机 Python 3）
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

$bin = Join-Path $root 'src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64'

Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host '== resolve system Python ==' -ForegroundColor Cyan
& (Join-Path $root 'scripts\resolve_system_python.ps1') -Quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host '== clean + build Release x64 ==' -ForegroundColor Cyan
dotnet clean src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 | Out-Null
dotnet build src/ZhuLong.App/ZhuLong.App.csproj -c Release -p:Platform=x64 `
    -p:PublishTrimmed=false -p:PublishReadyToRun=false
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($item in @('config.json', 'models', 'data', 'zhulong', 'ZhuLong.PythonEngine')) {
    $src = Join-Path $root $item
    $dst = Join-Path $bin $item
    if (-not (Test-Path $src)) { continue }
    if (Test-Path $dst) { continue }
    Copy-Item -Recurse -Force $src $dst
}

$pipeDll = Join-Path $root 'mql5\Libraries\ZhuLongMt5Pipe.dll'
Write-Host '== build ZhuLongMt5Pipe.dll ==' -ForegroundColor Cyan
& (Join-Path $root 'scripts\build-zhulong-mt5-pipe.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (Test-Path $pipeDll) {
    $libDst = Join-Path $bin 'mql5\Libraries'
    New-Item -ItemType Directory -Force -Path $libDst | Out-Null
    Copy-Item -Force $pipeDll (Join-Path $libDst 'ZhuLongMt5Pipe.dll')
}

Write-Host '== deploy MT5 indicator + pipe DLL ==' -ForegroundColor Cyan
& (Join-Path $root 'scripts\deploy-mt5-indicator.ps1')
if ($LASTEXITCODE -ne 0) { Write-Host 'WARN: MT5 deploy skipped or failed' -ForegroundColor Yellow }

$mt5Ok = $false
try {
    & py -3 -c "import MetaTrader5" 2>$null
    if ($LASTEXITCODE -eq 0) { $mt5Ok = $true }
} catch { }
if (-not $mt5Ok) {
    Write-Host 'WARN: MetaTrader5 未安装，运行 .\scripts\install_python_deps.ps1' -ForegroundColor Yellow
}

$exe = Join-Path $bin 'ZhuLong.exe'
Write-Host "启动: $exe" -ForegroundColor Green
Write-Host "PYTHONNET_PYDLL=$env:PYTHONNET_PYDLL" -ForegroundColor Gray
$p = Start-Process -FilePath $exe -WorkingDirectory $bin -PassThru
Start-Sleep -Seconds 6
$proc = Get-Process -Id $p.Id -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Write-Host '启动失败：进程已退出。' -ForegroundColor Red
    $log = Join-Path $env:LOCALAPPDATA 'ZhuLong\startup.log'
    if (Test-Path $log) {
        Write-Host "详见 $log" -ForegroundColor Yellow
        Get-Content $log -Tail 8
    }
    exit 1
}
if ($proc.MainWindowHandle -eq 0) {
    Write-Host '进程在运行但窗口尚未就绪，请稍候或检查任务栏。' -ForegroundColor Yellow
} else {
    Write-Host "Started PID $($proc.Id) title=$($proc.MainWindowTitle)" -ForegroundColor Green
}
