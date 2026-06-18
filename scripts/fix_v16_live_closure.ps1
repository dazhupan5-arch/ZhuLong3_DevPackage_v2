# 实机 V16+KN2 闭合修复：同步 Python/C# 补丁 + 校验
param(
    [string]$InstallDir = "C:\Program Files\ZhuLong",
    [string]$DevRoot = "D:\trae_projects\ZhuLong3_DevPackage_v2"
)

$ErrorActionPreference = "Stop"
$appData = Join-Path $env:APPDATA "ZhuLong"

Write-Host "=== V16 Live Closure Fix ===" -ForegroundColor Cyan

$pyFiles = @(
    "ZhuLong.PythonEngine\inference_cli.py",
    "zhulong\agent\trading_agent.py",
    "zhulong\agent\horizon_predictor.py",
    "zhulong\agent\knowledge_net_kn2.py",
    "zhulong\agent\kn2_location_labels.py",
    "zhulong\agent\knowledge_net.py",
    "zhulong\agent\tick_brief.py",
    "zhulong\agent\cognition.py",
    "zhulong\agent\trader_mind.py",
    "zhulong\agent\structure_service.py"
)

foreach ($rel in $pyFiles) {
    $src = Join-Path $DevRoot $rel
    if (-not (Test-Path $src)) { Write-Warning "skip missing $rel"; continue }
    foreach ($base in @($InstallDir, $appData)) {
        if (-not (Test-Path $base)) { continue }
        $dst = Join-Path $base $rel
        $dir = Split-Path $dst -Parent
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        try {
            Copy-Item -Force $src $dst
            Write-Host "OK $rel -> $base" -ForegroundColor Green
        } catch {
            Write-Warning "skip $base $rel (admin?)"
        }
    }
}

& (Join-Path $DevRoot "scripts\deploy_kn2_v16_when_ready.ps1") -EnableLive -SkipAcceptCheck -InstallDir $InstallDir

# AppData 热更新目录须含 mt5_ops（旧版 exe 曾将 PythonEngineDir 指向此处）
$engineHot = Join-Path $appData "ZhuLong.PythonEngine"
New-Item -ItemType Directory -Force -Path $engineHot | Out-Null
$mt5Ops = Join-Path $InstallDir "ZhuLong.PythonEngine\mt5_ops.py"
if (Test-Path $mt5Ops) {
    Copy-Item -Force $mt5Ops (Join-Path $engineHot "mt5_ops.py") -ErrorAction SilentlyContinue
}

$adminScript = Join-Path $DevRoot "scripts\fix_v16_live_closure_admin.ps1"
$binExe = Join-Path $DevRoot "src\ZhuLong.App\bin\x64\Release\net8.0-windows10.0.19041.0\win-x64\ZhuLong.exe"
if (Test-Path $binExe) {
    try {
        Copy-Item -Force $binExe (Join-Path $InstallDir "ZhuLong.exe")
        Write-Host "OK ZhuLong.exe" -ForegroundColor Green
    } catch {
        Write-Warning "需要管理员权限部署 ZhuLong.exe，正在请求 UAC…"
        if (Test-Path $adminScript) {
            Start-Process powershell -Verb RunAs -Wait -ArgumentList @(
                "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $adminScript,
                "-InstallDir", "`"$InstallDir`"", "-DevRoot", "`"$DevRoot`""
            )
        } else {
            Write-Warning "Run as admin: $adminScript"
        }
    }
} else {
    Write-Warning "Build first: dotnet build src\ZhuLong.App\ZhuLong.App.csproj -c Release -p:Platform=x64"
}

Get-Process ZhuLong -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*Python*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Start-Process (Join-Path $InstallDir "ZhuLong.exe") -WorkingDirectory $InstallDir

Set-Location $DevRoot
py -3 scripts\kn2_v16_live_boot.py
exit $LASTEXITCODE
