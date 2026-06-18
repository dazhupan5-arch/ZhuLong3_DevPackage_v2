# 安装后：系统 Python + VC++ + Python 热更新（不打包内置 python_runtime）
param(
    [string]$InstallDir = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ((Split-Path -Leaf $InstallDir) -eq 'scripts') {
        $InstallDir = Split-Path -Parent $InstallDir
    }
}
$InstallDir = (Get-Item -LiteralPath $InstallDir).FullName

Write-Host "ZhuLong post-install: $InstallDir"

$legacyRt = Join-Path $InstallDir 'python_runtime'
if (Test-Path -LiteralPath $legacyRt) {
    Remove-Item -Recurse -Force $legacyRt
    Write-Host '  removed legacy python_runtime (system Python mode)' -ForegroundColor Yellow
}

$staleHostFiles = @(
    'coreclr.dll', 'hostfxr.dll', 'hostpolicy.dll', 'clrjit.dll',
    'mscordaccore.dll', 'mscordbi.dll', 'createdump.exe'
)
$rcPath = Join-Path $InstallDir 'ZhuLong.runtimeconfig.json'
$isFrameworkDependent = $false
if (Test-Path -LiteralPath $rcPath) {
    try {
        $rc = Get-Content -LiteralPath $rcPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $isFrameworkDependent = $null -ne $rc.runtimeOptions.frameworks -and -not $rc.runtimeOptions.includedFrameworks
    }
    catch { }
}
if ($isFrameworkDependent) {
    foreach ($f in $staleHostFiles) {
        $p = Join-Path $InstallDir $f
        if (Test-Path -LiteralPath $p) {
            Remove-Item -LiteralPath $p -Force
            Write-Host "  removed stale host file: $f" -ForegroundColor Yellow
        }
    }
}

$fixRc = Join-Path $InstallDir 'scripts\fix_runtimeconfig.ps1'
if (Test-Path -LiteralPath $fixRc) {
    & $fixRc -StageDir $InstallDir
}

$vc = Join-Path $InstallDir 'redist\VC_redist.x64.exe'
if (Test-Path -LiteralPath $vc) {
    Write-Host '  repairing VC++ 2015-2022 x64 …' -ForegroundColor Cyan
    $vcProc = Start-Process -FilePath $vc -ArgumentList '/repair', '/passive', '/norestart' -Wait -PassThru
    if ($vcProc.ExitCode -notin 0, 1638, 3010) {
        Write-Host "  VC++ repair exit=$($vcProc.ExitCode), trying install…" -ForegroundColor Yellow
        Start-Process -FilePath $vc -ArgumentList '/install', '/passive', '/norestart' -Wait | Out-Null
    }
}

$appData = Join-Path $env:APPDATA 'ZhuLong'
foreach ($stale in @('python_exe.txt', 'python_dll.txt', 'python_bundled.txt')) {
    $p = Join-Path $appData $stale
    if (Test-Path $p) { Remove-Item -Force $p }
}

$resolvePy = Join-Path $InstallDir 'scripts\resolve_system_python.ps1'
if (-not (Test-Path -LiteralPath $resolvePy)) {
    $resolvePy = Join-Path $InstallDir 'resolve_system_python.ps1'
}
& $resolvePy -InstallDir $InstallDir
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'System Python not found — install Python 3.10+ then run install_python_deps.ps1'
}
else {
    $deps = Join-Path $InstallDir 'install_python_deps.ps1'
    if (Test-Path -LiteralPath $deps) {
        Write-Host '  installing Python deps (may take a few minutes)…' -ForegroundColor Cyan
        & $deps -InstallDir $InstallDir
        if ($LASTEXITCODE -ne 0) {
            Write-Warning 'install_python_deps failed — run it manually after installing Python'
        }
    }
}

$macroCsv = Join-Path $InstallDir 'data\macro_events.csv'
$macroDst = Join-Path $appData 'data\macro_events.csv'
if (Test-Path $macroCsv) {
    New-Item -ItemType Directory -Force -Path (Split-Path $macroDst -Parent) | Out-Null
    Copy-Item -Force $macroCsv $macroDst
    Write-Host '  synced data/macro_events.csv' -ForegroundColor Green
}

$hotEngine = Join-Path $appData 'ZhuLong.PythonEngine'
New-Item -ItemType Directory -Force -Path $hotEngine | Out-Null
foreach ($rel in @(
        'ZhuLong.PythonEngine\inference_cli.py',
        'ZhuLong.PythonEngine\inference_worker.py',
        'ZhuLong.PythonEngine\hotfix_loader.py',
        'ZhuLong.PythonEngine\mt5_ops.py',
        'zhulong\utils\json_safe.py',
        'zhulong\utils\py_syntax_gate.py',
        'zhulong\engine\agent_engine.py',
        'zhulong\agent\trading_agent.py',
        'zhulong\agent\horizon_predictor.py',
        'zhulong\agent\knowledge_net_kn2.py',
        'zhulong\agent\kn2_location_labels.py',
        'zhulong\agent\knowledge_net.py',
        'zhulong\agent\tick_brief.py',
        'zhulong\agent\cognition.py',
        'zhulong\agent\trader_mind.py',
        'zhulong\agent\structure_service.py'
    )) {
    $src = Join-Path $InstallDir $rel
    if (-not (Test-Path $src)) { continue }
    $dst = Join-Path $appData $rel
    $dir = Split-Path $dst -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Copy-Item -Force $src $dst
    Write-Host "  hotfix OK $rel" -ForegroundColor Yellow
}

$installCfg = Join-Path $InstallDir 'config\config_agent.json'
$userCfg = Join-Path $appData 'config_agent.json'
$installMainCfg = Join-Path $InstallDir 'config.json'
$userMainCfg = Join-Path $appData 'config.json'

New-Item -ItemType Directory -Force -Path $appData | Out-Null

# 升级安装：AppData 配置必须被安装包全覆盖（避免 legacy config_agent 残留）
# 必须在 Python 校验之前执行 — 校验失败时仍要保证配置已覆盖
if (Test-Path -LiteralPath $installCfg) {
    Copy-Item -LiteralPath $installCfg -Destination $userCfg -Force
    Write-Host '  overwritten AppData config_agent.json from install' -ForegroundColor Green
}
else {
    Write-Warning 'install config_agent.json missing — skip AppData sync'
}

if (Test-Path -LiteralPath $installMainCfg) {
    Copy-Item -LiteralPath $installMainCfg -Destination $userMainCfg -Force
    Write-Host '  overwritten AppData config.json from install' -ForegroundColor Green
}

$ver = 'unknown'
if (Test-Path -LiteralPath $installMainCfg) {
    try {
        $mj = Get-Content -LiteralPath $installMainCfg -Raw -Encoding UTF8 | ConvertFrom-Json
        $ver = [string]$mj.app.version
    }
    catch { }
}
Set-Content -LiteralPath (Join-Path $appData 'install_version.txt') -Value $ver -Encoding ASCII -NoNewline
Write-Host "  install_version.txt = $ver" -ForegroundColor Green

Write-Host '  verifying Python hotfix syntax…' -ForegroundColor Cyan
& (Join-Path $InstallDir 'scripts\validate_python_agent_bundle.ps1') -Root $InstallDir
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'Python hotfix syntax check FAILED (config already synced; fix Python deps manually)'
}

Write-Host '  post-install OK (system Python + hotfix)' -ForegroundColor Green
exit 0
