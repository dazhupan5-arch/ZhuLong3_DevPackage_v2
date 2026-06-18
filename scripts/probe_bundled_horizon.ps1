# Bundled Python Horizon ONNX session probe (same path as app startup validation)
#Requires -Version 5.1
param(
    [string]$InstallDir = '',
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    if ((Split-Path -Leaf $InstallDir) -eq 'scripts') {
        $InstallDir = Split-Path -Parent $InstallDir
    }
}
$InstallDir = (Get-Item -LiteralPath $InstallDir).FullName

$resolve = Join-Path $InstallDir 'scripts\resolve_bundled_python.ps1'
if (-not (Test-Path -LiteralPath $resolve)) {
    $resolve = Join-Path $InstallDir 'resolve_bundled_python.ps1'
}
& $resolve -InstallDir $InstallDir -Quiet
if ($LASTEXITCODE -ne 0) { exit 1 }

$py = Join-Path $InstallDir 'python_runtime\python.exe'
$cfg = Join-Path $InstallDir 'config\config_agent.json'
if (-not (Test-Path -LiteralPath $cfg)) {
    $cfg = Join-Path $env:APPDATA 'ZhuLong\config_agent.json'
}

$installPy = ($InstallDir -replace '\\', '/')
$cfgPy = ($cfg -replace '\\', '/')
$probeFile = Join-Path $env:TEMP 'zhulong_bundled_horizon_probe.py'
@(
    'import json, os, sys',
    'from pathlib import Path',
    "install = Path('$installPy')",
    "os.environ['ZHULONG_INSTALL_DIR'] = str(install)",
    'sys.path.insert(0, str(install))',
    'sys.path.insert(0, str(install / "ZhuLong.PythonEngine"))',
    'from zhulong.utils.win_dll import configure_native_dll_paths',
    'configure_native_dll_paths()',
    'import onnxruntime  # preload before numpy',
    'from hotfix_loader import apply_appdata_hotfixes',
    'apply_appdata_hotfixes()',
    "cfg = Path('$cfgPy')",
    'if not cfg.is_file():',
    "    print('HORIZON_PROBE_FAIL config_not_found', cfg)",
    '    sys.exit(1)',
    "config = json.loads(cfg.read_text(encoding='utf-8-sig'))",
    'from inference_cli import _cmd_agent_validate',
    "out = _cmd_agent_validate({'cmd':'agent_validate','root':str(install),'config_path':str(cfg),'quick':True})",
    "if not out.get('ok'):",
    "    print('HORIZON_PROBE_FAIL', out.get('error','unknown'))",
    '    sys.exit(1)',
    "print('HORIZON_PROBE_OK')"
) | Set-Content -LiteralPath $probeFile -Encoding UTF8

Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
$prevEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    $out = & $py $probeFile 2>&1 | ForEach-Object { if ($_ -is [System.Management.Automation.ErrorRecord]) { $_.ToString() } else { "$_" } }
    $exitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $prevEap
}
$text = ($out -join [Environment]::NewLine).Trim()
if ($exitCode -ne 0 -or $text -notmatch 'HORIZON_PROBE_OK') {
    if (-not $Quiet) {
        Write-Host '[FAIL] Bundled Horizon probe failed:' -ForegroundColor Red
        Write-Host $text
    }
    exit 1
}

if (-not $Quiet) { Write-Host '[OK] Bundled Horizon probe OK' -ForegroundColor Green }
exit 0
