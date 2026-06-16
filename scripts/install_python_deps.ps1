# Install ZhuLong runtime deps on system Python 3.10+ (resolved python.exe, no py -3 PATH)
param(
    [string]$PythonExe = '',
    [string]$Root = ''
)

$ErrorActionPreference = 'Stop'

function Get-ZhuLongRoot {
    param([string]$ScriptDir)
    if ($Root -and (Test-Path (Join-Path $Root 'requirements_runtime.txt'))) { return $Root }
    if (Test-Path (Join-Path $ScriptDir 'requirements_runtime.txt')) { return $ScriptDir }
    if (Test-Path (Join-Path $ScriptDir 'requirements.txt')) { return $ScriptDir }
    $parent = Split-Path $ScriptDir -Parent
    if (Test-Path (Join-Path $parent 'requirements_runtime.txt')) { return $parent }
    if (Test-Path (Join-Path $parent 'requirements.txt')) { return $parent }
    throw "Cannot find ZhuLong root from $ScriptDir"
}

function Resolve-PythonExe {
    param([string]$Hint)
    if ($Hint -and (Test-Path $Hint)) { return (Resolve-Path $Hint).Path }

    $cache = Join-Path $env:APPDATA 'ZhuLong\python_exe.txt'
    if (Test-Path $cache) {
        $cached = (Get-Content $cache -Raw).Trim()
        if ($cached -and (Test-Path $cached)) { return $cached }
    }

    $resolve = Join-Path (Get-ZhuLongRoot -ScriptDir $PSScriptRoot) 'resolve_system_python.ps1'
    if (-not (Test-Path $resolve)) {
        $resolve = Join-Path (Get-ZhuLongRoot -ScriptDir $PSScriptRoot) 'scripts\resolve_system_python.ps1'
    }
    if (Test-Path $resolve) {
        & $resolve -Quiet
        if ($LASTEXITCODE -ne 0) { throw 'resolve_system_python failed' }
        if (Test-Path $cache) {
            $cached = (Get-Content $cache -Raw).Trim()
            if ($cached -and (Test-Path $cached)) { return $cached }
        }
    }

    throw 'Python 3 not found. Install Python 3.10+ with Add to PATH.'
}

$root = Get-ZhuLongRoot -ScriptDir $PSScriptRoot
Set-Location $root

$python = Resolve-PythonExe -Hint $PythonExe
Write-Host "Python: $python" -ForegroundColor Green

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $python @Args
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Pip {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Invoke-Python -m pip @Args
}

Write-Host '== pip upgrade ==' -ForegroundColor Cyan
Invoke-Pip install --upgrade pip wheel setuptools

$runtimeReq = Join-Path $root 'requirements_runtime.txt'
if (Test-Path $runtimeReq) {
    Write-Host '== pip install requirements_runtime.txt (prefer binary) ==' -ForegroundColor Cyan
    Invoke-Pip install --prefer-binary -r $runtimeReq
} else {
    Write-Host '== pip install fallback packages ==' -ForegroundColor Cyan
    Invoke-Pip install --prefer-binary torch xgboost "pandas==2.2.3" "pyarrow==17.0.0" "numpy>=1.26,<2" scikit-learn joblib MetaTrader5 fredapi requests
}

Write-Host '== pip ensure MT5/macro ==' -ForegroundColor Cyan
Invoke-Pip install --prefer-binary --upgrade MetaTrader5 fredapi requests

Write-Host '== import smoke ==' -ForegroundColor Cyan
$smoke = @'
import importlib, sys
mods = ["MetaTrader5", "fredapi", "onnxruntime", "xgboost", "pandas", "sklearn", "joblib", "requests"]
optional = ["pyarrow", "torch", "gymnasium", "stable_baselines3"]
failed = []
for m in mods:
    try:
        importlib.import_module(m)
        print("  OK", m)
    except Exception as ex:
        failed.append(f"{m}: {ex}")
        print("  FAIL", m, ex)
for m in optional:
    try:
        importlib.import_module(m)
        print("  OK", m, "(optional)")
    except Exception as ex:
        print("  SKIP", m, ex)
if failed:
    print("SMOKE_FAIL")
    sys.exit(1)
print("SMOKE_OK")
'@
Invoke-Python -c $smoke

Write-Host '== IMF cache read smoke ==' -ForegroundColor Cyan
$rootEsc = $root -replace '\\', '\\\\'
$imfSmoke = @"
import sys
from pathlib import Path
root = Path(r'$rootEsc')
sys.path.insert(0, str(root))
import pandas as pd
csv = root / 'models' / 'XAUUSD' / 'imf_vmd.csv'
pq = root / 'models' / 'XAUUSD' / 'imf_vmd.parquet'
ok = False
if csv.is_file():
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    ok = len(df) > 0
    print('  OK CSV', len(df))
if not ok and pq.is_file():
    from zhulong.utils.parquet_io import read_parquet_safe
    df = read_parquet_safe(pq)
    ok = df is not None and len(df) > 0
    print('  OK Parquet', 0 if df is None else len(df))
if not ok:
    print('IMF_FAIL')
    sys.exit(1)
print('IMF_OK')
"@
Invoke-Python -c $imfSmoke

Write-Host '== agent validate smoke ==' -ForegroundColor Cyan
$agentSmoke = @"
import json, sys
from pathlib import Path
root = Path(r'$rootEsc')
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'ZhuLong.PythonEngine'))
try:
    from zhulong.utils.win_dll import configure_native_dll_paths
    configure_native_dll_paths()
except Exception:
    pass
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass
from zhulong.engine.agent_engine import run_agent_tick
from zhulong.agent.knowledge_net import KnowledgeNetInference
from zhulong.agent.rl_agent import resolve_knowledge_paths
from zhulong.engine.agent_engine import load_agent_config
import pandas as pd
import numpy as np
cfg_path = root / 'config' / 'config_agent.json'
if not cfg_path.is_file():
    print('AGENT_FAIL missing config')
    sys.exit(1)
cfg = load_agent_config(cfg_path)
if not cfg.get('enabled', True):
    print('AGENT_FAIL agent disabled in config')
    sys.exit(1)
kn_path, kn_scaler = resolve_knowledge_paths('XAUUSD', cfg, root)
kn = KnowledgeNetInference(kn_path, scaler_path=kn_scaler)
if not kn.is_ready:
    print('AGENT_FAIL knowledge_net not ready', kn_path)
    sys.exit(1)
idx = pd.date_range('2024-01-01', periods=120, freq='5min', tz='UTC')
close = 2400 + np.cumsum(np.random.randn(120) * 0.1)
m5 = pd.DataFrame({'open': close, 'high': close+0.3, 'low': close-0.3, 'close': close, 'volume': 50.0}, index=idx)
req = {'config_path': str(cfg_path), 'symbols': ['XAUUSD'], 'primary_symbol': 'XAUUSD'}
out = run_agent_tick({'XAUUSD': m5}, req, root)
if not out.get('ok') or not out.get('results'):
    print('AGENT_FAIL', out)
    sys.exit(1)
if out['results'][0].get('skipped'):
    print('AGENT_FAIL skipped', out['results'][0].get('reason'))
    sys.exit(1)
print('AGENT_OK')
"@
Invoke-Python -c $agentSmoke

Write-Host 'Python deps OK' -ForegroundColor Green
