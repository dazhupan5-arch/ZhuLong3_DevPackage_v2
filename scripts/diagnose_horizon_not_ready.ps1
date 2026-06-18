# 同事机 Horizon horizon_not_ready 一键诊断
param(
    [string]$InstallDir = 'C:\Program Files\ZhuLong'
)

$ErrorActionPreference = 'Continue'
$appdata = Join-Path $env:APPDATA 'ZhuLong'
Write-Host "=== ZhuLong Horizon 诊断 ===" -ForegroundColor Cyan
Write-Host "Install: $InstallDir"
Write-Host "AppData: $appdata"

$onnx = Join-Path $InstallDir 'models\horizon_v16.onnx'
$scaler = Join-Path $InstallDir 'models\horizon_v16_scaler.pkl'
foreach ($p in @($onnx, $scaler)) {
    if (Test-Path $p) {
        $fi = Get-Item $p
        Write-Host "[OK] $($fi.Name) size=$($fi.Length) path=$($fi.FullName)" -ForegroundColor Green
    }
    else {
        $alt = Join-Path $appdata (Split-Path $p -Leaf)
        if (Test-Path $alt) {
            $fi = Get-Item $alt
            Write-Host "[OK] $($fi.Name) (AppData) size=$($fi.Length)" -ForegroundColor Yellow
        }
        else {
            Write-Host "[FAIL] missing $p" -ForegroundColor Red
        }
    }
}

Write-Host "`n--- Python / onnxruntime ---" -ForegroundColor Cyan
py -3 -c "import sys; print('python', sys.version.split()[0]); import onnxruntime as ort; print('onnxruntime', ort.__version__)"

Write-Host "`n--- ONNX Session probe ---" -ForegroundColor Cyan
$probe = @"
import os, sys
install = r'$InstallDir'
os.environ['ZHULONG_INSTALL_DIR'] = install
sys.path.insert(0, install)
from zhulong.utils.win_dll import configure_native_dll_paths
configure_native_dll_paths()
import onnxruntime as ort
onnx = os.path.join(install, 'models', 'horizon_v16.onnx')
sess = ort.InferenceSession(onnx, ort.SessionOptions(), providers=['CPUExecutionProvider'])
print('ONNX Session OK', onnx)
"@
py -3 -c $probe

Write-Host "`n--- agent_validate (install CLI) ---" -ForegroundColor Cyan
$cfg = Join-Path $appdata 'config_agent.json'
if (-not (Test-Path $cfg)) { $cfg = Join-Path $InstallDir 'config\config_agent.json' }
$req = @{ cmd = 'agent_validate'; config_path = $cfg; root = $InstallDir; quick = $true } | ConvertTo-Json -Compress
$rf = Join-Path $env:TEMP 'zhulong_diag_req.json'
$of = Join-Path $env:TEMP 'zhulong_diag_out.json'
$req | Set-Content -Encoding UTF8 $rf
$cli = Join-Path $InstallDir 'ZhuLong.PythonEngine\inference_cli.py'
if (Test-Path (Join-Path $appdata 'ZhuLong.PythonEngine\inference_cli.py')) {
    $cli = Join-Path $appdata 'ZhuLong.PythonEngine\inference_cli.py'
    Write-Host "Using AppData CLI: $cli" -ForegroundColor Yellow
}
$env:ZHULONG_INSTALL_DIR = $InstallDir
$env:PYTHONDONTWRITEBYTECODE = '1'
py -3 $cli --input $rf --output $of
Get-Content $of
Write-Host "`n--- 说明 ---" -ForegroundColor Cyan
Write-Host "设置页「环境自检」的 import onnxruntime 通过，不等于下方探针会通过。" -ForegroundColor Yellow
Write-Host "agent_validate 与开机校验使用相同 Horizon Session 探针。" -ForegroundColor Yellow
Write-Host "`n若 ONNX Session 失败：以管理员运行 install_python_deps.ps1，并安装 VC++ 2015-2022 x64 运行库。" -ForegroundColor Yellow
Write-Host "若 AppData 有陈旧热更新，可删除 %APPDATA%\ZhuLong\ZhuLong.PythonEngine 后重试。" -ForegroundColor Yellow
