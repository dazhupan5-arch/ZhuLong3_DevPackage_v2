# 打包/热更新门禁：智能体 Python 热更新文件必须语法正确，且不得向安装目录写 __pycache__
param(
    [string]$Root = ''
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Root)) {
    $Root = Split-Path -Parent $PSScriptRoot
}
$Root = (Resolve-Path -LiteralPath $Root).Path

$pyFiles = @(
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
)

$checkScript = "import ast,sys; from pathlib import Path; p=Path(sys.argv[1]); ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))"

Write-Host "Python syntax gate (read-only ast.parse): root=$Root" -ForegroundColor Cyan
$failed = @()
foreach ($rel in $pyFiles) {
    $path = Join-Path $Root $rel
    if (-not (Test-Path -LiteralPath $path)) {
        $failed += "MISSING $rel"
        continue
    }
    & py -3 -c $checkScript $path 2>&1 | Out-String | ForEach-Object {
        if ($_ -match '\S') { Write-Host $_ }
    }
    if ($LASTEXITCODE -ne 0) { $failed += $rel }
}

if ($failed.Count -gt 0) {
    Write-Error ("Python agent bundle syntax FAILED:`n  " + ($failed -join "`n  "))
}
Write-Host "Python agent bundle syntax OK ($($pyFiles.Count) files)" -ForegroundColor Green
