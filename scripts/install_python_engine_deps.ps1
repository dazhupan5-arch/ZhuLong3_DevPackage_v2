# Deprecated: use install_python_deps.ps1 (system Python, no python_runtime)
Write-Host 'This script is deprecated. Use: .\scripts\install_python_deps.ps1' -ForegroundColor Yellow
& (Join-Path $PSScriptRoot 'install_python_deps.ps1')
exit $LASTEXITCODE
