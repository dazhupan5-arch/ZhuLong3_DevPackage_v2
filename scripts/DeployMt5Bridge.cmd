@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo Deploying ZhuLong MT5 bridge from %CD%
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\deploy-mt5-indicator.ps1" -InstallDir "%CD%"
echo.
echo Done. Restart MT5 and reload ZhuLongIndicator on XAUUSD M1.
pause
endlocal
