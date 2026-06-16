@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "DOTNET_ROLL_FORWARD=LatestPatch"
set "DOTNET_MULTILEVEL_LOOKUP=1"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\check_runtime.ps1" -InstallDir "%CD%" -AutoRepair
if errorlevel 1 (
  echo.
  echo 运行环境未就绪，请查看 %APPDATA%\ZhuLong\startup.log
  pause
  exit /b 1
)

if exist "%APPDATA%\ZhuLong\dotnet_root.txt" (
  for /f "usebackq delims=" %%D in ("%APPDATA%\ZhuLong\dotnet_root.txt") do set "DOTNET_ROOT=%%D"
)

start "" /D "%CD%" "%CD%\ZhuLong.exe"
endlocal
