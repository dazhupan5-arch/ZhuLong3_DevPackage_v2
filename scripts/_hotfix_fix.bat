@echo off
echo ==========================================
echo  ZL3 Hotfix - KN2 dictator fixes
echo ==========================================

copy /Y "D:\trae_projects\ZhuLong3_DevPackage_v2\zhulong\agent\trading_agent.py" "C:\Program Files\ZhuLong\zhulong\agent\trading_agent.py"
if %ERRORLEVEL% EQU 0 (echo [OK] trading_agent.py) else (echo [FAIL] trading_agent.py)

copy /Y "D:\trae_projects\ZhuLong3_DevPackage_v2\config\config_agent.json" "C:\Program Files\ZhuLong\config\config_agent.json"
if %ERRORLEVEL% EQU 0 (echo [OK] config_agent.json -- install) else (echo [FAIL] config_agent.json -- install)

copy /Y "D:\trae_projects\ZhuLong3_DevPackage_v2\config\config_agent.json" "%APPDATA%\ZhuLong\config_agent.json"
if %ERRORLEVEL% EQU 0 (echo [OK] config_agent.json -- appdata) else (echo [FAIL] config_agent.json -- appdata)

echo DONE
