@echo off
cd /d D:\trae_projects\ZhuLong3_DevPackage_v2
C:\Users\xiaomi\AppData\Local\Programs\Python\Python311\python.exe scripts\deep_audit_all_scenarios.py > scripts\_audit_output.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> scripts\_audit_output.txt
