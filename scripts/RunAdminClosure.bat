@echo off
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\trae_projects\ZhuLong3_DevPackage_v2\scripts\fix_v16_live_closure_admin.ps1'"
pause
