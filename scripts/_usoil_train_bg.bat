@echo off
chcp 65001 >nul
set PYTHONUNBUFFERED=1
cd /d "D:\ZhuLong3_Migration_20260609.zip"
py -3 -u scripts/train_usoil_agent_until_pass.py > D:\ZhuLong3_Migration_20260609.zip\logs\training\usoil_run_20260612_153825.log 2>&1
