# Launch XAU PPO train-until-pass loop (background-friendly)
$ErrorActionPreference = "Stop"
$root = "D:\ZhuLong3_Migration_20260609.zip"
$log = Join-Path $root "logs\training\xau_rl_until_pass_runner.log"
Set-Location $root
$env:PYTHONUNBUFFERED = "1"
py -3 -u scripts/train_xau_rl_until_pass.py 2>&1 | Tee-Object -FilePath $log
