#Requires -Version 5.1
param([string] $LogDir = "logs/training")
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'run_full_agent_training.ps1') `
    -LogDir $LogDir `
    -SkipXauPrepare -SkipXauKnowledge -SkipXauRl -SkipXauBacktest
