param(
    [string]$Source = (Join-Path $PSScriptRoot 'LaunchZhuLong.cmd'),
    [string[]]$Destinations = @()
)

function Write-WindowsBatch {
    param([string]$Src, [string]$Dest)
    $raw = [System.IO.File]::ReadAllText($Src)
    $raw = $raw -replace "`r`n", "`n" -replace "`n", "`r`n"
    $enc = New-Object System.Text.ASCIIEncoding
    $dir = Split-Path -Parent $Dest
    if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    [System.IO.File]::WriteAllText($Dest, $raw, $enc)
    Write-Host "Wrote $Dest"
}

if ($Destinations.Count -eq 0) {
    $root = Split-Path -Parent $PSScriptRoot
    $Destinations = @(
        (Join-Path $root 'publish\win-x64\LaunchZhuLong.cmd'),
        'D:\Program Files\ZhuLong\LaunchZhuLong.cmd'
    )
}

foreach ($d in $Destinations) {
    Write-WindowsBatch -Src $Source -Dest $d
}

$bytes = [System.IO.File]::ReadAllBytes($Destinations[0])
$text = [System.Text.Encoding]::ASCII.GetString($bytes)
$crlf = ([regex]::Matches($text, "`r`n")).Count
Write-Host "CRLF lines: $crlf"
