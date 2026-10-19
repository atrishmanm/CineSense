# CineSense Startup Script — DEPRECATED wrapper.
#
# This script is kept for backwards compatibility only. It delegates to the
# canonical launcher start.ps1 (which uses paths relative to the repo root).
#
# NOTE: the old $CheckMySQL / $OpenBrowser / $Port parameters are accepted for
# call-site compatibility but are ignored; use start.ps1 instead.

param([switch]$CheckMySQL = $true, [switch]$OpenBrowser = $true, [string]$Port = "5000")

$canonical = Join-Path $PSScriptRoot "start.ps1"
if (-not (Test-Path $canonical)) {
    Write-Error "Canonical launcher not found: $canonical"
    exit 1
}

Write-Host "start-cinesense.ps1 is deprecated; delegating to start.ps1" -ForegroundColor Yellow
& $canonical
exit $LASTEXITCODE
