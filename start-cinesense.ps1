# CineSense Application Startup Script
param([switch]$CheckMySQL = $true, [switch]$OpenBrowser = $true, [string]$Port = "5000")

$ProjectRoot = "c:\Users\Atrishman\Documents\VS CODE\PROJECTS\CineSense"
$AppScript = "app_integrated.py"
$ServerURL = "http://localhost:$Port"

function Write-ColorOutput { param([string]$Message, [string]$Color = "White"); Write-Host $Message -ForegroundColor $Color }

Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "      CineSense Startup Manager        " "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

Write-ColorOutput "[1/4] Navigating to project directory..." "Yellow"
if (-not (Test-Path $ProjectRoot)) { Write-ColorOutput "ERROR: Project directory not found: $ProjectRoot" "Red"; exit 1 }
Set-Location $ProjectRoot
Write-ColorOutput "       Changed to: $ProjectRoot" "Green"
Write-Host ""

if ($CheckMySQL) {
    Write-ColorOutput "[2/4] Checking MySQL service..." "Yellow"
    try {
        $mysqlService = Get-Service -Name "MySQL80" -ErrorAction SilentlyContinue
        if ($mysqlService -and $mysqlService.Status -eq "Running") {
            Write-ColorOutput "       MySQL is running" "Green"
        } elseif ($mysqlService) {
            Write-ColorOutput "       MySQL is not running. Attempting to start..." "Yellow"
            Start-Service -Name "MySQL80"
            Start-Sleep -Seconds 3
            Write-ColorOutput "       MySQL started successfully" "Green"
        } else {
            Write-ColorOutput "       MySQL service not found" "Yellow"
        }
    } catch {
        Write-ColorOutput "       Could not check MySQL: $($_.Exception.Message)" "Yellow"
    }
} else {
    Write-ColorOutput "[2/4] Skipping MySQL check..." "Yellow"
}
Write-Host ""

Write-ColorOutput "[3/4] Verifying Python installation..." "Yellow"
try {
    $pythonVersion = py --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python launcher not found" }
    Write-ColorOutput "       Python found: $pythonVersion" "Green"
} catch {
    Write-ColorOutput "      ERROR: Python not found. Please install Python from python.org" "Red"
    exit 1
}
Write-Host ""

Write-ColorOutput "[4/4] Starting CineSense application..." "Yellow"
Write-ColorOutput "      Server URL: $ServerURL" "Cyan"
Write-ColorOutput "      Press Ctrl+C to stop the server" "Gray"
Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

if ($OpenBrowser) {
    Start-Job -ScriptBlock { param($url); Start-Sleep -Seconds 3; Start-Process $url } -ArgumentList $ServerURL | Out-Null
}

py $AppScript
