# CineSense Application Startup Script
# This script automatically starts the CineSense movie recommendation platform

param(
    [switch]$CheckMySQL = $true,
    [switch]$OpenBrowser = $true,
    [string]$Port = "5000"
)

# Script configuration
$ProjectRoot = "c:\Users\Atrishman\Documents\VS CODE\PROJECTS\CineSense"
$AppScript = "app.py"
$ServerURL = "http://localhost:$Port"

# Colors for output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Banner
Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "      CineSense Startup Manager        " "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

# Change to project directory
Write-ColorOutput "[1/4] Navigating to project directory..." "Yellow"
if (-not (Test-Path $ProjectRoot)) {
    Write-ColorOutput "ERROR: Project directory not found: $ProjectRoot" "Red"
    exit 1
}
Set-Location $ProjectRoot
Write-ColorOutput "      ✓ Changed to: $ProjectRoot" "Green"
Write-Host ""

# Check if MySQL is running (optional)
if ($CheckMySQL) {
    Write-ColorOutput "[2/4] Checking MySQL service..." "Yellow"
    try {
        $mysqlService = Get-Service -Name "MySQL80" -ErrorAction SilentlyContinue
        if ($mysqlService -and $mysqlService.Status -eq "Running") {
            Write-ColorOutput "      ✓ MySQL is running" "Green"
        }
        elseif ($mysqlService) {
            Write-ColorOutput "      ⚠ MySQL is not running. Attempting to start..." "Yellow"
            Start-Service -Name "MySQL80"
            Start-Sleep -Seconds 3
            Write-ColorOutput "      ✓ MySQL started successfully" "Green"
        }
        else {
            Write-ColorOutput "      ⚠ MySQL service not found" "Yellow"
        }
    }
    catch {
        Write-ColorOutput "      ⚠ Could not check MySQL: $($_.Exception.Message)" "Yellow"
    }
}
else {
    Write-ColorOutput "[2/4] Skipping MySQL check..." "Yellow"
}
Write-Host ""

# Check if Python is available
Write-ColorOutput "[3/4] Verifying Python installation..." "Yellow"
try {
    $pythonVersion = py --version 2>&1
    Write-ColorOutput "      ✓ Python found: $pythonVersion" "Green"
}
catch {
    Write-ColorOutput "      ERROR: Python not found" "Red"
    exit 1
}
Write-Host ""

# Start the Flask application
Write-ColorOutput "[4/4] Starting CineSense application..." "Yellow"
Write-ColorOutput "      Server URL: $ServerURL" "Cyan"
Write-ColorOutput "      Press Ctrl+C to stop the server" "Gray"
Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

# Open browser after a short delay (in background job)
if ($OpenBrowser) {
    Start-Job -ScriptBlock {
        param($url)
        Start-Sleep -Seconds 3
        Start-Process $url
    } -ArgumentList $ServerURL | Out-Null
}

# Start the application
py $AppScript
