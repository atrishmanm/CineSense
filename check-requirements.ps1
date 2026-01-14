# CineSense Requirements Checker
param([switch]$AutoInstall = $false)

$ProjectRoot = "c:\Users\Atrishman\Documents\VS CODE\PROJECTS\CineSense"
$RequirementsFile = "requirements.txt"

function Write-ColorOutput { param([string]$Message, [string]$Color = "White"); Write-Host $Message -ForegroundColor $Color }

Write-Host ""
Write-ColorOutput "========================================" "Cyan"
Write-ColorOutput "   CineSense Requirements Checker      " "Cyan"
Write-ColorOutput "========================================" "Cyan"
Write-Host ""

if (-not (Test-Path $ProjectRoot)) { Write-ColorOutput "ERROR: Project directory not found: $ProjectRoot" "Red"; exit 1 }
Set-Location $ProjectRoot

Write-ColorOutput "[1/4] Checking Python installation..." "Yellow"
try {
    $pythonVersion = py --version 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Python launcher not found" }
    Write-ColorOutput "       Python found: $pythonVersion" "Green"
} catch {
    Write-ColorOutput "      ERROR: Python not found" "Red"
    exit 1
}
Write-Host ""

Write-ColorOutput "[2/4] Checking requirements file..." "Yellow"
if (-not (Test-Path $RequirementsFile)) { Write-ColorOutput "      ERROR: requirements.txt not found" "Red"; exit 1 }
Write-ColorOutput "       requirements.txt found" "Green"
Write-Host ""

Write-ColorOutput "[3/4] Checking installed packages..." "Yellow"
$installedPackages = py -m pip list --format=freeze 2>&1 | Out-String
$requirements = Get-Content $RequirementsFile | Where-Object { $_ -notmatch '^\s*#' -and $_ -match '\S' }
$missingPackages = @()
$installedCount = 0

foreach ($req in $requirements) {
    $packageName = ($req -split '==|>=|<=|>|<|~=')[0].Trim()
    if ($installedPackages -match "(?i)^$packageName==") {
        Write-ColorOutput "       $req" "Green"
        $installedCount++
    } else {
        Write-ColorOutput "       $req" "Red"
        $missingPackages += $req
    }
}

Write-Host ""
Write-ColorOutput "Summary: $installedCount/$($requirements.Count) packages installed" $(if ($missingPackages.Count -eq 0) { "Green" } else { "Yellow" })
Write-Host ""

if ($missingPackages.Count -gt 0) {
    Write-ColorOutput "[4/4] Missing packages detected" "Yellow"
    Write-Host ""
    Write-ColorOutput "The following packages need to be installed:" "White"
    foreach ($pkg in $missingPackages) { Write-ColorOutput "  - $pkg" "Yellow" }
    Write-Host ""
    
    $install = $false
    if ($AutoInstall) {
        $install = $true
        Write-ColorOutput "Auto-install enabled. Installing packages..." "Cyan"
    } else {
        $response = Read-Host "Do you want to install missing packages? (Y/N)"
        $install = $response -match '^[Yy]'
    }
    
    if ($install) {
        Write-Host ""
        Write-ColorOutput "Installing packages from requirements.txt..." "Cyan"
        Write-Host ""
        py -m pip install -r $RequirementsFile
        if ($LASTEXITCODE -eq 0) {
            Write-Host ""
            Write-ColorOutput "========================================" "Green"
            Write-ColorOutput "   All packages installed successfully" "Green"
            Write-ColorOutput "========================================" "Green"
        } else {
            Write-Host ""
            Write-ColorOutput " Some packages may have failed to install" "Yellow"
        }
    } else {
        Write-Host ""
        Write-ColorOutput "Installation cancelled by user." "Yellow"
        Write-ColorOutput "Run this script again when ready to install." "Gray"
    }
} else {
    Write-ColorOutput "[4/4] All requirements satisfied" "Green"
    Write-Host ""
    Write-ColorOutput "========================================" "Green"
    Write-ColorOutput "   All required packages are installed" "Green"
    Write-ColorOutput "========================================" "Green"
}
Write-Host ""
