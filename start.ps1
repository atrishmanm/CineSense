# CineSense Startup Script (PowerShell)
# Run this script to start the CineSense application

Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║         CineSense - AI Movie Recommendation Platform         ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "⚠️  Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv .venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".venv\Scripts\Activate.ps1"

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  .env file not found. Copying from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env file created. Please update with your configuration." -ForegroundColor Green
    Write-Host "📝 Edit .env file with your database credentials and TMDB API key" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter after updating .env file"
}

# Install/update dependencies
Write-Host "`nChecking dependencies..." -ForegroundColor Cyan
$answer = Read-Host "Do you want to install/update dependencies? (y/n)"
if ($answer -eq "y") {
    Write-Host "Installing dependencies..." -ForegroundColor Cyan
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
}

# Check if Redis is running
Write-Host "`nChecking Redis..." -ForegroundColor Cyan
$redis_running = Get-Process redis-server -ErrorAction SilentlyContinue
if (-not $redis_running) {
    Write-Host "⚠️  Redis is not running. Starting Redis..." -ForegroundColor Yellow
    Write-Host "   If Redis is not installed, install it from:" -ForegroundColor Gray
    Write-Host "   https://github.com/microsoftarchive/redis/releases" -ForegroundColor Gray
    Write-Host ""
    $start_redis = Read-Host "Try to start Redis? (y/n)"
    if ($start_redis -eq "y") {
        Start-Process redis-server -WindowStyle Hidden
        Start-Sleep -Seconds 2
        Write-Host "✓ Redis started" -ForegroundColor Green
    }
} else {
    Write-Host "✓ Redis is running" -ForegroundColor Green
}

# Check database connection
Write-Host "`nChecking database connection..." -ForegroundColor Cyan
python -c "from database.db_manager import DatabaseManager; db = DatabaseManager(); print('✓ Database connection successful')" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Database connected" -ForegroundColor Green
} else {
    Write-Host "⚠️  Could not connect to database" -ForegroundColor Yellow
    Write-Host "   Make sure MySQL is running and .env is configured correctly" -ForegroundColor Gray
}

# Ask which app to start
Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "Select application mode:" -ForegroundColor Cyan
Write-Host "  [1] Standard App (app.py)" -ForegroundColor White
Write-Host "  [2] Fully Integrated App with ALL Features (app_integrated.py)" -ForegroundColor Yellow
Write-Host "  [3] Quick Start Script (scripts/quick_start.py)" -ForegroundColor White
Write-Host "════════════════════════════════════════════════════════════════" -ForegroundColor White

$choice = Read-Host "Enter your choice (1-3)"

Write-Host ""
Write-Host "🚀 Starting CineSense..." -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════`n" -ForegroundColor Green

switch ($choice) {
    "1" {
        Write-Host "Starting Standard App..." -ForegroundColor Cyan
        python app.py
    }
    "2" {
        Write-Host "Starting Fully Integrated App..." -ForegroundColor Cyan
        Write-Host "⚡ Loading AI models (may take a few moments)..." -ForegroundColor Yellow
        python app_integrated.py
    }
    "3" {
        Write-Host "Starting Quick Start..." -ForegroundColor Cyan
        python scripts/quick_start.py
    }
    default {
        Write-Host "Starting Fully Integrated App (default)..." -ForegroundColor Cyan
        python app_integrated.py
    }
}

Write-Host "`n════════════════════════════════════════════════════════════════" -ForegroundColor White
Write-Host "Application stopped." -ForegroundColor Yellow
Write-Host "════════════════════════════════════════════════════════════════`n" -ForegroundColor White
