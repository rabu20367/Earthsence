# Setup script for EarthSense AI environment

# Check if Python is installed
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python is not installed or not in PATH. Please install Python 3.9 or higher." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path -Path $activateScript) {
    . $activateScript
} else {
    Write-Host "Failed to activate virtual environment. Script not found at: $activateScript" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt
pip install -r web_requirements.txt

# Install package in development mode
Write-Host "Installing EarthSense in development mode..."
pip install -e .

# Set up environment variables
if (-not (Test-Path -Path .\.env) -and (Test-Path -Path .\.env.example)) {
    Write-Host "Creating .env file from .env.example..."
    Copy-Item -Path .\.env.example -Destination .\.env
    Write-Host "Please update the .env file with your configuration." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete! Virtual environment is now active." -ForegroundColor Green
Write-Host "To activate the virtual environment in a new terminal, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start the development server:" -ForegroundColor Cyan
Write-Host "  uvicorn app:app --reload" -ForegroundColor White
