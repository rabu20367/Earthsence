# Script to start the EarthSense development server

# Check if virtual environment exists
if (-not (Test-Path -Path .\.venv)) {
    Write-Host "Virtual environment not found. Please run setup_environment.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
$activateScript = ".\.venv\Scripts\Activate.ps1"
if (Test-Path -Path $activateScript) {
    . $activateScript
} else {
    Write-Host "Failed to activate virtual environment. Script not found at: $activateScript" -ForegroundColor Red
    exit 1
}

# Load environment variables
if (Test-Path -Path .\.env) {
    Get-Content .\.env | ForEach-Object {
        $name, $value = $_.split('=', 2)
        if ($name -and $value) {
            [System.Environment]::SetEnvironmentVariable($name.Trim(), $value.Trim())
        }
    }
}

# Start the FastAPI server with auto-reload
Write-Host "Starting EarthSense development server..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "API Documentation:" -ForegroundColor Cyan
Write-Host "- Swagger UI:      http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "- ReDoc:           http://127.0.0.1:8000/redoc" -ForegroundColor White
Write-Host ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
