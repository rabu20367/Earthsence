#!/bin/bash
# Script to start the EarthSense development server (Linux/macOS)

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup_environment.sh first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Start the FastAPI server with auto-reload
echo "Starting EarthSense development server..."
echo ""
echo "🌍 API Documentation:"
echo "- Swagger UI:      http://127.0.0.1:8000/docs"
echo "- ReDoc:           http://127.0.0.1:8000/redoc"
echo "- Web Interface:   http://127.0.0.1:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
