#!/bin/bash
# Setup script for EarthSense AI environment (Linux/macOS)

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
pip install -r web_requirements.txt

# Install package in development mode
echo "Installing EarthSense in development mode..."
pip install -e .

# Set up environment variables
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "Please update the .env file with your configuration."
fi

echo ""
echo "✅ Setup complete! Virtual environment is now active."
echo "To activate the virtual environment in a new terminal, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the development server:"
echo "  ./run.sh"
echo ""
