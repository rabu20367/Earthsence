#!/usr/bin/env python3
"""
Initialize the EarthSense database and create necessary directories.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path(__file__).parent
    dirs = [
        base_dir / "data",
        base_dir / "data/raw",
        base_dir / "data/processed",
        base_dir / "data/models",
        base_dir / "logs",
        base_dir / "static",
        base_dir / "static/images",
        base_dir / "static/uploads",
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to initialize the database and directories."""
    print("Initializing EarthSense database and directories...")
    create_directories()
    print("\nInitialization complete!")

if __name__ == "__main__":
    main()
