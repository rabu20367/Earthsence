"""Configuration settings for EarthSense AI."""
from pathlib import Path
from pydantic import BaseSettings, Field, PostgresDsn, validator
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Project metadata
    PROJECT_NAME: str = "EarthSense AI"
    VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # Database settings
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "earthsense")
    DATABASE_URI: Optional[PostgresDsn] = None
    
    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            user=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )
    
    # AWS S3 settings
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_S3_BUCKET: str = os.getenv("AWS_S3_BUCKET", "earthsense-data")
    
    # Sentinel Hub settings
    SENTINEL_HUB_CLIENT_ID: Optional[str] = os.getenv("SENTINEL_HUB_CLIENT_ID")
    SENTINEL_HUB_CLIENT_SECRET: Optional[str] = os.getenv("SENTINEL_HUB_CLIENT_SECRET")
    
    # Model settings
    MODEL_CACHE_DIR: Path = BASE_DIR / "models"
    MODEL_CHECKPOINT: str = "google/vit-base-patch16-224-in21k"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create settings instance
settings = Settings()

# Create necessary directories
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(BASE_DIR / "data/raw", exist_ok=True)
os.makedirs(BASE_DIR / "data/processed", exist_ok=True)
