from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

from config.config import config
from models.base import Base, engine

# Import routers
from .endpoints import data, detections, satellite, sensors, dashboard

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="EarthSense AI API",
    description="API for EarthSense AI environmental monitoring system",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))

# Include routers
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(detections.router, prefix="/api/detections", tags=["detections"])
app.include_router(satellite.router, prefix="/api/satellite", tags=["satellite"])
app.include_router(sensors.router, prefix="/api/sensors", tags=["sensors"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
