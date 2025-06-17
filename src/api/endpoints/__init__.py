"""EarthSense API endpoints."""

from fastapi import APIRouter

# Import all endpoint modules
from . import data
from . import detections
from . import satellite
from . import sensors
from . import dashboard

# Create a router for all API endpoints
router = APIRouter()

# Include all endpoint routers
router.include_router(data.router, prefix="/data", tags=["data"])
router.include_router(detections.router, prefix="/detections", tags=["detections"])
router.include_router(satellite.router, prefix="/satellite", tags=["satellite"])
router.include_router(sensors.router, prefix="/sensors", tags=["sensors"])
router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])

__all__ = [
    'router',
    'data',
    'detections',
    'satellite',
    'sensors',
    'dashboard'
]
