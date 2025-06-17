"""EarthSense API package."""

from fastapi import FastAPI
from .main import app
from .endpoints import data, detections, satellite, sensors, dashboard

__all__ = [
    'app',
    'data',
    'detections',
    'satellite',
    'sensors',
    'dashboard'
]
