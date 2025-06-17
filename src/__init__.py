"""EarthSense AI: Environmental Monitoring System."""

__version__ = "0.1.0"

# Import main components
from . import api
from . import data_ingestion
from . import data_processing
from . import models

__all__ = [
    'api',
    'data_ingestion',
    'data_processing',
    'models'
]
