from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum, Boolean
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from geoalchemy2 import Geometry
from .base import Base
import enum

class DataSourceType(enum.Enum):
    SATELLITE = "satellite"
    DRONE = "drone"
    IOT = "iot_sensor"

class ThreatType(enum.Enum):
    ILLEGAL_LOGGING = "illegal_logging"
    WILDFIRE = "wildfire"
    WATER_POLLUTION = "water_pollution"
    AIR_QUALITY = "air_quality"
    OTHER = "other"

class DataSource(Base):
    """Represents a data source (satellite, drone, or IoT sensor)"""
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    type = Column(Enum(DataSourceType), nullable=False)
    description = Column(String, nullable=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=True)
    metadata = Column(JSONB, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SatelliteImage(Base):
    """Stores metadata about satellite images"""
    __tablename__ = "satellite_images"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    bounds = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=False)
    cloud_cover = Column(Float, nullable=True)
    resolution = Column(Float, nullable=True)  # in meters
    bands = Column(ARRAY(String), nullable=False)
    storage_path = Column(String, nullable=False)
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class SensorReading(Base):
    """Stores readings from IoT sensors"""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    temperature = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    air_quality = Column(JSONB, nullable=True)  # PM2.5, PM10, CO2, etc.
    water_quality = Column(JSONB, nullable=True)  # pH, turbidity, etc.
    metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class Detection(Base):
    """Stores detected environmental threats"""
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    threat_type = Column(Enum(ThreatType), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)
    bbox = Column(Geometry(geometry_type='POLYGON', srid=4326), nullable=True)  # For object detection
    source_type = Column(Enum(DataSourceType), nullable=False)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=True)
    image_id = Column(Integer, ForeignKey("satellite_images.id"), nullable=True)
    sensor_reading_id = Column(Integer, ForeignKey("sensor_readings.id"), nullable=True)
    metadata = Column(JSONB, default={})
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
