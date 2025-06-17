from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
import shutil
import os
from datetime import datetime
from pathlib import Path

from models.base import get_db
from models.data_models import DataSource, SatelliteImage, SensorReading, DataSourceType
from config.config import config

router = APIRouter()

@router.post("/upload/satellite")
async def upload_satellite_image(
    file: UploadFile = File(...),
    source_id: int = None,
    timestamp: str = None,
    db: Session = Depends(get_db)
):
    """
    Upload a satellite image with metadata
    """
    try:
        # Validate source
        if source_id:
            source = db.query(DataSource).filter(
                DataSource.id == source_id, 
                DataSource.type == DataSourceType.SATELLITE
            ).first()
            if not source:
                raise HTTPException(status_code=400, detail="Invalid satellite source ID")
        
        # Parse timestamp or use current time
        img_timestamp = datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow()
        
        # Create upload directory if it doesn't exist
        upload_dir = Path(config.UPLOAD_FOLDER) / "satellite" / str(img_timestamp.date())
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_extension = Path(file.filename).suffix
        filename = f"{int(img_timestamp.timestamp())}_{source_id or 'unknown'}{file_extension}"
        file_path = upload_dir / filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Create database record
        # Note: In a real implementation, you would extract actual metadata from the image
        image = SatelliteImage(
            source_id=source_id,
            timestamp=img_timestamp,
            bounds=None,  # Should be extracted from image metadata
            cloud_cover=None,  # Should be extracted from image metadata
            resolution=None,  # Should be extracted from image metadata
            bands=[],  # Should be extracted from image metadata
            storage_path=str(file_path.relative_to(config.UPLOAD_FOLDER)),
            metadata={
                "original_filename": file.filename,
                "content_type": file.content_type,
                "size": file.size
            }
        )
        
        db.add(image)
        db.commit()
        db.refresh(image)
        
        return {"message": "File uploaded successfully", "id": image.id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload/sensor")
async def upload_sensor_data(
    data: dict,
    source_id: int,
    db: Session = Depends(get_db)
):
    """
    Upload sensor data
    """
    try:
        # Validate source
        source = db.query(DataSource).filter(
            DataSource.id == source_id,
            DataSource.type == DataSourceType.IOT
        ).first()
        
        if not source:
            raise HTTPException(status_code=400, detail="Invalid sensor source ID")
        
        # Parse timestamp or use current time
        timestamp = data.get('timestamp')
        if timestamp:
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.utcnow()
        else:
            timestamp = datetime.utcnow()
        
        # Create sensor reading
        reading = SensorReading(
            source_id=source_id,
            timestamp=timestamp,
            location=None,  # Should be provided in data or from source
            temperature=data.get('temperature'),
            humidity=data.get('humidity'),
            air_quality=data.get('air_quality'),
            water_quality=data.get('water_quality'),
            metadata=data.get('metadata', {})
        )
        
        db.add(reading)
        db.commit()
        db.refresh(reading)
        
        return {"message": "Sensor data saved successfully", "id": reading.id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
