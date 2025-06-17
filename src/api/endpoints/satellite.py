from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json
import rasterio
from rasterio.features import bounds as feature_bounds
from shapely.geometry import shape, mapping
import numpy as np
import os
from pathlib import Path

from models.base import get_db
from models.data_models import SatelliteImage, DataSource, Detection, ThreatType
from config.config import config

router = APIRouter()

@router.get("/images")
async def list_satellite_images(
    db: Session = Depends(get_db),
    source_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    bbox: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List satellite images with optional filters
    """
    try:
        query = db.query(SatelliteImage)
        
        # Apply filters
        if source_id is not None:
            query = query.filter(SatelliteImage.source_id == source_id)
        if start_time:
            query = query.filter(SatelliteImage.timestamp >= start_time)
        if end_time:
            query = query.filter(SatelliteImage.timestamp <= end_time)
        if bbox:
            # Convert bbox string to WKT format: "min_lon,min_lat,max_lon,max_lat"
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
                bbox_wkt = f'POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))'
                query = query.filter(SatelliteImage.bounds.ST_Intersects(bbox_wkt))
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="Invalid bbox format. Use 'min_lon,min_lat,max_lon,max_lat'")
        
        # Apply ordering and pagination
        query = query.order_by(SatelliteImage.timestamp.desc())
        total = query.count()
        images = query.offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "count": len(images),
            "offset": offset,
            "limit": limit,
            "items": images
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/images/{image_id}")
async def get_satellite_image(
    image_id: int,
    db: Session = Depends(get_db)
):
    """
    Get details of a specific satellite image
    """
    try:
        image = db.query(SatelliteImage).filter(SatelliteImage.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
            
        return image
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/{image_id}")
async def process_satellite_image(
    image_id: int,
    process_type: str = Query(..., description="Type of processing to apply"),
    db: Session = Depends(get_db)
):
    """
    Process a satellite image (e.g., detect features, calculate indices)
    """
    try:
        # Get the image
        image = db.query(SatelliteImage).filter(SatelliteImage.id == image_id).first()
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Load the image using rasterio
        image_path = Path(config.UPLOAD_FOLDER) / image.storage_path
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        
        with rasterio.open(image_path) as src:
            # Process based on the requested type
            if process_type == "ndvi":
                # Example: Calculate NDVI (Normalized Difference Vegetation Index)
                if len(src.indexes) < 3:
                    raise HTTPException(status_code=400, detail="Image doesn't have enough bands for NDVI calculation")
                
                # Assuming bands are in standard order (e.g., RGB, NIR)
                red = src.read(3).astype('float32')
                nir = src.read(4).astype('float32')
                
                # Calculate NDVI
                ndvi = (nir - red) / (nir + red + 1e-10)  # Add small value to avoid division by zero
                
                # Convert to uint8 for visualization
                ndvi_vis = ((ndvi + 1) * 127.5).astype('uint8')
                
                # Save the result
                output_path = image_path.with_stem(f"{image_path.stem}_ndvi")
                
                # Update metadata for the output
                meta = src.meta.copy()
                meta.update({
                    'driver': 'GTiff',
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': None
                })
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(ndvi_vis, 1)
                
                return {"message": f"NDVI calculation complete. Output saved to {output_path}"}
                
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported process type: {process_type}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tiles/{z}/{x}/{y}.png")
async def get_map_tile(
    z: int, x: int, y: int,
    source_id: Optional[int] = None,
    time_range: str = Query("30d", description="Time range: 24h, 7d, 30d, 90d"),
    db: Session = Depends(get_db)
):
    """
    Serve map tiles with satellite imagery overlay
    """
    try:
        # This is a simplified example - in a real implementation, you would use a tile server
        # like Geoserver, Mapbox, or implement your own tile generation
        
        # Calculate time range
        now = datetime.utcnow()
        if time_range.endswith('h'):
            time_ago = now - timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            time_ago = now - timedelta(days=int(time_range[:-1]))
        else:
            time_ago = now - timedelta(days=30)  # Default to 30 days
        
        # Query for images in the time range
        query = db.query(SatelliteImage).filter(
            SatelliteImage.timestamp >= time_ago
        )
        
        if source_id is not None:
            query = query.filter(SatelliteImage.source_id == source_id)
        
        # Get the most recent image
        image = query.order_by(SatelliteImage.timestamp.desc()).first()
        
        if not image:
            # Return transparent tile if no image found
            return Response(
                b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A\x00\x00\x00\x0D\x49\x48\x44\x52\x00\x00\x01\x00\x00\x00\x01\x00\x01\x03\x00\x00\x00\x66\xBC\x3A\x25\x00\x00\x00\x06\x50\x4C\x54\x45\x00\x00\x00\x00\x00\x00\xA5\x67\xB9\x93\x00\x00\x00\x01\x74\x52\x4E\x53\x00\x40\xE6\xD8\x66\x00\x00\x00\x01\x62\x4B\x47\x44\x00\x88\x05\x1D\x48\x00\x00\x00\x09\x70\x48\x59\x73\x00\x00\x0B\x13\x00\x00\x0B\x13\x01\x00\x9A\x9C\x18\x00\x00\x00\x07\x74\x49\x4D\x45\x07\xE5\x0B\x0F\x0B\x15\x0B\x5D\x1C\x1F\x8F\xC7\x00\x00\x00\x0F\x49\x44\x41\x54\x08\xD7\x63\x60\x20\x00\x00\x00\x40\x00\x01\xE0\xE6\x7C\x8C\x00\x00\x00\x00\x49\x45\x4E\x44\xAE\x42\x60\x82",
                media_type="image/png"
            )
        
        # In a real implementation, you would generate or fetch the tile for the requested coordinates
        # This is a placeholder that returns a simple tile with the image ID
        from fastapi.responses import Response
        from PIL import Image, ImageDraw
        import io
        
        # Create a simple tile with the image ID
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"Image {image.id}", fill=(255, 255, 255, 255))
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        return Response(img_byte_arr, media_type="image/png")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
