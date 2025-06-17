from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from models.base import get_db
from models.data_models import Detection, SensorReading, DataSource, ThreatType, DataSourceType

router = APIRouter()

@router.get("/overview")
async def get_dashboard_overview(
    db: Session = Depends(get_db),
    time_range: str = "7d"
):
    """
    Get overview statistics for the dashboard
    """
    try:
        # Calculate time range
        now = datetime.utcnow()
        
        if time_range.endswith('h'):
            time_ago = now - timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            time_ago = now - timedelta(days=int(time_range[:-1]))
        else:
            time_ago = now - timedelta(days=7)  # Default to 7 days
        
        # Get detection counts by type
        detection_counts = db.query(
            Detection.threat_type,
            func.count(Detection.id).label('count')
        ).filter(
            Detection.timestamp >= time_ago
        ).group_by(
            Detection.threat_type
        ).all()
        
        # Get sensor counts by type
        sensor_counts = db.query(
            DataSource.type,
            func.count(DataSource.id).label('count')
        ).filter(
            DataSource.is_active == True
        ).group_by(
            DataSource.type
        ).all()
        
        # Get latest detections
        latest_detections = db.query(Detection).order_by(
            Detection.timestamp.desc()
        ).limit(5).all()
        
        # Get latest sensor readings
        latest_readings = db.query(SensorReading).order_by(
            SensorReading.timestamp.desc()
        ).limit(5).all()
        
        # Format response
        return {
            "time_range": {
                "start": time_ago.isoformat(),
                "end": now.isoformat()
            },
            "detection_counts": [
                {"threat_type": d[0].value, "count": d[1]} 
                for d in detection_counts
            ],
            "sensor_counts": [
                {"type": s[0].value, "count": s[1]} 
                for s in sensor_counts
            ],
            "latest_detections": latest_detections,
            "latest_readings": latest_readings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_recent_alerts(
    db: Session = Depends(get_db),
    limit: int = 10,
    threshold: float = 0.7
):
    """
    Get recent high-confidence detections (alerts)
    """
    try:
        alerts = db.query(Detection).filter(
            Detection.confidence >= threshold
        ).order_by(
            Detection.timestamp.desc()
        ).limit(limit).all()
        
        return {
            "count": len(alerts),
            "alerts": alerts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensor-health")
async def get_sensor_health(
    db: Session = Depends(get_db),
    time_window: int = 24  # hours
):
    """
    Get sensor health status based on recent activity
    """
    try:
        # Get the timestamp for the time window
        time_ago = datetime.utcnow() - timedelta(hours=time_window)
        
        # Get all active sensors
        sensors = db.query(DataSource).filter(
            DataSource.is_active == True,
            DataSource.type.in_([DataSourceType.IOT, DataSourceType.DRONE])
        ).all()
        
        sensor_status = []
        
        for sensor in sensors:
            # Get the most recent reading
            last_reading = db.query(SensorReading).filter(
                SensorReading.source_id == sensor.id
            ).order_by(
                SensorReading.timestamp.desc()
            ).first()
            
            # Determine status
            status = "offline"
            last_seen = None
            
            if last_reading:
                last_seen = last_reading.timestamp
                time_since_last_seen = (datetime.utcnow() - last_reading.timestamp).total_seconds()
                
                if time_since_last_seen < 3600:  # 1 hour
                    status = "online"
                elif time_since_last_seen < 86400:  # 24 hours
                    status = "warning"
                else:
                    status = "error"
            
            sensor_status.append({
                "id": sensor.id,
                "name": sensor.name,
                "type": sensor.type.value,
                "status": status,
                "last_seen": last_seen.isoformat() if last_seen else None,
                "location": sensor.location
            })
        
        return {
            "sensor_count": len(sensors),
            "sensors": sensor_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/threat-map")
async def get_threat_map_data(
    db: Session = Depends(get_db),
    time_range: str = "7d",
    threat_type: Optional[ThreatType] = None
):
    """
    Get data for the threat map visualization
    """
    try:
        # Calculate time range
        now = datetime.utcnow()
        
        if time_range.endswith('h'):
            time_ago = now - timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            time_ago = now - timedelta(days=int(time_range[:-1]))
        else:
            time_ago = now - timedelta(days=7)  # Default to 7 days
        
        # Build base query
        query = db.query(Detection).filter(
            Detection.timestamp >= time_ago
        )
        
        # Filter by threat type if provided
        if threat_type:
            query = query.filter(Detection.threat_type == threat_type)
        
        # Execute query
        detections = query.all()
        
        # Format response for map
        features = []
        for d in detections:
            # Extract coordinates from location (assuming POINT format: "POINT(lon lat)")
            if d.location:
                # Parse WKT POINT format
                coords = d.location.ST_AsText().replace('POINT(', '').replace(')', '').split()
                if len(coords) == 2:
                    lon, lat = map(float, coords)
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lon, lat]
                        },
                        "properties": {
                            "id": d.id,
                            "threat_type": d.threat_type.value,
                            "confidence": d.confidence,
                            "timestamp": d.timestamp.isoformat(),
                            "source_type": d.source_type.value if d.source_type else None,
                            "is_verified": d.is_verified
                        }
                    }
                    features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "time_range": {
                "start": time_ago.isoformat(),
                "end": now.isoformat()
            },
            "threat_type": threat_type.value if threat_type else "all"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
