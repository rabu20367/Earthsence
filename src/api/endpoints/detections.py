from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import json

from models.base import get_db
from models.data_models import Detection, ThreatType, DataSourceType

router = APIRouter()

@router.get("/")
async def list_detections(
    db: Session = Depends(get_db),
    threat_type: Optional[ThreatType] = None,
    source_type: Optional[DataSourceType] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    bbox: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all detections with optional filters
    """
    try:
        query = db.query(Detection)
        
        # Apply filters
        if threat_type:
            query = query.filter(Detection.threat_type == threat_type)
        if source_type:
            query = query.filter(Detection.source_type == source_type)
        if start_time:
            query = query.filter(Detection.timestamp >= start_time)
        if end_time:
            query = query.filter(Detection.timestamp <= end_time)
        if bbox:
            # Convert bbox string to WKT format: "min_lon,min_lat,max_lon,max_lat"
            try:
                min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(','))
                bbox_wkt = f'POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))'
                query = query.filter(Detection.location.ST_Intersects(bbox_wkt))
            except (ValueError, IndexError):
                raise HTTPException(status_code=400, detail="Invalid bbox format. Use 'min_lon,min_lat,max_lon,max_lat'")
        
        # Apply ordering and pagination
        query = query.order_by(Detection.timestamp.desc())
        total = query.count()
        detections = query.offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "count": len(detections),
            "offset": offset,
            "limit": limit,
            "items": detections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def create_detection(
    detection_data: dict = Body(...),
    db: Session = Depends(get_db)
):
    """
    Create a new detection
    """
    try:
        # Validate required fields
        required_fields = ['threat_type', 'confidence', 'location', 'source_type']
        for field in required_fields:
            if field not in detection_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create detection
        detection = Detection(
            threat_type=detection_data['threat_type'],
            confidence=detection_data['confidence'],
            timestamp=detection_data.get('timestamp', datetime.utcnow()),
            location=f"POINT({detection_data['location']['lng']} {detection_data['location']['lat']})",
            bbox=detection_data.get('bbox'),
            source_type=detection_data['source_type'],
            source_id=detection_data.get('source_id'),
            image_id=detection_data.get('image_id'),
            sensor_reading_id=detection_data.get('sensor_reading_id'),
            metadata=detection_data.get('metadata', {}),
            is_verified=detection_data.get('is_verified', False)
        )
        
        db.add(detection)
        db.commit()
        db.refresh(detection)
        
        return {"message": "Detection created successfully", "id": detection.id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_detection_stats(
    db: Session = Depends(get_db),
    time_range: str = Query("7d", description="Time range: 24h, 7d, 30d, 90d"),
    group_by: str = Query("day", description="Group by: hour, day, week, month")
):
    """
    Get detection statistics
    """
    try:
        # Calculate time range
        now = datetime.utcnow()
        time_ago = now
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            time_ago = now - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            time_ago = now - timedelta(days=days)
        else:
            raise HTTPException(status_code=400, detail="Invalid time_range. Use format like '24h' or '7d'")
        
        # Build group by clause
        if group_by == 'hour':
            time_format = "YYYY-MM-DD HH24:00:00"
            group_expr = "date_trunc('hour', timestamp)"
        elif group_by == 'day':
            time_format = "YYYY-MM-DD"
            group_expr = "date_trunc('day', timestamp)"
        elif group_by == 'week':
            time_format = "IYYY-IW"
            group_expr = "date_trunc('week', timestamp)"
        elif group_by == 'month':
            time_format = "YYYY-MM"
            group_expr = "date_trunc('month', timestamp)"
        else:
            raise HTTPException(status_code=400, detail="Invalid group_by. Must be one of: hour, day, week, month")
        
        # Execute raw SQL for better performance with time-based grouping
        sql = f"""
        SELECT 
            {group_expr} as time_period,
            threat_type,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM detections
        WHERE timestamp >= :time_ago
        GROUP BY time_period, threat_type
        ORDER BY time_period
        """
        
        result = db.execute(sql, {"time_ago": time_ago}).fetchall()
        
        # Format results
        stats = {}
        for row in result:
            time_str = row[0].strftime(time_format)
            if time_str not in stats:
                stats[time_str] = {}
            stats[time_str][row[1]] = {
                "count": row[2],
                "avg_confidence": float(row[3]) if row[3] else None
            }
        
        return {
            "time_range": {"start": time_ago.isoformat(), "end": now.isoformat()},
            "group_by": group_by,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
