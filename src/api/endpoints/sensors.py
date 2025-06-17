from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json

from models.base import get_db
from models.data_models import SensorReading, DataSource, DataSourceType
from config.config import config

router = APIRouter()

@router.get("/readings")
async def list_sensor_readings(
    db: Session = Depends(get_db),
    source_id: Optional[int] = None,
    sensor_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    offset: int = 0
):
    """
    List sensor readings with optional filters
    """
    try:
        query = db.query(SensorReading)
        
        # Apply filters
        if source_id is not None:
            query = query.filter(SensorReading.source_id == source_id)
        if sensor_type:
            query = query.filter(SensorReading.metadata['type'].astext == sensor_type)
        if start_time:
            query = query.filter(SensorReading.timestamp >= start_time)
        if end_time:
            query = query.filter(SensorReading.timestamp <= end_time)
        
        # Apply ordering and pagination
        query = query.order_by(SensorReading.timestamp.desc())
        total = query.count()
        readings = query.offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "count": len(readings),
            "offset": offset,
            "limit": limit,
            "items": readings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_sensor_stats(
    db: Session = Depends(get_db),
    source_id: Optional[int] = None,
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    aggregation: str = Query("hour", description="Aggregation: minute, hour, day")
):
    """
    Get aggregated sensor statistics
    """
    try:
        # Calculate time range
        now = datetime.utcnow()
        
        if time_range.endswith('h'):
            time_ago = now - timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith('d'):
            time_ago = now - timedelta(days=int(time_range[:-1]))
        else:
            time_ago = now - timedelta(hours=24)  # Default to 24 hours
        
        # Build aggregation SQL
        if aggregation == 'minute':
            time_expr = "date_trunc('minute', timestamp)"
        elif aggregation == 'hour':
            time_expr = "date_trunc('hour', timestamp)"
        elif aggregation == 'day':
            time_expr = "date_trunc('day', timestamp)"
        else:
            raise HTTPException(status_code=400, detail="Invalid aggregation. Must be one of: minute, hour, day")
        
        # Base query
        sql = f"""
        SELECT 
            {time_expr} as time_period,
            source_id,
            AVG(temperature) as avg_temperature,
            AVG(humidity) as avg_humidity,
            COUNT(*) as reading_count
        FROM sensor_readings
        WHERE timestamp >= :time_ago
        """
        
        params = {"time_ago": time_ago}
        
        # Add source filter if provided
        if source_id is not None:
            sql += " AND source_id = :source_id"
            params["source_id"] = source_id
        
        # Add grouping
        sql += f"""
        GROUP BY time_period, source_id
        ORDER BY time_period
        """
        
        # Execute query
        result = db.execute(sql, params).fetchall()
        
        # Format results
        stats = []
        for row in result:
            stats.append({
                "timestamp": row[0].isoformat(),
                "source_id": row[1],
                "avg_temperature": float(row[2]) if row[2] is not None else None,
                "avg_humidity": float(row[3]) if row[3] is not None else None,
                "reading_count": row[4]
            })
        
        return {
            "time_range": {"start": time_ago.isoformat(), "end": now.isoformat()},
            "aggregation": aggregation,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sources")
async def list_sensor_sources(
    db: Session = Depends(get_db),
    type: Optional[DataSourceType] = None,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all sensor data sources
    """
    try:
        query = db.query(DataSource)
        
        # Apply filters
        if type is not None:
            query = query.filter(DataSource.type == type)
        if is_active is not None:
            query = query.filter(DataSource.is_active == is_active)
        
        # Apply ordering and pagination
        query = query.order_by(DataSource.name)
        total = query.count()
        sources = query.offset(offset).limit(limit).all()
        
        return {
            "total": total,
            "count": len(sources),
            "offset": offset,
            "limit": limit,
            "items": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sources")
async def create_sensor_source(
    source_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Create a new sensor data source
    """
    try:
        # Validate required fields
        required_fields = ['name', 'type']
        for field in required_fields:
            if field not in source_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Create source
        source = DataSource(
            name=source_data['name'],
            type=source_data['type'],
            description=source_data.get('description'),
            location=source_data.get('location'),
            metadata=source_data.get('metadata', {}),
            is_active=source_data.get('is_active', True)
        )
        
        db.add(source)
        db.commit()
        db.refresh(source)
        
        return {"message": "Sensor source created successfully", "id": source.id}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
