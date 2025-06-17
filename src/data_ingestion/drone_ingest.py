"""Drone data ingestion module for EarthSense."""
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
import cv2
from PIL import Image
import pyproj
from shapely.geometry import Polygon, box
import geopandas as gpd
import pandas as pd

from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class DroneDataIngestor:
    """Class for ingesting and processing drone data including LIDAR and thermal imagery."""
    
    def __init__(self):
        """Initialize the DroneDataIngestor with default settings."""
        self.supported_lidar_formats = ['.las', '.laz']
        self.supported_thermal_formats = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
        self.output_crs = "EPSG:4326"  # WGS84
    
    def load_lidar_data(self, file_path: str) -> Optional[Dict]:
        """
        Load LIDAR data from a LAS/LAZ file.
        
        Args:
            file_path: Path to the LIDAR file
            
        Returns:
            Dictionary containing point cloud data and metadata, or None if failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"LIDAR file not found: {file_path}")
                return None
                
            if file_path.suffix.lower() not in self.supported_lidar_formats:
                logger.error(f"Unsupported LIDAR format: {file_path.suffix}")
                return None
            
            # Read the LAS/LAZ file
            with laspy.open(file_path) as las_file:
                las = las_file.read()
                
                # Extract point data
                points = np.vstack((
                    las.x,
                    las.y,
                    las.z,
                    las.intensity if hasattr(las, 'intensity') else np.zeros_like(las.x),
                    las.classification if hasattr(las, 'classification') else np.zeros_like(las.x, dtype=np.uint8)
                )).T
                
                # Create a GeoDataFrame
                gdf = gpd.GeoDataFrame(
                    {
                        'intensity': points[:, 3],
                        'classification': points[:, 4].astype(np.uint8),
                        'geometry': gpd.points_from_xy(points[:, 0], points[:, 1], points[:, 2])
                    },
                    crs=f"EPSG:{las.header.parse_crs().to_epsg()}" if las.header.parse_crs() else None
                )
                
                # If the CRS is not defined, we need to assume it's in the same CRS as the output
                if gdf.crs is None:
                    logger.warning("No CRS information found in LIDAR file. Assuming output CRS.")
                    gdf.crs = self.output_crs
                
                # Transform to output CRS if needed
                if gdf.crs != self.output_crs:
                    gdf = gdf.to_crs(self.output_crs)
                
                # Extract metadata
                metadata = {
                    'file_path': str(file_path),
                    'point_count': len(points),
                    'crs': str(gdf.crs),
                    'bounds': gdf.total_bounds.tolist(),
                    'min_z': float(np.min(points[:, 2])),
                    'max_z': float(np.max(points[:, 2])),
                    'intensity_range': (float(np.min(points[:, 3])), float(np.max(points[:, 3]))),
                    'classification_codes': np.unique(points[:, 4]).tolist(),
                    'acquisition_date': self._extract_lidar_metadata(las),
                    'original_crs': f"EPSG:{las.header.parse_crs().to_epsg()}" if las.header.parse_crs() else None
                }
                
                return {
                    'points': points,
                    'gdf': gdf,
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Error loading LIDAR data: {e}")
            return None
    
    def _extract_lidar_metadata(self, las) -> Optional[str]:
        """Extract metadata from LAS file."""
        try:
            # Try to get acquisition date from global encoding or file creation date
            if hasattr(las.header, 'creation_date'):
                return las.header.creation_date.strftime('%Y-%m-%d')
            return None
        except:
            return None
    
    def load_thermal_image(self, file_path: str) -> Optional[Dict]:
        """
        Load thermal image data from a file.
        
        Args:
            file_path: Path to the thermal image file
            
        Returns:
            Dictionary containing thermal image data and metadata, or None if failed
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Thermal image not found: {file_path}")
                return None
                
            if file_path.suffix.lower() not in self.supported_thermal_formats:
                logger.error(f"Unsupported thermal image format: {file_path.suffix}")
                return None
            
            # Read the image
            if file_path.suffix.lower() in ['.tif', '.tiff']:
                # For GeoTIFF, use rasterio to preserve georeferencing
                with rasterio.open(file_path) as src:
                    # Read the first band (assuming single-band thermal data)
                    image = src.read(1)
                    
                    # Get georeferencing info
                    transform = src.transform
                    crs = src.crs
                    
                    # Create a polygon representing the image bounds
                    height, width = image.shape
                    corners = [
                        transform * (0, 0),
                        transform * (width, 0),
                        transform * (width, height),
                        transform * (0, height)
                    ]
                    bounds_polygon = Polygon(corners)
                    
            else:
                # For other formats, use OpenCV/PIL (no georeferencing)
                image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                if image is None:
                    # Try with PIL if OpenCV fails
                    pil_image = Image.open(file_path)
                    image = np.array(pil_image)
                
                # Create a dummy transform and CRS
                transform = from_origin(0, 0, 1, 1)  # Default 1m/pixel
                crs = None
                bounds_polygon = box(0, 0, image.shape[1], image.shape[0])
            
            # Create a GeoDataFrame for the image footprint
            gdf = gpd.GeoDataFrame(
                {'geometry': [bounds_polygon]},
                crs=crs if crs else self.output_crs
            )
            
            # If the CRS is not defined, use the output CRS
            if gdf.crs is None:
                logger.warning("No CRS information found in thermal image. Assuming output CRS.")
                gdf.crs = self.output_crs
            
            # Transform to output CRS if needed
            if gdf.crs != self.output_crs:
                gdf = gdf.to_crs(self.output_crs)
            
            # Extract metadata
            metadata = {
                'file_path': str(file_path),
                'width': image.shape[1],
                'height': image.shape[0],
                'dtype': str(image.dtype),
                'crs': str(gdf.crs),
                'bounds': gdf.total_bounds.tolist(),
                'min_temp': float(np.min(image)),
                'max_temp': float(np.max(image)),
                'mean_temp': float(np.mean(image)),
                'acquisition_date': self._extract_thermal_metadata(file_path)
            }
            
            return {
                'image': image,
                'transform': transform,
                'gdf': gdf,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading thermal image: {e}")
            return None
    
    def _extract_thermal_metadata(self, file_path: Path) -> Optional[str]:
        """Extract metadata from thermal image file."""
        try:
            # Try to get creation date from file metadata
            creation_time = os.path.getmtime(file_path)
            return datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d')
        except:
            return None
    
    def process_drone_flight(
        self,
        lidar_path: Optional[str] = None,
        thermal_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process a drone flight with both LIDAR and thermal data.
        
        Args:
            lidar_path: Path to LIDAR file
            thermal_path: Path to thermal image file
            output_dir: Directory to save processed outputs
            
        Returns:
            Dictionary containing processed data and metadata
        """
        result = {
            'lidar': None,
            'thermal': None,
            'metadata': {}
        }
        
        # Process LIDAR data if provided
        if lidar_path:
            lidar_data = self.load_lidar_data(lidar_path)
            if lidar_data:
                result['lidar'] = lidar_data
                result['metadata'].update({
                    'lidar_metadata': lidar_data['metadata']
                })
        
        # Process thermal data if provided
        if thermal_path:
            thermal_data = self.load_thermal_image(thermal_path)
            if thermal_data:
                result['thermal'] = thermal_data
                result['metadata'].update({
                    'thermal_metadata': thermal_data['metadata']
                })
        
        # Save outputs if output directory is provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save LIDAR data as GeoPackage
            if result['lidar']:
                output_path = output_dir / 'lidar_points.gpkg'
                result['lidar']['gdf'].to_file(output_path, driver='GPKG')
                result['metadata']['lidar_output_path'] = str(output_path)
            
            # Save thermal data as GeoTIFF
            if result['thermal']:
                output_path = output_dir / 'thermal_image.tif'
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=result['thermal']['image'].shape[0],
                    width=result['thermal']['image'].shape[1],
                    count=1,
                    dtype=result['thermal']['image'].dtype,
                    crs=result['thermal']['gdf'].crs,
                    transform=result['thermal']['transform'],
                ) as dst:
                    dst.write(result['thermal']['image'], 1)
                result['metadata']['thermal_output_path'] = str(output_path)
            
            # Save metadata as JSON
            import json
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(result['metadata'], f, indent=2)
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize the ingestor
    ingestor = DroneDataIngestor()
    
    # Example file paths (replace with actual paths)
    lidar_path = "path/to/your/lidar_file.las"
    thermal_path = "path/to/your/thermal_image.tif"
    output_dir = "data/processed/drone_data"
    
    # Process the drone data
    result = ingestor.process_drone_flight(
        lidar_path=lidar_path,
        thermal_path=thermal_path,
        output_dir=output_dir
    )
    
    if result['lidar']:
        print(f"Processed LIDAR data with {len(result['lidar']['points'])} points")
    if result['thermal']:
        print(f"Processed thermal image with shape {result['thermal']['image'].shape}")
