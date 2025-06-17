"""Satellite data ingestion module for EarthSense."""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import rasterio
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    SHConfig
)
import geopandas as gpd
from shapely.geometry import box

from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class SatelliteIngestor:
    """Class for ingesting satellite imagery from Sentinel Hub."""
    
    def __init__(self):
        """Initialize the SatelliteIngestor with configuration."""
        self.config = SHConfig()
        if settings.SENTINEL_HUB_CLIENT_ID and settings.SENTINEL_HUB_CLIENT_SECRET:
            self.config.sh_client_id = settings.SENTINEL_HUB_CLIENT_ID
            self.config.sh_client_secret = settings.SENTINEL_HUB_CLIENT_SECRET
        else:
            logger.warning("Sentinel Hub credentials not found. Some functionality may be limited.")
    
    def get_bbox_from_geojson(self, geojson_path: str) -> BBox:
        """
        Create a bounding box from a GeoJSON file.
        
        Args:
            geojson_path: Path to the GeoJSON file
            
        Returns:
            BBox: Bounding box for the area of interest
        """
        gdf = gpd.read_file(geojson_path)
        bounds = gdf.total_bounds  # minx, miny, maxx, maxy
        return BBox(bbox=bounds, crs=CRS.WGS84)
    
    def get_time_interval(self, days_back: int = 30) -> Tuple[str, str]:
        """
        Get a time interval for satellite data query.
        
        Args:
            days_back: Number of days to look back from now
            
        Returns:
            Tuple of (start_date, end_date) in ISO format
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        return start_date.isoformat(), end_date.isoformat()
    
    def get_evalscript(self, bands: List[str] = None) -> str:
        """
        Generate an evalscript for Sentinel Hub request.
        
        Args:
            bands: List of bands to include in the output
            
        Returns:
            str: Evalscript for Sentinel Hub
        """
        if bands is None:
            bands = ["B02", "B03", "B04", "B08"]  # Default to RGB + NIR
            
        band_vars = ""
        for band in bands:
            band_vars += f"var {band} = B0{band[1]} || B8A;\n"
            
        return f"""
        //VERSION=3
        function setup() {{
            return {{
                input: [{{
                    bands: {str(bands + ['dataMask']).replace("'", '"')},
                    units: "DN"
                }}],
                output: [
                    {{ id: "default", bands: 4 }},
                    {{ id: "dataMask", bands: 1 }}
                ]
            }};
        }}
        
        function evaluatePixel(sample) {{
            return {{
                default: [{', '.join([f'sample.{band}' for band in bands])}],
                dataMask: [sample.dataMask]
            }};
        }}
        """
    
    def download_satellite_image(
        self,
        bbox: BBox,
        time_interval: Tuple[str, str],
        resolution: int = 10,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Download satellite imagery from Sentinel Hub.
        
        Args:
            bbox: Bounding box for the area of interest
            time_interval: Tuple of (start_date, end_date) in ISO format
            resolution: Resolution in meters per pixel
            output_path: Optional path to save the downloaded image
            **kwargs: Additional arguments for the request
            
        Returns:
            Optional[np.ndarray]: Downloaded image as a numpy array, or None if failed
        """
        try:
            # Calculate image dimensions based on bbox and resolution
            size = bbox_to_dimensions(bbox, resolution=resolution)
            
            # Create request
            evalscript = self.get_evalscript()
            
            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval,
                        mosaicking_order=MosaickingOrder.LEAST_CC,
                        **kwargs
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                    SentinelHubRequest.output_response('dataMask', MimeType.TIFF)
                ],
                bbox=bbox,
                size=size,
                config=self.config
            )
            
            # Download the data
            data = request.get_data()[0]  # Get the first (and only) time step
            
            # Save to file if output path is provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    width=size[0],
                    height=size[1],
                    count=4,  # RGB + NIR
                    dtype=data.dtype,
                    crs=bbox.crs.ogc_string(),
                    transform=rasterio.transform.from_origin(
                        bbox.min_x, bbox.max_y, resolution, resolution
                    ),
                ) as dst:
                    dst.write(data.transpose(2, 0, 1))  # Change from HWC to CHW
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading satellite image: {e}")
            return None
    
    def get_cloud_mask(self, bbox: BBox, time_interval: Tuple[str, str], resolution: int = 20) -> Optional[np.ndarray]:
        """
        Get cloud mask for the specified area and time.
        
        Args:
            bbox: Bounding box for the area of interest
            time_interval: Tuple of (start_date, end_date) in ISO format
            resolution: Resolution in meters per pixel
            
        Returns:
            Optional[np.ndarray]: Cloud mask as a numpy array, or None if failed
        """
        try:
            size = bbox_to_dimensions(bbox, resolution=resolution)
            
            cloud_mask_evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: [{
                        bands: ["B02", "B03", "B04", "B8A", "SCL"],
                        units: "DN"
                    }],
                    output: { id: "default", bands: 1, sampleType: "UINT8" }
                };
            }
            
            function evaluatePixel(sample) {
                // Class 3: cloud medium probability
                // Class 8: cloud high probability
                // Class 9: cloud shadows
                const cloudShadow = 3;
                const cloudHighProb = 9;
                const cloudMediumProb = 8;
                
                const isCloud = sample.SCL === cloudHighProb || 
                               sample.SCL === cloudMediumProb ||
                               sample.SCL === cloudShadow;
                
                return [isCloud ? 255 : 0];
            }
            """
            
            request = SentinelHubRequest(
                evalscript=cloud_mask_evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=time_interval
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=size,
                config=self.config
            )
            
            cloud_mask = request.get_data()[0]
            return cloud_mask
            
        except Exception as e:
            logger.error(f"Error getting cloud mask: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize the ingestor
    ingestor = SatelliteIngestor()
    
    # Define a bounding box (example: San Francisco area)
    bbox = BBox(bbox=[-122.52, 37.70, -122.36, 37.84], crs=CRS.WGS84)
    
    # Get time interval (last 30 days)
    time_interval = ingestor.get_time_interval(days_back=30)
    
    # Download the image
    image = ingestor.download_satellite_image(
        bbox=bbox,
        time_interval=time_interval,
        resolution=10,
        output_path="data/raw/satellite_image.tif"
    )
    
    if image is not None:
        print(f"Successfully downloaded image with shape: {image.shape}")
        
        # Get cloud mask
        cloud_mask = ingestor.get_cloud_mask(bbox, time_interval)
        if cloud_mask is not None:
            print(f"Cloud mask shape: {cloud_mask.shape}")
