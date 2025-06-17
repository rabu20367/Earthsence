import os
import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd
from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
from typing import Dict, List, Tuple, Optional, Union, Any
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import warnings

class GeoProcessor:
    """A utility class for processing geospatial data."""
    
    def __init__(self, default_crs: str = 'EPSG:4326'):
        self.default_crs = default_crs
    
    def read_vector_file(self, file_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """Read a vector file and return a GeoDataFrame."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Vector file not found: {file_path}")
            
        try:
            gdf = gpd.read_file(file_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(self.default_crs)
            return gdf
        except Exception as e:
            raise ValueError(f"Failed to read vector file: {str(e)}")
    
    def read_raster_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Read a raster file and return its data and metadata."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Raster file not found: {file_path}")
            
        try:
            with rasterio.open(file_path) as src:
                data = src.read()
                return {
                    'data': data,
                    'crs': src.crs or self.default_crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'count': src.count,
                    'width': src.width,
                    'height': src.height,
                    'nodata': src.nodata
                }
        except Exception as e:
            raise ValueError(f"Failed to read raster file: {str(e)}")
    
    def clip_raster(self, 
                   raster_data: np.ndarray,
                   transform: List[float],
                   geometry: Union[Polygon, MultiPolygon],
                   nodata: float = None) -> Tuple[np.ndarray, List[float]]:
        """Clip a raster using a polygon geometry."""
        try:
            out_image, out_transform = rasterio.mask.mask(
                {'driver': 'GTiff', 'height': raster_data.shape[1], 
                 'width': raster_data.shape[2], 'transform': transform},
                [mapping(geometry)],
                crop=True,
                nodata=nodata
            )
            return out_image, out_transform
        except Exception as e:
            raise ValueError(f"Failed to clip raster: {str(e)}")
    
    def calculate_ndvi(self, 
                      red_band: np.ndarray, 
                      nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Vegetation Index."""
        red = red_band.astype(float)
        nir = nir_band.astype(float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red + 1e-10)
            ndvi = np.nan_to_num(ndvi, nan=-1.0, posinf=1.0, neginf=-1.0)
            return np.clip(ndvi, -1.0, 1.0)

    def calculate_ndwi(self, 
                      green_band: np.ndarray,
                      nir_band: np.ndarray) -> np.ndarray:
        """Calculate Normalized Difference Water Index."""
        green = green_band.astype(float)
        nir = nir_band.astype(float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir + 1e-10)
            ndwi = np.nan_to_num(ndwi, nan=-1.0, posinf=1.0, neginf=-1.0)
            return np.clip(ndwi, -1.0, 1.0)

# Example usage:
if __name__ == "__main__":
    # Initialize the processor
    processor = GeoProcessor()
    
    # Example: Calculate NDVI from red and NIR bands
    try:
        # Load your raster bands here
        # red_band = ...
        # nir_band = ...
        # ndvi = processor.calculate_ndvi(red_band, nir_band)
        pass
    except Exception as e:
        print(f"Error: {str(e)}")
