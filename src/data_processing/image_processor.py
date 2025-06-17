"""Image processing module for EarthSense."""
import os
import logging
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.plot import reshape_as_raster, reshape_as_image
from skimage import exposure, filters, transform
from skimage.morphology import disk, opening, closing
import geopandas as gpd
from shapely.geometry import box
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from config.settings import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Class for processing and analyzing satellite and drone imagery."""
    
    def __init__(self):
        """Initialize the ImageProcessor with default settings."""
        self.default_resolution = 10  # meters per pixel
        self.default_bands = {
            'coastal': 1, 'blue': 2, 'green': 3, 'red': 4,
            'rededge1': 5, 'rededge2': 6, 'rededge3': 7,
            'nir': 8, 'nir08': 8, 'nir09': 9, 'swir16': 10, 'swir22': 11
        }
    
    def load_image(self, file_path: str) -> Optional[Dict]:
        """
        Load an image from a file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing the image data and metadata, or None if failed
        """
        try:
            with rasterio.open(file_path) as src:
                # Read all bands
                image = src.read()
                
                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'bounds': src.bounds,
                    'driver': src.driver,
                    'nodata': src.nodata
                }
                
                # Create a GeoDataFrame for the image footprint
                bounds = src.bounds
                geometry = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                gdf = gpd.GeoDataFrame(
                    {'geometry': [geometry]},
                    crs=src.crs
                )
                
                return {
                    'image': image,
                    'metadata': metadata,
                    'gdf': gdf
                }
                
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
        clip_limit: float = 0.03,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        Preprocess an image for analysis.
        
        Args:
            image: Input image as a numpy array (C, H, W) or (H, W, C)
            target_size: Optional target size (height, width) for resizing
            normalize: Whether to normalize pixel values to [0, 1]
            clip_limit: Clip limit for CLAHE (Contrast Limited Adaptive Histogram Equalization)
            kernel_size: Size of the kernel for median filtering
            
        Returns:
            Preprocessed image
        """
        try:
            # Handle different input shapes
            if len(image.shape) == 2:  # Single channel (H, W)
                image = np.expand_dims(image, axis=0)  # Add channel dimension
            elif len(image.shape) == 3 and image.shape[2] <= 4:  # (H, W, C) with C <= 4
                image = image.transpose(2, 0, 1)  # Convert to (C, H, W)
            
            # Convert to float32 if needed
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            # Apply preprocessing to each band
            processed_bands = []
            for band in image:
                # Remove noise with median filter
                denoised = cv2.medianBlur(band, kernel_size)
                
                # Apply CLAHE for contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                
                # Convert to uint8 for CLAHE
                band_min = np.min(denoised)
                band_max = np.max(denoised)
                if band_max > band_min:
                    normalized = ((denoised - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                else:
                    normalized = denoised.astype(np.uint8)
                
                enhanced = clahe.apply(normalized)
                
                # Convert back to float32
                enhanced = enhanced.astype(np.float32) / 255.0
                
                processed_bands.append(enhanced)
            
            # Stack bands back together
            processed_image = np.stack(processed_bands, axis=0)
            
            # Resize if target size is provided
            if target_size is not None:
                processed_image = np.stack([
                    transform.resize(
                        band,
                        target_size,
                        order=3,  # Bicubic interpolation
                        preserve_range=True,
                        anti_aliasing=True
                    )
                    for band in processed_image
                ])
            
            # Normalize to [0, 1] if requested
            if normalize:
                min_val = np.min(processed_image, axis=(1, 2), keepdims=True)
                max_val = np.max(processed_image, axis=(1, 2), keepdims=True)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1  # Avoid division by zero
                processed_image = (processed_image - min_val) / range_val
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image  # Return original if processing fails
    
    def calculate_ndvi(
        self,
        image: np.ndarray,
        red_band: int = 3,  # 1-based index for red band
        nir_band: int = 8   # 1-based index for NIR band
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI).
        
        Args:
            image: Input image as a numpy array (C, H, W)
            red_band: 1-based index of the red band
            nir_band: 1-based index of the near-infrared band
            
        Returns:
            NDVI array with values in range [-1, 1]
        """
        try:
            # Convert to 0-based indexing
            red = image[red_band - 1].astype(np.float32)
            nir = image[nir_band - 1].astype(np.float32)
            
            # Calculate NDVI
            ndvi = (nir - red) / (nir + red + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Clip to valid range
            return np.clip(ndvi, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating NDVI: {e}")
            return np.zeros_like(image[0])  # Return array of zeros with same shape as input height and width
    
    def calculate_ndwi(
        self,
        image: np.ndarray,
        green_band: int = 3,  # 1-based index for green band
        nir_band: int = 8     # 1-based index for NIR band
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index (NDWI).
        
        Args:
            image: Input image as a numpy array (C, H, W)
            green_band: 1-based index of the green band
            nir_band: 1-based index of the near-infrared band
            
        Returns:
            NDWI array with values in range [-1, 1]
        """
        try:
            # Convert to 0-based indexing
            green = image[green_band - 1].astype(np.float32)
            nir = image[nir_band - 1].astype(np.float32)
            
            # Calculate NDWI
            ndwi = (green - nir) / (green + nir + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Clip to valid range
            return np.clip(ndwi, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating NDWI: {e}")
            return np.zeros_like(image[0])
    
    def detect_anomalies(
        self,
        image: np.ndarray,
        method: str = 'isolation_forest',
        **kwargs
    ) -> np.ndarray:
        """
        Detect anomalies in an image.
        
        Args:
            image: Input image as a numpy array (C, H, W)
            method: Method to use for anomaly detection ('isolation_forest', 'autoencoder', 'one_class_svm')
            **kwargs: Additional arguments for the anomaly detection method
            
        Returns:
            Anomaly score map with values in range [0, 1]
        """
        try:
            # Reshape image to 2D array (n_samples, n_features)
            n_bands, height, width = image.shape
            X = image.reshape(n_bands, -1).T  # Shape: (height * width, n_bands)
            
            if method == 'isolation_forest':
                from sklearn.ensemble import IsolationForest
                
                # Initialize and fit the model
                model = IsolationForest(
                    n_estimators=kwargs.get('n_estimators', 100),
                    contamination=kwargs.get('contamination', 0.01),
                    random_state=42
                )
                
                # Predict anomaly scores
                scores = -model.fit_predict(X)  # Convert to 0 (normal) and 1 (anomaly)
                
            elif method == 'autoencoder':
                from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
                from tensorflow.keras.models import Model
                
                # Reshape for convolutional layers
                X_reshaped = X.reshape(-1, 1, 1, n_bands)
                
                # Define autoencoder architecture
                input_img = Input(shape=(1, 1, n_bands))
                x = Flatten()(input_img)
                encoded = Dense(32, activation='relu')(x)
                decoded = Dense(n_bands, activation='sigmoid')(encoded)
                
                # Create and compile the model
                autoencoder = Model(input_img, decoded)
                autoencoder.compile(optimizer='adam', loss='mse')
                
                # Train the autoencoder
                autoencoder.fit(
                    X_reshaped, X_reshaped,
                    epochs=kwargs.get('epochs', 10),
                    batch_size=kwargs.get('batch_size', 32),
                    shuffle=True,
                    verbose=0
                )
                
                # Calculate reconstruction error
                X_pred = autoencoder.predict(X_reshaped, verbose=0)
                scores = np.mean(np.power(X_reshaped - X_pred, 2), axis=(1, 2, 3))
                
                # Normalize scores to [0, 1]
                scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)
                
            elif method == 'one_class_svm':
                from sklearn.svm import OneClassSVM
                from sklearn.preprocessing import StandardScaler
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Initialize and fit the model
                model = OneClassSVM(
                    nu=kwargs.get('nu', 0.01),
                    kernel=kwargs.get('kernel', 'rbf'),
                    gamma=kwargs.get('gamma', 'scale')
                )
                
                # Predict anomaly scores
                scores = -model.fit_predict(X_scaled)  # Convert to 0 (normal) and 1 (anomaly)
                
            else:
                raise ValueError(f"Unsupported anomaly detection method: {method}")
            
            # Reshape scores back to image dimensions
            anomaly_map = scores.reshape(height, width)
            
            # Apply Gaussian smoothing to reduce noise
            anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)
            
            return anomaly_map
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return np.zeros((image.shape[1], image.shape[2]))
    
    def segment_image(
        self,
        image: np.ndarray,
        method: str = 'kmeans',
        n_clusters: int = 5,
        **kwargs
    ) -> np.ndarray:
        """
        Segment an image into regions.
        
        Args:
            image: Input image as a numpy array (C, H, W)
            method: Segmentation method ('kmeans', 'meanshift', 'slic')
            n_clusters: Number of clusters/segments
            **kwargs: Additional arguments for the segmentation method
            
        Returns:
            Label map with segment IDs
        """
        try:
            # Reshape image to 2D array (n_samples, n_features)
            n_bands, height, width = image.shape
            X = image.reshape(n_bands, -1).T  # Shape: (height * width, n_bands)
            
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                
                # Initialize and fit KMeans
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    n_init=10
                )
                labels = kmeans.fit_predict(X)
                
            elif method == 'meanshift':
                from sklearn.cluster import MeanShift
                
                # Initialize and fit MeanShift
                bandwidth = kwargs.get('bandwidth', 0.5)
                ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)
                labels = ms.fit_predict(X)
                
            elif method == 'slic':
                from skimage.segmentation import slic
                
                # Convert to (H, W, C) for skimage
                img_rgb = image[[2, 1, 0], :, :].transpose(1, 2, 0)  # Assuming RGB order
                
                # Apply SLIC superpixels
                segments = slic(
                    img_rgb,
                    n_segments=n_clusters,
                    compactness=kwargs.get('compactness', 10),
                    sigma=kwargs.get('sigma', 1.0),
                    start_label=0
                )
                
                # Flatten and return
                return segments
                
            else:
                raise ValueError(f"Unsupported segmentation method: {method}")
            
            # Reshape labels back to image dimensions
            label_map = labels.reshape(height, width)
            
            return label_map
            
        except Exception as e:
            logger.error(f"Error segmenting image: {e}")
            return np.zeros((height, width), dtype=np.int32)
    
    def extract_features(
        self,
        image: np.ndarray,
        method: str = 'efficientnet',
        **kwargs
    ) -> np.ndarray:
        """
        Extract deep features from an image.
        
        Args:
            image: Input image as a numpy array (C, H, W)
            method: Feature extraction method ('efficientnet', 'resnet', 'vgg')
            **kwargs: Additional arguments for the feature extraction method
            
        Returns:
            Feature vector
        """
        try:
            # Ensure image has 3 channels (for pretrained models)
            if image.shape[0] == 1:  # Single channel
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] > 3:  # More than 3 channels (e.g., multispectral)
                # Take first 3 channels or use PCA to reduce to 3
                image = image[:3, :, :]
            
            # Resize to expected input size
            target_size = (224, 224)  # Default for most pretrained models
            resized_image = np.stack([
                transform.resize(
                    band,
                    target_size,
                    order=3,  # Bicubic interpolation
                    preserve_range=True,
                    anti_aliasing=True
                )
                for band in image
            ])
            
            # Normalize to [0, 1] if needed
            if np.max(resized_image) > 1.0:
                resized_image = resized_image / 255.0
            
            # Add batch dimension
            batch_image = np.expand_dims(resized_image.transpose(1, 2, 0), axis=0)
            
            if method == 'efficientnet':
                # Load pretrained EfficientNet without top layers
                base_model = EfficientNetB0(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Add custom layers
                x = base_model(batch_image)
                x = GlobalAveragePooling2D()(x)
                features = x.numpy().flatten()
                
            elif method == 'resnet':
                from tensorflow.keras.applications import ResNet50
                
                # Load pretrained ResNet50 without top layers
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Add custom layers
                x = base_model(batch_image)
                x = GlobalAveragePooling2D()(x)
                features = x.numpy().flatten()
                
            elif method == 'vgg':
                from tensorflow.keras.applications import VGG16
                
                # Load pretrained VGG16 without top layers
                base_model = VGG16(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(224, 224, 3)
                )
                
                # Add custom layers
                x = base_model(batch_image)
                x = GlobalAveragePooling2D()(x)
                features = x.numpy().flatten()
                
            else:
                raise ValueError(f"Unsupported feature extraction method: {method}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(512)  # Return zeros with default size

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = ImageProcessor()
    
    # Example: Load and process an image
    image_data = processor.load_image("path/to/your/image.tif")
    if image_data:
        print(f"Loaded image with shape: {image_data['image'].shape}")
        
        # Preprocess the image
        processed_image = processor.preprocess_image(
            image_data['image'],
            target_size=(256, 256),
            normalize=True
        )
        print(f"Processed image shape: {processed_image.shape}")
        
        # Calculate NDVI
        ndvi = processor.calculate_ndvi(processed_image)
        print(f"NDVI shape: {ndvi.shape}, min: {np.min(ndvi):.2f}, max: {np.max(ndvi):.2f}")
        
        # Detect anomalies
        anomaly_map = processor.detect_anomalies(processed_image, method='isolation_forest')
        print(f"Anomaly map shape: {anomaly_map.shape}")
        
        # Segment the image
        segments = processor.segment_image(processed_image, method='kmeans', n_clusters=5)
        print(f"Segmentation result shape: {segments.shape}, {len(np.unique(segments))} segments")
        
        # Extract features
        features = processor.extract_features(processed_image, method='efficientnet')
        print(f"Extracted features shape: {features.shape}")
