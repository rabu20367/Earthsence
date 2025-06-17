import os
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SatelliteImageDataset(Dataset):
    """
    Dataset for loading and processing satellite/drone images for environmental monitoring.
    Supports both RGB and multi-spectral imagery.
    """
    
    def __init__(
        self, 
        root_dir: Union[str, Path],
        split: str = 'train',
        bands: List[str] = None,
        transform: Optional[A.Compose] = None,
        target_transform=None,
        normalize: bool = True,
        max_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            bands: List of band names to load (e.g., ['B04', 'B03', 'B02'] for RGB)
            transform: Albumentations transform pipeline
            target_transform: Transform to apply to the target
            normalize: Whether to normalize the images
            max_samples: Maximum number of samples to load (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.bands = bands or ['B04', 'B03', 'B02']  # Default to RGB
        self.normalize = normalize
        self.max_samples = max_samples
        
        # Set up transforms
        self.transform = transform
        self.target_transform = target_transform
        
        # Load dataset metadata
        self.samples = self._load_metadata()
        
        # If no transform is provided, use a default one
        if self.transform is None:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet stats
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=1.0
                ),
                ToTensorV2()
            ])
    
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load dataset metadata.
        
        Returns:
            List of dictionaries containing metadata for each sample
        """
        # This is a placeholder implementation
        # In a real scenario, you would load this from a CSV, JSON, or other metadata file
        samples = []
        
        # Example: Look for all TIFF files in the directory
        image_files = list(self.root_dir.glob(f'**/*.tif'))
        
        for img_path in image_files:
            # Skip files that don't match the expected pattern
            if not self._is_valid_image(img_path):
                continue
                
            # Get corresponding label/mask path if it exists
            label_path = self._get_label_path(img_path)
            
            samples.append({
                'image_path': str(img_path),
                'label_path': str(label_path) if label_path else None,
                'image_id': img_path.stem
            })
            
            # Early stopping if max_samples is set
            if self.max_samples and len(samples) >= self.max_samples:
                break
        
        return samples
    
    def _is_valid_image(self, image_path: Path) -> bool:
        """Check if the image file is valid and should be included in the dataset."""
        # Check file extension
        if image_path.suffix.lower() not in ['.tif', '.tiff', '.jp2']:
            return False
            
        # Check if all required bands are present
        # This is a simplified check - in practice, you'd want to verify the actual bands
        return True
    
    def _get_label_path(self, image_path: Path) -> Optional[Path]:
        """Get the path to the corresponding label/mask for an image."""
        # This is a placeholder implementation
        # In a real scenario, you would implement logic to find the corresponding label file
        label_path = image_path.parent / 'labels' / f'{image_path.stem}_mask.tif'
        return label_path if label_path.exists() else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the image and its metadata
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load label if available
        target = None
        if sample['label_path'] and os.path.exists(sample['label_path']):
            target = self._load_label(sample['label_path'])
        
        # Apply transforms
        if self.transform is not None:
            if target is not None:
                # For segmentation tasks
                transformed = self.transform(image=image, mask=target)
                image = transformed['image']
                target = transformed['mask']
            else:
                # For classification tasks
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Apply target transform if provided
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        
        # Prepare output
        output = {
            'image': image,
            'image_id': sample['image_id'],
            'image_path': sample['image_path']
        }
        
        if target is not None:
            output['target'] = target
            
        return output
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            NumPy array containing the image data
        """
        with rasterio.open(image_path) as src:
            # Read all bands if not specified, otherwise read selected bands
            if self.bands is None:
                image = src.read()
            else:
                # Get band indices for the requested bands
                band_indices = [i+1 for i, desc in enumerate(src.descriptions) 
                              if desc in self.bands]
                if not band_indices:
                    # If no matching bands found, read first 3 bands as fallback
                    band_indices = [1, 2, 3]
                image = src.read(band_indices)
            
            # Convert to float32 and normalize to [0, 1]
            image = image.astype(np.float32)
            
            # Handle no-data values
            if src.nodata is not None:
                image[image == src.nodata] = np.nan
            
            # Clip and normalize
            if self.normalize:
                # Clip to 2nd and 98th percentiles to remove outliers
                p2, p98 = np.percentile(image, (2, 98))
                image = np.clip(image, p2, p98)
                
                # Normalize to [0, 1]
                min_val = np.nanmin(image)
                max_val = np.nanmax(image)
                if max_val > min_val:  # Avoid division by zero
                    image = (image - min_val) / (max_val - min_val)
                
                # Fill any remaining NaNs with 0
                image = np.nan_to_num(image, nan=0.0)
            
            # Transpose to (H, W, C) for compatibility with albumentations
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            
            return image
    
    def _load_label(self, label_path: str) -> np.ndarray:
        """
        Load a label/mask from disk.
        
        Args:
            label_path: Path to the label file
            
        Returns:
            NumPy array containing the label data
        """
        with rasterio.open(label_path) as src:
            label = src.read(1)  # Read first band for single-channel masks
            
            # Convert to int64 for PyTorch compatibility
            return label.astype(np.int64)

def get_transforms(
    split: str = 'train',
    img_size: Tuple[int, int] = (256, 256),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> A.Compose:
    """
    Get data augmentation transforms for the specified split.
    
    Args:
        split: Dataset split ('train', 'val', 'test')
        img_size: Target image size (height, width)
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Albumentations Compose object with the transforms
    """
    if split == 'train':
        return A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2()
        ])
    else:  # val/test
        return A.Compose([
            A.Resize(*img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2()
        ])

def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: Tuple[int, int] = (256, 256),
    bands: List[str] = None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Root directory containing the dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        img_size: Target image size (height, width)
        bands: List of band names to load
        
    Returns:
        Dictionary containing the data loaders
    """
    data_dir = Path(data_dir)
    
    # Define transforms
    train_transforms = get_transforms('train', img_size)
    val_transforms = get_transforms('val', img_size)
    test_transforms = get_transforms('test', img_size)
    
    # Create datasets
    train_dataset = SatelliteImageDataset(
        root_dir=data_dir / 'train',
        transform=train_transforms,
        bands=bands,
        **kwargs
    )
    
    val_dataset = SatelliteImageDataset(
        root_dir=data_dir / 'val',
        transform=val_transforms,
        bands=bands,
        **kwargs
    )
    
    test_dataset = SatelliteImageDataset(
        root_dir=data_dir / 'test',
        transform=test_transforms,
        bands=bands,
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': train_dataset.num_classes if hasattr(train_dataset, 'num_classes') else None
    }
