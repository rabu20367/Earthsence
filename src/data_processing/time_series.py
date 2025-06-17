import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import signal
from scipy.interpolate import interp1d
import pywt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class TimeSeriesProcessor:
    """
    A utility class for processing and analyzing time series data from environmental sensors.
    Supports operations like resampling, filtering, feature extraction, and anomaly detection.
    """
    
    def __init__(self, 
                 freq: str = '1H',
                 fill_method: str = 'linear',
                 window_size: int = 24,
                 **kwargs):
        """
        Initialize the TimeSeriesProcessor.
        
        Args:
            freq: Target frequency for resampling (pandas frequency string)
            fill_method: Method for filling missing values ('linear', 'ffill', 'bfill', 'mean', etc.)
            window_size: Size of the sliding window for feature extraction
        """
        self.freq = freq
        self.fill_method = fill_method
        self.window_size = window_size
        self.scaler = None
        self.feature_columns = None
        self.target_columns = None
    
    def preprocess_dataframe(self, 
                           df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp',
                           value_cols: List[str] = None) -> pd.DataFrame:
        """
        Preprocess a DataFrame containing time series data.
        
        Args:
            df: Input DataFrame with time series data
            timestamp_col: Name of the timestamp column
            value_cols: List of column names containing time series values
            
        Returns:
            Processed DataFrame with resampled and cleaned time series
        """
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
            
        # Make a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Convert timestamp to datetime and set as index
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()
        
        # If value_cols is not specified, use all columns except the timestamp
        if value_cols is None:
            value_cols = [col for col in df.columns if col != timestamp_col]
        
        # Resample the data to the target frequency
        resampled = df[value_cols].resample(self.freq)
        
        # Apply the specified fill method
        if self.fill_method == 'linear':
            processed = resampled.interpolate(method='linear')
        elif self.fill_method in ['ffill', 'bfill', 'pad']:
            processed = resampled.fillna(method=self.fill_method)
        elif self.fill_method == 'mean':
            processed = resampled.mean()
        else:
            processed = resampled.asfreq()
        
        # Forward fill any remaining NaNs at the beginning
        processed = processed.ffill().bfill()
        
        return processed
    
    def extract_features(self, 
                        series: Union[pd.Series, np.ndarray], 
                        window_size: int = None) -> Dict[str, float]:
        """
        Extract time-series features from a window of data.
        
        Args:
            series: Input time series data
            window_size: Size of the sliding window
            
        Returns:
            Dictionary of extracted features
        """
        if window_size is None:
            window_size = self.window_size
            
        if len(series) < window_size:
            warnings.warn(f"Series length ({len(series)}) is less than window size ({window_size})")
            window_size = len(series)
        
        # Convert to numpy array if needed
        if isinstance(series, pd.Series):
            values = series.values[-window_size:]
        else:
            values = np.asarray(series)[-window_size:]
        
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(values)
        features['std'] = np.std(values, ddof=1)
        features['min'] = np.min(values)
        features['max'] = np.max(values)
        features['median'] = np.median(values)
        features['q25'] = np.percentile(values, 25)
        features['q75'] = np.percentile(values, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Trend and seasonality
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        features['trend_slope'] = coeffs[0]
        features['trend_intercept'] = coeffs[1]
        
        # Wavelet transform for frequency analysis
        try:
            coeffs = pywt.wavedec(values, 'db1', level=min(3, pywt.dwt_max_level(len(values), 'db1')))
            features['wavelet_energy'] = sum(np.sum(c**2) for c in coeffs)
        except:
            features['wavelet_energy'] = np.nan
        
        # Spectral features
        f, Pxx = signal.welch(values, nperseg=min(len(values), 256))
        if len(Pxx) > 0:
            features['spectral_energy'] = np.sum(Pxx)
            features['dominant_freq'] = f[np.argmax(Pxx)]
        else:
            features['spectral_energy'] = np.nan
            features['dominant_freq'] = np.nan
        
        # Time-domain features
        features['zero_crossings'] = len(np.where(np.diff(np.signbit(values)))[0])
        features['autocorr_lag1'] = pd.Series(values).autocorr(lag=1) if len(values) > 1 else np.nan
        
        # Entropy
        hist, _ = np.histogram(values, bins=min(10, len(values)//2))
        hist = hist / hist.sum()
        features['entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        return features
    
    def create_sequences(self, 
                        data: Union[pd.DataFrame, np.ndarray], 
                        seq_length: int, 
                        target_col: str = None,
                        target_length: int = 1,
                        step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            data: Input time series data
            seq_length: Length of input sequences
            target_col: Name of the target column (if data is a DataFrame)
            target_length: Length of target sequences
            step: Step size between sequences
            
        Returns:
            Tuple of (sequences, targets) as numpy arrays
        """
        if isinstance(data, pd.DataFrame):
            if target_col is None:
                values = data.values
            else:
                values = data[target_col].values
        else:
            values = np.asarray(data)
        
        sequences = []
        targets = []
        
        for i in range(0, len(values) - seq_length - target_length + 1, step):
            sequences.append(values[i:i + seq_length])
            targets.append(values[i + seq_length:i + seq_length + target_length])
        
        return np.array(sequences), np.array(targets)
    
    def detect_anomalies(self, 
                        values: np.ndarray, 
                        method: str = 'zscore',
                        threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies in a time series.
        
        Args:
            values: Input time series
            method: Detection method ('zscore', 'iqr', 'ewma')
            threshold: Threshold for anomaly detection
            
        Returns:
            Boolean array indicating anomalies
        """
        if method == 'zscore':
            z_scores = (values - np.mean(values)) / (np.std(values) + 1e-10)
            return np.abs(z_scores) > threshold
        
        elif method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (values < lower_bound) | (values > upper_bound)
        
        elif method == 'ewma':
            # Exponentially weighted moving average
            ewma = pd.Series(values).ewm(span=30).mean().values
            std = pd.Series(values).ewm(span=30).std().values
            return np.abs(values - ewma) > (threshold * std)
        
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    Supports both forecasting and classification tasks.
    """
    
    def __init__(self, 
                 data: Union[pd.DataFrame, np.ndarray],
                 seq_length: int,
                 target_col: str = None,
                 target_length: int = 1,
                 step: int = 1,
                 transform=None,
                 target_transform=None):
        """
        Initialize the TimeSeriesDataset.
        
        Args:
            data: Input time series data (DataFrame or numpy array)
            seq_length: Length of input sequences
            target_col: Name of the target column (if data is a DataFrame)
            target_length: Length of target sequences
            step: Step size between sequences
            transform: Optional transform to apply to the features
            target_transform: Optional transform to apply to the targets
        """
        self.seq_length = seq_length
        self.target_length = target_length
        self.step = step
        self.transform = transform
        self.target_transform = target_transform
        
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            if target_col is not None:
                self.features = data.drop(columns=[target_col]).values if target_col in data.columns else data.values
                self.targets = data[target_col].values.reshape(-1, 1)
            else:
                self.features = data.values
                self.targets = None
        else:
            self.features = np.asarray(data)
            self.targets = None
        
        # Create sequences
        self.sequences = []
        self.target_sequences = []
        
        for i in range(0, len(self.features) - seq_length - target_length + 1, step):
            self.sequences.append(self.features[i:i + seq_length])
            if self.targets is not None:
                self.target_sequences.append(self.targets[i + seq_length:i + seq_length + target_length])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> tuple:
        sequence = self.sequences[idx]
        
        if self.targets is not None and len(self.target_sequences) > 0:
            target = self.target_sequences[idx]
        else:
            # For unsupervised learning or when targets are not provided
            target = np.array([0])
        
        # Convert to torch tensors
        sequence = torch.FloatTensor(sequence)
        target = torch.FloatTensor(target)
        
        # Apply transforms if specified
        if self.transform:
            sequence = self.transform(sequence)
            
        if self.target_transform and self.targets is not None:
            target = self.target_transform(target)
        
        return sequence, target


def create_time_series_dataloader(
    data: Union[pd.DataFrame, np.ndarray],
    seq_length: int,
    batch_size: int = 32,
    target_col: str = None,
    target_length: int = 1,
    step: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for time series data.
    
    Args:
        data: Input time series data
        seq_length: Length of input sequences
        batch_size: Batch size
        target_col: Name of the target column (if data is a DataFrame)
        target_length: Length of target sequences
        step: Step size between sequences
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        
    Returns:
        DataLoader for the time series data
    """
    dataset = TimeSeriesDataset(
        data=data,
        seq_length=seq_length,
        target_col=target_col,
        target_length=target_length,
        step=step,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
