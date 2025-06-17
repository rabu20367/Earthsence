"""Sensor data processing module for EarthSense."""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataProcessor:
    """
    A class for processing and analyzing sensor data from various sources.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the SensorDataProcessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.scalers_ = {}
        self.imputers_ = {}
        self.timezone = pytz.timezone(self.config.get('timezone', 'UTC'))
        self.default_freq = self.config.get('default_freq', '5T')  # 5 minutes
    
    def load_sensor_data(
        self, 
        file_path: str, 
        file_format: Optional[str] = None,
        time_col: str = 'timestamp',
        value_cols: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Load sensor data from a file.
        
        Args:
            file_path: Path to the sensor data file
            file_format: File format ('csv', 'json', 'parquet', 'feather', 'hdf'). 
                        If None, inferred from file extension
            time_col: Name of the timestamp column
            value_cols: List of value columns to load. If None, all columns except timestamp are loaded
            **kwargs: Additional arguments passed to the underlying pandas read function
            
        Returns:
            DataFrame containing the sensor data, or None if loading fails
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        if file_format is None:
            _, ext = os.path.splitext(file_path)
            file_format = ext.lstrip('.').lower()
            
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_format == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_format == 'feather':
                df = pd.read_feather(file_path, **kwargs)
            elif file_format == 'hdf':
                df = pd.read_hdf(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return None
                
            # Convert timestamp column to datetime if it exists
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.set_index(time_col)
                
                # Select only specified value columns if provided
                if value_cols is not None:
                    if time_col in value_cols:
                        value_cols.remove(time_col)
                    df = df[value_cols]
                
                return df
                
            logger.error(f"Timestamp column '{time_col}' not found in the data")
            return None
                
        except Exception as e:
            logger.error(f"Error loading sensor data: {e}")
            return None
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        value_cols: Optional[List[str]] = None,
        freq: Optional[str] = '1H',
        fill_method: str = 'linear',
        scale_method: str = 'standard',
        **kwargs
    ) -> pd.DataFrame:
        """
        Preprocess sensor data by handling missing values, resampling, and scaling.
        
        Args:
            df: Input DataFrame with sensor data
            timestamp_col: Name of the timestamp column
            value_cols: List of column names to process. If None, all numeric columns are used
            freq: Resampling frequency (e.g., '1H' for hourly, '15T' for 15 minutes)
            fill_method: Method for filling missing values ('linear', 'ffill', 'bfill', 'mean', 'median', 'zero')
            scale_method: Scaling method ('standard', 'minmax', 'robust', or None)
            **kwargs: Additional arguments for scaling
            
        Returns:
            Preprocessed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure timestamp is datetime and set as index
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
        
        # Select numeric columns if not specified
        if value_cols is None:
            value_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Resample to regular frequency if specified
        if freq is not None:
            df = df[value_cols].resample(freq).mean()
        else:
            df = df[value_cols].copy()
        
        # Handle missing values
        if fill_method == 'linear':
            df = df.interpolate(method='linear')
        elif fill_method == 'ffill':
            df = df.ffill()
        elif fill_method == 'bfill':
            df = df.bfill()
        elif fill_method == 'mean':
            df = df.fillna(df.mean())
        elif fill_method == 'median':
            df = df.fillna(df.median())
        elif fill_method == 'zero':
            df = df.fillna(0)
        
        # Scale the data if requested
        if scale_method is not None:
            if scale_method == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif scale_method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif scale_method == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler(**kwargs)
            else:
                raise ValueError(f"Unsupported scale method: {scale_method}")
            
            # Scale each column
            scaled_data = scaler.fit_transform(df[value_cols])
            df[value_cols] = scaled_data
            self.scalers_ = {col: scaler for col in value_cols}
        
        return df
    
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect anomalies in sensor data using various methods.
        
        Args:
            df: Input DataFrame with sensor data
            method: Anomaly detection method ('zscore', 'iqr', 'isolation_forest', 'one_class_svm')
            columns: List of columns to analyze. If None, all numeric columns are used
            **kwargs: Additional arguments for the anomaly detection method
            
        Returns:
            Tuple of (DataFrame with anomaly flags, dictionary of anomaly information)
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        # Initialize results
        results = {}
        anomaly_flags = pd.DataFrame(index=df.index)
        
        if method == 'zscore':
            threshold = kwargs.get('threshold', 3.0)
            for col in columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                anomaly_flags[f'anomaly_{col}'] = z_scores > threshold
                results[col] = {
                    'method': 'zscore',
                    'threshold': threshold,
                    'anomaly_count': int(anomaly_flags[f'anomaly_{col}'].sum()),
                    'anomaly_indices': df[anomaly_flags[f'anomaly_{col}']].index.tolist()
                }
                
        elif method == 'iqr':
            threshold = kwargs.get('threshold', 1.5)
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                anomaly_flags[f'anomaly_{col}'] = (df[col] < lower_bound) | (df[col] > upper_bound)
                results[col] = {
                    'method': 'iqr',
                    'threshold': threshold,
                    'anomaly_count': int(anomaly_flags[f'anomaly_{col}'].sum()),
                    'anomaly_indices': df[anomaly_flags[f'anomaly_{col}']].index.tolist()
                }
                
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            contamination = kwargs.get('contamination', 'auto')
            model = IsolationForest(contamination=contamination, random_state=42, **kwargs)
            
            # Fit on all specified columns
            X = df[columns].values
            preds = model.fit_predict(X)
            anomaly_flags['anomaly'] = preds == -1
            results['isolation_forest'] = {
                'method': 'isolation_forest',
                'contamination': contamination,
                'anomaly_count': int(anomaly_flags['anomaly'].sum()),
                'anomaly_indices': df[anomaly_flags['anomaly']].index.tolist(),
                'model': model
            }
            
        elif method == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            nu = kwargs.get('nu', 0.05)
            model = OneClassSVM(nu=nu, **kwargs)
            
            # Scale data first
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[columns])
            
            preds = model.fit_predict(X_scaled)
            anomaly_flags['anomaly'] = preds == -1
            results['one_class_svm'] = {
                'method': 'one_class_svm',
                'nu': nu,
                'anomaly_count': int(anomaly_flags['anomaly'].sum()),
                'anomaly_indices': df[anomaly_flags['anomaly']].index.tolist(),
                'model': model,
                'scaler': scaler
            }
            
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        return anomaly_flags, results
        
    def extract_features(
        self,
        df: pd.DataFrame,
        window_size: int = 24,
        columns: Optional[List[str]] = None,
        include_time_features: bool = True,
        include_statistical: bool = True,
        include_spectral: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from time series data.
        
        Args:
            df: Input DataFrame with time series data
            window_size: Size of the rolling window for feature extraction
            columns: List of columns to process. If None, all numeric columns are used
            include_time_features: Whether to include time-based features
            include_statistical: Whether to include statistical features
            include_spectral: Whether to include spectral features
            
        Returns:
            DataFrame with extracted features
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        features = pd.DataFrame(index=df.index)
        
        # Time-based features
        if include_time_features:
            if isinstance(df.index, pd.DatetimeIndex):
                features['hour'] = df.index.hour
                features['day_of_week'] = df.index.dayofweek
                features['day_of_month'] = df.index.day
                features['month'] = df.index.month
                features['is_weekend'] = df.index.weekday >= 5
        
        # Statistical features
        if include_statistical:
            for col in columns:
                # Rolling statistics
                rolling = df[col].rolling(window=window_size)
                
                # Basic statistics
                features[f'{col}_rolling_mean'] = rolling.mean()
                features[f'{col}_rolling_std'] = rolling.std()
                features[f'{col}_rolling_min'] = rolling.min()
                features[f'{col}_rolling_max'] = rolling.max()
                features[f'{col}_rolling_median'] = rolling.median()
                
                # Rate of change
                features[f'{col}_pct_change'] = df[col].pct_change()
                
                # Volatility
                features[f'{col}_volatility'] = df[col].rolling(window_size).std() / \
                                             df[col].rolling(window_size).mean()
        
        # Spectral features
        if include_spectral:
            for col in columns:
                # Fast Fourier Transform
                fft_vals = np.fft.fft(df[col].fillna(0).values)
                fft_freq = np.fft.fftfreq(len(fft_vals))
                
                # Dominant frequencies
                idx = np.argmax(np.abs(fft_vals)[1:]) + 1
                features[f'{col}_dominant_freq'] = fft_freq[idx]
                features[f'{col}_dominant_power'] = np.abs(fft_vals[idx])
                
                # Spectral entropy
                psd = np.abs(fft_vals) ** 2
                psd_norm = psd / psd.sum()
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                features[f'{col}_spectral_entropy'] = spectral_entropy
        
        return features.dropna()
                
    def detect_events(
        self,
        df: pd.DataFrame,
        method: str = 'threshold',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        """
        Detect events in time series data.
        
        Args:
            df: Input DataFrame with time series data
            method: Event detection method ('threshold', 'cwt', 'change_point')
            columns: List of columns to analyze. If None, all numeric columns are used
            **kwargs: Additional arguments for the event detection method
            
        Returns:
            Dictionary with event detection results
        """
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
            
        results = {}
        
        if method == 'threshold':
            threshold = kwargs.get('threshold', 3.0)
            for col in columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold_val = mean_val + threshold * std_val
                events = df[df[col] > threshold_val]
                results[col] = {
                    'method': 'threshold',
                    'threshold': threshold_val,
                    'event_count': len(events),
                    'event_indices': events.index.tolist(),
                    'event_values': events[col].tolist()
                }
                
        elif method == 'cwt':
            from scipy import signal
            widths = kwargs.get('widths', np.arange(1, 31))
            min_snr = kwargs.get('min_snr', 1.0)
            
            for col in columns:
                data = df[col].fillna(method='ffill').fillna(method='bfill').values
                data = (data - data.mean()) / data.std()
                
                # Find peaks using continuous wavelet transform
                peak_indices = signal.find_peaks_cwt(
                    vector=data,
                    widths=widths,
                    min_snr=min_snr
                )
                
                if len(peak_indices) > 0:
                    events = df.iloc[peak_indices]
                    results[col] = {
                        'method': 'cwt',
                        'event_count': len(events),
                        'event_indices': events.index.tolist(),
                        'event_values': events[col].tolist()
                    }
                
        elif method == 'change_point':
            from ruptures import Pelt
            from ruptures.costs import CostL2
            
            min_size = kwargs.get('min_size', 2)
            jump = kwargs.get('jump', 5)
            pen = kwargs.get('pen', 10)
            
            for col in columns:
                data = df[col].fillna(method='ffill').fillna(method='bfill').values
                data = data.reshape(-1, 1)
                
                # Detect change points
                algo = Pelt(model='l2', min_size=min_size, jump=jump).fit(data)
                result = algo.predict(pen=pen)
                
                if len(result) > 0:
                    change_points = result[:-1]  # Last point is the end of the series
                    events = df.iloc[change_points]
                    results[col] = {
                        'method': 'change_point',
                        'event_count': len(events),
                        'event_indices': events.index.tolist(),
                        'event_values': events[col].tolist()
                    }
        else:
            raise ValueError(f"Unsupported event detection method: {method}")
            
        return results

# Example usage
if __name__ == "__main__":
    # Example of how to use the SensorDataProcessor
    processor = SensorDataProcessor()
    
    # Example: Load and preprocess data
    data = {
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(50, 10, 100)
    }
    df = pd.DataFrame(data)
    
    # Save to CSV for example
    df.to_csv('sensor_data.csv', index=False)
    
    # Load and preprocess
    loaded_data = processor.load_sensor_data('sensor_data.csv', 'csv')
    if loaded_data is not None:
        # Preprocess
        processed_data = processor.preprocess_data(
            loaded_data,
            timestamp_col='timestamp',
            freq='1H',
            fill_method='linear',
            scale_method='standard'
        )
        
        # Detect anomalies
        anomalies, anomaly_info = processor.detect_anomalies(processed_data, method='zscore')
        print(f"Detected {anomalies.sum().sum()} anomalies")
        
        # Extract features
        features = processor.extract_features(processed_data)
        print(f"Extracted {features.shape[1]} features")
        
        # Detect events
        events = processor.detect_events(processed_data, method='threshold')
        print(f"Detected {sum(len(v['event_indices']) for v in events.values())} events")
