"""
Data loader for signal prediction tasks.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List

class SignalDataset(Dataset):
    """
    PyTorch Dataset for windowed time series data for transformer models.
    """
    def __init__(self, data: pd.DataFrame, context_length: int, prediction_length: int, value_col: str = "value", time_col: str = "timestamp"):
        self.data = data.sort_values(time_col).reset_index(drop=True)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.value_col = value_col
        self.time_col = time_col
        self.values = self.data[value_col].values.astype(np.float32)
        self.timestamps = pd.to_datetime(self.data[time_col])
        self.time_features, self.features_used = self._extract_time_features(self.timestamps)
        self.indices = self._compute_indices()

    def _detect_frequency(self, timestamps: pd.Series) -> str:
        """
        Detect the frequency of the time series data.
        Returns: 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
        """
        if len(timestamps) < 2:
            return 'unknown'
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        
        # Get the most common time difference by converting to seconds first
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        # Filter out 0-second differences (which occur when multiple series have same timestamps)
        non_zero_diffs = time_diffs_seconds[time_diffs_seconds > 0]
        
        if len(non_zero_diffs) == 0:
            return 'unknown'
        
        most_common_seconds = non_zero_diffs.mode().iloc[0]
        
        # Determine frequency based on time difference with better thresholds
        if most_common_seconds < 60:  # Less than 1 minute
            return 'minute'
        elif most_common_seconds < 3600:  # Less than 1 hour (but >= 1 minute)
            return 'minute'
        elif most_common_seconds < 86400:  # Less than 1 day (but >= 1 hour)
            return 'hour'
        elif most_common_seconds < 604800:  # Less than 1 week (but >= 1 day)
            return 'day'
        elif most_common_seconds < 2592000:  # Less than 30 days (but >= 1 week)
            return 'week'
        elif most_common_seconds < 7776000:  # Less than 90 days (but >= 30 days)
            return 'month'
        else:
            return 'year'

    def _extract_time_features(self, timestamps: pd.Series) -> Tuple[np.ndarray, list]:
        """
        Extract time features based on detected frequency.
        Only includes features that are meaningful for the data frequency.
        """
        frequency = self._detect_frequency(timestamps)
        features = []
        features_used = []
        
        # Always include day of week (relevant for most frequencies)
        dow = timestamps.dt.dayofweek.values.reshape(-1, 1) / 6.0
        features.append(dow)
        features_used.append('day_of_week (normalized)')
        
        # Always include month (relevant for most frequencies)
        month = (timestamps.dt.month.values.reshape(-1, 1) - 1) / 11.0
        features.append(month)
        features_used.append('month (normalized)')
        
        # Always include is_weekend (relevant for most frequencies)
        is_weekend = ((timestamps.dt.dayofweek >= 5).values.reshape(-1, 1)).astype(float)
        features.append(is_weekend)
        features_used.append('is_weekend')
        
        # Frequency-specific features
        if frequency in ['minute', 'hour']:
            # Include hour for minute and hour frequency data
            hour = timestamps.dt.hour.values.reshape(-1, 1) / 23.0
            features.append(hour)
            features_used.append('hour_of_day (normalized)')
            
            if frequency == 'minute':
                # Include minute only for minute frequency data
                minute = timestamps.dt.minute.values.reshape(-1, 1) / 59.0
                features.append(minute)
                features_used.append('minute_of_hour (normalized)')
        
        if frequency in ['day', 'week', 'month']:
            # Include day of month for daily and longer frequencies
            dom = (timestamps.dt.day.values.reshape(-1, 1) - 1) / 30.0
            features.append(dom)
            features_used.append('day_of_month (normalized)')
        
        if frequency in ['month', 'quarter', 'year']:
            # Include quarter for monthly and longer frequencies
            quarter = (timestamps.dt.quarter.values.reshape(-1, 1) - 1) / 3.0
            features.append(quarter)
            features_used.append('quarter (normalized)')
            
            # Include day of year for monthly and longer frequencies
            day_of_year = (timestamps.dt.dayofyear.values.reshape(-1, 1) - 1) / 365.0
            features.append(day_of_year)
            features_used.append('day_of_year (normalized)')
        
        # Add frequency-specific business features
        if frequency in ['hour', 'day']:
            # Business hours indicator for hourly and daily data
            is_business_hour = ((timestamps.dt.hour >= 9) & 
                               (timestamps.dt.hour <= 17) & 
                               (timestamps.dt.dayofweek < 5)).values.reshape(-1, 1).astype(float)
            features.append(is_business_hour)
            features_used.append('is_business_hour')
        
        if frequency in ['day', 'week', 'month']:
            # Month-end indicator for daily and longer frequencies
            is_month_end = timestamps.dt.is_month_end.values.reshape(-1, 1).astype(float)
            features.append(is_month_end)
            features_used.append('is_month_end')
        
        print(f"[INFO] Detected frequency: {frequency}")
        print(f"[INFO] Features used: {features_used}")
        
        return np.concatenate(features, axis=1), features_used

    def _compute_indices(self) -> List[int]:
        # Indices where a full context+prediction window fits
        total_length = self.context_length + self.prediction_length
        return [i for i in range(len(self.values) - total_length + 1)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        context = self.values[i:i+self.context_length]
        prediction = self.values[i+self.context_length:i+self.context_length+self.prediction_length]
        context_time_feat = self.time_features[i:i+self.context_length]
        pred_time_feat = self.time_features[i+self.context_length:i+self.context_length+self.prediction_length]
        return (
            torch.tensor(context, dtype=torch.float32),
            torch.tensor(prediction, dtype=torch.float32),
            torch.tensor(context_time_feat, dtype=torch.float32),
            torch.tensor(pred_time_feat, dtype=torch.float32)
        )

class SignalDataLoader:
    """
    Loads and preprocesses signal data for training and evaluation.
    """
    def __init__(self, filepath: str, context_length: int, prediction_length: int, value_col: str = "value", time_col: str = "timestamp"):
        self.filepath = filepath
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.value_col = value_col
        self.time_col = time_col
        self.data = None

    def load(self) -> pd.DataFrame:
        """Load data from file."""
        self.data = pd.read_csv(self.filepath, parse_dates=[self.time_col])
        return self.data

    def get_dataset(self) -> SignalDataset:
        if self.data is None:
            self.load()
        assert self.data is not None, "Data must be loaded."
        return SignalDataset(self.data, self.context_length, self.prediction_length, self.value_col, self.time_col)

    def train_test_split(self, test_size: float = 0.2) -> Tuple[SignalDataset, SignalDataset]:
        """
        Create time series aware train/test split.
        Ensures no future data is used to predict past data.
        """
        if self.data is None:
            self.load()
        assert self.data is not None, "Data must be loaded."
        
        # Sort by timestamp to ensure chronological order
        sorted_data = self.data.sort_values(self.time_col).reset_index(drop=True)
        
        # Calculate split point based on time (not index)
        n = len(sorted_data)
        split_idx = int(n * (1 - test_size))
        
        # Split data chronologically
        train_data = sorted_data.iloc[:split_idx].copy()
        test_data = sorted_data.iloc[split_idx:].copy()
        
        # Add sample tags
        train_data['sample'] = 'train'
        test_data['sample'] = 'test'
        
        print(f"[INFO] Time series split:")
        print(f"  Train period: {train_data[self.time_col].min()} to {train_data[self.time_col].max()}")
        print(f"  Test period: {test_data[self.time_col].min()} to {test_data[self.time_col].max()}")
        print(f"  Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        return (
            SignalDataset(train_data, self.context_length, self.prediction_length, self.value_col, self.time_col),
            SignalDataset(test_data, self.context_length, self.prediction_length, self.value_col, self.time_col)
        )
