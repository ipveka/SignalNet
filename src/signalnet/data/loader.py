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

    def _extract_time_features(self, timestamps: pd.Series) -> Tuple[np.ndarray, list]:
        # More features: day of week, hour, minute, month, day of month, is_weekend
        dow = timestamps.dt.dayofweek.values.reshape(-1, 1) / 6.0
        hour = timestamps.dt.hour.values.reshape(-1, 1) / 23.0
        minute = timestamps.dt.minute.values.reshape(-1, 1) / 59.0
        month = (timestamps.dt.month.values.reshape(-1, 1) - 1) / 11.0
        dom = (timestamps.dt.day.values.reshape(-1, 1) - 1) / 30.0
        is_weekend = ((timestamps.dt.dayofweek >= 5).values.reshape(-1, 1)).astype(float)
        features = [dow, hour, minute, month, dom, is_weekend]
        features_used = [
            'day_of_week (normalized)',
            'hour_of_day (normalized)',
            'minute_of_hour (normalized)',
            'month (normalized)',
            'day_of_month (normalized)',
            'is_weekend'
        ]
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
        if self.data is None:
            self.load()
        assert self.data is not None, "Data must be loaded."
        n = len(self.data)
        split = int(n * (1 - test_size))
        train_data = self.data.iloc[:split]
        test_data = self.data.iloc[split:]
        return (
            SignalDataset(train_data, self.context_length, self.prediction_length, self.value_col, self.time_col),
            SignalDataset(test_data, self.context_length, self.prediction_length, self.value_col, self.time_col)
        )
