"""
Evaluation metrics for SignalNet.
"""
from typing import Any
import numpy as np

def mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Compute mean squared error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Compute mean absolute error."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
