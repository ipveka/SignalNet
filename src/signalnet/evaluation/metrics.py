"""
Evaluation metrics for SignalNet.
"""
from typing import Union, List, Tuple
import numpy as np
import pandas as pd

def mean_squared_error(y_true: Union[np.ndarray, List[float], pd.Series], 
                      y_pred: Union[np.ndarray, List[float], pd.Series]) -> float:
    """
    Compute mean squared error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Mean squared error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    return float(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true: Union[np.ndarray, List[float], pd.Series], 
                       y_pred: Union[np.ndarray, List[float], pd.Series]) -> float:
    """
    Compute mean absolute error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Mean absolute error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_true: Union[np.ndarray, List[float], pd.Series], 
                           y_pred: Union[np.ndarray, List[float], pd.Series]) -> float:
    """
    Compute root mean squared error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Root mean squared error
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_percentage_error(y_true: Union[np.ndarray, List[float], pd.Series], 
                                 y_pred: Union[np.ndarray, List[float], pd.Series]) -> float:
    """
    Compute mean absolute percentage error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Mean absolute percentage error
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
