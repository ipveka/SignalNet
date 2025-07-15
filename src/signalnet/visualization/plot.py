"""
Plotting utilities for SignalNet.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, List

def plot_series_df(df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', value_col: str = 'value', features: Optional[List[str]] = None, title: str = 'Time Series Data', save_path: Optional[str] = None):
    """
    Plot raw time series data from a DataFrame, one subplot per series.
    """
    series_list = df[series_col].unique()
    num_series = len(series_list)
    fig, axes = plt.subplots(num_series, 1, figsize=(12, 4 * num_series), sharex=False)
    if num_series == 1:
        axes = [axes]
    for i, sid in enumerate(series_list):
        ax = axes[i]
        sdf = df[df[series_col] == sid]
        ax.plot(pd.to_datetime(sdf[time_col]), sdf[value_col], label='Value', alpha=0.8)
        if features:
            for feat in features:
                if feat in sdf.columns:
                    ax.plot(pd.to_datetime(sdf[time_col]), sdf[feat], label=feat, linestyle='--', alpha=0.5)
        ax.set_title(f'{title} - Series {sid}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_predictions_df(pred_df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', true_col: str = 'ground_truth', pred_col: str = 'prediction', title: str = 'Predictions vs. Ground Truth', save_path: Optional[str] = None):
    """
    Plot predictions vs. ground truth from a DataFrame, one subplot per series, arranged in 2 columns.
    """
    series_list = pred_df[series_col].unique()
    num_series = len(series_list)
    ncols = 2
    nrows = int(np.ceil(num_series / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), sharex=False)
    axes = np.array(axes).reshape(nrows, ncols)
    for idx, sid in enumerate(series_list):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        sdf = pred_df[pred_df[series_col] == sid]
        ts = pd.to_datetime(sdf[time_col])
        ax.plot(ts, sdf[true_col], label='True', alpha=0.7)
        ax.plot(ts, sdf[pred_col], label='Predicted', alpha=0.7)
        mse = np.mean((sdf[true_col] - sdf[pred_col]) ** 2)
        ax.set_title(f'{title} - Series {sid} (MSE: {mse:.4f})')
        # ax.set_xlabel('Timestamp')  # REMOVE x-axis label
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment('right')
    # Hide unused subplots
    for idx in range(num_series, nrows * ncols):
        row, col = divmod(idx, ncols)
        fig.delaxes(axes[row, col])
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        plt.savefig(save_path)
    plt.show()
