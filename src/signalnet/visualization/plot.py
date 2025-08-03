"""
Plotting utilities for SignalNet.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, List

def plot_series_df(df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', value_col: str = 'value', title: str = 'Time Series Data', save_path: Optional[str] = None):
    """
    Plot raw time series data from a DataFrame, one subplot per series.
    Clean, simple design with consistent styling.
    """
    series_list = sorted(df[series_col].unique())
    num_series = len(series_list)
    
    # Create subplots
    fig, axes = plt.subplots(num_series, 1, figsize=(12, 3 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]
    
    # Colors for different series
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, sid in enumerate(series_list):
        ax = axes[i]
        sdf = df[df[series_col] == sid].sort_values(time_col)
        
        # Plot the time series
        ax.plot(pd.to_datetime(sdf[time_col]), sdf[value_col], 
                color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)
        

        
        # Styling
        ax.set_title(f'Series {sid}', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Only show x-axis labels on the bottom subplot
        if i == num_series - 1:
            ax.set_xlabel('Time', fontsize=12)
            # Use auto date locator to avoid too many ticks
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        else:
            ax.set_xticklabels([])
        
        ax.set_ylabel('Value', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_series_df(df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', 
                           value_col: str = 'value', sample_col: str = 'sample', 
                           title: str = 'Combined Time Series Data', save_path: Optional[str] = None):
    """
    Plot combined time series data with different colors for train/test samples.
    One subplot per series, with train and test data colored differently.
    """
    series_list = sorted(df[series_col].unique())
    num_series = len(series_list)
    
    # Create subplots
    fig, axes = plt.subplots(num_series, 1, figsize=(12, 3 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]
    
    # Colors for train and test data
    train_color = '#1f77b4'  # Blue for train
    test_color = '#ff7f0e'   # Orange for test
    
    for i, sid in enumerate(series_list):
        ax = axes[i]
        sdf = df[df[series_col] == sid].sort_values(time_col)
        
        # Separate train and test data
        train_data = sdf[sdf[sample_col] == 'train']
        test_data = sdf[sdf[sample_col] == 'test']
        
        # Plot train data
        if not train_data.empty:
            ax.plot(pd.to_datetime(train_data[time_col]), train_data[value_col], 
                    color=train_color, linewidth=1.5, alpha=0.8, label='Train')
        
        # Plot test data
        if not test_data.empty:
            ax.plot(pd.to_datetime(test_data[time_col]), test_data[value_col], 
                    color=test_color, linewidth=1.5, alpha=0.8, label='Test')
        
        # Styling
        ax.set_title(f'Series {sid}', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', fontsize=10)
        
        # Only show x-axis labels on the bottom subplot
        if i == num_series - 1:
            ax.set_xlabel('Time', fontsize=12)
            # Use auto date locator to avoid too many ticks
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        else:
            ax.set_xticklabels([])
        
        ax.set_ylabel('Value', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()