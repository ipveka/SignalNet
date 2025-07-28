"""
Plotting utilities for SignalNet.
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import Optional, List

def plot_series_df(df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', value_col: str = 'value', title: str = 'Time Series Data', save_path: Optional[str] = None, train_test_split_time: Optional[str] = None):
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
        
        # Add train/test split line if provided
        if train_test_split_time:
            split_time = pd.to_datetime(train_test_split_time)
            ax.axvline(x=split_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvspan(split_time, ax.get_xlim()[1], alpha=0.1, color='red')
        
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

def plot_combined_data(df: pd.DataFrame, series_col: str = 'series_id', time_col: str = 'timestamp', 
                      ground_truth_col: str = 'ground_truth', prediction_col: str = 'prediction', 
                      sample_col: str = 'sample', title: str = 'Combined Train/Test Data with Predictions', 
                      save_path: Optional[str] = None):
    """
    Plot combined train/test data with ground truth and predictions, one subplot per series.
    Clean, simple design with consistent styling matching plot_series_df.
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
        
        # Separate train and test data
        train_data = sdf[sdf[sample_col] == 'train']
        test_data = sdf[sdf[sample_col] == 'test']
        
        # Plot train data (ground truth only)
        if not train_data.empty:
            ax.plot(pd.to_datetime(train_data[time_col]), train_data[ground_truth_col], 
                    color=colors[i % len(colors)], linewidth=1.5, alpha=0.8, label='Train (Ground Truth)')
        
        # Plot test data (ground truth and predictions)
        if not test_data.empty:
            # Ground truth for test set
            ax.plot(pd.to_datetime(test_data[time_col]), test_data[ground_truth_col], 
                    color=colors[i % len(colors)], linewidth=1.5, alpha=0.8, label='Test (Ground Truth)')
            
            # Predictions for test set
            ax.plot(pd.to_datetime(test_data[time_col]), test_data[prediction_col], 
                    color='red', linewidth=1.5, alpha=0.8, linestyle='--', label='Test (Predictions)')
        
        # Add train/test split line
        if not train_data.empty and not test_data.empty:
            split_time = pd.to_datetime(test_data[time_col].min())
            ax.axvline(x=split_time, color='red', linestyle='-', alpha=0.7, linewidth=1)
            ax.axvspan(split_time, ax.get_xlim()[1], alpha=0.1, color='red')
        
        # Styling
        ax.set_title(f'Series {sid}', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
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
