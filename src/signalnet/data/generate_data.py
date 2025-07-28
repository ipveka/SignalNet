"""
Synthetic data generator for SignalNet.
Generates time series data and saves as CSV in the 'input' folder.
"""
import numpy as np
import pandas as pd
import os
import argparse

def generate_signal_data(n_series: int, length: int, output_file: str, frequency: str = 'h'):
    """
    Generate synthetic time series data with specified frequency.
    
    Args:
        n_series: Number of time series to generate
        length: Length of each time series
        output_file: Output CSV file path
        frequency: Data frequency ('h'=hourly, 'D'=daily, 'W'=weekly, 'M'=monthly, etc.)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_data = []
    
    # Adjust seasonality based on frequency
    if frequency == 'h':  # Hourly
        daily_seasonality = 24
        weekly_seasonality = 24 * 7
    elif frequency == 'D':  # Daily
        daily_seasonality = 1
        weekly_seasonality = 7
    elif frequency == 'W':  # Weekly
        daily_seasonality = 1/7
        weekly_seasonality = 1
    elif frequency == 'M':  # Monthly
        daily_seasonality = 1/30
        weekly_seasonality = 1/4
    else:
        daily_seasonality = 24
        weekly_seasonality = 24 * 7
    
    for series_id in range(n_series):
        t = np.arange(length)
        # Compose signal: trend + seasonality + noise
        trend = 0.05 * t
        seasonality = np.sin(2 * np.pi * t / daily_seasonality) + 0.5 * np.sin(2 * np.pi * t / weekly_seasonality)
        noise = 0.2 * np.random.randn(length)
        value = trend + seasonality + noise
        
        # Generate timestamps based on frequency, with offset for each series to avoid duplicates
        start_date = pd.Timestamp('2023-01-01') + pd.Timedelta(days=series_id)
        timestamps = pd.date_range(start_date, periods=length, freq=frequency)
        
        df = pd.DataFrame({
            'series_id': series_id,
            'timestamp': timestamps,
            'value': value
        })
        all_data.append(df)
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(output_file, index=False)
    print(f"Saved {n_series} series of length {length} with {frequency} frequency to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic signal data for SignalNet.")
    parser.add_argument('--n_series', type=int, default=10, help='Number of time series')
    parser.add_argument('--length', type=int, default=200, help='Length of each series')
    parser.add_argument('--output', type=str, default='input/signal_data.csv', help='Output CSV file')
    parser.add_argument('--frequency', type=str, default='h', help='Data frequency (h=hourly, D=daily, W=weekly, M=monthly)')
    args = parser.parse_args()
    generate_signal_data(args.n_series, args.length, args.output, args.frequency) 