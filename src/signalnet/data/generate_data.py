"""
Synthetic data generator for SignalNet.
Generates time series data and saves as CSV in the 'input' folder.
"""
import numpy as np
import pandas as pd
import os
import argparse

def generate_signal_data(n_series: int, length: int, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_data = []
    for series_id in range(n_series):
        t = np.arange(length)
        # Compose signal: trend + seasonality + noise
        trend = 0.05 * t
        seasonality = np.sin(2 * np.pi * t / 24) + 0.5 * np.sin(2 * np.pi * t / 6)
        noise = 0.2 * np.random.randn(length)
        value = trend + seasonality + noise
        df = pd.DataFrame({
            'series_id': series_id,
            'timestamp': pd.date_range('2023-01-01', periods=length, freq='h'),
            'value': value
        })
        all_data.append(df)
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv(output_file, index=False)
    print(f"Saved {n_series} series of length {length} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic signal data for SignalNet.")
    parser.add_argument('--n_series', type=int, default=10, help='Number of time series')
    parser.add_argument('--length', type=int, default=200, help='Length of each series')
    parser.add_argument('--output', type=str, default='input/signal_data.csv', help='Output CSV file')
    args = parser.parse_args()
    generate_signal_data(args.n_series, args.length, args.output) 