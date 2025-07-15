"""
Example usage of SignalNet: generate new data, use trained model (if available), predict using the predict() utility, evaluate, and visualize all predictions.
Also visualizes and prints all key objects and data.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from signalnet.data.loader import SignalDataLoader
from signalnet.evaluation.metrics import mean_squared_error, mean_absolute_error
from signalnet.data.generate_data import generate_signal_data
from signalnet.predict import predict
from signalnet.visualization.plot import plot_series_df, plot_predictions_df
import os

# Step 1: Generate new synthetic data for evaluation
print("[EXAMPLE] Generating new synthetic data for evaluation...")
new_data_path = 'input/example_signal_data.csv'
generate_signal_data(n_series=3, length=150, output_file=new_data_path)

# Step 2: Load and visualize the generated data
print("[EXAMPLE] Loading and visualizing the generated data...")
df = pd.read_csv(new_data_path)
print("[EXAMPLE] DataFrame head:")
print(df.head())
print(f"[EXAMPLE] DataFrame shape: {df.shape}")
print(f"[EXAMPLE] Unique series_id: {df['series_id'].unique()}")
# Plot the raw generated data for each series using the utility
plot_series_df(df, title='Generated Signal Data (Raw)')

# Step 3: Prepare and print the SignalDataLoader
context_length = 24
prediction_length = 6
loader = SignalDataLoader(new_data_path, context_length, prediction_length)
print("[EXAMPLE] SignalDataLoader object:")
print(loader)

# Step 4: Prepare and print the SignalDataset
print("[EXAMPLE] SignalDataset object:")
dataset = loader.get_dataset()
print(dataset)
print(f"[EXAMPLE] Dataset length: {len(dataset)}")
print("[EXAMPLE] First item in dataset (context, prediction, context_time_feat, pred_time_feat):")
first_item = dataset[0]
print(f"  context shape: {first_item[0].shape}")
print(f"  prediction shape: {first_item[1].shape}")
print(f"  context_time_feat shape: {first_item[2].shape}")
print(f"  pred_time_feat shape: {first_item[3].shape}")

# Step 5: Prepare and print the DataLoader
print("[EXAMPLE] DataLoader object:")
dataloader = DataLoader(dataset, batch_size=16)
print(dataloader)

# Step 6: Use the simple prediction utility to make predictions and get all info
print("[EXAMPLE] Using predict() utility to make predictions (returns DataFrame)...")
model_path = 'output/signalnet_model.pth'
pred_df = predict(dataloader, model_path=model_path, return_all_info=False)

# Step 7: Save predictions DataFrame to CSV
print("[EXAMPLE] Saving predictions DataFrame to example_output.csv...")
pred_df.to_csv('example_output.csv', index=False)
print("[EXAMPLE] Saved predictions to example_output.csv.")

# Step 8: Compute metrics directly
mse = mean_squared_error(pred_df['ground_truth'], pred_df['prediction'])
mae = mean_absolute_error(pred_df['ground_truth'], pred_df['prediction'])
print(f"[EXAMPLE] MSE: {mse:.4f}, MAE: {mae:.4f}")

# Step 9: Visualize predictions vs. ground truth using the same utility
print("[EXAMPLE] Visualizing predictions vs. ground truth for each series...")
plot_predictions_df(pred_df, title='SignalNet: Predictions vs. Ground Truth for Each Series') 