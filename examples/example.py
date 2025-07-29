"""
Example usage of SignalNet: generate data, demonstrate frequency-aware feature selection, 
train model, predict using the predict() utility, evaluate, and visualize all predictions.
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
from signalnet.visualization.plot import plot_series_df
from signalnet.training.train import train
import os

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
CONFIG = {
    # Data generation
    'n_series': 3,
    'length': 168,
    'frequency': 'h', 
    
    # Model training
    'train_model': False, 
    'epochs': 5,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'context_length': 24,
    'prediction_length': 6, 
    
    # File paths
    'data_path': 'input/example_signal_data.csv',
    'model_path': 'output/signalnet_model.pth',
    'output_dir': 'output'
}

# ============================================================================
# SETUP AND ENVIRONMENT PREPARATION
# ============================================================================
print("=" * 60)
print("SIGNALNET EXAMPLE EXECUTION")
print("=" * 60)

# Ensure output directory exists
os.makedirs(CONFIG['output_dir'], exist_ok=True)
print("[SETUP] Output directory ready")

# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================
print("\n[DATA] Generating new synthetic data for evaluation...")
print(f"[DATA] Frequency: {CONFIG['frequency']}, Length: {CONFIG['length']} periods")

generate_signal_data(
    n_series=CONFIG['n_series'], 
    length=CONFIG['length'], 
    output_file=CONFIG['data_path'], 
    frequency=CONFIG['frequency']
)
print(f"[DATA] Data generated and saved to {CONFIG['data_path']}")

# ============================================================================
# STEP 2: DATA LOADING AND ANALYSIS
# ============================================================================
print("\n[DATA] Loading and analyzing the generated data...")

df = pd.read_csv(CONFIG['data_path'])

# Print data overview
print("[DATA] DataFrame head:")
print(df.head(15))
print(f"[DATA] DataFrame shape: {df.shape}")
print(f"[DATA] Unique series_id: {df['series_id'].unique()}")
print(f"[DATA] Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Visualize the raw data
plot_series_df(df, title=f'Input {CONFIG["frequency"]} Data', 
               save_path=f'{CONFIG["output_dir"]}/example_input.png')
print(f"[DATA] Visualization saved to {CONFIG['output_dir']}/example_input.png")

# ============================================================================
# STEP 3: DATA LOADER PREPARATION
# ============================================================================
print("\n[LOADER] Preparing SignalDataLoader...")

loader = SignalDataLoader(CONFIG['data_path'], CONFIG['context_length'], CONFIG['prediction_length'])
print("[LOADER] SignalDataLoader object:")
print(loader)

# ============================================================================
# STEP 4: DATASET AND FEATURE ANALYSIS
# ============================================================================
print("\n[FEATURES] Analyzing SignalDataset with frequency-aware feature selection...")

dataset = loader.get_dataset()

# Print dataset information
print("[FEATURES] SignalDataset object:")
print(dataset)
print(f"[FEATURES] Dataset length: {len(dataset)}")
print(f"[FEATURES] Detected frequency: {dataset._detect_frequency(dataset.timestamps)}")
print(f"[FEATURES] Features used: {dataset.features_used}")
print(f"[FEATURES] Number of features: {len(dataset.features_used)}")

# Show first item structure
print("[FEATURES] First item in dataset (context, prediction, context_time_feat, pred_time_feat):")
first_item = dataset[0]
print(f"  context shape: {first_item[0].shape}")
print(f"  prediction shape: {first_item[1].shape}")
print(f"  context_time_feat shape: {first_item[2].shape}")
print(f"  pred_time_feat shape: {first_item[3].shape}")

# Feature analysis
print("\n[FEATURES] Feature analysis:")
print("=" * 40)
for i, feature_name in enumerate(dataset.features_used):
    feature_values = dataset.time_features[:, i]
    unique_values = len(np.unique(feature_values))
    print(f"  {feature_name}:")
    print(f"    - Unique values: {unique_values}")
    print(f"    - Range: [{feature_values.min():.3f}, {feature_values.max():.3f}]")

# ============================================================================
# STEP 5: TRAIN/TEST SPLIT
# ============================================================================
print("\n[SPLIT] Creating time series aware train/test split...")

train_dataset, test_dataset = loader.train_test_split(test_size=0.2)

print(f"[SPLIT] Training set size: {len(train_dataset)}")
print(f"[SPLIT] Test set size: {len(test_dataset)}")

# Show sample tags in the data
print("\n[SPLIT] Sample tags in data:")
train_data = train_dataset.data
test_data = test_dataset.data
print(f"  Train data sample column: {train_data['sample'].unique()}")
print(f"  Test data sample column: {test_data['sample'].unique()}")
print(f"  Train data shape: {train_data.shape}")
print(f"  Test data shape: {test_data.shape}")

# Get train/test split time for plotting
train_test_split_time = test_data['timestamp'].min()
print(f"  Train/test split time: {train_test_split_time}")

# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================
if CONFIG['train_model']:
    print(f"\n[TRAIN] Training model for {CONFIG['epochs']} epochs...")
    print("=" * 50)
    
    train(
        data_path=CONFIG['data_path'],
        context_length=CONFIG['context_length'],
        prediction_length=CONFIG['prediction_length'],
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        lr=CONFIG['learning_rate'],
        save_model=True
    )
    print(f"[TRAIN] Training completed. Model saved to {CONFIG['model_path']}")
else:
    print(f"\n[TRAIN] Skipping training. Using existing model: {CONFIG['model_path']}")

# ============================================================================
# STEP 7: PREDICTION PREPARATION
# ============================================================================
print("\n[PREDICT] Preparing test DataLoader...")

test_dataloader = DataLoader(test_dataset, batch_size=16)
print("[PREDICT] Test DataLoader object:")
print(test_dataloader)

# ============================================================================
# STEP 8: MAKING PREDICTIONS
# ============================================================================
print("[PREDICT] Using predict() utility to make predictions on test set...")

test_pred_df = predict(test_dataloader, model_path=CONFIG['model_path'], return_all_info=False)

# Add sample
test_pred_df['sample'] = 'test'

# Show test set predictions
print("\n[PREDICT] Test set predictions head:")
print(test_pred_df.head(15))
print(f"\n[PREDICT] Test set predictions shape: {test_pred_df.shape}")

# ============================================================================
# STEP 9: EVALUATION METRICS
# ============================================================================
print("\n[EVAL] Computing evaluation metrics...")

test_mse = mean_squared_error(test_pred_df['ground_truth'], test_pred_df['prediction'])
test_mae = mean_absolute_error(test_pred_df['ground_truth'], test_pred_df['prediction'])

print(f"[EVAL] Test Set MSE: {test_mse:.4f}")
print(f"[EVAL] Test Set MAE: {test_mae:.4f}")

# ============================================================================
# STEP 10: DATA COMBINATION AND ANALYSIS
# ============================================================================
print("\n[SAVE] Combining and analyzing datasets...")

# Create combined dataset with train/test tags
print("[SAVE] Creating combined dataset with sample tags...")
combined_data = pd.concat([train_data, test_data], ignore_index=True)

print(f"[SAVE] Combined data shape: {combined_data.shape}")
print(f"[SAVE] Sample distribution: {combined_data['sample'].value_counts().to_dict()}")
print(f"[SAVE] Time range: {combined_data['timestamp'].min()} to {combined_data['timestamp'].max()}")
print(f"[SAVE] Series distribution: {combined_data['series_id'].value_counts().to_dict()}")

# Print head of combined data
print(f"\n[SAVE] Combined data head:")
print(combined_data.head(15))

# ============================================================================
# STEP 11: SAVING ALL DATASETS
# ============================================================================
print("\n[SAVE] Saving all datasets to output directory...")

# Save input data
print("[SAVE] Saving input data...")
df.to_csv(f'{CONFIG["output_dir"]}/example_input.csv', index=False)
print("[SAVE] Saved input data to output/example_input.csv")

# Save forecast data
print("[SAVE] Saving forecast data...")
test_pred_df.to_csv(f'{CONFIG["output_dir"]}/example_forecast.csv', index=False)
print("[SAVE] Saved forecast data to output/example_forecast.csv")

# Save combined data
print("[SAVE] Saving combined data...")
combined_data.to_csv(f'{CONFIG["output_dir"]}/example_combined.csv', index=False)
print("[SAVE] Saved combined data to output/example_combined.csv")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 60)
print("EXAMPLE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 60)