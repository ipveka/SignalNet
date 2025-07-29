"""
SignalNet Complete Example: End-to-End Time Series Prediction Pipeline

This example demonstrates the complete SignalNet workflow:
1. Synthetic data generation with customizable frequency and length
2. Data loading and preprocessing with automatic feature engineering
3. Model training with improved transformer architecture
4. Prediction generation and evaluation
5. Comprehensive visualization and analysis
6. Data export for further analysis

Key Features Demonstrated:
- Frequency-aware time feature extraction (hour, day, week, etc.)
- Automatic data normalization and preprocessing
- Train/test split with time series awareness
- Enhanced transformer model with 256 dimensions
- Comprehensive evaluation metrics
- Professional visualization with train/test split indicators

Usage:
    python examples/example.py
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from signalnet.data.loader import SignalDataLoader
from signalnet.evaluation.metrics import mean_squared_error, mean_absolute_error
from signalnet.data.generate_data import generate_signal_data
from signalnet.predict import predict
from signalnet.visualization.plot import plot_series_df, plot_combined_series_df
from signalnet.training.train import train
import os

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
CONFIG = {
    # Data Generation Parameters
    'n_series': 3, 
    'length': 168,  
    'frequency': 'h',
    
    # Model Training Parameters
    'train_model': True,
    'epochs': 5,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'context_length': 24, 
    'prediction_length': 6,
    
    # File Paths and Output Configuration
    'data_path': 'input/example_signal_data.csv',
    'model_path': 'output/signalnet_model.pth',
    'output_dir': 'output'
}

# ============================================================================
# SETUP AND ENVIRONMENT PREPARATION
# ============================================================================
# Initialize the execution environment and create necessary directories
print("=" * 60)
print("SIGNALNET EXAMPLE EXECUTION")
print("=" * 60)

# Create output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)
print("[SETUP] Output directory ready")

# ============================================================================
# STEP 1: DATA GENERATION
# ============================================================================
# Generate synthetic time series data for demonstration purposes
print("\n[DATA] Generating new synthetic data for evaluation...")
print(f"[DATA] Frequency: {CONFIG['frequency']}, Length: {CONFIG['length']} periods")

# Generate synthetic data with specified parameters
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
# Load the generated data and perform initial analysis
print("\n[DATA] Loading and analyzing the generated data...")

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(CONFIG['data_path'])

# Perform comprehensive data analysis and display key statistics
print("[DATA] DataFrame head:")
print(df.head())
print(f"[DATA] DataFrame shape: {df.shape}")
print(f"[DATA] Unique series_id: {df['series_id'].unique()}")
print(f"[DATA] Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Create and save visualization of the raw data
plot_series_df(df, title=f'Input {CONFIG["frequency"]} Data', 
               save_path=f'{CONFIG["output_dir"]}/example_input.png')
print(f"[DATA] Visualization saved to {CONFIG['output_dir']}/example_input.png")

# ============================================================================
# STEP 3: DATA LOADER PREPARATION
# ============================================================================
# Initialize the SignalDataLoader which handles data preprocessing and windowing
print("\n[LOADER] Preparing SignalDataLoader...")

# Create SignalDataLoader instance with specified parameters
loader = SignalDataLoader(CONFIG['data_path'], CONFIG['context_length'], CONFIG['prediction_length'])
print("[LOADER] SignalDataLoader object:")
print(loader)

# ============================================================================
# STEP 4: DATASET AND FEATURE ANALYSIS
# ============================================================================
# Analyze the dataset and extract frequency-aware time features
print("\n[FEATURES] Analyzing SignalDataset with frequency-aware feature selection...")

# Get the processed dataset with automatic feature extraction
dataset = loader.get_dataset()

# Analyze and display dataset properties and structure
print("[FEATURES] SignalDataset object:")
print(dataset)
print(f"[FEATURES] Dataset length: {len(dataset)}") 
print(f"[FEATURES] Detected frequency: {dataset._detect_frequency(dataset.timestamps)}")
print(f"[FEATURES] Features used: {dataset.features_used}")  
print(f"[FEATURES] Number of features: {len(dataset.features_used)}") 

# Examine the structure of a single training example
print("[FEATURES] First item in dataset (context, prediction, context_time_feat, pred_time_feat):")
first_item = dataset[0]
print(f"  context shape: {first_item[0].shape}") 
print(f"  prediction shape: {first_item[1].shape}")  
print(f"  context_time_feat shape: {first_item[2].shape}")  
print(f"  pred_time_feat shape: {first_item[3].shape}")    

# Detailed analysis of each extracted time feature
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
# Create a time series aware train/test split to prevent data leakage
print("\n[SPLIT] Creating time series aware train/test split...")

# Split the data chronologically with 80% for training and 20% for testing
train_dataset, test_dataset = loader.train_test_split(test_size=0.2)

# Display split statistics
print(f"[SPLIT] Training set size: {len(train_dataset)}") 
print(f"[SPLIT] Test set size: {len(test_dataset)}") 

# Analyze the split data structure and verify proper tagging
print("\n[SPLIT] Sample tags in data:")
train_data = train_dataset.data
test_data = test_dataset.data   
print(f"  Train data sample column: {train_data['sample'].unique()}") 
print(f"  Test data sample column: {test_data['sample'].unique()}")
print(f"  Train data shape: {train_data.shape}")
print(f"  Test data shape: {test_data.shape}")

# Store the split time for visualization purposes
train_test_split_time = test_data['timestamp'].min()
print(f"  Train/test split time: {train_test_split_time}")

# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================
# Train the enhanced transformer model with improved architecture
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
# Prepare the test dataset for prediction using PyTorch DataLoader
print("\n[PREDICT] Preparing test DataLoader...")

# Create a DataLoader for the test dataset
test_dataloader = DataLoader(test_dataset, batch_size=16)
print("[PREDICT] Test DataLoader object:")
print(test_dataloader)

# ============================================================================
# STEP 8: MAKING PREDICTIONS
# ============================================================================
# Generate predictions on the test set using the trained model
print("[PREDICT] Using predict() utility to make predictions on test set...")

# Make predictions using the SignalNet predict function
test_pred_df = predict(test_dataloader, model_path=CONFIG['model_path'], return_all_info=False)

# Add sample tag to identify these as test predictions
test_pred_df['sample'] = 'test'

# Display prediction results and statistics
print("\n[PREDICT] Test set predictions head:")
print(test_pred_df.head())
print(f"\n[PREDICT] Test set predictions shape: {test_pred_df.shape}")

# ============================================================================
# STEP 9: EVALUATION METRICS
# ============================================================================
# Calculate and display model performance metrics
print("\n[EVAL] Computing evaluation metrics...")

# Calculate Mean Squared Error (MSE) - measures average squared prediction error
test_mse = mean_squared_error(test_pred_df['ground_truth'], test_pred_df['prediction'])

# Calculate Mean Absolute Error (MAE) - measures average absolute prediction error
test_mae = mean_absolute_error(test_pred_df['ground_truth'], test_pred_df['prediction'])

# Display the evaluation results
print(f"[EVAL] Test Set MSE: {test_mse:.4f}") 
print(f"[EVAL] Test Set MAE: {test_mae:.4f}") 

# ============================================================================
# STEP 10: DATA COMBINATION AND ANALYSIS
# ============================================================================
# Combine train and test datasets for comprehensive analysis and visualization
print("\n[SAVE] Combining and analyzing datasets...")

# Merge train and test datasets into a single DataFrame
print("[SAVE] Creating combined dataset with sample tags...")
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Sort the combined data by series_id and timestamp for better organization
combined_data = combined_data.sort_values(['series_id', 'timestamp']).reset_index(drop=True)

# Show a sample of the combined data structure
print(f"\n[SAVE] Combined data head:")
print(combined_data.head())

# Perform detailed analysis of the combined dataset structure
print(f"\n[SAVE] Combined data structure analysis:")
print("=" * 50)

# Analyze DataFrame structure and metadata
print(f"DataFrame Info:")
print(f"  - Shape: {combined_data.shape}")
print(f"  - Columns: {list(combined_data.columns)}")
print(f"  - Data types:")
for col, dtype in combined_data.dtypes.items():
    print(f"    {col}: {dtype}")

# Display a formatted sample of the actual data
print(f"\n[SAVE] Sample of combined data (first 20 rows):")
print(combined_data.head().to_string(index=False))

# ============================================================================
# STEP 11: PLOTTING COMBINED DATASET
# ============================================================================
# Create a complete visualization showing the complete output
print("\n[PLOT] Creating visualization of combined dataset...")
plot_combined_series_df(
    combined_data, 
    title=f'Combined Dataset: Train/Test Split ({CONFIG["frequency"]} Data)', 
    save_path=f'{CONFIG["output_dir"]}/example_combined.png'
)
print(f"[PLOT] Combined dataset visualization saved to {CONFIG['output_dir']}/example_combined.png")

# ============================================================================
# STEP 12: SAVING ALL DATASETS
# ============================================================================
# Export all generated data and results for further analysis
print("\n[SAVE] Saving all datasets to output directory...")

# Save the original input data (synthetic data)
print("[SAVE] Saving input data...")
df.to_csv(f'{CONFIG["output_dir"]}/example_input.csv', index=False)
print("[SAVE] Saved input data to output/example_input.csv")

# Save the model predictions with ground truth values
print("[SAVE] Saving forecast data...")
test_pred_df.to_csv(f'{CONFIG["output_dir"]}/example_output.csv', index=False)
print("[SAVE] Saved forecast data to output/example_output.csv")

# Save the combined dataset with train/test tags
print("[SAVE] Saving combined data...")
combined_data.to_csv(f'{CONFIG["output_dir"]}/example_combined.csv', index=False)
print("[SAVE] Saved combined data to output/example_combined.csv")

# ============================================================================
# COMPLETION
# ============================================================================
print("\n" + "=" * 60)
print("EXAMPLE EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 60)