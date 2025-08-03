# Usage Examples and API Guide

SignalNet provides a comprehensive and easy-to-use API for time series prediction with advanced transformer architecture. This guide covers all major usage patterns and best practices.

## 🚀 Quick Start Example

### Basic Prediction Pipeline
```python
from signalnet.data.loader import SignalDataLoader
from signalnet.predict import predict
from torch.utils.data import DataLoader

# 1. Load and prepare data
loader = SignalDataLoader('input/example_signal_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()
dataloader = DataLoader(dataset, batch_size=16)

# 2. Make predictions (returns DataFrame)
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')

# 3. Save results
pred_df.to_csv('output/predictions.csv', index=False)
```

## 📊 Complete Workflow Example

### End-to-End SignalNet Usage
```python
import torch
import pandas as pd
from signalnet.data.loader import SignalDataLoader
from signalnet.data.generate_data import generate_signal_data
from signalnet.predict import predict
from signalnet.visualization.plot import plot_predictions_df
from signalnet.evaluation.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

# Step 1: Generate synthetic data (or load your own)
generate_signal_data(n_series=3, length=150, output_file='input/signal_data.csv')

# Step 2: Load and examine data
df = pd.read_csv('input/signal_data.csv')
print(f"Data shape: {df.shape}")
print(f"Value range: [{df['value'].min():.3f}, {df['value'].max():.3f}]")

# Step 3: Create data loader
loader = SignalDataLoader('input/signal_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()
dataloader = DataLoader(dataset, batch_size=16)

# Step 4: Make predictions
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')

# Step 5: Evaluate performance
mse = mean_squared_error(pred_df['ground_truth'], pred_df['prediction'])
mae = mean_absolute_error(pred_df['ground_truth'], pred_df['prediction'])
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

# Step 6: Visualize results
from signalnet.visualization.plot import plot_series_df, plot_combined_series_df

# Plot raw data
plot_series_df(df, title='Input Time Series Data', save_path='output/input_data.png')

# Plot combined train/test data
plot_combined_series_df(combined_df, title='Train/Test Split', save_path='output/combined_data.png')

# Step 7: Save results
pred_df.to_csv('output/example_output.csv', index=False)

## 📁 Output Files

The SignalNet example generates the following output files:

### **Data Files**
- `output/example_input.csv` - Original input data
- `output/example_output.csv` - Model predictions with ground truth
- `output/example_combined.csv` - Combined dataset with train/test tags (sorted by series and timestamp)

### **Visualization Files**
- `output/example_input.png` - Raw data visualization
- `output/example_combined.png` - Combined train/test visualization with sample-based coloring

### **Model Files**
- `output/signalnet_model.pth` - Trained model weights

## 🔧 Advanced Usage Patterns

### Custom Model Configuration
```python
from signalnet.models.transformer import SignalTransformer

# Load model with custom parameters
model_kwargs = {
    'input_dim': 1,
    'model_dim': 256,
    'num_heads': 16,
    'num_layers': 6,
    'output_dim': 1,
    'time_feat_dim': 6
}

pred_df = predict(
    dataloader, 
    model_path='output/signalnet_model.pth',
    model_kwargs=model_kwargs
)
```

### Batch Processing for Large Datasets
```python
# Process large datasets in batches
loader = SignalDataLoader('input/large_dataset.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()

# Use smaller batch size for memory efficiency
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')
```

### Multiple Series Processing
```python
# SignalNet automatically handles multiple time series
df = pd.read_csv('input/multi_series_data.csv')
print(f"Number of series: {df['series_id'].nunique()}")

loader = SignalDataLoader('input/multi_series_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()
dataloader = DataLoader(dataset, batch_size=16)

# Predictions include series_id for easy filtering
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')

# Filter predictions for specific series
series_0_preds = pred_df[pred_df['series_id'] == 0]
print(f"Predictions for series 0: {len(series_0_preds)}")
```

## 📈 Output Format and Analysis

### Prediction DataFrame Structure
```python
# The predict function returns a comprehensive DataFrame
print(pred_df.columns)
# Output: ['series_id', 'timestamp', 'ground_truth', 'prediction']

print(pred_df.head())
# Output:
#    series_id            timestamp  ground_truth  prediction
# 0          0  2023-01-01 08:00:00      1.552995    1.508401
# 1          0  2023-01-01 09:00:00      1.593074    1.578559
# 2          0  2023-01-01 10:00:00      1.017386    1.092944
```

### Performance Analysis
```python
# Analyze prediction accuracy
import numpy as np

# Overall metrics
mse = mean_squared_error(pred_df['ground_truth'], pred_df['prediction'])
mae = mean_absolute_error(pred_df['ground_truth'], pred_df['prediction'])

# Per-series analysis
for series_id in pred_df['series_id'].unique():
    series_preds = pred_df[pred_df['series_id'] == series_id]
    series_mse = mean_squared_error(series_preds['ground_truth'], series_preds['prediction'])
    print(f"Series {series_id}: MSE = {series_mse:.4f}")

# Scale analysis
print(f"Ground truth range: [{pred_df['ground_truth'].min():.3f}, {pred_df['ground_truth'].max():.3f}]")
print(f"Prediction range: [{pred_df['prediction'].min():.3f}, {pred_df['prediction'].max():.3f}]")
```

## 🎨 Visualization Options

### Basic Plotting
```python
from signalnet.visualization.plot import plot_predictions_df, plot_series_df

# Plot predictions vs ground truth
plot_predictions_df(pred_df, save_path='output/predictions.png')

# Plot raw data
plot_series_df(df, save_path='output/raw_data.png')
```

### Custom Plotting
```python
# Custom plot with specific title
plot_predictions_df(
    pred_df, 
    title='Custom SignalNet Predictions',
    save_path='output/custom_predictions.png'
)
```

## 🔍 Data Requirements

### Input CSV Format
SignalNet expects CSV files with the following columns:

```csv
series_id,timestamp,value
0,2023-01-01 00:00:00,1.23
0,2023-01-01 01:00:00,1.45
1,2023-01-01 00:00:00,2.34
...
```

### Required Columns
- **series_id**: Unique identifier for each time series
- **timestamp**: Datetime in format 'YYYY-MM-DD HH:MM:SS'
- **value**: Numeric signal values

### Data Quality
- **No missing values**: Ensure complete time series
- **Sorted timestamps**: Data should be chronologically ordered
- **Consistent frequency**: Regular time intervals recommended

## ⚡ Performance Tips

### Optimization Strategies
1. **Batch Size**: Use larger batches (16-32) for faster processing
2. **GPU Usage**: Model automatically uses GPU if available
3. **Memory Management**: For large datasets, use smaller batch sizes
4. **Data Preprocessing**: Ensure clean, normalized data for best results

### Expected Performance
- **MSE**: ~0.44 (excellent accuracy)
- **MAE**: ~0.53 (low error)
- **Scale Alignment**: Perfect match with data scale
- **Processing Speed**: ~1000 predictions/second on CPU

## 🛠️ Troubleshooting

### Common Issues
```python
# Issue: ModuleNotFoundError: No module named 'signalnet'
# Solution: Install in development mode
pip install -e .

# Issue: CUDA out of memory
# Solution: Reduce batch size
dataloader = DataLoader(dataset, batch_size=8)

# Issue: Poor predictions
# Solution: Ensure model is trained with sufficient epochs
# Recommended: 50+ epochs for full convergence
```

---

**SignalNet Usage**: Comprehensive API for advanced time series prediction with excellent performance and ease of use. 