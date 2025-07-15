# SignalNet: Transformer-based Time Series Signal Prediction

**Work in Progress: This project is under active development. APIs, features, and outputs may change.**

SignalNet is a modular, professional-grade Python package for time series signal prediction using a transformer architecture. It features robust data loading, synthetic data generation, model training, prediction, evaluation, and visualization tools.

## Key Features
- Modern `src/` layout, modular codebase
- PyTorch-based encoder-decoder transformer model
- DataLoader with windowing and rich time features (day of week, hour, minute, month, day of month, is_weekend)
- Prediction utility that returns a DataFrame for easy postprocessing
- Consistent, per-series, time-aware plotting utilities
- Sphinx documentation, pytest, and CI-ready

## Example Workflow
```python
from signalnet.data.loader import SignalDataLoader
from signalnet.predict import predict
from torch.utils.data import DataLoader

# Load data and create DataLoader
loader = SignalDataLoader('input/example_signal_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()
dataloader = DataLoader(dataset, batch_size=16)

# Predict (returns DataFrame)
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')
pred_df.to_csv('example_output.csv', index=False)
```

## Output Example (`example_output.csv`)
| series_id | timestamp           | ground_truth | prediction |
|-----------|---------------------|--------------|------------|
| 0         | 2023-01-02 00:00:00 | 1.23         | 1.18       |
| 0         | 2023-01-02 01:00:00 | 1.45         | 1.39       |
| ...       | ...                 | ...          | ...        |

## Expanded Features
- **Time features:** day_of_week, hour_of_day, minute_of_hour, month, day_of_month, is_weekend (all normalized)
- **Prediction output:** DataFrame for seamless postprocessing and analysis
- **Plotting:** Improved subplot layout, no x-axis label clutter

## Documentation
See `docs/usage_example.md` and `docs/training.md` for more details and code examples.

---

**Note:** This project is a work in progress. Expect rapid changes and improvements. Contributions and feedback are welcome!
