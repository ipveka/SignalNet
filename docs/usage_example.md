# Usage Example (Work in Progress)

SignalNet now outputs predictions as a DataFrame for easy postprocessing and analysis. The project is under active development; APIs and outputs may change.

## Example: Predict and Save Output
```python
from signalnet.data.loader import SignalDataLoader
from signalnet.predict import predict
from torch.utils.data import DataLoader

loader = SignalDataLoader('input/example_signal_data.csv', context_length=24, prediction_length=6)
dataset = loader.get_dataset()
dataloader = DataLoader(dataset, batch_size=16)

# Predict (returns DataFrame)
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')
pred_df.to_csv('example_output.csv', index=False)
```

## Output CSV Example
| series_id | timestamp           | ground_truth | prediction |
|-----------|---------------------|--------------|------------|
| 0         | 2023-01-02 00:00:00 | 1.23         | 1.18       |
| ...       | ...                 | ...          | ...        |

---
**Note:** This is a work in progress. See the main README for more details and updates. 