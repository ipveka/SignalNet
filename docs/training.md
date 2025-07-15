# Training and Data Flow (Work in Progress)

SignalNet is under active development. The feature set, model, and outputs are evolving rapidly.

## Expanded Feature Set
- **Time features:**
    - day_of_week (normalized)
    - hour_of_day (normalized)
    - minute_of_hour (normalized)
    - month (normalized)
    - day_of_month (normalized)
    - is_weekend
- These features are automatically extracted and used in both training and prediction.

## Prediction Output
- The `predict` utility now returns a DataFrame with columns:
    - `series_id`, `timestamp`, `ground_truth`, `prediction`
- This makes it easy to save, analyze, and visualize results:

```python
pred_df = predict(dataloader, model_path='output/signalnet_model.pth')
pred_df.to_csv('example_output.csv', index=False)
```

## Model and Training
- The transformer model now expects 6 time features in addition to the value.
- See the main README for a full workflow and example.

---
**Note:** This documentation is a work in progress and will be updated as the project evolves. 