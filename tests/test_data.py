"""
Unit tests for data loader.
"""
import pandas as pd
import numpy as np
from signalnet.data.loader import SignalDataset, SignalDataLoader

def test_signal_dataset_windowing():
    # Create dummy data
    n = 30
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='H'),
        'value': np.arange(n, dtype=np.float32)
    })
    context_length = 5
    prediction_length = 2
    dataset = SignalDataset(df, context_length, prediction_length)
    # There should be n - (context + pred) + 1 windows
    assert len(dataset) == n - (context_length + prediction_length) + 1
    # Check shapes
    context, prediction, context_time_feat, pred_time_feat = dataset[0]
    assert context.shape == (context_length,)
    assert prediction.shape == (prediction_length,)
    assert context_time_feat.shape == (context_length, 6)
    assert pred_time_feat.shape == (prediction_length, 6)

def test_signal_dataloader_split():
    n = 50
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n, freq='H'),
        'value': np.random.randn(n)
    })
    context_length = 6
    prediction_length = 3
    # Save to CSV
    csv_path = 'test_signal.csv'
    df.to_csv(csv_path, index=False)
    loader = SignalDataLoader(csv_path, context_length, prediction_length)
    train_ds, test_ds = loader.train_test_split(test_size=0.2)
    # Check that both splits are non-empty
    assert len(train_ds) > 0
    assert len(test_ds) > 0
    # The sum of windows should be less than or equal to the total possible windows
    total_windows = len(train_ds) + len(test_ds)
    expected = n - (context_length + prediction_length) + 1
    assert total_windows <= expected
