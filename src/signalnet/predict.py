"""
Prediction utility for SignalNet.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from signalnet.models.transformer import SignalTransformer
from typing import Optional, Dict, Any, List, Tuple

def predict(
    dataloader: DataLoader,
    model_path: Optional[str] = 'output/signalnet_model.pth',
    device: Optional[torch.device] = None,
    model_kwargs: Optional[dict] = None,
    return_all_info: bool = False
) -> Any:
    """
    Load a trained SignalNet model and make predictions on a dataset.
    Now returns a DataFrame directly for easy postprocessing.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_kwargs is None:
        model_kwargs = dict(input_dim=1, model_dim=32, num_heads=2, num_layers=2, output_dim=1, time_feat_dim=6)
    model = SignalTransformer(**model_kwargs)
    model.to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_preds = []
    all_targets = []
    window_indices = []
    series_ids = []
    timestamps = []
    batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1
    # Print features used (if available)
    ds = dataloader.dataset
    features_used = getattr(ds, 'features_used', None)
    if features_used is not None:
        print(f"[PREDICT] Features used in the model: {features_used}")
    else:
        print("[PREDICT] Features used in the model: ['day_of_week (normalized)', 'hour_of_day (normalized)', 'minute_of_hour (normalized)', 'month (normalized)', 'day_of_month (normalized)', 'is_weekend']")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            context, target, context_time_feat, target_time_feat = batch
            context = context.to(device)
            context_time_feat = context_time_feat.to(device)
            target_time_feat = target_time_feat.to(device)
            output = model(context, context_time_feat, target_time_feat)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.numpy())
            ds_data = getattr(ds, 'data', None)
            ds_indices = getattr(ds, 'indices', None)
            if ds_data is not None and hasattr(ds_data, 'columns') and 'series_id' in ds_data.columns and 'timestamp' in ds_data.columns:
                start_idx = batch_idx * batch_size
                for i in range(len(context)):
                    idx = ds_indices[start_idx + i] if ds_indices is not None else start_idx + i
                    sid = ds_data.iloc[idx]['series_id']
                    series_ids.append(sid)
                    window_indices.append(idx)
                    ts_start = idx + ds.context_length
                    ts_end = ts_start + ds.prediction_length
                    ts_list = list(ds_data.iloc[ts_start:ts_end]['timestamp'])
                    timestamps.append(ts_list)
            else:
                for i in range(len(context)):
                    window_indices.append(batch_idx * batch_size + i)
                    series_ids.append(None)
                    timestamps.append([None] * (target.shape[1] if len(target.shape) > 1 else 1))
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    # Build DataFrame directly
    rows = []
    for i in range(preds_arr.shape[0]):
        sid = series_ids[i]
        for j in range(preds_arr.shape[1]):
            ts = timestamps[i][j]
            rows.append({
                'series_id': sid,
                'timestamp': ts,
                'ground_truth': targets_arr[i, j],
                'prediction': preds_arr[i, j]
            })
    import pandas as pd
    pred_df = pd.DataFrame(rows)
    if return_all_info:
        return pred_df, {
            'predictions': preds_arr,
            'ground_truth': targets_arr,
            'window_indices': window_indices,
            'series_ids': series_ids,
            'timestamps': timestamps
        }
    else:
        return pred_df