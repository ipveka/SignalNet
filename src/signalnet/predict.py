"""
Prediction utility for SignalNet.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from signalnet.models.transformer import SignalTransformer
from typing import Optional, Dict, Any, List, Tuple
import os

def predict(
    dataloader: DataLoader,
    model_path: Optional[str] = 'output/signalnet_model.pth',
    device: Optional[torch.device] = None,
    model_kwargs: Optional[dict] = None,
    return_all_info: bool = False,
    ensemble_preds: bool = True
) -> Any:
    """
    Load a trained SignalNet model and make predictions on a dataset.
    Now returns a DataFrame directly for easy postprocessing.
    
    Args:
        dataloader: DataLoader containing the test dataset
        model_path: Path to the trained model
        device: Device to run predictions on
        model_kwargs: Model configuration parameters
        return_all_info: Whether to return additional info
        ensemble_preds: Whether to average duplicate predictions by timestamp (default: True)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the dataset to determine feature dimensions
    dataset = dataloader.dataset
    time_feat_dim = dataset.time_features.shape[1] if hasattr(dataset, 'time_features') else 6
    
    if model_kwargs is None:
        model_kwargs = dict(input_dim=1, model_dim=256, num_heads=16, num_layers=6, output_dim=1, time_feat_dim=time_feat_dim)
    else:
        # Update time_feat_dim if not provided or different
        if 'time_feat_dim' not in model_kwargs or model_kwargs['time_feat_dim'] != time_feat_dim:
            model_kwargs['time_feat_dim'] = time_feat_dim
    
    model = SignalTransformer(**model_kwargs)
    model.to(device)
    
    # Try to load the model, but handle compatibility issues
    model_loaded = False
    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model_loaded = True
            print(f"[PREDICT] Successfully loaded model from {model_path}")
        except RuntimeError as e:
            print(f"[PREDICT] Model compatibility issue: {e}")
            print(f"[PREDICT] Expected {time_feat_dim} features, but model was trained with different dimensions")
            print(f"[PREDICT] Using untrained model for predictions (results may be poor)")
            model_loaded = False
    
    if not model_loaded:
        print(f"[PREDICT] Using untrained model with {time_feat_dim} features")
    
    model.eval()
    all_preds = []
    all_targets = []
    window_indices = []
    series_ids = []
    timestamps = []
    batch_size = dataloader.batch_size if dataloader.batch_size is not None else 1
    # Print features used (if available)
    features_used = getattr(dataset, 'features_used', None)
    if features_used is not None:
        print(f"[PREDICT] Features used in the model: {features_used}")
        print(f"[PREDICT] Number of features: {len(features_used)}")
    else:
        print(f"[PREDICT] Features used in the model: {time_feat_dim} features (auto-detected)")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            context, target, context_time_feat, target_time_feat = batch
            context = context.to(device)
            context_time_feat = context_time_feat.to(device)
            target_time_feat = target_time_feat.to(device)
            output = model(context, context_time_feat, target_time_feat)
            
            # Denormalize predictions and targets if dataset has normalization
            if hasattr(dataset, 'denormalize'):
                output_denorm = dataset.denormalize(output.cpu().numpy())
                target_denorm = dataset.denormalize(target.numpy())
            else:
                output_denorm = output.cpu().numpy()
                target_denorm = target.numpy()
            
            all_preds.append(output_denorm)
            all_targets.append(target_denorm)
            ds_data = getattr(dataset, 'data', None)
            ds_indices = getattr(dataset, 'indices', None)
            if ds_data is not None and hasattr(ds_data, 'columns') and 'series_id' in ds_data.columns and 'timestamp' in ds_data.columns:
                start_idx = batch_idx * batch_size
                for i in range(len(context)):
                    idx = ds_indices[start_idx + i] if ds_indices is not None else start_idx + i
                    sid = ds_data.iloc[idx]['series_id']
                    series_ids.append(sid)
                    window_indices.append(idx)
                    ts_start = idx + dataset.context_length
                    ts_end = ts_start + dataset.prediction_length
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
    
    # Apply ensemble averaging if requested
    if ensemble_preds and len(pred_df) > 0:
        print(f"[PREDICT] Applying ensemble averaging to {len(pred_df)} predictions...")
        
        # Group by series_id, timestamp and average ground_truth and prediction
        pred_df_ensemble = pred_df.groupby(['series_id', 'timestamp']).agg({
            'ground_truth': 'mean',
            'prediction': 'mean'
        }).reset_index()
        
        # Sort by series_id and timestamp for clean output
        pred_df_ensemble = pred_df_ensemble.sort_values(['series_id', 'timestamp'])
        
        print(f"[PREDICT] Ensemble averaging reduced to {len(pred_df_ensemble)} unique predictions")
        print(f"[PREDICT] Removed {len(pred_df) - len(pred_df_ensemble)} duplicate timestamps")
        
        pred_df = pred_df_ensemble
    
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