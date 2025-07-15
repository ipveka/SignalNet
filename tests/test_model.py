"""
Unit tests for model.
"""
import torch
from signalnet.models.transformer import SignalTransformer

def test_signal_transformer_forward():
    batch_size = 4
    context_len = 8
    pred_len = 3
    input_dim = 1
    model_dim = 16
    num_heads = 2
    num_layers = 2
    output_dim = 1
    time_feat_dim = 6
    model = SignalTransformer(input_dim, model_dim, num_heads, num_layers, output_dim, time_feat_dim)
    context = torch.randn(batch_size, context_len)
    context_time_feat = torch.randn(batch_size, context_len, time_feat_dim)
    target_time_feat = torch.randn(batch_size, pred_len, time_feat_dim)
    # Forward without teacher forcing
    out = model(context, context_time_feat, target_time_feat)
    assert out.shape == (batch_size, pred_len)
    # Forward with teacher forcing
    target = torch.randn(batch_size, pred_len)
    out2 = model(context, context_time_feat, target_time_feat, target)
    assert out2.shape == (batch_size, pred_len)
