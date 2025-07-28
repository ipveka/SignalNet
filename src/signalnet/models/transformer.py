"""
Transformer model for signal prediction.
"""
import torch
import torch.nn as nn
from typing import Optional

class SignalTransformer(nn.Module):
    """
    Transformer-based model for signal value prediction (regression).
    """
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, output_dim: int, time_feat_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.time_feat_dim = time_feat_dim
        
        # Data normalization layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.time_feat_norm = nn.LayerNorm(time_feat_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim + time_feat_dim, model_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, model_dim))  # max seq len 512
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1,  # Add dropout for regularization
            dim_feedforward=model_dim * 4  # Larger feedforward network
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1,  # Add dropout for regularization
            dim_feedforward=model_dim * 4  # Larger feedforward network
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection with scaling
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, output_dim)
        )
        
        # Output scaling layer to match data scale
        self.output_scale = nn.Parameter(torch.ones(1) * 5.0)  # Learnable scale factor
        self.output_bias = nn.Parameter(torch.zeros(1))  # Learnable bias

    def forward(
        self,
        context: torch.Tensor,  # (batch, context_len, input_dim)
        context_time_feat: torch.Tensor,  # (batch, context_len, time_feat_dim)
        target_time_feat: torch.Tensor,  # (batch, pred_len, time_feat_dim)
        target: Optional[torch.Tensor] = None  # (batch, pred_len, input_dim) for teacher forcing
    ) -> torch.Tensor:
        # Normalize inputs - LayerNorm expects the normalized dimension to be the last
        # context: (batch, seq_len) -> (batch, seq_len, 1) for LayerNorm
        context = context.unsqueeze(-1)  # Add input_dim dimension
        context = self.input_norm(context)
        context = context.squeeze(-1)  # Remove the added dimension
        
        context_time_feat = self.time_feat_norm(context_time_feat)
        target_time_feat = self.time_feat_norm(target_time_feat)
        
        # Concatenate value and time features
        # context: (batch, seq_len) -> (batch, seq_len, 1) to match context_time_feat: (batch, seq_len, time_feat_dim)
        context = context.unsqueeze(-1)
        enc_in = torch.cat([context, context_time_feat], dim=-1)
        enc_in = self.input_proj(enc_in) + self.pos_encoder[:, :enc_in.size(1), :]
        
        # Encode context
        memory = self.encoder(enc_in)
        
        # Prepare decoder input (use zeros if not teacher forcing)
        if target is not None:
            target_norm = target.unsqueeze(-1)  # Add input_dim dimension
            target_norm = self.input_norm(target_norm)
            target_norm = target_norm.squeeze(-1)  # Remove the added dimension
            target_norm = target_norm.unsqueeze(-1)  # Add input_dim dimension for concatenation
            dec_in = torch.cat([target_norm, target_time_feat], dim=-1)
        else:
            dec_in = torch.cat([torch.zeros_like(target_time_feat[..., :self.input_dim]), target_time_feat], dim=-1)
        
        dec_in = self.input_proj(dec_in) + self.pos_encoder[:, :dec_in.size(1), :]
        
        # Decode
        out = self.decoder(dec_in, memory)
        
        # Project to output
        out = self.output_proj(out)
        
        # Apply scaling to match data scale
        out = out * self.output_scale + self.output_bias
        
        return out.squeeze(-1)
