"""
Transformer model for signal prediction.
"""
import torch
import torch.nn as nn
from typing import Optional

class SignalTransformer(nn.Module):
    """
    Enhanced transformer-based model for signal value prediction (regression).
    Features improved architecture with better normalization and scaling.
    """
    def __init__(self, input_dim: int, model_dim: int, num_heads: int, num_layers: int, output_dim: int, time_feat_dim: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.time_feat_dim = time_feat_dim
        
        # Enhanced data normalization layers
        self.input_norm = nn.LayerNorm(input_dim)
        self.time_feat_norm = nn.LayerNorm(time_feat_dim)
        
        # Improved input projection with residual connection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_feat_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Learnable positional encoding with better initialization
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, model_dim) * 0.02)  # Smaller initialization
        
        # Enhanced transformer layers with better configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1,
            dim_feedforward=model_dim * 4,
            activation='gelu'  # Use GELU for better performance
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=0.1,
            dim_feedforward=model_dim * 4,
            activation='gelu'  # Use GELU for better performance
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Enhanced output projection with residual connections
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, output_dim)
        )
        
        # Improved output scaling with better initialization
        self.output_scale = nn.Parameter(torch.ones(1) * 1.0)  # Start with scale 1
        self.output_bias = nn.Parameter(torch.zeros(1))  # Start with zero bias
        
        # Initialize weights for better training
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
