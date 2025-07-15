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
        self.input_proj = nn.Linear(input_dim + time_feat_dim, model_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, model_dim))  # max seq len 512
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(
        self,
        context: torch.Tensor,  # (batch, context_len, input_dim)
        context_time_feat: torch.Tensor,  # (batch, context_len, time_feat_dim)
        target_time_feat: torch.Tensor,  # (batch, pred_len, time_feat_dim)
        target: Optional[torch.Tensor] = None  # (batch, pred_len, input_dim) for teacher forcing
    ) -> torch.Tensor:
        # Concatenate value and time features
        enc_in = torch.cat([context.unsqueeze(-1), context_time_feat], dim=-1)
        enc_in = self.input_proj(enc_in) + self.pos_encoder[:, :enc_in.size(1), :]
        memory = self.encoder(enc_in)
        # Prepare decoder input (use zeros if not teacher forcing)
        if target is not None:
            dec_in = torch.cat([target.unsqueeze(-1), target_time_feat], dim=-1)
        else:
            dec_in = torch.cat([torch.zeros_like(target_time_feat[..., :1]), target_time_feat], dim=-1)
        dec_in = self.input_proj(dec_in) + self.pos_encoder[:, :dec_in.size(1), :]
        out = self.decoder(dec_in, memory)
        return self.output_head(out).squeeze(-1)
