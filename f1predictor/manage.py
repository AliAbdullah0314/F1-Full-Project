#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


class TemporalFusionTransformer(nn.Module):
    def __init__(self, input_dim, static_dim, d_model=256, nhead=8, 
                 num_encoder_layers=6, dropout=0.1):
        super().__init__()
        
        # Static variable processing
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Variable selection network
        self.var_select = nn.Sequential(
            nn.Linear(input_dim + d_model, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # Temporal processing
        self.lstm = nn.LSTM(input_size=input_dim, 
                          hidden_size=d_model, 
                          num_layers=2,
                          batch_first=True,
                          dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model*2,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Gated residual connections
        self.gate = nn.Sequential(
            nn.Linear(d_model*2, d_model*2),
            nn.Sigmoid()
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, static, attention_mask=None):
        # Static context
        static_ctx = self.static_encoder(static)  # [batch_size, d_model]
        
        # Variable selection
        static_expanded = static_ctx.unsqueeze(1).expand(-1, x.size(1), -1)
        var_weights = self.var_select(torch.cat([x, static_expanded], dim=-1))
        x_selected = x * var_weights
        
        # Temporal processing
        lstm_out, _ = self.lstm(x_selected)  # [batch_size, seq_len, d_model]
        
        # Static context fusion
        static_fused = torch.cat([
            lstm_out,
            static_ctx.unsqueeze(1).expand(-1, lstm_out.size(1), -1)
        ], dim=-1)
        
        # Transformer processing
        if attention_mask is not None:
            padding_mask = ~attention_mask.bool()
            static_fused = static_fused.masked_fill(padding_mask.unsqueeze(-1), 0)
            
        transformer_out = self.transformer(static_fused, src_key_padding_mask=padding_mask)
        
        # Gated residual
        gate = self.gate(transformer_out)
        residual_out = gate * transformer_out + (1 - gate) * static_fused
        
        # Output projection
        output = self.output_layer(residual_out)
        return output.squeeze(-1)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        # Multiple LSTM layers with dropout
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  # Dropout between LSTM layers
        )
        
        # Dropout layer for the output of the last LSTM layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Batch normalization layer
        self.batch_norm = torch.nn.BatchNorm1d(hidden_size)
        
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # Pack the padded sequences
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Pass through LSTM layers
        packed_output, (hn, cn) = self.lstm(packed_input)
        
        # Unpack the output
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=87)
        
        # Apply batch normalization
        output = self.batch_norm(output.transpose(1, 2)).transpose(1, 2)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Apply the fully connected layer
        output = self.fc(output)
        
        # Squeeze the last dimension to match the shape of batch_y
        output = output.squeeze(-1)
        
        return output
    

    
def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'f1predictor.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
