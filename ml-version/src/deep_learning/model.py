"""
Deep Learning model for Bitcoin price prediction.
Uses LSTM/GRU with historical price data, technical indicators, and news sentiment.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import os

class LSTMPredictor(nn.Module):
    """LSTM-based price prediction model"""
    
    def __init__(self, input_size: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout: float = 0.2, output_size: int = 1):
        """
        Initialize LSTM model
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            output_size: Number of outputs (1 for price prediction)
        """
        super(LSTMPredictor, self).__init__()
        
        self.hidden_layers = hidden_layers
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(
            nn.LSTM(input_size, hidden_layers[0], batch_first=True, dropout=dropout if len(hidden_layers) > 1 else 0)
        )
        
        # Additional LSTM layers
        for i in range(1, len(hidden_layers)):
            self.lstm_layers.append(
                nn.LSTM(hidden_layers[i-1], hidden_layers[i], batch_first=True, 
                       dropout=dropout if i < len(hidden_layers) - 1 else 0)
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Take only the last output
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer
        x = self.fc(x)
        
        return x

class GRUPredictor(nn.Module):
    """GRU-based price prediction model"""
    
    def __init__(self, input_size: int, hidden_layers: List[int] = [128, 64, 32], 
                 dropout: float = 0.2, output_size: int = 1):
        """
        Initialize GRU model
        
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            output_size: Number of outputs
        """
        super(GRUPredictor, self).__init__()
        
        self.hidden_layers = hidden_layers
        
        # Build GRU layers
        self.gru_layers = nn.ModuleList()
        
        # First GRU layer
        self.gru_layers.append(
            nn.GRU(input_size, hidden_layers[0], batch_first=True, dropout=dropout if len(hidden_layers) > 1 else 0)
        )
        
        # Additional GRU layers
        for i in range(1, len(hidden_layers)):
            self.gru_layers.append(
                nn.GRU(hidden_layers[i-1], hidden_layers[i], batch_first=True, 
                      dropout=dropout if i < len(hidden_layers) - 1 else 0)
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        """Forward pass"""
        # Pass through GRU layers
        for gru in self.gru_layers:
            x, _ = gru(x)
        
        # Take only the last output
        x = x[:, -1, :]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output layer
        x = self.fc(x)
        
        return x

class TransformerPredictor(nn.Module):
    """Transformer-based price prediction model"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.2, output_size: int = 1,
                 max_seq_length: int = 100):
        """
        Initialize Transformer model
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            output_size: Number of outputs
            max_seq_length: Maximum sequence length for positional encoding
        """
        super(TransformerPredictor, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        self.max_seq_length = max_seq_length
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Forward pass"""
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_length}")
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Transformer
        x = self.transformer(x)
        
        # Take mean of sequence
        x = x.mean(dim=1)
        
        # Dropout and output
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def create_model(model_type: str, input_size: int, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'transformer')
        input_size: Number of input features
        **kwargs: Additional arguments for the model
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMPredictor(input_size, **kwargs)
    elif model_type == 'gru':
        return GRUPredictor(input_size, **kwargs)
    elif model_type == 'transformer':
        return TransformerPredictor(input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def save_model(model: nn.Module, filepath: str):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model.__class__.__name__
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str, model: nn.Module) -> nn.Module:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {filepath}")
    return model
