"""
Deep Learning trainer for Bitcoin price prediction.
Handles training, validation, and model checkpointing.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
import os
import json
from datetime import datetime
from tqdm import tqdm

class DeepLearningTrainer:
    """Trainer for deep learning models"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
            device: Device to train on ('cpu', 'cuda', 'mps')
        """
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size
            
        Returns:
            Training and validation data loaders
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 100, checkpoint_dir: str = 'checkpoints',
             save_interval: int = 10):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save checkpoint every N epochs
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"\nStarting training on {self.device}...")
        print(f"{'='*60}")
        
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch + 1)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                self.save_checkpoint(best_model_path, epoch, val_loss, is_best=True)
                print(f"  → New best model! Saved to {best_model_path}")
            else:
                patience_counter += 1
            
            # Save checkpoint at intervals
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
                self.save_checkpoint(checkpoint_path, epoch, val_loss)
                print(f"  → Checkpoint saved to {checkpoint_path}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
                break
        
        print(f"\n{'='*60}")
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")
        
        # Save training history
        self.save_history(checkpoint_dir)
    
    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'history': self.history,
            'config': self.config,
            'is_best': is_best
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['val_loss']
        print(f"Checkpoint loaded from {filepath}")
        print(f"Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.6f}")
    
    def save_history(self, output_dir: str):
        """Save training history"""
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
