from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from pathlib import Path
import json

class BaseModel(ABC, nn.Module):
    """
    Base class for all EarthSense AI models.
    Provides common functionality for model loading, saving, and inference.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model with configuration.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the model architecture and load weights if available"""
        pass
    
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model"""
        pass
    
    def save(self, path: str):
        """
        Save the model and its configuration.
        
        Args:
            path: Directory path where to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.model.state_dict(), path / "model.pth")
        
        # Save configuration
        with open(path / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """
        Load a saved model.
        
        Args:
            path: Directory path where the model is saved
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = cls(config)
        
        # Load weights
        model.model.load_state_dict(torch.load(path / "model.pth", map_location=model.device))
        model.model.to(model.device)
        model.model.eval()
        
        return model
    
    def predict(self, x, **kwargs):
        """
        Run inference on input data.
        
        Args:
            x: Input data (tensor or list of tensors)
            **kwargs: Additional arguments for the forward pass
            
        Returns:
            Model predictions
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, (list, tuple)):
                x = [xi.to(self.device) if torch.is_tensor(xi) else xi for xi in x]
            else:
                x = x.to(self.device) if torch.is_tensor(x) else x
            return self.forward(x, **kwargs)
    
    def train_model(self, train_loader, val_loader, num_epochs: int, optimizer, criterion, 
                   scheduler=None, device=None, log_interval: int = 10):
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Number of training epochs
            optimizer: Optimizer to use
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
            device: Device to train on (default: use self.device)
            log_interval: Log training progress every N batches
            
        Returns:
            Training history
        """
        if device is None:
            device = self.device
            
        self.model.train()
        history = {'train_loss': [], 'val_loss': [], 'metrics': []}
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            
            for i, batch in enumerate(train_loader):
                # Move data to device
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                
                if i % log_interval == log_interval - 1:
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / log_interval:.4f}')
                    running_loss = 0.0
            
            # Step the scheduler if provided
            if scheduler is not None:
                scheduler.step()
            
            # Validation
            val_loss, metrics = self.evaluate(val_loader, criterion, device)
            
            # Save epoch statistics
            epoch_train_loss = running_loss / len(train_loader)
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(metrics)
            
            print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Validation Metrics: {metrics}')
        
        return history
    
    def evaluate(self, data_loader, criterion, device=None):
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            criterion: Loss function
            device: Device to evaluate on (default: use self.device)
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        if device is None:
            device = self.device
            
        self.model.eval()
        running_loss = 0.0
        
        # Initialize metrics
        metrics = {}
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Update metrics here based on your task
                # Example: accuracy, precision, recall, etc.
        
        avg_loss = running_loss / len(data_loader.dataset)
        
        return avg_loss, metrics
