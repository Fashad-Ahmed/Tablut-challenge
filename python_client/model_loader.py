"""
PyTorch model loader for value network.

Provides:
- Lightweight CNN/MLP architecture for board evaluation
- CPU-optimized inference (M1 MPS support)
- Model save/load utilities
- evaluate(state) method returning scalar value

Test with: pytest python_client/tests/test_model_loader.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from .utils import GameState, Pawn
from .tablut_env import TablutEnv


class ValueNetwork(nn.Module):
    """
    Neural network for evaluating Tablut board positions.
    
    Architecture:
    - Input: 9x9x5 one-hot encoded board (or 9x9x6 with turn)
    - CNN layers for spatial patterns
    - Fully connected layers for final evaluation
    - Output: Single scalar value (white perspective)
    """
    
    def __init__(self, input_channels: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Global average pooling + flatten
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Activation
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch_size, channels, 9, 9) or (channels, 9, 9)
        
        Returns:
            (batch_size, 1) or (1,) scalar value
        """
        # Add batch dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Pool and flatten
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)


class SimpleValueNetwork(nn.Module):
    """
    Simpler MLP-based value network (faster, less parameters).
    
    Good for initial training or when compute is limited.
    """
    
    def __init__(self, input_dim: int = 9 * 9 * 5, hidden_dim: int = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x.squeeze(-1)


class ModelLoader:
    """
    Loader and evaluator for value network models.
    
    Handles:
    - Device selection (CPU/MPS/CUDA)
    - Model loading from checkpoint
    - State evaluation
    - Batch inference
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, 
                 use_simple: bool = False):
        """
        Initialize model loader.
        
        Args:
            model_path: Path to .pt checkpoint file
            device: Override device ("cpu", "mps", "cuda")
            use_simple: Use SimpleValueNetwork instead of ValueNetwork
        """
        # Device selection (CPU-first for competition, MPS for M1 dev)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Set CPU threads for efficiency (important for competition)
        if self.device.type == "cpu":
            torch.set_num_threads(4)  # Match VM CPU count
        
        # Initialize model
        if use_simple:
            self.model = SimpleValueNetwork()
        else:
            self.model = ValueNetwork()
        
        self.model.to(self.device)
        self.model.eval()  # Inference mode
        
        # Load checkpoint if provided
        if model_path is not None:
            self.load_model(model_path)
        else:
            # Initialize with random weights (for training from scratch)
            self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights (Xavier uniform)."""
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def load_model(self, model_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to .pt file
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # Assume entire dict is state_dict
                self.model.load_state_dict(checkpoint)
        else:
            # Assume checkpoint is state_dict directly
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
    
    def save_model(self, model_path: str, metadata: Optional[dict] = None) -> None:
        """
        Save model to checkpoint.
        
        Args:
            model_path: Output path
            metadata: Optional dict with training info (epoch, loss, etc.)
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_type": type(self.model).__name__,
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        torch.save(checkpoint, model_path)
    
    def evaluate(self, state: GameState) -> float:
        """
        Evaluate a game state and return scalar value.
        
        Args:
            state: GameState to evaluate
        
        Returns:
            Scalar value from white's perspective:
            - Positive = good for white
            - Negative = good for black
        """
        # Convert state to tensor
        obs = self._state_to_tensor(state)
        
        with torch.no_grad():
            value = self.model(obs)
            return value.item()
    
    def evaluate_batch(self, states: list[GameState]) -> np.ndarray:
        """
        Evaluate multiple states in batch (faster).
        
        Args:
            states: List of GameState objects
        
        Returns:
            Array of scalar values
        """
        # Convert to batch tensor
        obs_list = [self._state_to_tensor(s) for s in states]
        obs_batch = torch.stack(obs_list).to(self.device)
        
        with torch.no_grad():
            values = self.model(obs_batch)
            return values.cpu().numpy()
    
    def _state_to_tensor(self, state: GameState) -> torch.Tensor:
        """Convert GameState to model input tensor."""
        # Use TablutEnv's observation conversion
        env = TablutEnv()
        obs = env._state_to_observation(state)
        
        # Convert to tensor and add channel dimension if needed
        tensor = torch.from_numpy(obs).float()
        
        # Model expects (C, H, W) format
        if tensor.dim() == 3:
            tensor = tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        return tensor.to(self.device)


def create_model(use_simple: bool = False, hidden_dim: int = 128) -> nn.Module:
    """
    Factory function to create a new model.
    
    Args:
        use_simple: Use SimpleValueNetwork
        hidden_dim: Hidden dimension size
    
    Returns:
        Initialized model
    """
    if use_simple:
        return SimpleValueNetwork(hidden_dim=hidden_dim)
    else:
        return ValueNetwork(hidden_dim=hidden_dim)

