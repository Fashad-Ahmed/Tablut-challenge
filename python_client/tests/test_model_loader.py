"""
Tests for ModelLoader.

Run with: pytest python_client/tests/test_model_loader.py -v
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_client.model_loader import ModelLoader, ValueNetwork, SimpleValueNetwork
from python_client.tablut_env import TablutEnv
from python_client.utils import GameState, Turn, Pawn


def test_value_network_forward():
    """Test ValueNetwork forward pass."""
    model = ValueNetwork()
    
    # Create dummy input (batch_size=1, channels=5, height=9, width=9)
    x = torch.randn(1, 5, 9, 9)
    output = model(x)
    
    assert output.shape == (1,)
    assert isinstance(output.item(), float)


def test_simple_value_network_forward():
    """Test SimpleValueNetwork forward pass."""
    model = SimpleValueNetwork()
    
    # Create dummy input (flattened)
    x = torch.randn(1, 9 * 9 * 5)
    output = model(x)
    
    assert output.shape == (1,)
    assert isinstance(output.item(), float)


def test_model_loader_initialization():
    """Test ModelLoader initialization."""
    loader = ModelLoader(device="cpu")
    
    assert loader.device.type == "cpu"
    assert loader.model is not None


def test_model_loader_evaluate():
    """Test model evaluation on game state."""
    env = TablutEnv()
    env.reset()
    
    loader = ModelLoader(device="cpu")
    value = loader.evaluate(env.state)
    
    assert isinstance(value, float)
    assert not np.isnan(value)
    assert not np.isinf(value)


def test_model_save_load():
    """Test model save and load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pt"
        
        # Create and save model
        loader1 = ModelLoader(device="cpu")
        loader1.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loader2 = ModelLoader(model_path=str(model_path), device="cpu")
        
        # Compare outputs
        env = TablutEnv()
        env.reset()
        
        value1 = loader1.evaluate(env.state)
        value2 = loader2.evaluate(env.state)
        
        # Values should be similar (may differ slightly due to float precision)
        assert abs(value1 - value2) < 1e-5


def test_model_loader_batch_evaluation():
    """Test batch evaluation."""
    env = TablutEnv()
    states = []
    
    # Create a few states
    for _ in range(3):
        env.reset()
        states.append(env.state)
    
    loader = ModelLoader(device="cpu")
    values = loader.evaluate_batch(states)
    
    assert len(values) == 3
    assert all(isinstance(v, (float, np.floating)) for v in values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

