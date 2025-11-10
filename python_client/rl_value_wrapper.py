"""
Wrapper to use stable-baselines3 trained models as value functions in minimax.

This module provides a bridge between stable-baselines3 RL models and the
ModelLoader interface, allowing RL-trained policies to be used for evaluation.

Test with: python -c "from python_client.rl_value_wrapper import RLValueWrapper; ..."
"""

import torch
import numpy as np
import logging
from typing import Optional
from pathlib import Path

from .utils import GameState
from .tablut_env import TablutEnv

logger = logging.getLogger(__name__)


class RLValueWrapper:
    """
    Wrapper to use stable-baselines3 model's value function for minimax evaluation.
    
    This extracts the value function from a trained PPO/DQN policy and provides
    a clean interface matching ModelLoader.evaluate().
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize wrapper from saved stable-baselines3 model.
        
        Args:
            model_path: Path to .zip file saved by stable-baselines3
            device: Device for inference
        """
        try:
            from stable_baselines3 import PPO, DQN
        except ImportError:
            raise ImportError(
                "stable-baselines3 not installed. "
                "Install with: pip install stable-baselines3"
            )
        
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Load model
        logger.info(f"Loading SB3 model from {model_path}")
        try:
            # Try PPO first
            self.model = PPO.load(model_path, device=device)
            self.algo = "ppo"
            logger.info("Loaded PPO model")
        except Exception as e1:
            try:
                # Try DQN
                self.model = DQN.load(model_path, device=device)
                self.algo = "dqn"
                logger.info("Loaded DQN model")
            except Exception as e2:
                raise ValueError(
                    f"Could not load model from {model_path}. "
                    f"PPO error: {e1}, DQN error: {e2}"
                )
        
        # Extract value network
        self._extract_value_network()
        
        # Create environment for state conversion
        self.env = TablutEnv()
    
    def _extract_value_network(self):
        """Extract value network from policy."""
        policy = self.model.policy
        
        if self.algo == "ppo":
            # For PPO, we need to use the full policy to get value
            # The value_net requires features from mlp_extractor
            # So we'll use the policy's forward method or predict_values
            self.use_full_policy = True
            self.policy = policy
        else:  # DQN
            # For DQN, we use Q-network and take mean as value estimate
            if hasattr(policy, 'q_net'):
                self.value_net = policy.q_net
                self.use_full_policy = False
            else:
                raise ValueError("Cannot find Q-network in DQN policy")
    
    def evaluate(self, state: GameState) -> float:
        """
        Evaluate state using RL-trained value function.
        
        Args:
            state: GameState to evaluate
        
        Returns:
            Scalar value from white's perspective
        """
        # Convert state to observation
        obs = self.env._state_to_observation(state)
        
        # Flatten for MLP policy
        obs_flat = obs.flatten()
        obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0)
        
        # Move to device
        obs_tensor = obs_tensor.to(self.device)
        
        with torch.no_grad():
            if self.algo == "ppo":
                # For PPO, use the full policy to get value
                # The policy processes: obs -> features -> value
                if hasattr(self, 'use_full_policy') and self.use_full_policy:
                    # Use policy's forward method to get value
                    # PPO policy returns: (action, value, action_log_prob)
                    _, value, _ = self.policy(obs_tensor)
                    if value.dim() > 1:
                        value = value.squeeze()
                    return value.item()
                else:
                    # Fallback: try direct value_net (may not work due to architecture)
                    if hasattr(self.policy, 'value_net'):
                        # Need to extract features first
                        features = self.policy.mlp_extractor.forward_actor(obs_tensor)
                        value = self.policy.value_net(features)
                        if value.dim() > 1:
                            value = value.squeeze()
                        return value.item()
                    else:
                        raise ValueError("Cannot extract value from PPO policy")
            else:  # DQN
                # DQN Q-network: take mean over actions as value estimate
                q_values = self.value_net(obs_tensor)
                # Q-values are for all actions, take mean as state value
                value = q_values.mean().item()
                return value
    
    def evaluate_batch(self, states: list[GameState]) -> np.ndarray:
        """
        Evaluate multiple states in batch.
        
        Args:
            states: List of GameState objects
        
        Returns:
            Array of scalar values
        """
        # Convert states to observations
        obs_list = [self.env._state_to_observation(s) for s in states]
        obs_batch = np.array([obs.flatten() for obs in obs_list])
        obs_tensor = torch.from_numpy(obs_batch).float()
        
        # Move to device
        if hasattr(self.value_net, 'device'):
            obs_tensor = obs_tensor.to(self.value_net.device)
        else:
            obs_tensor = obs_tensor.to(self.device)
        
        with torch.no_grad():
            if self.algo == "ppo":
                values = self.value_net(obs_tensor)
                if isinstance(values, tuple):
                    values = values[0]
                if values.dim() > 1:
                    values = values.squeeze()
                return values.cpu().numpy()
            else:  # DQN
                q_values = self.value_net(obs_tensor)
                values = q_values.mean(dim=1)
                return values.cpu().numpy()


# Import logger
import logging
logger = logging.getLogger(__name__)

