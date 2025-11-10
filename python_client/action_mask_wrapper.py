"""
Action masking wrapper for Tablut environment.

Filters out illegal moves to improve RL training efficiency.
This wrapper ensures the agent only considers legal moves.

Test with: pytest python_client/tests/test_env.py
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from gymnasium import spaces
from gymnasium.core import Wrapper, ActType, ObsType

from .tablut_env import TablutEnv


class ActionMaskWrapper(Wrapper):
    """
    Wrapper that masks illegal actions for stable-baselines3.
    
    This improves training efficiency by preventing the agent from
    attempting illegal moves. The action space is modified to only
    include legal moves at each step.
    """
    
    def __init__(self, env: TablutEnv):
        """
        Initialize action mask wrapper.
        
        Args:
            env: TablutEnv to wrap
        """
        super().__init__(env)
        
        # Store original action space
        self.original_action_space = env.action_space
        
        # For now, keep the same action space but mask in step()
        # A more advanced implementation could use a dynamic action space
        self.action_space = env.action_space
        
        # Cache for legal moves
        self._legal_moves: Optional[list] = None
        self._action_mask: Optional[np.ndarray] = None
    
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[ObsType, Dict]:
        """Reset environment and update action mask."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Update legal moves
        self._update_action_mask()
        
        # Add action mask to info
        info['action_mask'] = self._action_mask
        info['legal_moves'] = self._legal_moves
        
        return obs, info
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict]:
        """
        Step environment with action masking.
        
        If action is illegal, return large penalty and terminate.
        """
        # Check if action is legal
        if self._action_mask is not None:
            from_idx, to_idx = action[0], action[1]
            action_idx = from_idx * 81 + to_idx
            
            # Check if this action is in legal moves
            is_legal = False
            for from_pos, to_pos in self._legal_moves:
                legal_from_idx = from_pos[0] * 9 + from_pos[1]
                legal_to_idx = to_pos[0] * 9 + to_pos[1]
                if legal_from_idx == from_idx and legal_to_idx == to_idx:
                    is_legal = True
                    break
            
            if not is_legal:
                # Illegal move: large penalty
                obs = self.env._state_to_observation(self.env.state)
                return obs, -1000.0, True, False, {
                    "error": "illegal_move",
                    "action_mask": self._action_mask,
                }
        
        # Execute legal action
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update action mask for next step
        self._update_action_mask()
        info['action_mask'] = self._action_mask
        info['legal_moves'] = self._legal_moves
        
        return obs, reward, terminated, truncated, info
    
    def _update_action_mask(self):
        """Update the action mask based on current legal moves."""
        if self.env.state is None:
            self._legal_moves = []
            self._action_mask = np.zeros(81 * 81, dtype=bool)
            return
        
        # Get legal moves
        self._legal_moves = self.env.get_legal_moves(self.env.state)
        
        # Create binary mask (1 = legal, 0 = illegal)
        self._action_mask = np.zeros(81 * 81, dtype=bool)
        for from_pos, to_pos in self._legal_moves:
            from_idx = from_pos[0] * 9 + from_pos[1]
            to_idx = to_pos[0] * 9 + to_pos[1]
            action_idx = from_idx * 81 + to_idx
            if 0 <= action_idx < len(self._action_mask):
                self._action_mask[action_idx] = True

