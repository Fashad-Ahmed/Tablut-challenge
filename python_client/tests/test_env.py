"""
Tests for TablutEnv.

Run with: pytest python_client/tests/test_env.py -v
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_client.tablut_env import TablutEnv, TablutConfig
from python_client.utils import GameState, Turn, Pawn
from python_client.heuristics import find_king


def test_env_reset():
    """Test environment reset."""
    env = TablutEnv()
    obs, info = env.reset()
    
    assert obs.shape == (9, 9, 5)
    assert env.state is not None
    assert env.state.turn == Turn.WHITE
    assert info["turn"] == "W"


def test_initial_board_setup():
    """Test initial board has correct pieces."""
    env = TablutEnv()
    env.reset()
    
    # Check king is at center
    king_pos = find_king(env.state)
    assert king_pos == (4, 4)
    assert env.state.board[4, 4] == Pawn.KING
    
    # Check white soldiers count
    white_count = np.sum(env.state.board == Pawn.WHITE)
    assert white_count == 8
    
    # Check black soldiers count
    black_count = np.sum(env.state.board == Pawn.BLACK)
    assert black_count == 16


def test_legal_moves_generation():
    """Test legal moves are generated correctly."""
    env = TablutEnv()
    env.reset()
    
    legal_moves = env.get_legal_moves(env.state)
    assert len(legal_moves) > 0
    
    # All moves should be tuples of (from_pos, to_pos)
    for move in legal_moves:
        assert isinstance(move, tuple)
        assert len(move) == 2
        from_pos, to_pos = move
        assert len(from_pos) == 2
        assert len(to_pos) == 2


def test_step():
    """Test environment step function."""
    env = TablutEnv()
    obs, info = env.reset()
    
    # Get first legal move
    legal_moves = env.get_legal_moves(env.state)
    assert len(legal_moves) > 0
    
    from_pos, to_pos = legal_moves[0]
    from_idx = from_pos[0] * 9 + from_pos[1]
    to_idx = to_pos[0] * 9 + to_pos[1]
    action = np.array([from_idx, to_idx])
    
    # Step
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (9, 9, 5)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_terminal_detection():
    """Test terminal state detection."""
    env = TablutEnv()
    env.reset()
    
    # Test non-terminal state
    terminated, reward, info = env._check_terminal(env.state)
    # Initial state should not be terminal
    assert not terminated or env.state.turn in (Turn.WHITEWIN, Turn.BLACKWIN, Turn.DRAW)
    
    # Test king escape (white win)
    board = np.zeros((9, 9), dtype=object)
    board.fill(Pawn.EMPTY)
    board[0, 0] = Pawn.KING  # King on escape tile
    test_state = GameState(board=board, turn=Turn.WHITE)
    terminated, reward, info = env._check_terminal(test_state)
    assert terminated
    assert reward > 0  # White wins
    assert info["result"] == "white_win"
    
    # Test king capture (black win)
    board2 = np.zeros((9, 9), dtype=object)
    board2.fill(Pawn.EMPTY)
    # No king on board = captured
    test_state2 = GameState(board=board2, turn=Turn.BLACK)
    terminated, reward, info = env._check_terminal(test_state2)
    assert terminated
    assert reward < 0  # Black wins
    assert info["result"] == "black_win"


def test_observation_shape():
    """Test observation has correct shape."""
    env = TablutEnv()
    obs, _ = env.reset()
    
    assert obs.shape == (9, 9, 5)
    assert obs.dtype == np.float32
    
    # Check one-hot encoding (each cell should sum to 1)
    for i in range(9):
        for j in range(9):
            cell_sum = obs[i, j, :].sum()
            assert abs(cell_sum - 1.0) < 1e-6, f"Cell ({i}, {j}) does not sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

