"""
Tests for MinimaxAgent.

Run with: pytest python_client/tests/test_minimax.py -v
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_client.minimax_agent import MinimaxAgent, TranspositionTable
from python_client.tablut_env import TablutEnv
from python_client.utils import GameState, Turn, Pawn
from python_client.heuristics import evaluate_heuristic


def test_transposition_table():
    """Test transposition table operations."""
    table = TranspositionTable(max_size=100)
    
    # Create dummy state
    board = np.zeros((9, 9), dtype=object)
    board.fill(Pawn.EMPTY)
    state = GameState(board=board, turn=Turn.WHITE)
    
    # Test put/get
    table.put(state, 42.0, depth=3)
    value = table.get(state, depth=3)
    assert value == 42.0
    
    # Test depth requirement
    value_shallow = table.get(state, depth=5)
    assert value_shallow is None  # Cached depth insufficient


def test_minimax_initialization():
    """Test MinimaxAgent initialization."""
    value_fn = lambda state: evaluate_heuristic(state, Turn.WHITE)
    agent = MinimaxAgent(
        value_fn=value_fn,
        max_depth=3,
        time_limit=10.0,
    )
    
    assert agent.max_depth == 3
    assert agent.time_limit == 10.0
    assert agent.value_fn is not None


def test_minimax_get_move():
    """Test minimax can compute a move."""
    env = TablutEnv()
    env.reset()
    
    value_fn = lambda state: evaluate_heuristic(state, Turn.WHITE)
    agent = MinimaxAgent(
        value_fn=value_fn,
        max_depth=2,  # Shallow for speed
        time_limit=5.0,
    )
    
    # Get move
    move = agent.get_move(env.state)
    
    assert move is not None
    assert isinstance(move, tuple)
    assert len(move) == 2
    from_pos, to_pos = move
    assert len(from_pos) == 2
    assert len(to_pos) == 2


def test_minimax_time_limit():
    """Test minimax respects time limit."""
    value_fn = lambda state: evaluate_heuristic(state, Turn.WHITE)
    agent = MinimaxAgent(
        value_fn=value_fn,
        max_depth=10,  # Deep enough to take time
        time_limit=0.1,  # Very short time limit
    )
    
    env = TablutEnv()
    env.reset()
    
    # Should return a move even if interrupted
    move = agent.get_move(env.state)
    assert move is not None


def test_minimax_handles_terminal():
    """Test minimax handles terminal states."""
    # Create a terminal state (white win)
    board = np.zeros((9, 9), dtype=object)
    board.fill(Pawn.EMPTY)
    # Place king on escape tile
    board[0, 0] = Pawn.KING
    state = GameState(board=board, turn=Turn.WHITEWIN)
    
    value_fn = lambda s: evaluate_heuristic(s, Turn.WHITE)
    agent = MinimaxAgent(value_fn=value_fn, max_depth=3, time_limit=5.0)
    
    # Should handle terminal state gracefully
    # (may raise error if no moves, which is expected)
    try:
        move = agent.get_move(state)
    except (ValueError, AttributeError):
        pass  # Expected for terminal state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

