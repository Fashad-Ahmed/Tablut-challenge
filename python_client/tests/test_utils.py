"""
Tests for utils module.

Run with: pytest python_client/tests/test_utils.py -v
"""

import pytest
import json
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python_client.utils import (
    GameState, Pawn, Turn,
    json_to_state, state_to_json,
    move_to_action, action_to_move,
    set_seed
)


def test_json_to_state():
    """Test JSON to state conversion."""
    # Create sample JSON (matching Java server format)
    json_data = {
        "board": [
            ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
            ["O", "O", "W", "W", "K", "W", "W", "O", "O"],
            ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
            ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
        ],
        "turn": "W"
    }
    
    json_str = json.dumps(json_data)
    state = json_to_state(json_str)
    
    assert state.board.shape == (9, 9)
    assert state.turn == Turn.WHITE
    assert state.board[4, 4] == Pawn.KING


def test_move_to_action():
    """Test move to action conversion."""
    from_pos = (4, 4)
    to_pos = (4, 5)
    action = move_to_action(from_pos, to_pos, Turn.WHITE)
    
    assert action["from"] == "e5"
    assert action["to"] == "e6"
    assert action["turn"] == "W"


def test_action_to_move():
    """Test action to move conversion."""
    action = {"from": "e5", "to": "e6", "turn": "W"}
    from_pos, to_pos = action_to_move(action)
    
    assert from_pos == (4, 4)  # e5 = row 4, col 4
    assert to_pos == (4, 5)   # e6 = row 4, col 5


def test_move_roundtrip():
    """Test move -> action -> move roundtrip."""
    original_from = (2, 3)
    original_to = (5, 7)
    
    action = move_to_action(original_from, original_to, Turn.BLACK)
    from_pos, to_pos = action_to_move(action)
    
    assert from_pos == original_from
    assert to_pos == original_to


def test_set_seed():
    """Test seed setting."""
    set_seed(42)
    val1 = np.random.rand()
    
    set_seed(42)
    val2 = np.random.rand()
    
    assert val1 == val2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

