"""
Gymnasium-compatible Tablut game environment for training.

Implements:
- reset(): Initialize game to starting position
- step(action): Apply move and return (obs, reward, done, truncated, info)
- render(): Optional visualization
- Legal move generation
- Terminal state detection
- Reward shaping

Test with: pytest python_client/tests/test_env.py
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces

from .utils import GameState, Pawn, Turn
from .heuristics import find_king

# Board constants
BOARD_SIZE = 9
CASTLE_ROW, CASTLE_COL = 4, 4

# Camp positions (Ashton rules)
CAMPS = {
    (0, 3), (0, 4), (0, 5), (1, 4),  # Top
    (8, 3), (8, 4), (8, 5), (7, 4),  # Bottom
    (3, 0), (4, 0), (5, 0), (4, 1),  # Left
    (3, 8), (4, 8), (5, 8), (4, 7),  # Right
}

# Escape tiles (edges excluding camps)
ESCAPE_TILES = set()
for i in range(9):
    for j in range(9):
        if (i == 0 or i == 8 or j == 0 or j == 8) and (i, j) not in CAMPS:
            ESCAPE_TILES.add((i, j))


@dataclass
class TablutConfig:
    """Configuration for Tablut environment."""
    board_size: int = 9
    max_steps: int = 200  # Max moves per game
    reward_win: float = 100.0
    reward_loss: float = -100.0
    reward_draw: float = 0.0
    reward_step: float = -0.1  # Small penalty per step to encourage efficiency


class TablutEnv(gym.Env):
    """
    Tablut game environment following Gymnasium API.
    
    Observation space: 9x9x5 (one-hot encoding of board state)
    Action space: Discrete(81*81) or MultiDiscrete for from/to positions
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    
    def __init__(self, config: Optional[TablutConfig] = None):
        super().__init__()
        self.config = config or TablutConfig()
        
        # Observation: 9x9x5 (one-hot: EMPTY, WHITE, BLACK, KING, THRONE)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(9, 9, 5), dtype=np.float32
        )
        
        # Action: from_pos (81) and to_pos (81) = 81*81 possible moves
        # Use MultiDiscrete for easier indexing
        self.action_space = spaces.MultiDiscrete([81, 81])
        
        self.state: Optional[GameState] = None
        self.step_count: int = 0
        self._legal_moves_cache: Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = None
        self._state_history: List[int] = []  # For repetition tracking
        self._black_starting_positions: set = set()  # Track black pieces that started in camps
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Returns:
            observation: 9x9x5 one-hot encoded board
            info: Dict with game metadata
        """
        super().reset(seed=seed)
        
        # Initialize board
        board = np.zeros((9, 9), dtype=object)
        board.fill(Pawn.EMPTY)
        
        # Place castle/throne
        board[CASTLE_ROW, CASTLE_COL] = Pawn.KING
        
        # Place white soldiers (8 around king)
        white_positions = [
            (2, 4), (3, 4), (5, 4), (6, 4),  # Vertical
            (4, 2), (4, 3), (4, 5), (4, 6),  # Horizontal
        ]
        for pos in white_positions:
            board[pos[0], pos[1]] = Pawn.WHITE
        
        # Place black soldiers (16 in camps)
        black_positions = [
            (0, 3), (0, 4), (0, 5), (1, 4),  # Top
            (8, 3), (8, 4), (8, 5), (7, 4),  # Bottom
            (3, 0), (4, 0), (5, 0), (4, 1),  # Left
            (3, 8), (4, 8), (5, 8), (4, 7),  # Right
        ]
        for pos in black_positions:
            board[pos[0], pos[1]] = Pawn.BLACK
        
        self.state = GameState(board=board, turn=Turn.WHITE)
        self.step_count = 0
        self._legal_moves_cache = None
        self._state_history = []
        
        # Track initial black positions (for camp movement rules)
        self._black_starting_positions = set(black_positions)
        
        # Add initial state to history
        self._state_history.append(self.state.hash())
        
        obs = self._state_to_observation(self.state)
        info = {"turn": self.state.turn.value, "step": self.step_count}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [from_idx, to_idx] where each is 0-80 (flattened 9x9)
        
        Returns:
            observation: New board state
            reward: Reward for this step
            terminated: Game ended (win/loss/draw)
            truncated: Max steps reached
            info: Additional info
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        from_idx, to_idx = action[0], action[1]
        from_row, from_col = from_idx // 9, from_idx % 9
        to_row, to_col = to_idx // 9, to_idx % 9
        
        from_pos = (from_row, from_col)
        to_pos = (to_row, to_col)
        
        # Validate move
        legal_moves = self.get_legal_moves(self.state)
        if (from_pos, to_pos) not in legal_moves:
            # Illegal move: large penalty and end episode
            obs = self._state_to_observation(self.state)
            return obs, -1000.0, True, False, {"error": "illegal_move"}
        
        # Apply move
        new_state = self._apply_move(self.state, from_pos, to_pos)
        self.state = new_state
        self.step_count += 1
        self._legal_moves_cache = None
        
        # Track state for repetition detection
        if hasattr(self, '_state_history'):
            self._state_history.append(new_state.hash())
            # Keep only last 50 states to prevent memory growth
            if len(self._state_history) > 50:
                self._state_history.pop(0)
        
        # Check terminal
        terminated, reward, info = self._check_terminal(new_state)
        truncated = self.step_count >= self.config.max_steps
        
        # Reward shaping
        if not terminated:
            reward = self.config.reward_step
        
        obs = self._state_to_observation(new_state)
        info.update({"turn": new_state.turn.value, "step": self.step_count})
        
        return obs, reward, terminated, truncated, info
    
    def get_legal_moves(self, state: GameState) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Generate all legal moves for current player.
        
        Returns:
            List of (from_pos, to_pos) tuples
        """
        if self._legal_moves_cache is not None and state == self.state:
            return self._legal_moves_cache
        
        moves = []
        player_pawn = Pawn.WHITE if state.turn == Turn.WHITE else Pawn.BLACK
        
        # Find all pieces of current player
        for i in range(9):
            for j in range(9):
                pawn = state.board[i, j]
                if pawn == player_pawn or (pawn == Pawn.KING and state.turn == Turn.WHITE):
                    # Generate moves for this piece
                    piece_moves = self._generate_moves_for_piece(state, (i, j))
                    moves.extend(piece_moves)
        
        self._legal_moves_cache = moves
        return moves
    
    def _generate_moves_for_piece(self, state: GameState, from_pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Generate legal moves for a single piece."""
        from_row, from_col = from_pos
        moves = []
        
        # Check all 4 directions (orthogonal)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            # Try moving in this direction
            for dist in range(1, 9):
                to_row = from_row + dr * dist
                to_col = from_col + dc * dist
                
                # Check bounds
                if to_row < 0 or to_row >= 9 or to_col < 0 or to_col >= 9:
                    break
                
                # Check if destination is valid
                if not self._is_valid_destination(state, from_pos, (to_row, to_col)):
                    break
                
                # Check if path is clear
                if not self._is_path_clear(state, from_pos, (to_row, to_col)):
                    break
                
                moves.append((from_pos, (to_row, to_col)))
        
        return moves
    
    def _is_camp(self, row: int, col: int) -> bool:
        """Check if position is a camp square."""
        return (row, col) in CAMPS
    
    def _is_valid_destination(self, state: GameState, from_pos: Tuple[int, int], 
                              to_pos: Tuple[int, int]) -> bool:
        """Check if destination square is valid for move."""
        to_row, to_col = to_pos
        from_row, from_col = from_pos
        
        # Cannot land on occupied square
        if state.board[to_row, to_col] != Pawn.EMPTY:
            return False
        
        # Cannot land on castle (unless it's the king already there)
        if to_row == CASTLE_ROW and to_col == CASTLE_COL and state.board[from_row, from_col] != Pawn.KING:
            return False
        
        # Black pieces: can only be in camps if they started there
        # Track which black pieces started in camps (set in reset())
        if state.turn == Turn.BLACK and self._is_camp(to_row, to_col):
            # Allow if it's a starting position for black
            if to_pos not in self._black_starting_positions:
                return False
        
        return True
    
    def _is_path_clear(self, state: GameState, from_pos: Tuple[int, int], 
                      to_pos: Tuple[int, int]) -> bool:
        """Check if path between from and to is clear (excluding endpoints)."""
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        
        # Determine direction
        if from_row == to_row:
            # Horizontal move
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                if state.board[from_row, col] != Pawn.EMPTY:
                    return False
        elif from_col == to_col:
            # Vertical move
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                if state.board[row, from_col] != Pawn.EMPTY:
                    return False
        else:
            # Not orthogonal (shouldn't happen)
            return False
        
        return True
    
    def _apply_move(self, state: GameState, from_pos: Tuple[int, int], 
                   to_pos: Tuple[int, int]) -> GameState:
        """Apply move and return new state (does not modify original)."""
        # Create new board
        new_board = state.board.copy()
        
        # Move piece
        piece = new_board[from_pos[0], from_pos[1]]
        new_board[from_pos[0], from_pos[1]] = Pawn.EMPTY
        new_board[to_pos[0], to_pos[1]] = piece
        
        # Check for captures after move
        new_board = self._check_captures(new_board, to_pos, state.turn)
        
        # Check for king escape (white win) - king on escape tile
        king_pos = find_king(GameState(board=new_board, turn=state.turn))
        if king_pos != (-1, -1):
            k_row, k_col = king_pos
            if (k_row, k_col) in ESCAPE_TILES:
                new_turn = Turn.WHITEWIN
            else:
                # Check for king capture (black win)
                if self._is_king_captured(new_board, king_pos):
                    new_turn = Turn.BLACKWIN
                else:
                    # Switch turn normally
                    new_turn = Turn.BLACK if state.turn == Turn.WHITE else Turn.WHITE
        else:
            # King not found - must be captured
            new_turn = Turn.BLACKWIN
        
        new_state = GameState(board=new_board, turn=new_turn)
        return new_state
    
    def _check_captures(self, board: np.ndarray, move_pos: Tuple[int, int], 
                       player_turn: Turn) -> np.ndarray:
        """
        Check for captures after a move and remove captured pieces.
        
        Capture rules (Ashton Tablut):
        - A piece is captured if it's between the moved piece and another friendly piece/barrier
        - The moved piece "pushes" the opponent piece into capture
        - Castle/Throne can act as a barrier
        - Camps can act as a barrier
        - King has special capture rules (4 sides if in castle, 3 if adjacent)
        """
        row, col = move_pos
        captured_positions = []
        
        # Check all 4 directions from the moved piece
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            # Check if there's an opponent piece adjacent to the moved piece
            check_row = row + dr
            check_col = col + dc
            
            if check_row < 0 or check_row >= 9 or check_col < 0 or check_col >= 9:
                continue
            
            check_pawn = board[check_row, check_col]
            
            # Determine if this is an opponent piece that might be captured
            if player_turn == Turn.WHITE:
                # White moved - check for black captures
                # Black piece is captured if it's between white piece and another white/barrier
                if check_pawn == Pawn.BLACK:
                    # Check the opposite side (behind the black piece)
                    opp_row = check_row + dr
                    opp_col = check_col + dc
                    
                    if 0 <= opp_row < 9 and 0 <= opp_col < 9:
                        opp_pawn = board[opp_row, opp_col]
                        # Captured if opposite side has white piece, king, or throne
                        if opp_pawn == Pawn.WHITE or opp_pawn == Pawn.KING:
                            captured_positions.append((check_row, check_col))
                        elif (opp_row == CASTLE_ROW and opp_col == CASTLE_COL):
                            # Throne/castle acts as barrier
                            captured_positions.append((check_row, check_col))
            else:
                # Black moved - check for white/king captures
                if check_pawn == Pawn.WHITE:
                    # White piece captured if between black piece and another black/barrier
                    opp_row = check_row + dr
                    opp_col = check_col + dc
                    
                    if 0 <= opp_row < 9 and 0 <= opp_col < 9:
                        opp_pawn = board[opp_row, opp_col]
                        if opp_pawn == Pawn.BLACK:
                            captured_positions.append((check_row, check_col))
                        elif (opp_row == CASTLE_ROW and opp_col == CASTLE_COL):
                            # Throne acts as barrier
                            captured_positions.append((check_row, check_col))
                        elif self._is_camp(opp_row, opp_col):
                            # Camp acts as barrier
                            captured_positions.append((check_row, check_col))
                elif check_pawn == Pawn.KING:
                    # King capture uses special rules (handled separately)
                    if self._is_king_captured(board, (check_row, check_col)):
                        captured_positions.append((check_row, check_col))
        
        # Remove captured pieces
        for cap_row, cap_col in captured_positions:
            board[cap_row, cap_col] = Pawn.EMPTY
        
        return board
    
    def _is_piece_captured(self, board: np.ndarray, pos: Tuple[int, int], 
                           piece_type: Optional[Turn]) -> bool:
        """
        Check if a piece at position is captured.
        
        Args:
            board: Current board state
            pos: Position to check (row, col)
            piece_type: Turn.WHITE, Turn.BLACK, or None for King
        
        Returns:
            True if piece is captured
        """
        row, col = pos
        pawn = board[row, col]
        
        if pawn == Pawn.KING:
            return self._is_king_captured(board, pos)
        
        # For regular pieces, check opposite sides
        # Horizontal capture
        left_blocked = self._is_position_blocked(board, (row, col - 1), pawn)
        right_blocked = self._is_position_blocked(board, (row, col + 1), pawn)
        if left_blocked and right_blocked:
            return True
        
        # Vertical capture
        up_blocked = self._is_position_blocked(board, (row - 1, col), pawn)
        down_blocked = self._is_position_blocked(board, (row + 1, col), pawn)
        if up_blocked and down_blocked:
            return True
        
        return False
    
    def _is_king_captured(self, board: np.ndarray, king_pos: Tuple[int, int]) -> bool:
        """
        Check if king is captured (special rules).
        
        King capture rules:
        - In castle: must be surrounded on all 4 sides
        - Adjacent to castle: must be surrounded on 3 free sides (castle acts as barrier)
        - Adjacent to camp: camp acts as barrier
        - Otherwise: normal capture (2 opposite sides)
        """
        row, col = king_pos
        
        # Check if king is in castle
        if row == CASTLE_ROW and col == CASTLE_COL:
            # Must be surrounded on all 4 sides
            sides_blocked = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if self._is_position_blocked(board, (row + dr, col + dc), Pawn.KING):
                    sides_blocked += 1
            return sides_blocked == 4
        
        # Check if king is adjacent to castle
        if abs(row - CASTLE_ROW) + abs(col - CASTLE_COL) == 1:
            # Castle acts as barrier, need 3 other sides blocked
            sides_blocked = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                check_pos = (row + dr, col + dc)
                # Castle position counts as blocked
                if check_pos == (CASTLE_ROW, CASTLE_COL):
                    sides_blocked += 1
                elif self._is_position_blocked(board, check_pos, Pawn.KING):
                    sides_blocked += 1
            return sides_blocked >= 3
        
        # Check if adjacent to camp
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            adj_pos = (row + dr, col + dc)
            if self._is_camp(adj_pos[0], adj_pos[1]):
                # Camp acts as barrier, check opposite side
                opp_pos = (row - dr, col - dc)
                if self._is_position_blocked(board, opp_pos, Pawn.KING):
                    return True
        
        # Normal capture: check opposite sides
        left_blocked = self._is_position_blocked(board, (row, col - 1), Pawn.KING)
        right_blocked = self._is_position_blocked(board, (row, col + 1), Pawn.KING)
        if left_blocked and right_blocked:
            return True
        
        up_blocked = self._is_position_blocked(board, (row - 1, col), Pawn.KING)
        down_blocked = self._is_position_blocked(board, (row + 1, col), Pawn.KING)
        if up_blocked and down_blocked:
            return True
        
        return False
    
    def _is_position_blocked(self, board: np.ndarray, pos: Tuple[int, int], 
                            piece_type: Pawn) -> bool:
        """
        Check if a position is blocked (by opponent piece, castle, or camp).
        
        Args:
            board: Current board
            pos: Position to check
            piece_type: Type of piece being checked (to determine opponent)
        
        Returns:
            True if position is blocked
        """
        row, col = pos
        
        # Out of bounds
        if row < 0 or row >= 9 or col < 0 or col >= 9:
            return False
        
        pawn = board[row, col]
        
        # Castle/Throne blocks (unless it's the king checking)
        if row == CASTLE_ROW and col == CASTLE_COL:
            return True
        
        # Camp blocks
        if self._is_camp(row, col):
            return True
        
        # Opponent piece blocks
        if piece_type == Pawn.WHITE or piece_type == Pawn.KING:
            return pawn == Pawn.BLACK
        elif piece_type == Pawn.BLACK:
            return pawn == Pawn.WHITE or pawn == Pawn.KING
        
        return False
    
    def _check_terminal(self, state: GameState) -> Tuple[bool, float, Dict]:
        """
        Check if state is terminal and return reward.
        
        Returns:
            (terminated, reward, info)
        """
        # Check for king escape (white win)
        king_pos = find_king(state)
        if king_pos != (-1, -1):
            k_row, k_col = king_pos
            if (k_row, k_col) in ESCAPE_TILES:
                return True, self.config.reward_win, {"result": "white_win"}
        
        # Check for king capture (black win)
        if king_pos == (-1, -1):
            return True, self.config.reward_loss, {"result": "black_win"}
        
        # Check for draw (same state twice)
        # State repetition tracking (if state history is available)
        if hasattr(self, '_state_history') and self._state_history:
            current_hash = state.hash()
            if self._state_history.count(current_hash) >= 2:
                return True, self.config.reward_draw, {"result": "draw_repetition"}
        
        # Check for no legal moves (current player loses)
        legal_moves = self.get_legal_moves(state)
        if len(legal_moves) == 0:
            # Current player has no moves - they lose
            if state.turn == Turn.WHITE:
                return True, self.config.reward_loss, {"result": "white_no_moves"}
            else:
                return True, self.config.reward_win, {"result": "black_no_moves"}
        
        return False, 0.0, {}
    
    def _state_to_observation(self, state: GameState) -> np.ndarray:
        """Convert GameState to observation array (9x9x5 one-hot)."""
        obs = np.zeros((9, 9, 5), dtype=np.float32)
        
        for i in range(9):
            for j in range(9):
                pawn = state.board[i, j]
                if pawn == Pawn.EMPTY:
                    obs[i, j, 0] = 1.0
                elif pawn == Pawn.WHITE:
                    obs[i, j, 1] = 1.0
                elif pawn == Pawn.BLACK:
                    obs[i, j, 2] = 1.0
                elif pawn == Pawn.KING:
                    obs[i, j, 3] = 1.0
                # THRONE is represented as EMPTY in board
        
        return obs
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state."""
        if self.state is None:
            return None
        
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            print(self._render_ansi())
            return None
        else:
            raise ValueError(f"Unknown render mode: {mode}")
    
    def _render_ansi(self) -> str:
        """Render as ANSI text."""
        lines = []
        lines.append("  " + " ".join([chr(97 + i) for i in range(9)]))
        
        for i in range(9):
            line = f"{i+1} "
            for j in range(9):
                pawn = self.state.board[i, j]
                if pawn == Pawn.EMPTY:
                    line += ". "
                elif pawn == Pawn.WHITE:
                    line += "W "
                elif pawn == Pawn.BLACK:
                    line += "B "
                elif pawn == Pawn.KING:
                    line += "K "
                else:
                    line += "? "
            lines.append(line)
        
        lines.append(f"\nTurn: {self.state.turn.value}")
        return "\n".join(lines)

