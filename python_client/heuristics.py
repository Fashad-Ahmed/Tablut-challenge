"""
Handcrafted heuristics for Tablut game.

Provides fast evaluation functions for:
- Move ordering (prioritize good moves)
- Fallback evaluation when model unavailable
- Quick position assessment

Test with: pytest python_client/tests/test_heuristics.py
"""

import numpy as np
from typing import Tuple, List
from .utils import GameState, Pawn, Turn


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


def is_camp(row: int, col: int) -> bool:
    """Check if position is a camp square."""
    return (row, col) in CAMPS


def is_escape(row: int, col: int) -> bool:
    """Check if position is an escape tile."""
    return (row, col) in ESCAPE_TILES


def is_castle(row: int, col: int) -> bool:
    """Check if position is the castle (throne)."""
    return row == CASTLE_ROW and col == CASTLE_COL


def find_king(state: GameState) -> Tuple[int, int]:
    """Find king position on board. Returns (-1, -1) if not found."""
    for i in range(9):
        for j in range(9):
            if state.board[i, j] == Pawn.KING:
                return (i, j)
    return (-1, -1)


def king_distance_to_edge(state: GameState) -> int:
    """
    Calculate minimum Manhattan distance from king to any edge (escape).
    
    Lower is better for white (king closer to escape).
    """
    king_pos = find_king(state)
    if king_pos == (-1, -1):
        return 100  # King captured, very bad for white
    
    k_row, k_col = king_pos
    # Distance to nearest edge
    dist_to_top = k_row
    dist_to_bottom = 8 - k_row
    dist_to_left = k_col
    dist_to_right = 8 - k_col
    
    return min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)


def king_safety_score(state: GameState) -> float:
    """
    Evaluate king safety (higher = safer for white).
    
    Considers:
    - Distance to center (center is more dangerous)
    - Proximity to escape tiles
    - Number of defending pieces nearby
    """
    king_pos = find_king(state)
    if king_pos == (-1, -1):
        return -1000.0  # King captured
    
    k_row, k_col = king_pos
    
    # Distance to escape (lower is better)
    escape_dist = king_distance_to_edge(state)
    escape_score = 10.0 / (1.0 + escape_dist)
    
    # Distance from center (further is safer)
    center_dist = abs(k_row - 4) + abs(k_col - 4)
    center_score = center_dist * 0.5
    
    # Count nearby white defenders (within 2 squares)
    defender_count = 0
    for i in range(max(0, k_row - 2), min(9, k_row + 3)):
        for j in range(max(0, k_col - 2), min(9, k_col + 3)):
            if state.board[i, j] == Pawn.WHITE:
                defender_count += 1
    
    defender_score = defender_count * 2.0
    
    # Count nearby black attackers (within 2 squares)
    attacker_count = 0
    for i in range(max(0, k_row - 2), min(9, k_row + 3)):
        for j in range(max(0, k_col - 2), min(9, k_col + 3)):
            if state.board[i, j] == Pawn.BLACK:
                attacker_count += 1
    
    attacker_penalty = attacker_count * -3.0
    
    return escape_score + center_score + defender_score + attacker_penalty


def capture_value(state: GameState, move: Tuple[Tuple[int, int], Tuple[int, int]], 
                  player: Turn) -> float:
    """
    Estimate value of captures this move might enable.
    
    Returns positive value if move threatens opponent pieces.
    """
    from_pos, to_pos = move
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    
    # Simulate move to check for captures
    # Create temporary board
    temp_board = state.board.copy()
    piece = temp_board[from_row, from_col]
    temp_board[from_row, from_col] = Pawn.EMPTY
    temp_board[to_row, to_col] = piece
    
    capture_score = 0.0
    
    # Check all 4 directions from destination
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dr, dc in directions:
        check_row = to_row + dr
        check_col = to_col + dc
        
        if check_row < 0 or check_row >= 9 or check_col < 0 or check_col >= 9:
            continue
        
        check_pawn = temp_board[check_row, check_col]
        
        # Check if this is an opponent piece that might be captured
        if player == Turn.WHITE:
            if check_pawn == Pawn.BLACK:
                # Check if black piece would be captured (surrounded on opposite sides)
                opp_row = check_row - dr
                opp_col = check_col - dc
                if 0 <= opp_row < 9 and 0 <= opp_col < 9:
                    opp_pawn = temp_board[opp_row, opp_col]
                    if opp_pawn == Pawn.WHITE or opp_pawn == Pawn.KING:
                        capture_score += 5.0  # Black soldier captured
        else:  # player == Turn.BLACK
            if check_pawn == Pawn.WHITE:
                # Check if white piece would be captured
                opp_row = check_row - dr
                opp_col = check_col - dc
                if 0 <= opp_row < 9 and 0 <= opp_col < 9:
                    opp_pawn = temp_board[opp_row, opp_col]
                    if opp_pawn == Pawn.BLACK:
                        capture_score += 3.0  # White soldier captured
            elif check_pawn == Pawn.KING:
                # King capture is very valuable
                capture_score += 50.0
    
    return capture_score


def mobility_score(state: GameState, player: Turn) -> int:
    """
    Count legal moves available to player.
    
    Higher mobility = more options = better position.
    """
    # Use TablutEnv to generate actual legal moves
    try:
        from .tablut_env import TablutEnv
        env = TablutEnv()
        legal_moves = env.get_legal_moves(state)
        return len(legal_moves)
    except Exception:
        # Fallback: rough estimate based on piece count
        if player == Turn.WHITE:
            white_count = np.sum(state.board == Pawn.WHITE)
            king_count = 1 if find_king(state) != (-1, -1) else 0
            return white_count + king_count
        else:
            return np.sum(state.board == Pawn.BLACK)


def material_balance(state: GameState) -> float:
    """
    Simple material count (white perspective).
    
    Positive = white advantage, negative = black advantage.
    """
    white_soldiers = np.sum(state.board == Pawn.WHITE)
    black_soldiers = np.sum(state.board == Pawn.BLACK)
    king_exists = find_king(state) != (-1, -1)
    
    # King is worth much more than soldiers
    king_value = 50.0 if king_exists else -1000.0
    soldier_diff = (white_soldiers - black_soldiers) * 1.0
    
    return king_value + soldier_diff


def evaluate_heuristic(state: GameState, player: Turn) -> float:
    """
    Combined heuristic evaluation (white perspective).
    
    Returns value from white's perspective:
    - Positive = good for white
    - Negative = good for black
    
    Args:
        state: Current game state
        player: Player to evaluate for (used to flip sign if needed)
    
    Returns:
        Evaluation score
    """
    # Check terminal states
    if state.turn == Turn.WHITEWIN:
        return 1000.0
    if state.turn == Turn.BLACKWIN:
        return -1000.0
    if state.turn == Turn.DRAW:
        return 0.0
    
    # Combine heuristics
    material = material_balance(state)
    safety = king_safety_score(state)
    mobility = mobility_score(state, Turn.WHITE) - mobility_score(state, Turn.BLACK)
    
    # Weighted combination
    score = material * 1.0 + safety * 2.0 + mobility * 0.5
    
    # Flip if evaluating for black
    if player == Turn.BLACK:
        score = -score
    
    return score


def move_ordering_key(state: GameState, move: Tuple[Tuple[int, int], Tuple[int, int]], 
                     player: Turn) -> float:
    """
    Generate ordering key for move (higher = try first).
    
    Used to prioritize moves in alpha-beta search.
    """
    from_pos, to_pos = move
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    
    score = 0.0
    
    # Prefer moves that bring king closer to edge
    if state.board[from_row, from_col] == Pawn.KING:
        old_dist = min(from_row, 8 - from_row, from_col, 8 - from_col)
        new_dist = min(to_row, 8 - to_row, to_col, 8 - to_col)
        score += (old_dist - new_dist) * 10.0  # Closer is better
    
    # Prefer captures (if we can detect them)
    capture_val = capture_value(state, move, player)
    score += capture_val * 5.0
    
    # Prefer moving toward center for black (surround king)
    if player == Turn.BLACK:
        center_dist_old = abs(from_row - 4) + abs(from_col - 4)
        center_dist_new = abs(to_row - 4) + abs(to_col - 4)
        score += (center_dist_old - center_dist_new) * 2.0
    
    # Prefer moving away from center for white king
    if player == Turn.WHITE and state.board[from_row, from_col] == Pawn.KING:
        center_dist_old = abs(from_row - 4) + abs(from_col - 4)
        center_dist_new = abs(to_row - 4) + abs(to_col - 4)
        score += (center_dist_new - center_dist_old) * 3.0
    
    return score

