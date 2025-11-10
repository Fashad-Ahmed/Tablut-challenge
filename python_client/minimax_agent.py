"""
Minimax agent with alpha-beta pruning for Tablut.

Features:
- Iterative deepening with time management
- Alpha-beta pruning
- Transposition table (Zobrist hashing)
- Move ordering heuristics
- Thread-safe time checks
- Always returns best move found (even if interrupted)

Test with: pytest python_client/tests/test_minimax.py
"""

import time
import threading
from typing import Callable, Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from .utils import GameState, Turn, move_to_action
from .heuristics import evaluate_heuristic, move_ordering_key
from .tablut_env import TablutEnv


@dataclass
class SearchStats:
    """Statistics from minimax search."""
    nodes_searched: int = 0
    depth_reached: int = 0
    transposition_hits: int = 0
    time_elapsed: float = 0.0
    best_move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    best_value: float = 0.0


class TranspositionTable:
    """
    Transposition table for caching evaluated positions.
    
    Uses Zobrist hashing for better hash distribution and collision resistance.
    """
    
    def __init__(self, max_size: int = 100000):
        self.table: Dict[int, Tuple[float, int]] = {}  # hash -> (value, depth)
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
        # Initialize Zobrist hash table (9x9 board, 5 piece types, 2 turn states)
        # Using a simple pseudo-random initialization
        import random
        random.seed(42)  # Deterministic for reproducibility
        self.zobrist_table = np.zeros((9, 9, 5, 2), dtype=np.uint64)
        for i in range(9):
            for j in range(9):
                for p in range(5):  # EMPTY, WHITE, BLACK, THRONE, KING
                    for t in range(2):  # WHITE turn, BLACK turn
                        self.zobrist_table[i, j, p, t] = random.getrandbits(64)
        random.seed()  # Reset seed
    
    def _zobrist_hash(self, state: GameState) -> int:
        """
        Compute Zobrist hash for state.
        
        Zobrist hashing provides better distribution than simple hash.
        """
        hash_value = 0
        from .utils import Pawn, Turn
        
        # Hash board positions
        for i in range(9):
            for j in range(9):
                pawn = state.board[i, j]
                pawn_idx = 0  # EMPTY
                if pawn == Pawn.WHITE:
                    pawn_idx = 1
                elif pawn == Pawn.BLACK:
                    pawn_idx = 2
                elif pawn == Pawn.KING:
                    pawn_idx = 4
                # THRONE is treated as EMPTY for hashing
                
                turn_idx = 0 if state.turn == Turn.WHITE else 1
                hash_value ^= int(self.zobrist_table[i, j, pawn_idx, turn_idx])
        
        return hash_value
    
    def get(self, state: GameState, depth: int) -> Optional[float]:
        """Get cached value if depth is sufficient."""
        # Use Zobrist hash for better distribution
        state_hash = self._zobrist_hash(state)
        if state_hash in self.table:
            cached_value, cached_depth = self.table[state_hash]
            if cached_depth >= depth:
                self.hits += 1
                return cached_value
        self.misses += 1
        return None
    
    def put(self, state: GameState, value: float, depth: int) -> None:
        """Store value in table."""
        if len(self.table) >= self.max_size:
            # Simple eviction: clear half (FIFO would be better but this is simpler)
            if len(self.table) > self.max_size * 0.9:
                # Clear oldest 50%
                keys_to_remove = list(self.table.keys())[:len(self.table) // 2]
                for key in keys_to_remove:
                    del self.table[key]
        
        # Use Zobrist hash for better distribution
        state_hash = self._zobrist_hash(state)
        self.table[state_hash] = (value, depth)
    
    def clear(self) -> None:
        """Clear table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0


class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning and iterative deepening.
    
    Uses value function (RL model or heuristic) for leaf evaluation.
    """
    
    def __init__(
        self,
        value_fn: Callable[[GameState], float],
        max_depth: int = 4,
        time_limit: float = 55.0,  # Leave 5s buffer for move transmission
        use_transposition: bool = True,
        move_ordering: bool = True,
    ):
        """
        Initialize minimax agent.
        
        Args:
            value_fn: Function(state) -> float (white perspective)
            max_depth: Maximum search depth
            time_limit: Time limit per move in seconds
            use_transposition: Enable transposition table
            move_ordering: Enable move ordering
        """
        self.value_fn = value_fn
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_transposition = use_transposition
        self.move_ordering = move_ordering
        
        self.transposition_table = TranspositionTable() if use_transposition else None
        self.env = TablutEnv()
        
        # Thread-safe time tracking
        self._start_time: Optional[float] = None
        self._time_lock = threading.Lock()
        
        # Value function cache for performance (RL models can be slow)
        self._value_cache: Dict[int, float] = {}
        self._cache_max_size = 10000  # Limit cache size
    
    def get_move(self, state: GameState) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Get best move for current state using iterative deepening.
        
        Args:
            state: Current game state
        
        Returns:
            (from_pos, to_pos) tuple
        """
        stats = SearchStats()
        self._start_time = time.time()
        
        # Get legal moves
        legal_moves = self.env.get_legal_moves(state)
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # If only one move, return it immediately (skip expensive search)
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Clear value cache at start of each move (prevent stale evaluations)
        # Initialize if it doesn't exist (for backward compatibility)
        if not hasattr(self, '_value_cache'):
            self._value_cache: Dict[int, float] = {}
            self._cache_max_size = 10000
        self._value_cache.clear()
        
        # Iterative deepening
        best_move = legal_moves[0]  # Fallback
        best_value = float('-inf') if state.turn == Turn.WHITE else float('inf')
        
        try:
            for depth in range(1, self.max_depth + 1):
                if self._is_time_up():
                    break
                
                # Search at this depth
                move, value = self._minimax_ab(
                    state, depth, float('-inf'), float('inf'), True, stats
                )
                
                if move is not None:
                    best_move = move
                    best_value = value
                    stats.depth_reached = depth
                
                # If we found a winning move, stop early
                if abs(value) > 900:  # Terminal state value
                    break
                
        except TimeoutError:
            pass  # Use best move found so far
        
        stats.best_move = best_move
        stats.best_value = best_value
        stats.time_elapsed = time.time() - self._start_time
        
        return best_move
    
    def _minimax_ab(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        stats: SearchStats,
    ) -> Tuple[Optional[Tuple[Tuple[int, int], Tuple[int, int]]], float]:
        """
        Minimax with alpha-beta pruning.
        
        Returns:
            (best_move, best_value) or (None, value) for non-root nodes
        """
        stats.nodes_searched += 1
        
        # Time check
        if self._is_time_up():
            raise TimeoutError("Time limit reached")
        
        # Check transposition table
        if self.use_transposition and self.transposition_table:
            cached_value = self.transposition_table.get(state, depth)
            if cached_value is not None:
                stats.transposition_hits += 1
                return None, cached_value
        
        # Terminal check
        legal_moves = self.env.get_legal_moves(state)
        is_terminal = len(legal_moves) == 0 or self._is_terminal(state)
        
        if depth == 0 or is_terminal:
            value = self._evaluate(state)
            if self.use_transposition and self.transposition_table:
                self.transposition_table.put(state, value, depth)
            return None, value
        
        # Get moves (ordered if enabled)
        if self.move_ordering:
            move_scores = [
                (move, move_ordering_key(state, move, state.turn))
                for move in legal_moves
            ]
            move_scores.sort(key=lambda x: x[1], reverse=maximizing)
            ordered_moves = [move for move, _ in move_scores]
        else:
            ordered_moves = legal_moves
        
        # Search moves
        best_move = None
        best_value = float('-inf') if maximizing else float('inf')
        
        for move in ordered_moves:
            # Apply move
            new_state = self.env._apply_move(state, move[0], move[1])
            
            # Recursive search
            _, value = self._minimax_ab(
                new_state, depth - 1, alpha, beta, not maximizing, stats
            )
            
            # Update best
            if maximizing:
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = max(alpha, value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                    beta = min(beta, value)
            
            # Alpha-beta cutoff
            if beta <= alpha:
                break
        
        # Store in transposition table
        if self.use_transposition and self.transposition_table:
            self.transposition_table.put(state, best_value, depth)
        
        return best_move, best_value
    
    def _evaluate(self, state: GameState) -> float:
        """Evaluate state using value function (with caching for performance)."""
        # Initialize cache if it doesn't exist (for backward compatibility)
        if not hasattr(self, '_value_cache'):
            self._value_cache: Dict[int, float] = {}
            self._cache_max_size = 10000
        
        # Use GameState's hash method (not built-in hash which fails for dataclasses with numpy arrays)
        state_hash = state.hash()
        
        # Check cache
        if state_hash in self._value_cache:
            return self._value_cache[state_hash]
        
        # Evaluate
        value = self.value_fn(state)
        
        # Cache result (with size limit)
        if len(self._value_cache) < self._cache_max_size:
            self._value_cache[state_hash] = value
        else:
            # Clear cache if too large (simple strategy: clear all)
            self._value_cache.clear()
            self._value_cache[state_hash] = value
        
        return value
    
    def _is_terminal(self, state: GameState) -> bool:
        """Check if state is terminal."""
        # Check turn enum
        if state.turn in (Turn.WHITEWIN, Turn.BLACKWIN, Turn.DRAW):
            return True
        
        # Check for no legal moves
        legal_moves = self.env.get_legal_moves(state)
        return len(legal_moves) == 0
    
    def _is_time_up(self) -> bool:
        """Check if time limit exceeded (thread-safe)."""
        with self._time_lock:
            if self._start_time is None:
                return False
            elapsed = time.time() - self._start_time
            return elapsed >= self.time_limit
    
    def clear_cache(self) -> None:
        """Clear transposition table and value cache."""
        if self.transposition_table:
            self.transposition_table.clear()
        if hasattr(self, '_value_cache'):
            self._value_cache.clear()

