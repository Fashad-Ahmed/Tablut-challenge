"""
Main entrypoint for Tablut Python agent.

Handles:
- TCP socket communication with Java referee server
- Game loop (read state -> compute move -> send move)
- Time management and timeout fallback
- Model loading and minimax agent initialization
- Structured logging

Usage:
    python agent.py WHITE 60 127.0.0.1 --model models/value_net.pt

Test with: pytest python_client/tests/test_agent.py (if tests exist)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from .utils import (
    GameState, Turn, Pawn, json_to_state, move_to_action,
    connect_to_server, send_string, recv_string, set_seed
)
from .minimax_agent import MinimaxAgent
from .model_loader import ModelLoader
from .heuristics import evaluate_heuristic, find_king


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Note: File logging disabled by default to prevent disk saturation in competition
    ]
)
logger = logging.getLogger(__name__)


class TablutAgent:
    """
    Main agent class handling communication and game logic.
    """
    
    def __init__(
        self,
        player: str,
        timeout: int,
        server_host: str,
        server_port: int,
        model_path: Optional[str] = None,
        max_depth: int = 4,
        seed: Optional[int] = None,
    ):
        """
        Initialize agent.
        
        Args:
            player: "WHITE" or "BLACK"
            timeout: Time limit per move in seconds
            server_host: Server hostname/IP
            server_port: Server port (5800 for white, 5801 for black)
            model_path: Path to value network model (optional)
            max_depth: Minimax search depth
            seed: Random seed for reproducibility
        """
        # Parse player
        player_upper = player.upper()
        if player_upper == "WHITE":
            self.turn = Turn.WHITE
        elif player_upper == "BLACK":
            self.turn = Turn.BLACK
        else:
            raise ValueError(f"Player must be 'WHITE' or 'BLACK', got '{player}'")
        
        self.timeout = timeout
        self.server_host = server_host
        self.server_port = server_port
        self.max_depth = max_depth
        
        # Set seed if provided
        if seed is not None:
            set_seed(seed)
            logger.info(f"Random seed set to {seed}")
        
        # Initialize value function
        if model_path and Path(model_path).exists():
            logger.info(f"Loading model from {model_path}")
            try:
                # Try loading as standard PyTorch model first
                if model_path.endswith('.pt'):
                    model_loader = ModelLoader(model_path, device="cpu")  # CPU for competition
                    self.value_fn = lambda state: model_loader.evaluate(state)
                    logger.info("Model loaded successfully (PyTorch format)")
                # Try loading as stable-baselines3 model
                elif model_path.endswith('.zip'):
                    from .rl_value_wrapper import RLValueWrapper
                    rl_wrapper = RLValueWrapper(model_path, device="cpu")
                    self.value_fn = lambda state: rl_wrapper.evaluate(state)
                    logger.info("Model loaded successfully (SB3 RL format)")
                else:
                    # Try both formats
                    try:
                        model_loader = ModelLoader(model_path, device="cpu")
                        self.value_fn = lambda state: model_loader.evaluate(state)
                        logger.info("Model loaded successfully (PyTorch format)")
                    except:
                        from .rl_value_wrapper import RLValueWrapper
                        rl_wrapper = RLValueWrapper(model_path, device="cpu")
                        self.value_fn = lambda state: rl_wrapper.evaluate(state)
                        logger.info("Model loaded successfully (SB3 RL format)")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using heuristic fallback.")
                self.value_fn = lambda state: evaluate_heuristic(state, self.turn)
        else:
            logger.info("Using heuristic evaluation (no model provided)")
            self.value_fn = lambda state: evaluate_heuristic(state, self.turn)
        
        # Initialize minimax agent
        # Leave 5 seconds buffer for move transmission
        move_time_limit = max(1.0, timeout - 5.0)
        self.minimax = MinimaxAgent(
            value_fn=self.value_fn,
            max_depth=max_depth,
            time_limit=move_time_limit,
            use_transposition=True,
            move_ordering=True,
        )
        
        # Socket will be created in run()
        self.socket = None
        self.player_name = f"PythonAgent_{player_upper}"
    
    def run(self) -> None:
        """Main game loop."""
        try:
            # Connect to server
            logger.info(f"Connecting to server {self.server_host}:{self.server_port}")
            try:
                self.socket = connect_to_server(self.server_host, self.server_port)
                logger.info("Connected to server")
            except ConnectionError as e:
                logger.error(f"Connection failed: {e}")
                logger.error(
                    "\nTo start the Java server:\n"
                    "  1. Open a new terminal\n"
                    "  2. cd Tablut/Executables\n"
                    "  3. java -jar Server.jar\n"
                    "  4. Then run this agent again"
                )
                raise
            
            # Send player name
            logger.info(f"Sending player name: {self.player_name}")
            send_string(self.socket, json.dumps(self.player_name))
            
            # Read initial state
            logger.info("Waiting for initial state...")
            initial_state_json = recv_string(self.socket)
            logger.debug(f"Received initial state: {initial_state_json[:200]}...")
            
            # Track game start time for timeout management
            self._game_start_time = time.time()
            
            state = json_to_state(initial_state_json)
            logger.info(f"Game started. Turn: {state.turn.value}")
            
            # Debug: Log board state
            logger.debug(f"Board shape: {state.board.shape}")
            king_pos = find_king(state)
            logger.debug(f"King position: {king_pos}")
            
            # Count pieces for debugging
            white_count = sum(1 for i in range(9) for j in range(9) if state.board[i, j] == Pawn.WHITE)
            black_count = sum(1 for i in range(9) for j in range(9) if state.board[i, j] == Pawn.BLACK)
            king_count = sum(1 for i in range(9) for j in range(9) if state.board[i, j] == Pawn.KING)
            logger.debug(f"Pieces - White: {white_count}, Black: {black_count}, King: {king_count}")
            
            # Game loop
            while True:
                # Check terminal first
                if state.turn in (Turn.WHITEWIN, Turn.BLACKWIN, Turn.DRAW):
                    result = state.turn.value
                    logger.info(f"Game ended: {result}")
                    break
                
                # Check if it's our turn
                if state.turn != self.turn:
                    # Wait for opponent's move
                    logger.info("Waiting for opponent's move...")
                    try:
                        state_json = recv_string(self.socket)
                        if state_json is None:
                            logger.info("Socket closed by server (game ended)")
                            break
                        state = json_to_state(state_json)
                        logger.info(f"Received state. Turn: {state.turn.value}")
                    except ConnectionError as e:
                        logger.warning(f"Connection error while waiting for opponent: {e}")
                        logger.info("Assuming game ended")
                        break
                    # After receiving state, check if it's now our turn
                    continue  # Loop back to check turn
                
                # It's our turn - compute and send move
                if state.turn == self.turn:
                    logger.info("Computing move...")
                    
                    # Use the minimax agent's environment (which is already initialized)
                    # Don't create a new environment - use the existing one
                    legal_moves_debug = self.minimax.env.get_legal_moves(state)
                    logger.info(f"Found {len(legal_moves_debug)} legal moves")
                    
                    if len(legal_moves_debug) == 0:
                        logger.error("No legal moves found! This might be a parsing issue.")
                        logger.error(f"State turn: {state.turn}, Expected turn: {self.turn}")
                        
                        # Debug: Check what pieces we're looking for
                        player_pawn = Pawn.WHITE if state.turn == Turn.WHITE else Pawn.BLACK
                        logger.error(f"Looking for pieces: {player_pawn.value}")
                        
                        # Count pieces of current player
                        found_pieces = []
                        for i in range(9):
                            for j in range(9):
                                pawn = state.board[i, j]
                                if pawn == player_pawn or (pawn == Pawn.KING and state.turn == Turn.WHITE):
                                    found_pieces.append((i, j, pawn))
                        logger.error(f"Found {len(found_pieces)} pieces of current player: {found_pieces[:10]}")
                        
                        # Log a sample of the board
                        logger.error("Board sample (first 3 rows):")
                        for i in range(min(3, 9)):
                            row_str = " ".join([str(state.board[i, j].value) if isinstance(state.board[i, j], Pawn) else "?" for j in range(9)])
                            logger.error(f"  Row {i}: {row_str}")
                        
                        # Try to log original JSON
                        try:
                            logger.error(f"Original JSON (first 1000 chars): {initial_state_json[:1000]}")
                        except:
                            pass
                        
                        # Don't break - try to continue with fallback
                        logger.warning("Attempting to continue with fallback...")
                    
                    start_time = time.time()
                    
                    # Adjust time limit based on remaining time
                    # Leave buffer for network communication
                    remaining_time = self.timeout - (time.time() - self._game_start_time)
                    safe_time_limit = max(5.0, remaining_time - 3.0)  # 3s buffer for network
                    self.minimax.time_limit = min(self.minimax.time_limit, safe_time_limit)
                    
                    try:
                        # Get best move (only if we have legal moves)
                        if len(legal_moves_debug) > 0:
                            from_pos, to_pos = self.minimax.get_move(state)
                            
                            elapsed = time.time() - start_time
                            logger.info(f"Move computed in {elapsed:.2f}s: {from_pos} -> {to_pos}")
                            
                            # Warn if move took too long
                            if elapsed > 50:
                                logger.warning(f"Move computation took {elapsed:.2f}s (close to timeout!)")
                            
                            # CRITICAL: Re-validate move on CURRENT state right before sending
                            # State might have changed, so we need to check again
                            current_legal_moves = self.minimax.env.get_legal_moves(state)
                            
                            if (from_pos, to_pos) not in current_legal_moves:
                                logger.warning(
                                    f"Computed move {from_pos} -> {to_pos} is no longer valid! "
                                    f"Recomputing on current state..."
                                )
                                
                                # Additional validation: check if destination is empty
                                if state.board[to_pos[0], to_pos[1]] != Pawn.EMPTY:
                                    logger.error(
                                        f"Destination {to_pos} is occupied by {state.board[to_pos[0], to_pos[1]]}! "
                                        f"State may have changed. Recomputing move..."
                                    )
                                
                                # Recompute move on current state (quick - just use first legal move)
                                if len(current_legal_moves) > 0:
                                    from_pos, to_pos = current_legal_moves[0]
                                    logger.warning(f"Using safe fallback move: {from_pos} -> {to_pos}")
                                else:
                                    logger.error("No legal moves available on current state!")
                                    break
                            
                            # Final validation before sending
                            final_check = self.minimax.env.get_legal_moves(state)
                            if (from_pos, to_pos) not in final_check:
                                logger.error(
                                    f"Move {from_pos} -> {to_pos} is still invalid! "
                                    f"Destination: {state.board[to_pos[0], to_pos[1]]}"
                                )
                                if final_check:
                                    from_pos, to_pos = final_check[0]
                                    logger.warning(f"Using emergency fallback: {from_pos} -> {to_pos}")
                                else:
                                    logger.error("No legal moves available!")
                                    break
                        else:
                            # No legal moves - this shouldn't happen but handle gracefully
                            logger.error("No legal moves available - game may have ended")
                            # Check if game is actually terminal
                            if state.turn in (Turn.WHITEWIN, Turn.BLACKWIN, Turn.DRAW):
                                logger.info(f"Game ended: {state.turn.value}")
                                break
                            # Otherwise, this is an error - exit
                            logger.error("Fatal: No moves but game not terminal. Exiting.")
                            break
                        
                        # Convert to action JSON
                        action = move_to_action(from_pos, to_pos, self.turn)
                        action_json = json.dumps(action)
                        
                        # Send move
                        logger.info(f"Sending move: {action_json}")
                        send_string(self.socket, action_json)
                        
                        # Read updated state (confirmation of our move)
                        logger.info("Waiting for state update after our move...")
                        max_retries = 3
                        retry_count = 0
                        state_updated = False
                        
                        while retry_count < max_retries and not state_updated:
                            try:
                                state_json = recv_string(self.socket)
                                if state_json is None:
                                    logger.info("Socket closed by server (game ended)")
                                    break
                                
                                new_state = json_to_state(state_json)
                                
                                # Debug: Log raw JSON turn value for troubleshooting
                                try:
                                    raw_data = json.loads(state_json)
                                    raw_turn = raw_data.get("turn", "unknown")
                                    logger.debug(f"Raw JSON turn value: '{raw_turn}' (type: {type(raw_turn)})")
                                except:
                                    pass
                                
                                logger.info(
                                    f"Received state update #{retry_count + 1}. "
                                    f"Turn: {new_state.turn.value} (our turn: {self.turn.value})"
                                )
                                
                                # CRITICAL: After we send a move, the turn MUST change to opponent
                                # If it doesn't, either:
                                # 1. The move was rejected (illegal)
                                # 2. We're reading a stale state update
                                # 3. There's a server bug
                                
                                if new_state.turn == self.turn:
                                    retry_count += 1
                                    logger.warning(
                                        f"Turn did not change after our move (retry {retry_count}/{max_retries})! "
                                        f"Still {new_state.turn.value}. This might be a stale state update."
                                    )
                                    
                                    if retry_count >= max_retries:
                                        logger.error(
                                            f"Turn did not change after {max_retries} state updates! "
                                            f"Move was likely rejected or there's a server issue. Exiting."
                                        )
                                        break
                                    # Wait for next state update
                                    continue
                                
                                # Turn changed correctly - update state
                                state = new_state
                                state_updated = True
                                logger.info(f"State updated successfully. Turn changed to {state.turn.value}")
                                
                                # Check if game ended
                                if state.turn in (Turn.WHITEWIN, Turn.BLACKWIN, Turn.DRAW):
                                    logger.info(f"Game ended: {state.turn.value}")
                                    break
                                    
                            except ConnectionError as e:
                                logger.warning(f"Connection error after sending move: {e}")
                                logger.info("Assuming game ended")
                                break
                        
                        if not state_updated:
                            # Failed to get valid state update
                            logger.error("Failed to receive valid state update after sending move. Exiting.")
                            break
                        
                        # Clear transposition table periodically (prevent memory growth)
                        if hasattr(self.minimax, 'transposition_table'):
                            if self.minimax.transposition_table and len(self.minimax.transposition_table.table) > 50000:
                                logger.debug("Clearing transposition table")
                                self.minimax.clear_cache()
                        
                        # Continue loop - wait for opponent's turn
                        # The turn should now be opponent's, so we'll enter the "wait for opponent" branch
                        continue
                    
                    except ValueError as e:
                        # This is the "No legal moves available" error from minimax
                        logger.error(f"Minimax error: {e}")
                        # Try fallback: use first legal move if available
                        if len(legal_moves_debug) > 0:
                            from_pos, to_pos = legal_moves_debug[0]
                            action = move_to_action(from_pos, to_pos, self.turn)
                            logger.warning(f"Using first legal move as fallback: {action}")
                            send_string(self.socket, json.dumps(action))
                            
                            # Read updated state
                            try:
                                state_json = recv_string(self.socket)
                                if state_json is None:
                                    logger.info("Socket closed by server (game ended)")
                                    break
                                state = json_to_state(state_json)
                            except ConnectionError as e:
                                logger.warning(f"Connection error: {e}")
                                break
                        else:
                            logger.error("No legal moves available and no fallback possible!")
                            break
                    except Exception as e:
                        logger.error(f"Error computing/sending move: {e}", exc_info=True)
                        # Fallback: send first legal move if available
                        if len(legal_moves_debug) > 0:
                            from_pos, to_pos = legal_moves_debug[0]
                            
                            # Validate the fallback move is actually legal
                            if (from_pos, to_pos) not in legal_moves_debug:
                                logger.error(f"Fallback move {from_pos} -> {to_pos} is not legal!")
                                # Try to find any legal move
                                if legal_moves_debug:
                                    from_pos, to_pos = legal_moves_debug[0]
                                else:
                                    logger.error("No legal moves available for fallback!")
                                    break
                            
                            # Validate destination is empty
                            if state.board[to_pos[0], to_pos[1]] != Pawn.EMPTY:
                                logger.error(
                                    f"Fallback destination {to_pos} is occupied by {state.board[to_pos[0], to_pos[1]]}!"
                                )
                                # Try to find a move to an empty square
                                found_valid = False
                                for fallback_from, fallback_to in legal_moves_debug:
                                    if state.board[fallback_to[0], fallback_to[1]] == Pawn.EMPTY:
                                        from_pos, to_pos = fallback_from, fallback_to
                                        found_valid = True
                                        break
                                if not found_valid:
                                    logger.error("No valid fallback move to empty square!")
                                    break
                            
                            action = move_to_action(from_pos, to_pos, self.turn)
                            logger.warning(f"Using fallback move: {action}")
                            send_string(self.socket, json.dumps(action))
                            
                            # Read updated state
                            try:
                                state_json = recv_string(self.socket)
                                if state_json is None:
                                    logger.info("Socket closed by server (game ended)")
                                    break
                                state = json_to_state(state_json)
                            except ConnectionError as e:
                                logger.warning(f"Connection error: {e}")
                                break
                        else:
                            logger.error("No legal moves available!")
                            break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            if self.socket:
                self.socket.close()
                logger.info("Socket closed")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Tablut Python Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play as white with default settings
  python agent.py WHITE 60 127.0.0.1
  
  # Play as black with model
  python agent.py BLACK 60 127.0.0.1 --model models/value_net.pt
  
  # Custom depth and seed
  python agent.py WHITE 60 127.0.0.1 --depth 5 --seed 42
        """
    )
    
    # Required arguments (matching competition spec)
    parser.add_argument(
        "player",
        type=str,
        help="Player role: WHITE or BLACK (case-insensitive)"
    )
    parser.add_argument(
        "timeout",
        type=int,
        help="Timeout per move in seconds"
    )
    parser.add_argument(
        "server_ip",
        type=str,
        help="Server IP address"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to value network model (.pt file)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Minimax search depth (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 5800 for white, 5801 for black)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine port
    if args.port is None:
        player_upper = args.player.upper()
        server_port = 5800 if player_upper == "WHITE" else 5801
    else:
        server_port = args.port
    
    # Create and run agent
    try:
        agent = TablutAgent(
            player=args.player,
            timeout=args.timeout,
            server_host=args.server_ip,
            server_port=server_port,
            model_path=args.model,
            max_depth=args.depth,
            seed=args.seed,
        )
        agent.run()
    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

