"""
Utilities for Tablut game client.

This module provides:
- JSON ↔ board state serialization/deserialization
- Move encoding/decoding (chess notation: "a1" ↔ (row, col))
- Time utilities for move timing
- Deterministic seeding
- Socket communication helpers

Test with: pytest python_client/tests/test_utils.py
"""

import json
import random
import struct
import socket
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Pawn(Enum):
    """Board cell content."""
    EMPTY = "O"
    WHITE = "W"
    BLACK = "B"
    THRONE = "T"
    KING = "K"


class Turn(Enum):
    """Game turn state."""
    WHITE = "W"
    BLACK = "B"
    WHITEWIN = "WW"
    BLACKWIN = "BW"
    DRAW = "D"


@dataclass
class GameState:
    """Internal game state representation."""
    board: np.ndarray  # 9x9 array of Pawn enum values
    turn: Turn
    # Cache for performance
    _board_hash: Optional[int] = None

    def __post_init__(self):
        """Ensure board is proper shape."""
        if self.board.shape != (9, 9):
            raise ValueError(f"Board must be 9x9, got {self.board.shape}")

    def hash(self) -> int:
        """Compute hash for transposition table."""
        if self._board_hash is None:
            # Use numpy's hash for efficiency
            self._board_hash = hash((self.board.tobytes(), self.turn))
        return self._board_hash


def json_to_state(json_str: str) -> GameState:
    """
    Parse JSON state from Java server into GameState.
    
    Expected JSON format:
    {
        "board": [["O", "O", ...],  // 9x9 array
        "turn": "W"  // "W", "B", "WW", "BW", "D"
    }
    
    Note: Field names verified against Java StateTablut.java serialization.
    The server sends: {"board": [[...]], "turn": "W"/"B"/"WW"/"BW"/"D"}
    """
    data = json.loads(json_str)
    
    # Parse board
    board_json = data.get("board", [])
    board = np.zeros((9, 9), dtype=object)
    
    # Mapping from server JSON format to Pawn enum values
    # Server sends: "EMPTY", "WHITE", "BLACK", "KING"
    # Pawn enum expects: "O", "W", "B", "K"
    pawn_mapping = {
        "EMPTY": "O",
        "WHITE": "W",
        "BLACK": "B",
        "KING": "K",
        "THRONE": "T",
        # Also handle single-letter format (backward compatibility)
        "O": "O",
        "W": "W",
        "B": "B",
        "K": "K",
        "T": "T",
    }
    
    for i in range(9):
        for j in range(9):
            pawn_str = board_json[i][j] if isinstance(board_json[i], list) else board_json[i * 9 + j]
            # Convert server format to enum format
            pawn_str_normalized = pawn_mapping.get(pawn_str.upper() if isinstance(pawn_str, str) else str(pawn_str).upper(), "O")
            try:
                board[i, j] = Pawn(pawn_str_normalized)
            except ValueError:
                # Fallback to EMPTY if conversion fails
                board[i, j] = Pawn.EMPTY
    
    # Parse turn
    turn_str = data.get("turn", "W")
    try:
        turn = Turn(turn_str)
    except ValueError:
        turn = Turn.WHITE  # Default fallback
    
    return GameState(board=board, turn=turn)


def state_to_json(state: GameState) -> str:
    """
    Convert GameState to JSON string (for debugging/logging).
    
    TODO: This may not match exact server format - use for local testing only.
    """
    board_list = []
    for i in range(9):
        row = []
        for j in range(9):
            pawn = state.board[i, j]
            row.append(pawn.value if isinstance(pawn, Pawn) else "O")
        board_list.append(row)
    
    return json.dumps({
        "board": board_list,
        "turn": state.turn.value
    })


def move_to_action(from_pos: Tuple[int, int], to_pos: Tuple[int, int], turn: Turn) -> Dict[str, Any]:
    """
    Convert (row, col) positions to Action JSON for server.
    
    Args:
        from_pos: (row, col) tuple, 0-indexed
        to_pos: (row, col) tuple, 0-indexed
        turn: Current turn
    
    Returns:
        Dict matching Action.java structure: {"from": "a1", "to": "b2", "turn": "W"}
    """
    row_from, col_from = from_pos
    row_to, col_to = to_pos
    
    # Convert to chess notation: column is letter (a-i), row is number (1-9)
    from_str = f"{chr(97 + col_from)}{row_from + 1}"
    to_str = f"{chr(97 + col_to)}{row_to + 1}"
    
    return {
        "from": from_str,
        "to": to_str,
        "turn": turn.value
    }


def action_to_move(action: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Parse Action JSON from server to (from_pos, to_pos) tuples.
    
    Args:
        action: Dict with "from" and "to" keys (e.g., {"from": "a1", "to": "b2"})
    
    Returns:
        ((row_from, col_from), (row_to, col_to)) tuples, 0-indexed
    """
    from_str = action["from"].lower()
    to_str = action["to"].lower()
    
    col_from = ord(from_str[0]) - 97
    row_from = int(from_str[1]) - 1
    col_to = ord(to_str[0]) - 97
    row_to = int(to_str[1]) - 1
    
    return ((row_from, col_from), (row_to, col_to))


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def recv_string(sock: socket.socket, timeout: Optional[float] = None) -> Optional[str]:
    """
    Receive length-prefixed UTF-8 string from socket.
    
    Protocol: 4-byte big-endian int (length), then UTF-8 bytes.
    Matches Java StreamUtils.readString().
    
    Args:
        sock: Socket to read from
        timeout: Optional timeout in seconds
    
    Returns:
        Received string, or None if socket closed gracefully
    
    Raises:
        ConnectionError: If connection fails unexpectedly
        socket.timeout: If timeout is exceeded
    """
    # Set timeout if provided
    old_timeout = sock.gettimeout()
    if timeout is not None:
        sock.settimeout(timeout)
    
    try:
        # Read length (4 bytes, big-endian)
        length_bytes = b""
        while len(length_bytes) < 4:
            try:
                chunk = sock.recv(4 - len(length_bytes))
                if not chunk:
                    # Socket closed gracefully (game ended)
                    return None
                length_bytes += chunk
            except socket.timeout:
                raise
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                raise ConnectionError(f"Connection error while reading: {e}")
        
        length = struct.unpack(">i", length_bytes)[0]
        
        # Validate length (sanity check)
        if length < 0 or length > 1000000:  # Max 1MB
            raise ConnectionError(f"Invalid message length: {length}")
        
        # Read string bytes
        data = b""
        while len(data) < length:
            try:
                chunk = sock.recv(length - len(data))
                if not chunk:
                    # Socket closed gracefully
                    return None
                data += chunk
            except socket.timeout:
                raise
            except (ConnectionResetError, BrokenPipeError, OSError) as e:
                raise ConnectionError(f"Connection error while reading data: {e}")
        
        return data.decode("utf-8")
    finally:
        # Restore original timeout
        if timeout is not None:
            sock.settimeout(old_timeout)


def send_string(sock: socket.socket, message: str) -> None:
    """
    Send length-prefixed UTF-8 string to socket.
    
    Protocol: 4-byte big-endian int (length), then UTF-8 bytes.
    Matches Java StreamUtils.writeString().
    """
    data = message.encode("utf-8")
    length = len(data)
    
    # Send length (4 bytes, big-endian)
    sock.sendall(struct.pack(">i", length))
    
    # Send data
    sock.sendall(data)


def connect_to_server(host: str, port: int, timeout: float = 10.0) -> socket.socket:
    """
    Create and connect TCP socket to server.
    
    Args:
        host: Server hostname/IP
        port: Server port (5800 for white, 5801 for black)
        timeout: Connection timeout in seconds
    
    Returns:
        Connected socket
    
    Raises:
        ConnectionError: If connection fails
        socket.timeout: If connection times out
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except socket.timeout:
        sock.close()
        raise ConnectionError(
            f"Connection to {host}:{port} timed out after {timeout}s. "
            f"Make sure the Java server is running and listening on this port."
        )
    except ConnectionRefusedError:
        sock.close()
        raise ConnectionError(
            f"Connection refused to {host}:{port}. "
            f"The Java server is not running or not listening on this port. "
            f"Start the server with: cd Tablut/Executables && java -jar Server.jar"
        )
    except OSError as e:
        sock.close()
        raise ConnectionError(
            f"Failed to connect to {host}:{port}: {e}. "
            f"Check that the server is running and the address is correct."
        )
    
    sock.settimeout(None)  # Remove timeout after connection
    return sock

