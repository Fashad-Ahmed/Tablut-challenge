# Architecture Documentation

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Java Referee Server                       │
│              (Game state management, validation)             │
└──────────────────────┬──────────────────────────────────────┘
                       │ TCP Socket (JSON)
                       │ Port 5800 (White) / 5801 (Black)
┌──────────────────────▼──────────────────────────────────────┐
│                  Python Agent (agent.py)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Socket Communication Layer                           │  │
│  │  - JSON serialization/deserialization                  │  │
│  │  - Length-prefixed UTF-8 protocol                    │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  Game Loop                                            │  │
│  │  1. Read state from server                            │  │
│  │  2. Compute move (MinimaxAgent)                       │  │
│  │  3. Send move to server                               │  │
│  │  4. Read updated state                                │  │
│  │  5. Repeat                                            │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  MinimaxAgent (minimax_agent.py)                      │  │
│  │  - Iterative deepening                                │  │
│  │  - Alpha-beta pruning                                 │  │
│  │  - Transposition table                                │  │
│  │  - Move ordering                                      │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                       │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │  Value Function (value_fn)                            │  │
│  │  ┌──────────────┐  ┌──────────────┐                  │  │
│  │  │ RLValueWrapper│  │ Heuristics   │                  │  │
│  │  │ (PPO model)  │  │ (fallback)   │                  │  │
│  │  └──────────────┘  └──────────────┘                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `agent.py`
- **Purpose**: Main entrypoint and game loop orchestration
- **Responsibilities**:
  - TCP socket connection management
  - JSON message parsing/serialization
  - Turn-based game loop
  - Time management and timeout handling
  - Error recovery and fallback moves
  - Logging and monitoring

### `minimax_agent.py`
- **Purpose**: Game tree search with alpha-beta pruning
- **Responsibilities**:
  - Iterative deepening search
  - Alpha-beta pruning optimization
  - Transposition table caching
  - Move ordering for better pruning
  - Time-aware search (always returns best move found)
  - Thread-safe time checks

### `model_loader.py`
- **Purpose**: Neural network model management for supervised learning models
- **Responsibilities**:
  - PyTorch model loading/saving
  - Device selection (CPU/MPS/CUDA)
  - State-to-tensor conversion
  - Batch inference optimization
  - Model architecture definitions

### `rl_value_wrapper.py`
- **Purpose**: Bridge between stable-baselines3 PPO models and minimax evaluation
- **Responsibilities**:
  - Loading PPO models from stable-baselines3 (.zip format)
  - Extracting value function from PPO policy
  - Converting GameState to observation format
  - Providing evaluate() interface compatible with minimax
  - Batch evaluation support

### `tablut_env.py`
- **Purpose**: Gymnasium-compatible game environment
- **Responsibilities**:
  - Game state representation
  - Legal move generation
  - Move validation and application
  - Terminal state detection
  - Reward shaping
  - Observation encoding (for RL)

### `trainer.py`
- **Purpose**: PPO self-play training loop
- **Responsibilities**:
  - PPO training using stable-baselines3
  - Self-play environment setup
  - Action masking integration
  - Model checkpointing and evaluation callbacks
  - Training metrics logging (TensorBoard/WandB)
  - Value function extraction from trained policy

### `heuristics.py`
- **Purpose**: Fast handcrafted evaluation functions
- **Responsibilities**:
  - Position evaluation (material, safety, mobility)
  - Move ordering scores
  - Fallback when model unavailable
  - Board feature detection (king position, camps, escapes)

### `utils.py`
- **Purpose**: Common utilities
- **Responsibilities**:
  - JSON ↔ GameState conversion
  - Move encoding/decoding (chess notation)
  - Socket communication helpers
  - Deterministic seeding
  - Time utilities

## Data Flow

### Game State Representation

**Java Server → Python Agent:**
```json
{
  "board": [
    ["O", "O", "O", ...],  // 9x9 array
    ...
  ],
  "turn": "W"  // "W", "B", "WW", "BW", "D"
}
```

**Python Agent → Java Server:**
```json
{
  "from": "e5",
  "to": "e6",
  "turn": "W"
}
```

**Internal Representation:**
```python
GameState(
    board: np.ndarray[9, 9]  # Pawn enum values
    turn: Turn enum
)
```

### Hybrid Decision Loop Pseudocode

```
function get_move(state: GameState) -> Move:
    best_move = None
    best_value = -inf
    
    for depth in 1..max_depth:
        if time_limit_exceeded():
            break
        
        move, value = minimax_ab(
            state, 
            depth, 
            alpha=-inf, 
            beta=+inf,
            maximizing=(state.turn == WHITE)
        )
        
        if move is not None:
            best_move = move
            best_value = value
        
        if abs(value) > 900:  # Terminal state
            break
    
    return best_move

function minimax_ab(state, depth, alpha, beta, maximizing):
    if depth == 0 or is_terminal(state):
        return None, value_fn(state)  # PPO value network or heuristic
    
    moves = get_legal_moves(state)
    moves = order_moves(moves)  # Heuristic ordering
    
    for move in moves:
        new_state = apply_move(state, move)
        _, value = minimax_ab(new_state, depth-1, alpha, beta, not maximizing)
        
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
        
        if beta <= alpha:  # Alpha-beta cutoff
            break
    
    return best_move, best_value
```

## Communication Protocol

### Connection Sequence

1. **Client connects** to server (TCP socket)
2. **Client sends** player name (JSON string)
3. **Server sends** initial game state (JSON)
4. **Game loop**:
   - Client reads state
   - Client computes move
   - Client sends move (JSON Action)
   - Server validates and applies move
   - Server sends updated state to both players
   - Repeat

### Message Format

**String Protocol (Java StreamUtils):**
- 4 bytes: Length (big-endian int)
- N bytes: UTF-8 encoded JSON string

**Python Implementation:**
```python
def send_string(sock, message: str):
    data = message.encode("utf-8")
    length = len(data)
    sock.sendall(struct.pack(">i", length))  # 4-byte big-endian
    sock.sendall(data)

def recv_string(sock) -> str:
    length_bytes = recv_exact(sock, 4)
    length = struct.unpack(">i", length_bytes)[0]
    data = recv_exact(sock, length)
    return data.decode("utf-8")
```

## Performance Considerations

### Time Management
- **Iterative deepening**: Always returns best move found, even if interrupted
- **Time buffer**: 5 seconds reserved for move transmission
- **Thread-safe checks**: Time limit checked during search

### Memory Management
- **Transposition table**: Limited size (100k entries), periodic clearing
- **State caching**: Minimal (only current state)
- **Model inference**: CPU-optimized, batch processing when possible

### Search Optimization
- **Move ordering**: Heuristic-based to improve alpha-beta pruning
- **Transposition table**: Cache evaluated positions
- **Early termination**: Stop on winning moves

## Extension Points

### Adding New Heuristics
1. Add function to `heuristics.py`
2. Integrate into `evaluate_heuristic()` or `move_ordering_key()`

### Custom Model Architecture
1. Define new `nn.Module` in `model_loader.py`
2. Update `ModelLoader` to support new architecture

### Alternative Training Algorithms
1. Extend `trainer.py` with new RL algorithms (currently supports PPO)
2. Implement value function extraction from different policy architectures

### Different Communication Protocols
1. Extend `utils.py` with new serialization functions
2. Update `agent.py` socket handling

## Testing Strategy

- **Unit tests**: Each module tested independently
- **Integration tests**: Full game loop with mock server
- **Performance tests**: Time limits, memory usage
- **Competition tests**: Against Java referee server

## Deployment Considerations

### Competition Environment
- **OS**: Linux Debian 64-bit
- **Resources**: 4 CPUs, 8GB RAM, 30GB disk
- **No GPU**: CPU-only inference
- **No internet**: All dependencies bundled

### Development Environment
- **OS**: macOS M1 Pro (ARM64)
- **Resources**: MPS acceleration available
- **Internet**: For package installation

### Model Deployment
- PPO models are saved as `.zip` files by stable-baselines3
- Include in VM image
- Load with `RLValueWrapper(model_path, device="cpu")`

