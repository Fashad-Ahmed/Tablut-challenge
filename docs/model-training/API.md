# API Reference

## JSON Message Protocol

### Game State (Server → Client)

**Format:**
```json
{
  "board": [
    ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
    ["O", "O", "W", "W", "K", "W", "W", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
  ],
  "turn": "W"
}
```

**Board Encoding:**
- `"O"`: Empty cell
- `"W"`: White soldier
- `"B"`: Black soldier
- `"K"`: King
- `"T"`: Throne (castle, usually empty or occupied by king)

**Turn Values:**
- `"W"`: White's turn
- `"B"`: Black's turn
- `"WW"`: White wins
- `"BW"`: Black wins
- `"D"`: Draw

### Action (Client → Server)

**Format:**
```json
{
  "from": "e5",
  "to": "e6",
  "turn": "W"
}
```

**Move Notation:**
- Chess-style notation: column letter (a-i) + row number (1-9)
- Example: `"e5"` = row 4, column 4 (0-indexed: row=4, col=4)
- `from`: Source square
- `to`: Destination square
- `turn`: Current player ("W" or "B")

**Coordinate Conversion:**
```python
# String to (row, col)
"e5" → (4, 4)  # row = 5 - 1, col = ord('e') - ord('a')

# (row, col) to string
(4, 4) → "e5"  # col letter = chr(97 + 4), row = 4 + 1
```

## Socket Communication

### Connection

**White Player:**
- Host: Server IP
- Port: 5800
- Protocol: TCP

**Black Player:**
- Host: Server IP
- Port: 5801
- Protocol: TCP

### Message Format

All messages use length-prefixed UTF-8 strings:

1. **Send message:**
   ```
   [4 bytes: length (big-endian int)]
   [N bytes: UTF-8 JSON string]
   ```

2. **Receive message:**
   ```
   Read 4 bytes → length
   Read length bytes → JSON string
   Decode UTF-8 → Parse JSON
   ```

### Communication Flow

1. **Connection**
   - Client connects to server
   - Client sends player name (JSON string)

2. **Initial State**
   - Server sends initial game state (JSON)

3. **Game Loop**
   - Client reads state
   - Client computes move
   - Client sends action (JSON)
   - Server validates and applies move
   - Server sends updated state to both players
   - Repeat

4. **Termination**
   - Game ends when turn is "WW", "BW", or "D"
   - Socket closes

## Python API

### `agent.py`

**Main Entrypoint:**
```python
from python_client.agent import TablutAgent

agent = TablutAgent(
    player="WHITE",
    timeout=60,
    server_host="127.0.0.1",
    server_port=5800,
    model_path="models/rl_value_net_5M.zip",
    max_depth=4,
    seed=42
)
agent.run()
```

### `minimax_agent.py`

**Minimax Agent:**
```python
from python_client.minimax_agent import MinimaxAgent

value_fn = lambda state: evaluate_heuristic(state, Turn.WHITE)
agent = MinimaxAgent(
    value_fn=value_fn,
    max_depth=4,
    time_limit=55.0,
    use_transposition=True,
    move_ordering=True
)

move = agent.get_move(state)
```

### `rl_value_wrapper.py`

**PPO Model Loading:**
```python
from python_client.rl_value_wrapper import RLValueWrapper

wrapper = RLValueWrapper(
    model_path="models/rl_value_net_5M.zip",
    device="cpu"
)
value = wrapper.evaluate(state)
```

### `tablut_env.py`

**Environment:**
```python
from python_client.tablut_env import TablutEnv

env = TablutEnv()
obs, info = env.reset()
legal_moves = env.get_legal_moves(env.state)
obs, reward, done, truncated, info = env.step(action)
```

### `utils.py`

**State Conversion:**
```python
from python_client.utils import json_to_state, move_to_action

state = json_to_state(json_string)
action = move_to_action(from_pos, to_pos, Turn.WHITE)
```

**Socket Helpers:**
```python
from python_client.utils import connect_to_server, send_string, recv_string

sock = connect_to_server("127.0.0.1", 5800)
send_string(sock, json.dumps(action))
state_json = recv_string(sock)
```

## Error Handling

### Socket Errors
- Connection refused: Server not running
- Timeout: Move computation exceeded time limit
- Invalid JSON: Malformed message from server

### Game Errors
- Illegal move: Server rejects move, player loses
- No legal moves: Current player loses
- State deserialization: Invalid board format

### Recovery
- Agent has fallback move (first legal move)
- Logging for debugging
- Graceful shutdown on errors

## Example Usage

### Complete Game Loop

```python
from python_client.agent import TablutAgent

# Initialize agent
agent = TablutAgent(
    player="WHITE",
    timeout=60,
    server_host="127.0.0.1",
    model_path="models/rl_value_net_5M.zip"
)

# Run game
agent.run()
```

### Custom Value Function

```python
from python_client.minimax_agent import MinimaxAgent
from python_client.rl_value_wrapper import RLValueWrapper

# Load PPO model
wrapper = RLValueWrapper("models/rl_value_net_5M.zip", device="cpu")

# Create value function
def value_fn(state):
    return wrapper.evaluate(state)

# Create minimax agent
agent = MinimaxAgent(value_fn=value_fn, max_depth=5)
move = agent.get_move(state)
```

## TODO: Verify Field Names

The exact JSON field names may differ from the Java server implementation. Verify:
- `board` vs `boardState`
- `turn` vs `currentTurn`
- Field structure matches `StateTablut.java` serialization

Check Java source:
- `Tablut/src/it/unibo/ai/didattica/competition/tablut/domain/StateTablut.java`
- `Tablut/src/it/unibo/ai/didattica/competition/tablut/domain/Action.java`

