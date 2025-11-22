# Playing Tablut with GUI

This guide shows you how to start the game with a visual GUI so you can watch your PPO-trained agent play in real-time.

## Quick Start (3 Terminals)

### Terminal 1: Start Java Server with GUI
```bash
cd Tablut/Executables
java -jar Server.jar -g
```

**What you'll see:**
- A window titled "Tablut" will open (415x415 pixels)
- The game board with pieces will be displayed
- The board updates automatically as moves are made

### Terminal 2: Run Your RL Agent (White Player)
```bash
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 4
```

### Terminal 3: Run Black Player (Random or Another Agent)
```bash
# Option A: Use random player (included)
cd Tablut/Executables
java -jar RandomPlayer.jar BLACK 60 127.0.0.1

# Option B: Use your agent as black
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent BLACK 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 4
```

## What You'll See

### GUI Window
- **Board**: 9x9 grid showing the Tablut board
- **Pieces**:
  - White pieces (pawns)
  - Black pieces (pawns)
  - King (in the center, white's goal)
- **Real-time updates**: Board refreshes after each move
- **Game status**: Shows current turn and game state

### Console Output
Your Python agent will log:
- Move computation time
- Selected moves
- Game state updates
- Model evaluation (if using RL model)

## Server Options

### Enable GUI (Default)
```bash
java -jar Server.jar -g
```

### Disable GUI (Headless Mode)
```bash
java -jar Server.jar
```

### Other Useful Options
```bash
# Set move time limit (default: 60 seconds)
java -jar Server.jar -g -t 120

# Set game rules (1=Tablut, 2=Modern, 3=Brandub, 4=Ashton)
java -jar Server.jar -g -r 4

# Show all options
java -jar Server.jar --help
```

## Testing Different Configurations

### 1. RL Agent vs Random Player
```bash
# Terminal 1: Server with GUI
cd Tablut/Executables && java -jar Server.jar -g

# Terminal 2: Your RL agent (White)
cd Tablut-challenge && source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip

# Terminal 3: Random player (Black)
cd Tablut/Executables && java -jar RandomPlayer.jar BLACK 60 127.0.0.1
```

### 2. RL Agent vs Heuristic-Only Agent
```bash
# Terminal 1: Server with GUI
cd Tablut/Executables && java -jar Server.jar -g

# Terminal 2: RL agent (White) - with model
cd Tablut-challenge && source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip

# Terminal 3: Heuristic-only agent (Black) - no model
cd Tablut-challenge && source venv/bin/activate
python -m python_client.agent BLACK 60 127.0.0.1
```

### 3. RL Agent vs RL Agent (Self-Play)
```bash
# Terminal 1: Server with GUI
cd Tablut/Executables && java -jar Server.jar -g

# Terminal 2: RL agent (White)
cd Tablut-challenge && source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 4

# Terminal 3: RL agent (Black)
cd Tablut-challenge && source venv/bin/activate
python -m python_client.agent BLACK 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 4
```

## Understanding the GUI

### Board Layout
```
    0 1 2 3 4 5 6 7 8
  0 . . . C C C . . .
  1 . . . . C . . . .
  2 . . . . . . . . .
  3 C . . . . . . . C
  4 C C . . K . . C C
  5 C . . . . . . . C
  6 . . . . . . . . .
  7 . . . . C . . . .
  8 . . . C C C . . .
```

- **C** = Camp squares (black starting positions)
- **K** = King (white's goal is to get king to edge)
- **.** = Regular squares

### Game Rules (Ashton - Default)
- **White (King + 8 pawns)**: Goal is to get king to any edge square
- **Black (16 pawns)**: Goal is to capture the king
- **Moves**: Pieces move like rooks (horizontally/vertically)
- **Captures**: Surround opponent pieces (2 vs 1)

## Troubleshooting

### GUI Window Doesn't Appear
```bash
# Make sure you're using the -g flag
java -jar Server.jar -g

# Check if Java GUI is supported
java -version
# Should show Java version, not headless
```

### Connection Refused
- Make sure server is running first (Terminal 1)
- Wait for "Server started" message before connecting clients
- Check that ports 5800 (white) and 5801 (black) are available

### Agent Not Making Moves
- Check console output for errors
- Verify model loads correctly: `python -c "from python_client.rl_value_wrapper import RLValueWrapper; RLValueWrapper('models/rl_value_net.zip')"`
- Try without model first: `python -m python_client.agent WHITE 60 127.0.0.1`

## Performance Tips

### Faster Moves (Lower Depth)
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip --depth 3
```

### Stronger Moves (Higher Depth)
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip --depth 5
```

### Compare Models
```bash
# Test PPO model (recommended)
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip

# Test heuristics only (fallback)
python -m python_client.agent WHITE 60 127.0.0.1
```

## Recording Games

The server automatically logs games. Check:
```bash
cd Tablut/Executables/logs
ls -la
```

## Next Steps

1. **Watch your RL agent play** - See how it makes decisions
2. **Compare strategies** - RL vs heuristics vs random
3. **Tune parameters** - Adjust depth, time limits
4. **Train longer** - Improve model with more training
5. **Analyze games** - Review logs to understand agent behavior

Enjoy watching your RL agent play! 🎉

