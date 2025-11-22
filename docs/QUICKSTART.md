# Quick Start Guide

## 1. Setup (First Time Only)

### Check Prerequisites
```bash
# Make sure you're in the project root (Tablut-challenge/)
cd Tablut-challenge

# Run prerequisite checker (note the ./ prefix)
./scripts/check_prerequisites.sh
```

### Install Java (if not installed)
```bash
# macOS with Homebrew (recommended)
brew install openjdk@17
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk

# Verify
java -version
```

See [INSTALL_JAVA.md](INSTALL_JAVA.md) for detailed Java installation instructions.

### Install Python Dependencies
```bash
# Navigate to project directory
cd Tablut-challenge

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run Tests (Verify Installation)

```bash
# Run all tests
pytest python_client/tests/ -v

# Or run specific test
pytest python_client/tests/test_env.py -v
```

## 3. Run the Agent

### Option A: Against Java Server (Recommended)

**Terminal 1 - Start Java Server:**
```bash
cd Tablut/Executables
java -jar Server.jar
```

**Terminal 2 - Run White Player:**
```bash
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1
```

**Terminal 3 - Run Black Player (or use RandomPlayer.jar):**
```bash
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent BLACK 60 127.0.0.1
```

### Option B: Using Helper Scripts

```bash
# Make scripts executable (first time only)
chmod +x scripts/*.sh

# Run agent
./scripts/run_agent.sh WHITE 60 127.0.0.1
```

### Option C: With Trained PPO Model

```bash
# First, train a PPO model (requires stable-baselines3)
pip install stable-baselines3
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip

# Then run with model
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip
```

## 4. Common Commands

### Test Environment
```bash
python -c "
from python_client.tablut_env import TablutEnv
env = TablutEnv()
obs, info = env.reset()
print('Environment initialized successfully!')
print(f'Turn: {info[\"turn\"]}')
"
```

### Test Model Loader
```bash
python -c "
from python_client.model_loader import ModelLoader
from python_client.tablut_env import TablutEnv
env = TablutEnv()
env.reset()
loader = ModelLoader(device='cpu')
value = loader.evaluate(env.state)
print(f'Model evaluation: {value:.4f}')
"
```

### Test Minimax
```bash
python -c "
from python_client.minimax_agent import MinimaxAgent
from python_client.tablut_env import TablutEnv
from python_client.heuristics import evaluate_heuristic
from python_client.utils import Turn

env = TablutEnv()
env.reset()
value_fn = lambda s: evaluate_heuristic(s, Turn.WHITE)
agent = MinimaxAgent(value_fn=value_fn, max_depth=2, time_limit=5.0)
move = agent.get_move(env.state)
print(f'Best move: {move}')
"
```

## 5. Training a Model

### PPO Training (Recommended)

The agent uses Proximal Policy Optimization (PPO) for training value networks through self-play.

**Basic training:**
```bash
pip install stable-baselines3
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --device cpu
```

**Extended training (5M timesteps):**
```bash
python -m python_client.trainer \
    --algo ppo \
    --timesteps 5000000 \
    --save models/rl_value_net_5M.zip \
    --device cpu \
    --checkpoint-interval 250000 \
    --wandb
```

**Using trained model:**
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip
```

## 6. Troubleshooting

### Java Not Found
```bash
# Error: "Unable to locate a Java Runtime"
# Solution: Install Java (see INSTALL_JAVA.md)
brew install openjdk@17
sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
java -version  # Verify
```

### Import Errors
```bash
# Make sure you're in the project root
cd Tablut-challenge

# Activate venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Socket Connection Errors
- Make sure Java server is running
- Check server is listening on port 5800 (white) or 5801 (black)
- Verify IP address is correct (127.0.0.1 for localhost)

### Module Not Found
```bash
# Run from project root
cd Tablut-challenge
python -m python_client.agent WHITE 60 127.0.0.1
# NOT: python python_client/agent.py
```

## 7. Example Full Workflow

```bash
# 1. Setup
cd Tablut-challenge
source venv/bin/activate
pip install -r requirements.txt

# 2. Test
pytest python_client/tests/ -v

# 3. Train PPO model (optional, requires stable-baselines3)
pip install stable-baselines3
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --device cpu

# 4. Start server (in separate terminal)
cd Tablut/Executables
java -jar Server.jar

# 5. Run agent (in another terminal)
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip --depth 4
```

## 8. Command-Line Options

```bash
# Basic usage
python -m python_client.agent WHITE 60 127.0.0.1

# With all options
python -m python_client.agent \
    WHITE \
    60 \
    127.0.0.1 \
    --model models/rl_value_net.zip \
    --depth 5 \
    --seed 42 \
    --log-level DEBUG

# Help
python -m python_client.agent --help
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
- See [docs/TUNING.md](docs/TUNING.md) for performance optimization
- Review [CHECKLIST.md](CHECKLIST.md) before submission

