# Tablut Challenge - Python Agent

Production-quality hybrid RL + Minimax agent for the Tablut Challenge (Fundamentals of AI and Knowledge Representation, A.A. 2025/2026).

## Overview

This agent combines:
- **Minimax search with alpha-beta pruning** for game tree exploration
- **RL-trained value network** for leaf position evaluation
- **Handcrafted heuristics** for move ordering and fallback evaluation
- **Robust socket communication** with Java referee server

## Quick Start

### Prerequisites

- **Python 3.10+** (check with `python3 --version`)
- **Java Runtime** (required for the referee server)
  - macOS: `brew install openjdk@17` or download from [Oracle](https://www.oracle.com/java/technologies/downloads/)
  - Linux: `sudo apt-get install default-jdk`
- **macOS M1 Pro (ARM64)** for development or **Linux Debian 64-bit** for competition
- Virtual environment (recommended)

**Quick check:** Run `./scripts/check_prerequisites.sh` to verify all prerequisites

```bash
# Make sure you're in the project root directory
cd Tablut-challenge
./scripts/check_prerequisites.sh
```

### Installation

```bash
# Clone repository
cd Tablut-challenge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

### Running the Agent

**Basic usage (heuristic evaluation only):**
```bash
python -m python_client.agent WHITE 60 127.0.0.1
```

**With trained model:**
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/value_net.pt
```

**Full options:**
```bash
python -m python_client.agent \
    WHITE \
    60 \
    127.0.0.1 \
    --model models/value_net.pt \
    --depth 5 \
    --seed 42 \
    --log-level INFO
```

### Training a Value Network

The agent supports two training approaches:

#### 1. Supervised Learning (Quick Start)

Trains a value network to approximate heuristic evaluation. Fast but limited by heuristic quality.

```bash
python -m python_client.trainer \
    --algo simple \
    --episodes 500 \
    --save models/value_net.pt \
    --lr 1e-3 \
    --batch-size 32
```

#### 2. Reinforcement Learning (Full Implementation)

**PPO (Proximal Policy Optimization) - Recommended:**
```bash
# Install stable-baselines3
pip install stable-baselines3

# Train with PPO
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --device mps \
    --checkpoint-interval 50000 \
    --seed 42

# With Weights & Biases logging (optional)
pip install wandb
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --wandb
```

**DQN (Deep Q-Network):**
```bash
python -m python_client.trainer \
    --algo dqn \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --device mps
```

**Using RL-trained models:**
```bash
# RL models are saved as .zip files
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip
```

**Training Features:**
- ✅ Self-play training (agent learns by playing against itself)
- ✅ Action masking (only legal moves considered)
- ✅ Automatic checkpointing
- ✅ Evaluation callbacks (saves best model)
- ✅ TensorBoard/WandB logging
- ✅ Value function extraction for minimax
- ✅ MPS/CUDA support for faster training

**Training Tips:**
- Start with 100k-500k timesteps for initial training
- Use `--device mps` on M1 Macs for faster training
- Monitor training with TensorBoard: `tensorboard --logdir tensorboard_logs/`
- Checkpoints are saved automatically for resuming training

## Project Structure

```
Tablut-challenge/
├── python_client/          # Main agent code
│   ├── agent.py           # Main entrypoint (socket communication)
│   ├── minimax_agent.py   # Minimax with alpha-beta
│   ├── model_loader.py    # PyTorch model loader
│   ├── tablut_env.py      # Gymnasium environment
│   ├── trainer.py         # Self-play training (RL + supervised)
│   ├── rl_value_wrapper.py # Bridge for SB3 models → minimax
│   ├── action_mask_wrapper.py # Action masking for RL training
│   ├── heuristics.py      # Handcrafted heuristics
│   ├── utils.py           # Utilities (JSON, sockets, etc.)
│   └── tests/             # Unit tests
├── scripts/               # Helper scripts
│   ├── run_agent.sh      # Agent launcher
│   ├── train.sh          # Training launcher
│   └── eval.sh           # Model evaluation
├── docs/                  # Documentation
│   ├── API.md            # JSON protocol reference
│   ├── DEPLOY.md         # Deployment guide
│   └── TUNING.md         # Hyperparameter tuning
├── requirements.txt      # Core dependencies
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## Development

### Running Tests

```bash
# All tests
pytest python_client/tests/ -v

# Specific test file
pytest python_client/tests/test_env.py -v

# With coverage
pytest python_client/tests/ --cov=python_client --cov-report=html
```

### Code Quality

```bash
# Format code
black python_client/

# Sort imports
isort python_client/

# Lint
ruff check python_client/

# Type check
mypy python_client/
```

### Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

**High-level flow:**
1. Agent connects to Java referee via TCP socket
2. Receives game state as JSON
3. Minimax agent searches game tree using value network for evaluation
4. Sends best move as JSON action
5. Repeats until game ends

## Configuration

### Agent Parameters

- `player`: "WHITE" or "BLACK" (required)
- `timeout`: Time limit per move in seconds (required)
- `server_ip`: Server IP address (required)
- `--model`: Path to value network model (optional)
- `--depth`: Minimax search depth (default: 4)
- `--seed`: Random seed for reproducibility (optional)
- `--port`: Server port override (default: 5800 for white, 5801 for black)

### Environment Variables

- `PYTHONUNBUFFERED=1`: Disable output buffering (recommended)

## Troubleshooting

### Common Issues on macOS M1

**Issue: "Unable to locate a Java Runtime"**
- **Solution:** Install Java:
  ```bash
  # Using Homebrew (recommended)
  brew install openjdk@17
  sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk
  
  # Or download from Oracle: https://www.oracle.com/java/technologies/downloads/
  ```
  Verify: `java -version`

**Issue: PyTorch MPS backend errors**
- Solution: Agent defaults to CPU for competition compatibility. MPS is only used during development if available.

**Issue: Import errors**
- Solution: Ensure virtual environment is activated and dependencies are installed:
  ```bash
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**Issue: Socket connection refused**
- Solution: Ensure Java server is running and listening on correct port (5800 for white, 5801 for black).
- Make sure Java is installed first (see above).

**Issue: Timeout errors**
- Solution: Reduce `--depth` parameter or ensure model inference is fast enough.

### Performance Tuning

See [docs/TUNING.md](docs/TUNING.md) for hyperparameter tuning guidelines.

## Competition Submission

1. **Prepare virtual machine image** (Linux Debian 64-bit)
2. **Create submission .txt file** with:
   - Link to virtual machine
   - Command-line execution instructions
   - Optional parameters documentation
3. **Test agent** against Java referee locally
4. **Verify time limits** (60s per move)

Example submission command:
```bash
python -m python_client.agent WHITE 60 <SERVER_IP> --model models/value_net.pt --depth 4
```

## License

MIT License

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## References

- [Tablut Competition Repository](https://github.com/AGalassi/TablutCompetition)
- [Ashton Tablut Rules](https://aagenielsen.dk/ashton.php)
- Course: Fundamentals of AI and Knowledge Representation, A.A. 2025/2026
