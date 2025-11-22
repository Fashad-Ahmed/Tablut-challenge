# Tablut Challenge - Python Agent

Hybrid reinforcement learning and minimax agent for the Tablut Challenge. Combines PPO-trained value networks with alpha-beta search for competitive play.

## Architecture

The agent uses a hybrid approach:
- **Minimax search** with alpha-beta pruning and iterative deepening
- **PPO value network** trained through self-play (5M timesteps) for leaf evaluation
- **Heuristics** for move ordering and fallback evaluation
- **TCP socket communication** with Java referee server (JSON protocol)

## Installation

**Prerequisites:** Python 3.10+, Java Runtime (for referee server)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

**Basic (heuristics only):**
```bash
python -m python_client.agent WHITE 60 127.0.0.1
```

**With PPO model:**
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 4
```

**Parameters:**
- `player`: WHITE or BLACK
- `timeout`: Time limit per move (seconds)
- `server_ip`: Referee server address
- `--model`: Path to PPO model (.zip file)
- `--depth`: Minimax search depth (default: 4)
- `--seed`: Random seed for reproducibility

## Training

PPO training through self-play. The value function is extracted from the trained policy and used for minimax leaf evaluation.

```bash
pip install stable-baselines3

# 5M timesteps (recommended for competition)
python -m python_client.trainer \
    --algo ppo \
    --timesteps 5000000 \
    --save models/rl_value_net_5M.zip \
    --device cpu \
    --checkpoint-interval 250000 \
    --wandb
```

**PPO Configuration:**
- Policy: MlpPolicy (multi-layer perceptron)
- Learning rate: 3e-4, Batch size: 64, Steps: 2048
- Epochs: 10, Gamma: 0.99, GAE lambda: 0.95
- Action masking for legal moves only
- Self-play training with automatic checkpointing

## Key Components

- `agent.py`: Main entrypoint, socket communication, game loop
- `minimax_agent.py`: Alpha-beta search with iterative deepening
- `rl_value_wrapper.py`: Extracts value function from PPO policy
- `trainer.py`: PPO self-play training with stable-baselines3
- `tablut_env.py`: Gymnasium environment for RL training
- `heuristics.py`: Move ordering and fallback evaluation

## Troubleshooting

**Java not found:** Install with `brew install openjdk@17` (macOS) or `sudo apt-get install default-jdk` (Linux)

**Socket connection refused:** Ensure Java server is running on port 5800 (white) or 5801 (black)

**Timeout errors:** Reduce `--depth` parameter

**Import errors:** Activate virtual environment and install dependencies

## Competition Submission

**VM Requirements:** Linux Debian 64-bit, Python 3.10+, Java Runtime

**Submission command:**
```bash
python -m python_client.agent WHITE 60 <SERVER_IP> --model models/rl_value_net_5M.zip --depth 4
```

See [docs/DEPLOY.md](docs/DEPLOY.md) for detailed deployment instructions.

## Documentation

- [Training Guide](docs/model-training/TRAINING_GUIDE.md): PPO training details
- [Deployment](docs/DEPLOY.md): Competition environment setup
- [API Reference](docs/model-training/API.md): Code examples and interfaces

## References

- [Tablut Competition Repository](https://github.com/AGalassi/TablutCompetition)
- [Ashton Tablut Rules](https://aagenielsen.dk/ashton.php)
