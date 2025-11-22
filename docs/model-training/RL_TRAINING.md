# Reinforcement Learning Training Guide

This document describes the PPO implementation using stable-baselines3.

## Overview

The agent uses Proximal Policy Optimization (PPO) for training value networks through self-play. The trained policy's value function is extracted and integrated into minimax search for leaf node evaluation.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Training Phase (trainer.py)                            │
├─────────────────────────────────────────────────────────┤
│  1. TablutEnv (Gymnasium environment)                   │
│  2. ActionMaskWrapper (filters illegal moves)           │
│  3. Monitor (tracks statistics)                        │
│  4. DummyVecEnv (vectorization)                         │
│  5. PPO (stable-baselines3)                            │
│  6. RLValueExtractor (extracts value function)         │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Inference Phase (agent.py)                            │
├─────────────────────────────────────────────────────────┤
│  1. RLValueWrapper (loads SB3 model)                   │
│  2. Extracts value network from policy                │
│  3. Provides evaluate(state) → float                  │
│  4. Used by MinimaxAgent for leaf evaluation          │
└─────────────────────────────────────────────────────────┘
```

## Training Workflow

### 1. Environment Setup

The `TablutEnv` provides:
- **Observation space**: `Box(9, 9, 5)` - one-hot encoded board
- **Action space**: `MultiDiscrete([81, 81])` - from/to positions
- **Rewards**: Win (+100), Loss (-100), Draw (0), Step penalty (-0.1)

### 2. Action Masking

`ActionMaskWrapper` filters illegal moves:
- Prevents agent from attempting invalid moves
- Significantly improves training efficiency
- Provides action mask in info dict

### 3. RL Training

**PPO Configuration:**
```python
PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
```

### 4. Value Function Extraction

After training, the value network is extracted from the PPO policy. The PPO policy's forward method provides both action probabilities and state values. The value function is extracted and wrapped in `RLValueWrapper` for use in minimax search.

## Usage Examples

### Basic Training

```bash
# Train with PPO (500k timesteps)
python -m python_client.trainer \
    --algo ppo \
    --timesteps 500000 \
    --save models/rl_value_net.zip \
    --device cpu
```

### Extended Training

```bash
# Extended training (5M timesteps) with WandB logging
python -m python_client.trainer \
    --algo ppo \
    --timesteps 5000000 \
    --save models/rl_value_net_5M.zip \
    --device cpu \
    --wandb \
    --checkpoint-interval 250000 \
    --seed 42
```

### Using Trained Model

```bash
# Load RL model in agent
python -m python_client.agent \
    WHITE \
    60 \
    127.0.0.1 \
    --model models/rl_value_net.zip \
    --depth 5
```

## Training Features

### Self-Play
The agent learns by playing against itself, discovering strategies through exploration.

### Action Masking
Only legal moves are considered, preventing wasted training on invalid actions.

### Checkpointing
Models are saved periodically:
- Checkpoints: `{model_path}_checkpoints/`
- Best model: `{model_path}_best/`
- Logs: `{model_path}_logs/`

### Evaluation Callbacks
Best model is automatically saved based on evaluation performance.

### Logging
- TensorBoard: `tensorboard --logdir tensorboard_logs/`
- WandB: `--wandb` flag (requires `pip install wandb`)

### Device Support
- CPU: Default, used for competition environment compatibility
- MPS: Apple Silicon (M1/M2) acceleration (development only)
- CUDA: NVIDIA GPU acceleration (development only)

## Value Function Integration

The RL-trained value function is seamlessly integrated into minimax:

```python
# In agent.py
if model_path.endswith('.zip'):
    from .rl_value_wrapper import RLValueWrapper
    rl_wrapper = RLValueWrapper(model_path, device="cpu")
    self.value_fn = lambda state: rl_wrapper.evaluate(state)
```

The value function provides:
- **Input**: `GameState` object
- **Output**: Scalar value (white's perspective)
- **Usage**: Leaf node evaluation in minimax search

## Hyperparameter Tuning

### PPO Hyperparameters

| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `learning_rate` | 3e-4 | Learning rate | Lower for stable training |
| `n_steps` | 2048 | Steps per update | Higher = more stable |
| `batch_size` | 64 | Batch size | Adjust based on memory |
| `n_epochs` | 10 | Epochs per update | Higher = more learning |
| `gamma` | 0.99 | Discount factor | 0.99 for long-term planning |
| `gae_lambda` | 0.95 | GAE lambda parameter | Controls bias-variance tradeoff |
| `clip_range` | 0.2 | PPO clip range | Prevents large policy updates |
| `ent_coef` | 0.01 | Entropy coefficient | Higher = more exploration |
| `vf_coef` | 0.5 | Value function coefficient | Balances policy and value learning |
| `max_grad_norm` | 0.5 | Gradient clipping | Prevents gradient explosion |

## Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir tensorboard_logs/

# View in browser: http://localhost:6006
```

### WandB

```bash
# Enable WandB logging
python -m python_client.trainer --algo ppo --wandb ...

# View dashboard: https://wandb.ai
```

## Troubleshooting

### Issue: "stable-baselines3 not installed"
```bash
pip install stable-baselines3
```

### Issue: "CUDA out of memory"
- Reduce `batch_size` or `n_steps`
- Use CPU: `--device cpu`

### Issue: "Training is slow"
- Training is CPU-based for competition compatibility
- Reduce `n_steps` for faster updates (may reduce stability)
- Consider longer training runs for better performance

### Issue: "Model not improving"
- Increase training timesteps
- Adjust learning rate
- Check reward shaping in `tablut_env.py`
- Verify action masking is working

## Best Practices

1. **Extended Training**: Use 5M+ timesteps for competitive performance
2. **Monitor Progress**: Use TensorBoard or WandB to track training metrics
3. **Save Checkpoints**: Enable checkpointing for long training runs
4. **Evaluate Regularly**: Test trained models against baseline opponents
5. **CPU Training**: Use CPU device for competition environment compatibility
6. **Action Masking**: Always enabled to ensure only legal moves are considered
7. **Test Integration**: Verify trained model works correctly in minimax agent

## Next Steps

- Experiment with hyperparameter tuning based on training curves
- Extend training runs to 5M+ timesteps for improved performance
- Fine-tune reward shaping in the environment
- Evaluate model performance using evaluation scripts
- Test trained models in actual game scenarios

