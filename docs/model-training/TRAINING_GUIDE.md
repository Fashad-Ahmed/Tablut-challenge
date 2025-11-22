# Tablut PPO Training Guide

## WandB Account Integration

### Current Setup (Offline by Default)
By default, training saves logs **locally** (offline mode) to avoid requiring WandB login. This means:
-  Logs are saved locally in `wandb/` directory
- **NOT automatically synced to your WandB account**
-  You can manually sync later with `wandb sync wandb/offline-run-*`

### To Log to Your WandB Account (Online Mode)

**Option 1: Set environment variable**
```bash
export WANDB_MODE=online
python -m python_client.trainer --algo ppo --timesteps 1000000 --wandb --save models/rl_value_net.zip
```

**Option 2: One-liner**
```bash
WANDB_MODE=online python -m python_client.trainer --algo ppo --timesteps 1000000 --wandb --save models/rl_value_net.zip
```

**First time setup:**
1. Install WandB: `pip install wandb`
2. Login: `wandb login` (you'll get an API key from https://wandb.ai/authorize)
3. Run training with `WANDB_MODE=online`

**What gets logged to your account:**
- All hyperparameters (learning rate, batch size, etc.)
- Episode rewards and lengths
- Training metrics (value loss, policy loss)
- Model checkpoints (if configured)
- System info (device, Python version)

**To disable WandB completely:**
```bash
export WANDB_MODE=disabled
# or just don't use --wandb flag
python -m python_client.trainer --algo ppo --timesteps 1000000 --save models/rl_value_net.zip
```

## Training Algorithm

The agent uses **Proximal Policy Optimization (PPO)** for training value networks. PPO is selected for its sample efficiency, training stability, and effective handling of sparse rewards in game environments.

## Why Your Model Needs More Training

### Current Issues
1. **Losing/Drawing**: Your model is under-trained
2. **100k timesteps is minimal**: Tablut is a complex game requiring millions of timesteps
3. **Self-play learning**: The agent needs to play many games to learn

### Recommended Training Steps

#### 1. **Initial Training (1M timesteps)**
```bash
python -m python_client.trainer \
    --algo ppo \
    --timesteps 1000000 \
    --save models/rl_value_net_v1.zip \
    --wandb \
    --checkpoint-interval 50000
```

#### 2. **Extended Training (5M timesteps)**
```bash
python -m python_client.trainer \
    --algo ppo \
    --timesteps 5000000 \
    --save models/rl_value_net_5M.zip \
    --wandb \
    --checkpoint-interval 250000 \
    --device cpu
```

#### 3. **Resume from Checkpoint**
```bash
# Load a checkpoint and continue training
python -c "
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from python_client.tablut_env import TablutEnv
from python_client.action_mask_wrapper import ActionMaskWrapper

def make_env():
    env = TablutEnv()
    env = ActionMaskWrapper(env)
    return env

env = DummyVecEnv([make_env])
model = PPO.load('models/rl_value_net_checkpoints/ppo_checkpoint_500000_steps.zip', env=env)

# Continue training
model.learn(total_timesteps=2000000, reset_num_timesteps=False)
model.save('models/rl_value_net_v2.zip')
"
```

## Hyperparameters Now Saved to WandB

**Fixed**: All hyperparameters are now logged to Weights & Biases when using `--wandb`:

### PPO Hyperparameters (logged to WandB):
- `learning_rate`: 3e-4
- `n_steps`: 2048
- `batch_size`: 64
- `n_epochs`: 10
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_range`: 0.2
- `ent_coef`: 0.01
- `vf_coef`: 0.5
- `max_grad_norm`: 0.5

### Additional PPO Hyperparameters:
- `max_grad_norm`: 0.5
- `gae_lambda`: 0.95
- `clip_range`: 0.2

## Improving Training Performance

### 1. **Use WandB to Monitor Training**

**Offline mode (default - saves locally, no account needed):**
```bash
python -m python_client.trainer --algo ppo --timesteps 1000000 --wandb --save models/rl_value_net.zip
```

**Online mode (syncs to your WandB account):**
```bash
# First: wandb login (one-time setup)
WANDB_MODE=online python -m python_client.trainer --algo ppo --timesteps 1000000 --wandb --save models/rl_value_net.zip
```

**View offline logs later:**
```bash
# Sync offline logs to your account
wandb sync wandb/offline-run-*
```

### 2. **Check Training Metrics**
- **Episode reward**: Should increase over time
- **Episode length**: Should stabilize (not too short/long)
- **Value loss**: Should decrease
- **Policy loss**: Should stabilize

### 3. **Hyperparameters for Tablut**

The default PPO hyperparameters are optimized for Tablut:
- Learning rate: 3e-4 (balanced for stable convergence)
- Steps per update: 2048 (good balance of stability and efficiency)
- Batch size: 64 (appropriate for CPU training)
- Epochs per update: 10 (sufficient for policy improvement)
- Entropy coefficient: 0.01 (maintains exploration)

### 4. **Training Tips**

1. **Extended training**: Use 5M+ timesteps for competitive performance

2. **Monitor evaluation**: Check `models/rl_value_net_5M_best/` for best model during training

3. **Use checkpoints**: Resume training from checkpoints if interrupted

4. **CPU training**: Use `--device cpu` for competition environment compatibility

5. **Evaluation**: Regularly test trained models using evaluation scripts

## Testing Your Model

### Quick Test
```bash
# Test against random player
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net.zip --depth 3
```

### Evaluation Script
```bash
# Use the provided evaluation script
python scripts/evaluate_model.py models/rl_value_net_5M.zip --games 100
```

## Troubleshooting

### Model Losing/Drawing
- **Not enough training**: Increase `--timesteps` to 5M+
- **Poor hyperparameters**: Try different learning rates
- **Check evaluation metrics**: Use WandB to see if training is progressing

### WandB Not Logging
- Check `WANDB_MODE` environment variable
- Verify `wandb` is installed: `pip install wandb`
- Check logs for "WandB logging enabled" message

### Slow Training
- Training is CPU-based for competition compatibility
- Reduce `n_steps` for faster updates (may reduce stability)
- Extended training (5M+ timesteps) is recommended for competitive performance

## Next Steps

1. **Extended training**: 5M+ timesteps for competitive performance
2. **Monitor with WandB**: Track training progress and metrics
3. **Evaluate regularly**: Use evaluation scripts to test model performance
4. **Test integration**: Verify trained models work correctly in minimax agent
5. **Use best model**: Check `models/rl_value_net_5M_best/` for best checkpoint

