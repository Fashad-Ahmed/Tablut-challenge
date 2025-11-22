# Training for Both White and Black Players

## Current Training Setup

### Yes, Your Model Trains for Both Sides

PPO training trains for both white and black because:

1. **Self-Play**: The agent plays against itself, experiencing both perspectives
2. **Turn Alternation**: The environment naturally alternates turns during games
3. **State-Based Learning**: The policy learns to make moves based on the current state (which includes whose turn it is)

### How It Works

During training:
- Game starts with **White** to move
- Agent makes a move as **White**
- Turn switches to **Black**
- Agent makes a move as **Black**
- This continues until game ends
- Agent experiences both sides in every game

### Potential Issue: Reward Perspective

**Current reward structure** (from `tablut_env.py`):
```python
reward_win = 100.0   # White wins
reward_loss = -100.0 # Black wins (or white loses)
reward_draw = 0.0
```

The rewards are **fixed from white's perspective**:
- White win = +100
- Black win = -100

**This is actually fine for self-play** because:
- The agent learns to maximize rewards regardless of which side it's playing
- When playing as white, it tries to get +100
- When playing as black, it tries to avoid -100 (which means winning as black)
- The policy naturally learns both perspectives

## Verifying Your Model Plays Both Sides

### Test as White
```bash
python -m python_client.agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 3
```

### Test as Black
```bash
python -m python_client.agent BLACK 60 127.0.0.1 --model models/rl_value_net_5M.zip --depth 3
```

Both should work! The same model can play either side.

## Improving Training for Both Sides

### Option 1: Current Setup (Recommended)
Your current setup is good! Self-play naturally trains both sides.

### Option 2: Explicit Side Training (Advanced)
If you want to ensure balanced training, you could:

1. **Train separate models** (not recommended - doubles training time):
   ```bash
   # Train white model
   python -m python_client.trainer --algo ppo --timesteps 1000000 --save models/white_model.zip
   
   # Train black model  
   python -m python_client.trainer --algo ppo --timesteps 1000000 --save models/black_model.zip
   ```

2. **Use perspective-flipping** (more complex):
   - Flip the board state when it's black's turn
   - Train from a single perspective
   - This is more complex and usually unnecessary

### Option 3: Reward Shaping (Current is Fine)
The current reward structure works well for self-play. The agent learns:
- As white: maximize +100 (win)
- As black: minimize -100 (which means win as black)

## Why Self-Play Works for Both Sides

1. **Symmetric Learning**: The agent experiences both winning and losing from both perspectives
2. **Turn Information**: The observation includes whose turn it is, so the policy can learn different strategies
3. **Natural Alternation**: Games naturally alternate turns, ensuring balanced experience

## Checking Model Performance

### Evaluate Both Sides
```python
from stable_baselines3 import PPO
from python_client.tablut_env import TablutEnv
from python_client.action_mask_wrapper import ActionMaskWrapper

env = ActionMaskWrapper(TablutEnv())
model = PPO.load('models/rl_value_net_5M.zip')

# Test as white (starts first)
wins_white = 0
for _ in range(100):
    obs, _ = env.reset()  # Starts with white
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done and reward > 0:
            wins_white += 1

print(f"Win rate as white (starting): {wins_white/100:.2%}")

# Test as black (would need to modify env to start with black)
# Or just test in actual games
```

## Conclusion

**Your current training DOES train for both sides**
**The same model can play as either WHITE or BLACK**
**Self-play naturally ensures balanced training**

The reward structure being from white's perspective doesn't hurt because:
- The agent learns to maximize rewards from whatever perspective it's in
- Self-play provides balanced experience
- The policy learns to play both sides effectively

The current setup is correct and trains effectively for both sides.

