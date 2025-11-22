# Hyperparameter Tuning Guide

## Minimax Agent Parameters

### Search Depth (`max_depth`)

**Default**: 4

**Impact**:
- Higher depth = better moves but slower
- Lower depth = faster but weaker play

**Recommendations**:
- **Fast games** (< 10s per move): depth 2-3
- **Standard** (30-50s per move): depth 4-5
- **Slow games** (full 60s): depth 5-6

**Tuning**:
```python
agent = MinimaxAgent(
    value_fn=value_fn,
    max_depth=5,  # Adjust based on time budget
    time_limit=55.0
)
```

### Time Limit (`time_limit`)

**Default**: 55.0 seconds (leaves 5s buffer)

**Impact**:
- Too high = risk of timeout
- Too low = incomplete search

**Recommendations**:
- Always leave 5-10s buffer for move transmission
- Use iterative deepening to maximize search within limit

### Transposition Table

**Size**: 100,000 entries (default)

**Impact**:
- Larger = more cache hits but more memory
- Smaller = less memory but fewer hits

**Tuning**:
```python
from python_client.minimax_agent import TranspositionTable

table = TranspositionTable(max_size=50000)  # Reduce for memory-constrained environments
```

### Move Ordering

**Default**: Enabled

**Impact**:
- Enabled = better alpha-beta pruning
- Disabled = faster move generation but worse pruning

**Recommendation**: Always enable

## Model Parameters

### Architecture

**ValueNetwork** (default):
- 3 CNN layers + 3 FC layers
- ~50k parameters
- Good for spatial patterns

**SimpleValueNetwork**:
- 3 FC layers only
- ~20k parameters
- Faster inference, less capacity

**Selection**:
- Use `ValueNetwork` if time allows
- Use `SimpleValueNetwork` for faster inference

### Hidden Dimension

**Default**: 128

**Impact**:
- Larger = more capacity but slower
- Smaller = faster but less expressive

**Tuning**:
```python
model = ValueNetwork(hidden_dim=256)  # Increase capacity
```

## PPO Training Parameters

### Learning Rate

**Default**: 3e-4

**Impact**:
- Too high = unstable training
- Too low = slow convergence

**Recommendations**:
- Default 3e-4 is optimized for Tablut
- Reduce if training is unstable
- Monitor training curves for convergence

### Batch Size

**Default**: 64

**Impact**:
- Larger = more stable gradients but slower
- Smaller = faster but noisier

**Recommendations**:
- Default 64 is appropriate for CPU training
- Adjust based on available memory

### Training Timesteps

**Default**: 5000000 (5M) for competition

**Impact**:
- More timesteps = better model performance
- Diminishing returns after extended training

**Recommendations**:
- Use 5M+ timesteps for competitive performance
- Monitor evaluation metrics during training
- Use checkpointing for long training runs

## Heuristic Weights

### Material Balance

**Weight**: 1.0 (default)

**Tuning**:
```python
# In heuristics.py, evaluate_heuristic()
material = material_balance(state) * 1.0  # Adjust weight
```

### King Safety

**Weight**: 2.0 (default)

**Impact**:
- Higher = prioritize king safety
- Lower = prioritize material/position

**Tuning**:
```python
safety = king_safety_score(state) * 2.0  # Adjust weight
```

### Mobility

**Weight**: 0.5 (default)

**Impact**:
- Higher = prioritize move options
- Lower = prioritize static evaluation

## Performance Tuning

### CPU Threads

**Default**: 4 (matches VM)

**Tuning**:
```python
# In model_loader.py
torch.set_num_threads(4)  # Match available CPUs
```

### Device Selection

**Development** (M1 Mac):
```python
# PPO training uses CPU for competition compatibility
# MPS/CUDA can be used for development but models must work on CPU
```

**Competition** (Linux VM):
```python
# PPO models must be trained and loaded on CPU
wrapper = RLValueWrapper(model_path, device="cpu")
```

## Time Budget Allocation

### Recommended Split (60s total)

1. **Move computation**: 50-55s
2. **Move transmission**: 3-5s
3. **Buffer**: 2-5s

### Iterative Deepening Strategy

```python
# Search depths: 1, 2, 3, 4, 5, ...
# Stop when time limit reached
# Always return best move from deepest completed search
```

## Memory Optimization

### Transposition Table

**Size limit**: Based on available RAM
- 8GB VM: ~100k entries
- 16GB: ~500k entries

**Eviction**: Clear when > 90% full

### State Caching

**Minimal**: Only current state
- Don't store full game history
- Clear cache between games

## Model Size vs Speed

### PPO Model Architecture

The PPO model uses a multi-layer perceptron (MlpPolicy) architecture:
- Input: Flattened 9x9x5 observation (one-hot encoded board)
- Hidden layers: Standard MLP architecture
- Output: Action probabilities and state value

**Inference Performance**:
- CPU inference: ~1-5ms per evaluation
- Optimized for competition environment
- Value function extracted from policy for minimax integration

## Hyperparameter Search

### Grid Search Example

```python
depths = [3, 4, 5]
time_limits = [50.0, 55.0, 58.0]

for depth in depths:
    for time_limit in time_limits:
        # Test configuration
        agent = MinimaxAgent(
            value_fn=value_fn,
            max_depth=depth,
            time_limit=time_limit
        )
        # Evaluate performance
```

### Random Search

```python
import random

for _ in range(10):
    depth = random.randint(3, 6)
    time_limit = random.uniform(50.0, 58.0)
    # Test configuration
```

## Monitoring Metrics

### During Training
- Loss (MSE, MAE)
- Validation accuracy
- Training time per episode

### During Gameplay
- Move computation time
- Nodes searched
- Transposition table hit rate
- Model inference time

### Logging

```python
# Enable debug logging
python -m python_client.agent WHITE 60 127.0.0.1 --log-level DEBUG
```

## Best Practices

1. **Start conservative**: Use defaults first
2. **Measure baseline**: Establish performance baseline
3. **Tune one at a time**: Change one parameter at a time
4. **Validate changes**: Test against baseline
5. **Document results**: Keep track of what works

## Example Configurations

### Fast Agent (10s per move)
```python
agent = MinimaxAgent(
    value_fn=value_fn,
    max_depth=2,
    time_limit=8.0,
    use_transposition=True,
    move_ordering=True
)
```

### Balanced Agent (30s per move)
```python
agent = MinimaxAgent(
    value_fn=value_fn,
    max_depth=4,
    time_limit=28.0,
    use_transposition=True,
    move_ordering=True
)
```

### Strong Agent (60s per move)
```python
agent = MinimaxAgent(
    value_fn=value_fn,
    max_depth=6,
    time_limit=58.0,
    use_transposition=True,
    move_ordering=True
)
```

