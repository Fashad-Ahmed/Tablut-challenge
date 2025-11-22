# Deployment Guide

## Local Development (macOS M1 Pro)

### Prerequisites
- Python 3.10+
- Virtual environment
- Java referee server running

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running Locally
```bash
# Start Java server (in separate terminal)
cd Tablut/Executables
java -jar Server.jar

# Run agent (in another terminal)
python -m python_client.agent WHITE 60 127.0.0.1
```

### Testing Against Java Server
```bash
# Terminal 1: Start server
cd Tablut/Executables
java -jar Server.jar

# Terminal 2: Run white player
python -m python_client.agent WHITE 60 127.0.0.1

# Terminal 3: Run black player (or use RandomPlayer.jar)
python -m python_client.agent BLACK 60 127.0.0.1
```

## Competition Environment (Linux Debian 64-bit)

### VM Setup
1. **Create VirtualBox VM**
   - OS: Linux Debian 64-bit
   - CPUs: 4
   - RAM: 8GB
   - Disk: 30GB

2. **Install Dependencies**
   ```bash
   sudo apt-get update
   sudo apt-get install python3.10 python3-pip python3-venv
   ```

3. **Copy Agent Files**
   ```bash
   # Copy entire project to VM
   scp -r Tablut-challenge/ user@vm:/home/user/
   ```

4. **Setup in VM**
   ```bash
   cd Tablut-challenge
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Model Deployment
1. **Train PPO model locally**
   ```bash
   pip install stable-baselines3
   python -m python_client.trainer \
       --algo ppo \
       --timesteps 5000000 \
       --save models/rl_value_net_5M.zip \
       --device cpu
   ```

2. **Copy model to VM**
   ```bash
   scp models/rl_value_net_5M.zip user@vm:/home/user/Tablut-challenge/models/
   ```

3. **Verify model loads**
   ```bash
   python -c "from python_client.rl_value_wrapper import RLValueWrapper; wrapper = RLValueWrapper('models/rl_value_net_5M.zip', device='cpu'); print('OK')"
   ```

### Competition Execution
```bash
# In VM, run agent
cd Tablut-challenge
source venv/bin/activate
python -m python_client.agent WHITE 60 <SERVER_IP> --model models/rl_value_net_5M.zip
```

## Submission Requirements

### Virtual Machine Image
- **Format**: VirtualBox `.ova` or `.vdi`
- **OS**: Linux Debian 64-bit
- **Size**: < 30GB (including all dependencies)
- **Contents**:
  - Python 3.10+
  - Virtual environment with dependencies
  - Agent code
  - Trained model (if using)
  - Java referee (for testing)

### Submission .txt File
Create `submission.txt` with:

```
Virtual Machine Link:
https://liveunibo-my.sharepoint.com/... (OneDrive link)

Execution Command:
python -m python_client.agent WHITE 60 <SERVER_IP> --model models/rl_value_net_5M.zip --depth 4

Optional Parameters:
--model: Path to PPO-trained value network model (default: heuristic only)
--depth: Minimax search depth (default: 4)
--seed: Random seed for reproducibility (optional)
--log-level: Logging level (default: INFO)
```

### Testing Before Submission
1. **Test locally** against Java server
2. **Test in VM** (same environment as competition)
3. **Verify time limits** (60s per move)
4. **Check disk usage** (avoid excessive logging)
5. **Test both colors** (WHITE and BLACK)

## Docker Deployment (Alternative)

### Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY python_client/ ./python_client/

# Copy PPO model
COPY models/rl_value_net_5M.zip ./models/

# Set entrypoint
ENTRYPOINT ["python", "-m", "python_client.agent"]
```

### Build and Run
```bash
docker build -t tablut-agent .
docker run tablut-agent WHITE 60 127.0.0.1 --model models/rl_value_net_5M.zip
```

## Performance Optimization

### CPU Optimization
```python
# Set CPU threads (in model_loader.py)
torch.set_num_threads(4)  # Match VM CPU count
```

### Memory Management
- Limit transposition table size
- Clear cache periodically
- Avoid storing full game history

### Time Management
- Reserve 5s buffer for move transmission
- Use iterative deepening
- Always return best move found

## Troubleshooting

### Connection Issues
- **Connection refused**: Server not running or wrong port
- **Timeout**: Network issues or server overload
- **Solution**: Check server status, verify IP/port

### Model Loading Issues
- **File not found**: Model path incorrect (ensure .zip extension for PPO models)
- **Device mismatch**: Model must be loaded on CPU for competition
- **Solution**: Use absolute paths, ensure CPU compatibility, verify model format

### Performance Issues
- **Slow moves**: Reduce search depth
- **High memory**: Limit transposition table
- **CPU usage**: Optimize model inference

## Monitoring

### Logging
```bash
# Enable debug logging
python -m python_client.agent WHITE 60 127.0.0.1 --log-level DEBUG
```

### Metrics
- Move computation time
- Nodes searched
- Transposition table hits
- Model inference time

## Security Considerations

- **No internet access**: All dependencies bundled
- **No external APIs**: Self-contained agent
- **Input validation**: Validate server messages
- **Error handling**: Graceful failure modes

