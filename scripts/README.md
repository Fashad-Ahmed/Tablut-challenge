# Scripts Directory

Helper scripts for the Tablut Challenge agent.

## Available Scripts

### `check_prerequisites.sh`
Check if all prerequisites are installed (Python, Java, dependencies).

**Usage:**
```bash
# From project root
./scripts/check_prerequisites.sh

# NOT: run check_prerequisites.sh
# NOT: check_prerequisites.sh
```

**Note:** Always use `./` prefix to run scripts in the current directory.

### `run_agent.sh`
Run the Tablut agent with proper environment setup.

**Usage:**
```bash
./scripts/run_agent.sh WHITE 60 127.0.0.1
```

### `train.sh`
Train the value network model.

**Usage:**
```bash
./scripts/train.sh --episodes 500 --save models/value_net.pt
```

### `eval.sh`
Evaluate a trained model.

**Usage:**
```bash
./scripts/eval.sh models/value_net.pt
```

## Common Issues

### "command not found: run"
- **Problem:** You typed `run script.sh` instead of `./script.sh`
- **Solution:** Always use `./` prefix: `./scripts/script.sh`

### "Permission denied"
- **Problem:** Script is not executable
- **Solution:** `chmod +x scripts/script.sh`

### "No such file or directory"
- **Problem:** You're not in the project root directory
- **Solution:** `cd Tablut-challenge` first

## Quick Reference

```bash
# Always run from project root
cd Tablut-challenge

# Check prerequisites
./scripts/check_prerequisites.sh

# Run agent
./scripts/run_agent.sh WHITE 60 127.0.0.1

# Train model
./scripts/train.sh --episodes 100
```

