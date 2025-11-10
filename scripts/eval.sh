#!/bin/bash
# Wrapper script to evaluate model performance
# Usage: ./eval.sh models/value_net.pt

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtualenv if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Change to project root
cd "$PROJECT_ROOT"

# TODO: Implement evaluation script
# For now, just test model loading
python -c "
from python_client.model_loader import ModelLoader
from python_client.tablut_env import TablutEnv
import sys

if len(sys.argv) < 2:
    print('Usage: eval.sh <model_path>')
    sys.exit(1)

model_path = sys.argv[1]
loader = ModelLoader(model_path, device='cpu')
env = TablutEnv()
env.reset()

value = loader.evaluate(env.state)
print(f'Model loaded successfully. Initial state value: {value:.4f}')
" "$@"

