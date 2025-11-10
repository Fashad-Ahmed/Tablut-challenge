#!/bin/bash
# Wrapper script to train Tablut value network
# Usage: ./train.sh [--episodes 1000] [--save models/value_net.pt]

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

# Run trainer
python -m python_client.trainer "$@"

