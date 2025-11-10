#!/bin/bash
# Wrapper script to run Tablut agent
# Usage: ./run_agent.sh WHITE 60 127.0.0.1 [--model models/value_net.pt]

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

# Run agent
python -m python_client.agent "$@"

