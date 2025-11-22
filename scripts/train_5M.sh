#!/bin/bash
# Train Tablut RL model for 5M timesteps with full WandB logging

set -e

cd /Users/fashad/fashad/projects/Tablut-challenge
source venv/bin/activate

echo "Starting 5M timestep training with WandB..."

# Train
WANDB_MODE=online python -m python_client.trainer \
    --algo ppo \
    --timesteps 5000000 \
    --save models/rl_value_net_5M.zip \
    --wandb \
    --checkpoint-interval 250000 \
    --device cpu

echo "Training complete!"

# Upload models to WandB (with error handling)
echo "Uploading models to WandB..."
python -c "
import wandb
import os
from pathlib import Path

try:
    # Get latest run
    api = wandb.Api()
    runs = api.runs('tablut-rl', order='-created_at')
    if runs:
        run_id = runs[0].id
        wandb.init(project='tablut-rl', id=run_id, resume='must')
        
        # Upload final model
        if os.path.exists('models/rl_value_net_5M.zip'):
            try:
                artifact = wandb.Artifact('rl_model_5M_final', type='model')
                artifact.add_file('models/rl_value_net_5M.zip')
                wandb.log_artifact(artifact)
                print('Uploaded final model to WandB')
            except Exception as e:
                print(f'Could not upload final model: {e}')
        
        # Upload best model (check if exists)
        best_model = 'models/rl_value_net_5M_best/best_model.zip'
        if os.path.exists(best_model):
            try:
                artifact = wandb.Artifact('rl_model_5M_best', type='model')
                artifact.add_file(best_model)
                wandb.log_artifact(artifact)
                print('Uploaded best model to WandB')
            except Exception as e:
                print(f'Could not upload best model: {e}')
        else:
            print('Best model not found (may not have been saved)')
        
        wandb.finish()
    else:
        print('No WandB run found')
except Exception as e:
    print(f'WandB upload failed (non-critical): {e}')
    print('   Models are saved locally. You can upload manually later.')
"

echo "All done! Check your WandB dashboard."