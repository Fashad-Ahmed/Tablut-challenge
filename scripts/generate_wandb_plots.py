#!/usr/bin/env python3
"""
Generate training visualization plots from WandB data.
This script fetches metrics from wandb and generates static plots for the README.
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import wandb
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required packages. Install with: pip install wandb matplotlib numpy")
    sys.exit(1)


def fetch_wandb_data(project_name="tablut-rl", entity=None, run_id=None):
    """Fetch training data from WandB."""
    api = wandb.Api()
    
    if run_id:
        # Fetch specific run
        if entity:
            run = api.run(f"{entity}/{project_name}/{run_id}")
        else:
            # Try to find run in default entity
            runs = api.runs(project_name, filters={"id": run_id})
            if not runs:
                raise ValueError(f"Run {run_id} not found in project {project_name}")
            run = runs[0]
    else:
        # Fetch latest run
        runs = api.runs(project_name, order="-created_at")
        if not runs:
            raise ValueError(f"No runs found in project {project_name}")
        run = runs[0]
        print(f"Using latest run: {run.id} ({run.name})")
    
    # Fetch history
    history = run.history()
    
    return run, history


def generate_plots(run, history, output_dir="docs/images"):
    """Generate visualization plots from wandb data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Episode Rewards
    if "episode_reward" in history.columns:
        plt.figure(figsize=(10, 6))
        rewards = history["episode_reward"].dropna()
        if len(rewards) > 0:
            plt.plot(rewards.index, rewards.values, alpha=0.6, linewidth=1)
            # Add moving average
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                moving_avg = rewards.rolling(window=window).mean()
                plt.plot(moving_avg.index, moving_avg.values, 
                        color='red', linewidth=2, label=f'Moving Average (window={window})')
            plt.xlabel("Episode")
            plt.ylabel("Episode Reward")
            plt.title("Training Progress: Episode Rewards")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / "wandb_episode_rewards.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_path / 'wandb_episode_rewards.png'}")
    
    # Plot 2: Episode Lengths
    if "episode_length" in history.columns:
        plt.figure(figsize=(10, 6))
        lengths = history["episode_length"].dropna()
        if len(lengths) > 0:
            plt.plot(lengths.index, lengths.values, alpha=0.6, linewidth=1, color='green')
            if len(lengths) > 10:
                window = min(50, len(lengths) // 10)
                moving_avg = lengths.rolling(window=window).mean()
                plt.plot(moving_avg.index, moving_avg.values, 
                        color='darkgreen', linewidth=2, label=f'Moving Average (window={window})')
            plt.xlabel("Episode")
            plt.ylabel("Episode Length")
            plt.title("Training Progress: Episode Lengths")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path / "wandb_episode_lengths.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: {output_path / 'wandb_episode_lengths.png'}")
    
    # Plot 3: Combined metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    if "episode_reward" in history.columns:
        rewards = history["episode_reward"].dropna()
        if len(rewards) > 0:
            ax1.plot(rewards.index, rewards.values, alpha=0.6, linewidth=1, label='Episode Reward')
            if len(rewards) > 10:
                window = min(50, len(rewards) // 10)
                moving_avg = rewards.rolling(window=window).mean()
                ax1.plot(moving_avg.index, moving_avg.values, 
                        color='red', linewidth=2, label=f'Moving Average')
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Episode Rewards Over Time")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
    
    if "episode_length" in history.columns:
        lengths = history["episode_length"].dropna()
        if len(lengths) > 0:
            ax2.plot(lengths.index, lengths.values, alpha=0.6, linewidth=1, 
                    color='green', label='Episode Length')
            if len(lengths) > 10:
                window = min(50, len(lengths) // 10)
                moving_avg = lengths.rolling(window=window).mean()
                ax2.plot(moving_avg.index, moving_avg.values, 
                        color='darkgreen', linewidth=2, label=f'Moving Average')
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Length")
            ax2.set_title("Episode Lengths Over Time")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "wandb_training_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path / 'wandb_training_metrics.png'}")
    
    # Print run summary
    print("\n" + "="*60)
    print(f"Run Summary: {run.name}")
    print("="*60)
    print(f"Run ID: {run.id}")
    print(f"Status: {run.state}")
    print(f"Created: {run.created_at}")
    print(f"Duration: {run.summary.get('_wandb', {}).get('runtime', 'N/A')}")
    if "episode_reward" in history.columns:
        rewards = history["episode_reward"].dropna()
        if len(rewards) > 0:
            print(f"\nEpisode Rewards:")
            print(f"  Mean: {rewards.mean():.2f}")
            print(f"  Max: {rewards.max():.2f}")
            print(f"  Min: {rewards.min():.2f}")
            print(f"  Final: {rewards.iloc[-1]:.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training visualization plots from WandB data"
    )
    parser.add_argument(
        "--project",
        default="tablut-rl",
        help="WandB project name (default: tablut-rl)"
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="WandB entity/username (optional)"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run ID to visualize (default: latest run)"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/images",
        help="Output directory for plots (default: docs/images)"
    )
    
    args = parser.parse_args()
    
    # Check if wandb is logged in
    if not wandb.api.api_key:
        print("Warning: WandB API key not found.")
        print("   Run 'wandb login' first, or set WANDB_API_KEY environment variable.")
        print("   Attempting to fetch data anyway...")
    
    try:
        print(f"Fetching data from WandB project: {args.project}")
        run, history = fetch_wandb_data(
            project_name=args.project,
            entity=args.entity,
            run_id=args.run_id
        )
        
        print(f"\nGenerating plots...")
        generate_plots(run, history, output_dir=args.output_dir)
        
        print(f"\n Success! Plots saved to {args.output_dir}/")
        print(f"\nTo add to README, use:")
        print(f"  ![Training Metrics]({args.output_dir}/wandb_training_metrics.png)")
        
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're logged in: wandb login")
        print("  2. Check project name is correct")
        print("  3. Verify run exists in the project")
        sys.exit(1)


if __name__ == "__main__":
    main()

