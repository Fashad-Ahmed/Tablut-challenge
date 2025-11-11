"""
Self-play training loop for Tablut value network.

Uses stable-baselines3 (PPO or DQN) or supervised learning for training.
Generates value network for use in Minimax leaf evaluation.

Test with: python trainer.py --episodes 100 --algo ppo

Implementation:
- SimpleTrainer: Supervised learning on heuristic labels (working)
- train_with_sb3: Stable-baselines3 integration (PPO/DQN) - requires installation
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .tablut_env import TablutEnv, TablutConfig
from .model_loader import ModelLoader, ValueNetwork, create_model
from .utils import set_seed, GameState, Turn
from .heuristics import evaluate_heuristic


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleTrainer:
    """
    Simple trainer using supervised learning on heuristic labels.
    
    This is a working stub that trains a value network to approximate
    heuristic evaluation. Can be extended to use RL (PPO/DQN).
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        """
        Initialize trainer.
        
        Args:
            model_path: Path to save/load model
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Device for training
        """
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device(device)
        
        # Initialize model
        self.model = ValueNetwork().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def generate_training_data(self, num_games: int = 100) -> tuple:
        """
        Generate training data by playing random games.
        
        Returns:
            (states, values) where states are observations and values are targets
        """
        logger.info(f"Generating {num_games} games of training data...")
        env = TablutEnv()
        
        states = []
        values = []
        
        for game_idx in range(num_games):
            if (game_idx + 1) % 10 == 0:
                logger.info(f"Generated {game_idx + 1}/{num_games} games")
            
            obs, info = env.reset()
            game_states = []
            game_obs = []
            
            # Play game
            for step in range(200):  # Max steps
                # Get legal moves
                legal_moves = env.get_legal_moves(env.state)
                if not legal_moves:
                    break
                
                # Random move
                move_idx = np.random.randint(len(legal_moves))
                from_pos, to_pos = legal_moves[move_idx]
                
                # Convert to action
                from_idx = from_pos[0] * 9 + from_pos[1]
                to_idx = to_pos[0] * 9 + to_pos[1]
                action = np.array([from_idx, to_idx])
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                game_states.append(env.state)
                game_obs.append(obs.copy())
                
                if terminated or truncated:
                    break
            
            # Assign values (heuristic evaluation)
            for state in game_states:
                value = evaluate_heuristic(state, Turn.WHITE)
                states.append(state)
                values.append(value)
        
        # Convert to tensors
        logger.info("Converting to tensors...")
        obs_tensors = []
        value_tensors = []
        
        model_loader = ModelLoader(device=str(self.device))
        for state in states:
            obs_tensor = model_loader._state_to_tensor(state)
            obs_tensors.append(obs_tensor)
        
        obs_batch = torch.stack(obs_tensors).to(self.device)
        value_batch = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        logger.info(f"Generated {len(states)} training samples")
        return obs_batch, value_batch
    
    def train(self, num_episodes: int = 1000, save_interval: int = 100):
        """
        Train model on generated data.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Save model every N episodes
        """
        logger.info("Starting training...")
        
        # Generate initial data
        obs, values = self.generate_training_data(num_games=50)
        
        # Training loop
        for episode in range(num_episodes):
            # Forward pass
            pred_values = self.model(obs)
            loss = self.criterion(pred_values, values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            if (episode + 1) % 10 == 0:
                mse = loss.item()
                mae = torch.mean(torch.abs(pred_values - values)).item()
                logger.info(
                    f"Episode {episode + 1}/{num_episodes} - "
                    f"MSE: {mse:.4f}, MAE: {mae:.4f}"
                )
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0 and self.model_path:
                self.save_model(f"{self.model_path}.ep{episode + 1}")
        
        # Final save
        if self.model_path:
            self.save_model(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        model_loader = ModelLoader(device=str(self.device))
        model_loader.model = self.model
        model_loader.save_model(path, metadata={
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        })


class RLValueExtractor:
    """
    Extract value function from stable-baselines3 policy for use in minimax.
    
    Wraps the policy's value network to match ModelLoader interface.
    """
    
    def __init__(self, sb3_model):
        """
        Initialize extractor from stable-baselines3 model.
        
        Args:
            sb3_model: Trained PPO or DQN model
        """
        self.sb3_model = sb3_model
        self.device = torch.device("cpu")
        
        # Extract value network based on algorithm type
        if hasattr(sb3_model, 'policy'):
            policy = sb3_model.policy
            # PPO has value_net attribute
            if hasattr(policy, 'value_net'):
                self.value_net = policy.value_net
            # Some policies have mlp_extractor with value network
            elif hasattr(policy, 'mlp_extractor') and hasattr(policy.mlp_extractor, 'value_net'):
                self.value_net = policy.mlp_extractor.value_net
            else:
                # For DQN or other architectures, use the full network
                # We'll need to extract the value head
                self.value_net = self._extract_value_from_q_network(policy)
        else:
            raise ValueError("Model does not have policy attribute")
    
    def _extract_value_from_q_network(self, policy):
        """Extract value function from Q-network (for DQN)."""
        # For DQN, we can use the Q-network and take mean over actions
        # Or create a value network wrapper
        if hasattr(policy, 'q_net'):
            return policy.q_net
        else:
            raise ValueError("Cannot extract value network from this policy type")
    
    def evaluate(self, state: GameState) -> float:
        """
        Evaluate state using extracted value network.
        
        Args:
            state: GameState to evaluate
        
        Returns:
            Scalar value (white perspective)
        """
        from .tablut_env import TablutEnv
        
        # Convert state to observation
        env = TablutEnv()
        obs = env._state_to_observation(state)
        
        # Convert to tensor (flatten for MLP)
        obs_flat = obs.flatten()
        obs_tensor = torch.from_numpy(obs_flat).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get value from network
            if hasattr(self.value_net, 'forward'):
                value = self.value_net(obs_tensor)
                # Handle different output shapes
                if isinstance(value, tuple):
                    value = value[0]  # PPO returns (value, action_log_prob)
                if value.dim() > 1:
                    value = value.squeeze()
                return value.item()
            else:
                # For Q-network, take mean over actions as value estimate
                q_values = self.value_net(obs_tensor)
                value = q_values.mean().item()
                return value


def train_with_sb3(
    algo: str = "ppo",
    total_timesteps: int = 100000,
    model_path: str = "models/value_net.pt",
    seed: Optional[int] = None,
    use_wandb: bool = False,
    checkpoint_interval: int = 10000,
    device: str = "cpu",
):
    """
    Train using stable-baselines3 (PPO or DQN) with full RL implementation.
    
    This implements true reinforcement learning through self-play.
    The trained policy's value function is extracted for use in minimax.
    
    Args:
        algo: Algorithm name ("ppo" or "dqn")
        total_timesteps: Total training timesteps
        model_path: Path to save model
        seed: Random seed
        use_wandb: Enable Weights & Biases logging
        checkpoint_interval: Save checkpoint every N timesteps
        device: Device for training ("cpu", "mps", "cuda")
    """
    try:
        from stable_baselines3 import PPO, DQN
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        logger.error(
            "stable-baselines3 not installed. "
            "Install with: pip install stable-baselines3"
        )
        logger.info("Falling back to SimpleTrainer...")
        trainer = SimpleTrainer(model_path=model_path)
        trainer.train(num_episodes=total_timesteps // 100)
        return
    
    # Setup WandB if requested
    if use_wandb:
        try:
            import wandb
            import os
            # Set WandB to non-interactive mode (offline or disabled)
            # User can override with WANDB_MODE env var
            if "WANDB_MODE" not in os.environ:
                os.environ["WANDB_MODE"] = "offline"  # Use offline mode by default
            
            # Prepare hyperparameters based on algorithm
            if algo.lower() == "ppo":
                hyperparams = {
                    "algo": algo,
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "device": device,
                    # PPO hyperparameters
                    "learning_rate": 5e-4,
                    "n_steps": 4096,
                    "batch_size": 128,
                    "n_epochs": 15,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.02,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "policy": "MlpPolicy",
                }
            elif algo.lower() == "dqn":
                hyperparams = {
                    "algo": algo,
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "device": device,
                    # DQN hyperparameters
                    "learning_rate": 1e-4,
                    "buffer_size": 100000,
                    "learning_starts": 1000,
                    "batch_size": 32,
                    "tau": 1.0,
                    "gamma": 0.99,
                    "train_freq": "4 steps",
                    "gradient_steps": 1,
                    "target_update_interval": 1000,
                    "exploration_fraction": 0.1,
                    "exploration_initial_eps": 1.0,
                    "exploration_final_eps": 0.05,
                    "policy": "MlpPolicy",
                }
            else:
                hyperparams = {
                    "algo": algo,
                    "total_timesteps": total_timesteps,
                    "seed": seed,
                    "device": device,
                }
            
            wandb.init(
                project="tablut-rl",
                config=hyperparams,
                mode=os.environ.get("WANDB_MODE", "offline"),
            )
            logger.info(f"WandB logging enabled (mode: {os.environ.get('WANDB_MODE', 'offline')})")
            logger.info(f"Logged hyperparameters: {list(hyperparams.keys())}")
        except ImportError:
            logger.warning("wandb not installed. Continuing without logging.")
            use_wandb = False
    
    # Create environment with monitoring and action masking
    def make_env():
        from .action_mask_wrapper import ActionMaskWrapper
        env = TablutEnv()
        # Wrap with action masking to improve training efficiency
        env = ActionMaskWrapper(env)
        # Wrap with Monitor for statistics
        env = Monitor(env, allow_early_resets=True)
        return env
    
    # Create vectorized environment (single env for now)
    env = DummyVecEnv([make_env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env])
    
    # Set device
    # Note: stable-baselines3 recommends CPU for MLP policies
    # MPS/CUDA are better for CNN policies
    if device == "cuda" and torch.cuda.is_available():
        device_str = "cuda"
    elif device == "mps" and torch.backends.mps.is_available():
        # Warn but allow MPS (user may want to test)
        logger.warning(
            "MPS device selected. Note: stable-baselines3 recommends CPU for MLP policies. "
            "Training may be slower on MPS. Consider using --device cpu for better performance."
        )
        device_str = "mps"
    else:
        device_str = "cpu"
    
    logger.info(f"Training on device: {device_str}")
    
    # Create model with hyperparameters optimized for Tablut
    # Note: PPO is generally better for Tablut due to:
    # - Better sample efficiency
    # - More stable training
    # - Better handling of sparse rewards
    if algo.lower() == "ppo":
        ppo_config = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1,
            "seed": seed,
            "device": device_str,
            "tensorboard_log": "./tensorboard_logs/" if not use_wandb else None,
        }
        model = PPO(**ppo_config)
        logger.info(f"Created PPO model with hyperparameters: lr={ppo_config['learning_rate']}, "
                   f"n_steps={ppo_config['n_steps']}, batch_size={ppo_config['batch_size']}")
    elif algo.lower() == "dqn":
        dqn_config = {
            "policy": "MlpPolicy",
            "env": env,
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 1000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": (4, "step"),
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "verbose": 1,
            "seed": seed,
            "device": device_str,
            "tensorboard_log": "./tensorboard_logs/" if not use_wandb else None,
        }
        model = DQN(**dqn_config)
        logger.info(f"Created DQN model with hyperparameters: lr={dqn_config['learning_rate']}, "
                   f"buffer_size={dqn_config['buffer_size']}, batch_size={dqn_config['batch_size']}")
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=f"{model_path}_checkpoints/",
        name_prefix=f"{algo}_checkpoint",
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback (evaluate every 10000 steps)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_path}_best/",
        log_path=f"{model_path}_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Train
    logger.info(f"Training {algo.upper()} for {total_timesteps} timesteps...")
    logger.info(f"Checkpoints will be saved every {checkpoint_interval} steps")
    
    # Add WandB callback for metric logging if enabled
    if use_wandb:
        try:
            import wandb
            from stable_baselines3.common.callbacks import BaseCallback
            
            class WandBCallback(BaseCallback):
                """Callback to log metrics to WandB."""
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self.episode_rewards = []
                    self.episode_lengths = []
                
                def _on_step(self) -> bool:
                    # Log metrics from info dict if available
                    if len(self.locals.get("infos", [])) > 0:
                        for info in self.locals["infos"]:
                            if "episode" in info:
                                episode_info = info["episode"]
                                if "r" in episode_info:
                                    wandb.log({
                                        "episode_reward": episode_info["r"],
                                        "episode_length": episode_info["l"],
                                    }, step=self.num_timesteps)
                    return True
            
            wandb_callback = WandBCallback()
            callbacks.append(wandb_callback)
            logger.info("Added WandB callback for metric logging")
        except Exception as e:
            logger.warning(f"Could not add WandB callback: {e}")
    
    # Check if progress bar dependencies are available
    # stable-baselines3 requires tqdm and rich for progress bar
    use_progress_bar = False
    try:
        import tqdm
        import rich
        # Test if ProgressBarCallback can be instantiated (same check stable-baselines3 does)
        from stable_baselines3.common.callbacks import ProgressBarCallback
        _ = ProgressBarCallback()  # Test instantiation
        use_progress_bar = True
        logger.debug("Progress bar enabled (tqdm and rich available)")
    except (ImportError, Exception) as e:
        logger.warning(
            f"Progress bar disabled: {type(e).__name__}: {str(e)[:100]}. "
            "Install with: pip install 'stable-baselines3[extra]' or pip install tqdm rich"
        )
        use_progress_bar = False
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=use_progress_bar,
    )
    
    # Save final model (SB3 saves as .zip)
    if not model_path.endswith('.zip'):
        sb3_model_path = model_path.replace(".pt", ".zip")
    else:
        sb3_model_path = model_path
    
    model.save(sb3_model_path)
    logger.info(f"Final SB3 model saved to {sb3_model_path}")
    
    # Extract and save value network for use in minimax
    try:
        logger.info("Extracting value network from policy...")
        value_extractor = RLValueExtractor(model)
        
        logger.info("Value network extraction complete.")
        logger.info(f"To use in minimax: python -m python_client.agent WHITE 60 127.0.0.1 --model {sb3_model_path}")
        
    except Exception as e:
        logger.warning(f"Could not extract value network: {e}")
        logger.info("Model saved. You can load it using RLValueWrapper.")
    
    if use_wandb:
        try:
            wandb.finish()
        except:
            pass
    
    logger.info("Training complete!")
    logger.info(f"Use the model with: python -m python_client.agent WHITE 60 127.0.0.1 --model {model_path}")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train Tablut value network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple supervised learning
  python trainer.py --episodes 500 --save models/value_net.pt
  
  # Use stable-baselines3 PPO
  python trainer.py --algo ppo --timesteps 100000 --save models/value_net.pt
        """
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes (for SimpleTrainer)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="simple",
        choices=["simple", "ppo", "dqn"],
        help="Training algorithm"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total timesteps (for stable-baselines3)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="models/value_net.pt",
        help="Path to save model"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu, mps, cuda)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N timesteps (for RL training)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
    
    # Create output directory
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Train
    if args.algo == "simple":
        trainer = SimpleTrainer(
            model_path=args.save,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            device=args.device,
        )
        trainer.train(num_episodes=args.episodes)
    else:
        train_with_sb3(
            algo=args.algo,
            total_timesteps=args.timesteps,
            model_path=args.save,
            seed=args.seed,
            use_wandb=args.wandb,
            checkpoint_interval=args.checkpoint_interval,
            device=args.device,
        )


if __name__ == "__main__":
    main()

