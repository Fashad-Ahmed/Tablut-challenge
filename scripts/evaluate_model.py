#!/usr/bin/env python3
"""
Evaluate trained RL model performance.

Usage:
    python scripts/evaluate_model.py models/rl_value_net_5M.zip
    python scripts/evaluate_model.py models/rl_value_net_5M_best/best_model.zip
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO, DQN
from python_client.tablut_env import TablutEnv
from python_client.action_mask_wrapper import ActionMaskWrapper
import numpy as np


def evaluate_model(model_path: str, num_games: int = 100, verbose: bool = True):
    """
    Evaluate model by playing self-play games.
    
    Args:
        model_path: Path to trained model (.zip file)
        num_games: Number of games to play
        verbose: Print progress
    """
    print(f"📊 Evaluating model: {model_path}")
    print(f"🎮 Playing {num_games} self-play games...\n")
    
    # Load model - handle .zip extension correctly
    model_path_clean = model_path
    if model_path.endswith('.zip.zip'):
        model_path_clean = model_path[:-4]  # Remove duplicate .zip
    elif not model_path.endswith('.zip'):
        model_path_clean = model_path + '.zip'
    
    # Check if file exists
    from pathlib import Path
    if not Path(model_path_clean).exists():
        print(f"❌ Model file not found: {model_path_clean}")
        print(f"   Tried: {model_path}")
        return
    
    try:
        model = PPO.load(model_path_clean)
        algo = "PPO"
    except:
        try:
            model = DQN.load(model_path_clean)
            algo = "DQN"
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    
    print(f"✅ Loaded {algo} model\n")
    
    # Create environment
    env = ActionMaskWrapper(TablutEnv())
    
    # Statistics
    white_wins = 0
    black_wins = 0
    draws = 0
    game_lengths = []
    
    # Play games
    illegal_moves = 0
    for game_idx in range(num_games):
        obs, info = env.reset()
        done = False
        steps = 0
        game_result = None
        
        while not done:
            # Get action from model (deterministic)
            action, _ = model.predict(obs, deterministic=True)
            
            # Ensure action is legal by checking against legal moves
            legal_moves = info.get('legal_moves', [])
            if legal_moves:
                # Convert action to (from_pos, to_pos) format
                from_idx, to_idx = action[0], action[1]
                from_row, from_col = from_idx // 9, from_idx % 9
                to_row, to_col = to_idx // 9, to_idx % 9
                action_pos = ((from_row, from_col), (to_row, to_col))
                
                # If action is not legal, pick first legal move
                if action_pos not in legal_moves:
                    # Pick first legal move as fallback
                    from_pos, to_pos = legal_moves[0]
                    from_idx = from_pos[0] * 9 + from_pos[1]
                    to_idx = to_pos[0] * 9 + to_pos[1]
                    action = np.array([from_idx, to_idx])
            
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            # Check for illegal moves
            if info.get("error") == "illegal_move":
                illegal_moves += 1
                # Game ended due to illegal move
                if reward < -500:  # Large penalty indicates illegal move
                    black_wins += 1  # Player who made illegal move loses
                    game_result = "illegal_move"
                break
            
            if done:
                # Check result based on reward and info
                result = info.get("result", "")
                if reward > 0 or result == "white_win":
                    white_wins += 1
                    game_result = "white_win"
                elif reward < -500:  # Illegal move penalty
                    black_wins += 1
                    game_result = "illegal_move"
                elif reward < 0 or result == "black_win":
                    black_wins += 1
                    game_result = "black_win"
                elif result == "draw_repetition" or result == "white_no_moves" or result == "black_no_moves":
                    draws += 1
                    game_result = "draw"
                else:
                    # Default: check reward sign
                    if reward > 0:
                        white_wins += 1
                    elif reward < 0:
                        black_wins += 1
                    else:
                        draws += 1
                
                game_lengths.append(steps)
                break
            
            if truncated:
                draws += 1
                game_lengths.append(steps)
                break
        
        # Progress update
        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  Played {game_idx + 1}/{num_games} games... "
                  f"(W: {white_wins}, B: {black_wins}, D: {draws})")
    
    # Print results
    print("\n" + "="*50)
    print("📈 EVALUATION RESULTS")
    print("="*50)
    print(f"Total games: {num_games}")
    print(f"White wins: {white_wins} ({white_wins/num_games:.1%})")
    print(f"Black wins: {black_wins} ({black_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")
    if illegal_moves > 0:
        print(f"⚠️  Illegal moves detected: {illegal_moves} ({illegal_moves/num_games:.1%})")
    if game_lengths:
        print(f"\nAverage game length: {np.mean(game_lengths):.1f} moves")
        print(f"Min game length: {min(game_lengths)} moves")
        print(f"Max game length: {max(game_lengths)} moves")
    else:
        print("\n⚠️  No valid games completed (all ended with illegal moves?)")
    print("="*50)
    
    # Model quality assessment
    if white_wins + black_wins > 0:
        win_rate = max(white_wins, black_wins) / (white_wins + black_wins)
        if win_rate > 0.7:
            print("✅ Model shows strong play (high win rate)")
        elif win_rate > 0.55:
            print("✅ Model shows decent play")
        else:
            print("⚠️  Model may need more training (low win rate)")
    
    if np.mean(game_lengths) < 20:
        print("⚠️  Games are very short - may indicate poor play")
    elif np.mean(game_lengths) > 150:
        print("⚠️  Games are very long - may indicate defensive play")
    else:
        print("✅ Game lengths look reasonable")


def compare_models(model_paths: list, num_games: int = 50):
    """Compare multiple models."""
    print(f"🔍 Comparing {len(model_paths)} models...\n")
    
    results = {}
    for model_path in model_paths:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_path}")
        print('='*50)
        
        # Clean path
        model_path_clean = model_path
        if model_path.endswith('.zip.zip'):
            model_path_clean = model_path[:-4]
        elif not model_path.endswith('.zip'):
            model_path_clean = model_path + '.zip'
        
        # Check if exists
        from pathlib import Path
        if not Path(model_path_clean).exists():
            print(f"⚠️  Skipping - file not found: {model_path_clean}")
            continue
        
        # Quick evaluation
        try:
            model = PPO.load(model_path_clean)
        except:
            try:
                model = DQN.load(model_path_clean)
            except Exception as e:
                print(f"⚠️  Skipping - error loading: {e}")
                continue
        
        env = ActionMaskWrapper(TablutEnv())
        white_wins = 0
        
        for _ in range(num_games):
            obs, info = env.reset()
            done = False
            while not done:
                # Use action mask if available
                if 'action_mask' in info and info['action_mask'] is not None:
                    action_mask = info['action_mask']
                    action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, done, truncated, info = env.step(action)
                
                # Check result
                result = info.get("result", "")
                if done and (reward > 0 or result == "white_win"):
                    white_wins += 1
        
        win_rate = white_wins / num_games
        results[model_path] = win_rate
        print(f"Win rate: {win_rate:.1%}")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 COMPARISON SUMMARY")
    print('='*50)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for i, (path, win_rate) in enumerate(sorted_results, 1):
        print(f"{i}. {Path(path).name}: {win_rate:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL model")
    parser.add_argument("model_path", type=str, help="Path to model (.zip file)")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--compare", nargs="+", help="Compare multiple models")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare, args.games)
    else:
        evaluate_model(args.model_path, args.games, verbose=not args.quiet)

