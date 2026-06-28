#!/usr/bin/env python3
"""
Create frozen evaluation set from DreamerV3 training data
This script interfaces with your DreamerV3 implementation to generate eval sets
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Optional
import gymnasium as gym

# Add DreamerV3 to path (adjust based on your structure)
sys.path.append('/hy-tmp/SPWM')

def create_eval_set_from_replay(replay_dir: str, n_samples: int = 5000, 
                                seed: int = 42) -> Dict:
    """
    Create evaluation set from saved replay buffer
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"Loading replay data from {replay_dir}")
    
    # Look for replay files (adjust based on your format)
    replay_files = list(Path(replay_dir).glob('*.npz')) + \
                   list(Path(replay_dir).glob('*.pkl')) + \
                   list(Path(replay_dir).glob('*.pt'))
    
    if not replay_files:
        raise ValueError(f"No replay files found in {replay_dir}")
    
    # Load replay data (adapt to your format)
    all_transitions = []
    
    for replay_file in replay_files[:10]:  # Limit to first 10 files
        try:
            if replay_file.suffix == '.npz':
                data = np.load(replay_file)
                # Assuming standard RL transition format
                if 'observations' in data:
                    obs = data['observations']
                    acts = data.get('actions', np.zeros((len(obs), 4)))
                    next_obs = data.get('next_observations', obs[1:])
                    rewards = data.get('rewards', np.zeros(len(obs)))
                    
                    for i in range(min(len(obs)-1, 1000)):
                        all_transitions.append({
                            'x': torch.tensor(obs[i], dtype=torch.float32),
                            'a': torch.tensor(acts[i] if i < len(acts) else acts[-1], dtype=torch.float32),
                            'x_next': torch.tensor(next_obs[i] if i < len(next_obs) else obs[i], dtype=torch.float32),
                            'r': torch.tensor(rewards[i] if i < len(rewards) else 0, dtype=torch.float32),
                        })
            
            elif replay_file.suffix in ['.pkl', '.pt']:
                with open(replay_file, 'rb') as f:
                    data = pickle.load(f) if replay_file.suffix == '.pkl' else torch.load(f)
                    # Process based on your replay format
                    pass
                    
        except Exception as e:
            print(f"Error loading {replay_file}: {e}")
            continue
    
    if not all_transitions:
        print("Warning: No transitions loaded, generating synthetic data")
        return create_synthetic_eval_set(n_samples)
    
    # Sample from collected transitions
    n_available = len(all_transitions)
    n_to_sample = min(n_samples, n_available)
    
    indices = np.random.choice(n_available, n_to_sample, replace=False)
    sampled = [all_transitions[i] for i in indices]
    
    # Post-processing: convert HWC→CHW and scalar actions→one-hot
    def to_chw01_tensor(t):
        """Convert observation to CHW format in [0,1] range"""
        if t.ndim == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)  # HWC to CHW
        if t.max() > 1.0:
            t = t / 255.0
        return t.clamp(0, 1)
    
    def to_onehot17_tensor(a):
        """Convert action to one-hot encoding for Crafter (17 actions)"""
        if a.ndim == 0:  # scalar
            idx = int(a.item())
            one = torch.zeros(17, dtype=torch.float32)
            one[idx] = 1.0
            return one
        if a.numel() == 17:
            return a.view(17).float()
        # fallback: argmax
        idx = int(a.view(-1).argmax().item())
        one = torch.zeros(17, dtype=torch.float32)
        one[idx] = 1.0
        return one
    
    # Apply post-processing to all sampled transitions
    sampled = [
        {
            'x': to_chw01_tensor(t['x']),
            'a': to_onehot17_tensor(t['a']),
            'x_next': to_chw01_tensor(t['x_next']),
            'r': torch.tensor([t['r']], dtype=torch.float32) if t['r'].ndim == 0 else t['r'].view(1),
        }
        for t in sampled
    ]
    
    # Convert to tensors
    eval_set = {
        'x': torch.stack([t['x'] for t in sampled]),
        'a': torch.stack([t['a'] for t in sampled]),
        'x_next': torch.stack([t['x_next'] for t in sampled]),
        'r': torch.stack([t['r'] for t in sampled]),
    }
    
    return eval_set


def create_eval_set_from_env(env_name: str = 'crafter', n_samples: int = 5000,
                             n_episodes: int = 50, seed: int = 42) -> Dict:
    """
    Create evaluation set by collecting new episodes from environment (Crafter).
    - Observations: CHW float32 in [0,1], shape (N, 3, 64, 64)
    - Actions: one-hot(17) float32, shape (N, 17)
    - Rewards: shape (N, 1)
    """
    import numpy as np
    import torch

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- create env (prefer dreamerv3 embodied wrapper if available) ---
    env = None
    try:
        import crafter
        import embodied
        from embodied.envs import from_gym
        base_env = crafter.Env(seed=seed)
        env = from_gym.FromGym(base_env)  # returns dict obs with key 'image'
    except Exception as e:
        print(f"Falling back to gymnasium due to: {e}")
        import gymnasium as gym
        if env_name == 'crafter':
            # If crafter is not installed, fail back to synthetic later
            env = None
        else:
            env = gym.make(env_name)

    transitions = []
    if env is None:
        print("Warning: Crafter env not available; creating synthetic set.")
        return create_synthetic_eval_set(n_samples)

    # --- collector ---
    def to_chw01(img):
        # img can be HWC [0..255] or [0..1]; return CHW in [0,1]
        t = torch.tensor(img, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] == 3:  # HWC -> CHW
            t = t.permute(2, 0, 1)
        if t.max() > 1.0:
            t = t / 255.0
        return t.clamp(0, 1)

    def to_onehot17(a):
        # Crafter has a flat categorical action space with 17 actions
        # https://arxiv.org/abs/2109.06780 (paper + supp)
        one = torch.zeros(17, dtype=torch.float32)
        if np.isscalar(a):
            idx = int(a)
        elif isinstance(a, (np.ndarray, list)) and np.array(a).ndim == 0:
            idx = int(a)
        else:
            # if already one-hot/prob vector, try to coerce
            arr = np.asarray(a, dtype=np.float32)
            if arr.shape == (17,):
                return torch.from_numpy(arr)
            idx = int(np.argmax(arr))
        one[idx] = 1.0
        return one

    steps_collected = 0
    for _ in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if isinstance(obs, dict) and 'image' in obs:
            curr = obs['image']
        else:
            curr = obs

        done = False
        while not done and steps_collected < n_samples:
            # random action from env (gym.Discrete(17) typically)
            if hasattr(env, 'action_space'):
                a_raw = env.action_space.sample()
            else:
                a_raw = rng.randint(0, 17)
            nxt, reward, done, *rest = env.step(a_raw)
            if isinstance(nxt, tuple):
                nxt = nxt[0]
            if isinstance(nxt, dict) and 'image' in nxt:
                nxt_img = nxt['image']
            else:
                nxt_img = nxt

            transitions.append({
                'x': to_chw01(curr),
                'a': to_onehot17(a_raw),
                'x_next': to_chw01(nxt_img),
                'r': torch.tensor([reward], dtype=torch.float32),
            })
            curr = nxt_img
            steps_collected += 1
            if steps_collected >= n_samples:
                break

        if steps_collected >= n_samples:
            break

    # stack
    eval_set = {
        'x': torch.stack([t['x'] for t in transitions], dim=0),
        'a': torch.stack([t['a'] for t in transitions], dim=0),
        'x_next': torch.stack([t['x_next'] for t in transitions], dim=0),
        'r': torch.stack([t['r'] for t in transitions], dim=0),
    }
    # sanity
    assert eval_set['x'].shape[1:] == (3, 64, 64), f"obs shape {eval_set['x'].shape}"
    assert eval_set['a'].shape[1] == 17, f"action dim {eval_set['a'].shape[1]}"
    return eval_set



def create_synthetic_eval_set(n_samples: int = 5000) -> Dict:
    """
    Create synthetic evaluation set for testing (Crafter-like).
    Produces:
      - x, x_next: [N, 3, 64, 64] in [0,1]
      - a:        [N, 17] one-hot actions
      - r:        [N, 1]
    """
    print("Creating synthetic evaluation set for testing (Crafter-like)")
    x = torch.rand(n_samples, 3, 64, 64)
    noise = torch.rand(n_samples, 3, 64, 64) * 0.1
    x_next = torch.clone(x)
    x_next[:-1] = (0.9 * x[1:] + noise[:-1]).clamp(0, 1)
    idx = torch.randint(low=0, high=17, size=(n_samples,))
    a = torch.zeros(n_samples, 17, dtype=torch.float32)
    a[torch.arange(n_samples), idx] = 1.0
    r = torch.randn(n_samples, 1)
    return {'x': x, 'a': a, 'x_next': x_next, 'r': r}



def load_from_dreamer_checkpoint(ckpt_path: str, n_samples: int = 5000) -> Dict:
    """
    Extract evaluation set from a DreamerV3 checkpoint that contains replay data
    """
    print(f"Loading from DreamerV3 checkpoint: {ckpt_path}")
    
    try:
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # DreamerV3 checkpoints may contain replay buffer
    if 'replay' in ckpt:
        replay = ckpt['replay']
        
        # Extract transitions (adapt based on actual structure)
        transitions = []
        
        if hasattr(replay, 'sample'):
            # Sample from replay buffer
            batch = replay.sample(n_samples)
            
            eval_set = {
                'x': batch['observation'],
                'a': batch['action'],
                'x_next': batch['next_observation'],
                'r': batch['reward'].unsqueeze(-1),
            }
        else:
            # Manual extraction
            # Adjust based on your replay buffer structure
            pass
    else:
        print("No replay buffer in checkpoint, generating synthetic data")
        return create_synthetic_eval_set(n_samples)
    
    return eval_set


def main():
    parser = argparse.ArgumentParser(description='Create evaluation set for SPWM analysis')
    
    parser.add_argument('--method', type=str, default='replay',
                       choices=['replay', 'env', 'checkpoint', 'synthetic'],
                       help='Method to create evaluation set')
    parser.add_argument('--source_dir', type=str,
                       default='/hy-tmp/SPWM/logdir/crafter_5M_20250914_103133',
                       help='Directory containing replay data or checkpoints')
    parser.add_argument('--env_name', type=str, default='crafter',
                       help='Environment name for collection')
    parser.add_argument('--n_samples', type=int, default=5000,
                       help='Number of samples in evaluation set')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Number of episodes to collect (for env method)')
    parser.add_argument('--output_path', type=str, default='./eval_set.npz',
                       help='Output path for evaluation set')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the created evaluation set')
    
    args = parser.parse_args()
    
    # Create evaluation set based on method
    if args.method == 'replay':
        # Look for replay directory
        replay_dir = os.path.join(args.source_dir, 'replay')
        if not os.path.exists(replay_dir):
            replay_dir = os.path.join(args.source_dir, 'buffer')
        if not os.path.exists(replay_dir):
            replay_dir = args.source_dir
            
        eval_set = create_eval_set_from_replay(replay_dir, args.n_samples, args.seed)
        
    elif args.method == 'env':
        eval_set = create_eval_set_from_env(args.env_name, args.n_samples, 
                                           args.n_episodes, args.seed)
        
    elif args.method == 'checkpoint':
        # Find latest checkpoint
        ckpt_files = list(Path(args.source_dir).glob('**/checkpoint.pkl')) + \
             list(Path(args.source_dir).glob('**/*.pkl')) + \
             list(Path(args.source_dir).glob('**/*.ckpt')) + \
             list(Path(args.source_dir).glob('**/*.pt'))
        if ckpt_files:
            latest_ckpt = max(ckpt_files, key=lambda x: x.stat().st_mtime)
            eval_set = load_from_dreamer_checkpoint(str(latest_ckpt), args.n_samples)
        else:
            print("No checkpoint found, using synthetic data")
            eval_set = create_synthetic_eval_set(args.n_samples)
            
    else:  # synthetic
        eval_set = create_synthetic_eval_set(args.n_samples)
    
    # Validate shapes
    if args.validate:
        print("\nValidating evaluation set:")
        for key, tensor in eval_set.items():
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, "
                  f"range=[{tensor.min():.3f}, {tensor.max():.3f}]")
        
        # Check consistency
        n_samples = eval_set['x'].shape[0]
        assert eval_set['a'].shape[0] == n_samples, "Action count mismatch"
        assert eval_set['x_next'].shape[0] == n_samples, "Next observation count mismatch"
        assert eval_set['r'].shape[0] == n_samples, "Reward count mismatch"
        
        # Check observation dimensions
        assert eval_set['x'].shape == eval_set['x_next'].shape, "Observation shape mismatch"
        
        print("✓ Validation passed!")
    
    # Save evaluation set
    print(f"\nSaving evaluation set to {args.output_path}")
    np.savez_compressed(
        args.output_path,
        x=eval_set['x'].numpy(),
        a=eval_set['a'].numpy(),
        x_next=eval_set['x_next'].numpy(),
        r=eval_set['r'].numpy(),
        metadata=json.dumps({
            'n_samples': args.n_samples,
            'method': args.method,
            'seed': args.seed,
            'source': args.source_dir,
        })
    )
    
    print(f"✓ Evaluation set saved successfully!")
    print(f"  Samples: {eval_set['x'].shape[0]}")
    print(f"  Observation shape: {eval_set['x'].shape[1:]}")
    print(f"  Action dimension: {eval_set['a'].shape[1]}")
    
    # Print usage instructions
    print("\nNext steps:")
    print("1. Run SPWM analysis on your checkpoints:")
    print(f"   python spwm_analysis.py --mode analyze --eval_set_path {args.output_path}")
    print("\n2. Or run the complete pipeline:")
    print(f"   python spwm_analysis.py --mode full --eval_set_path {args.output_path}")


if __name__ == "__main__":
    main()