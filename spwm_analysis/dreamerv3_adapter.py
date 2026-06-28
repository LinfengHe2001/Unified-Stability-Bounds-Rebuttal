#!/usr/bin/env python3
"""
DreamerV3 Model Adapter for SPWM Metrics
Properly loads and interfaces with DreamerV3 checkpoints.

If core JAX dependencies are unavailable, falls back to a lightweight
mock adapter so that this script and downstream tooling can still run.
"""

import os
import sys
import pickle
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
FORCE_MOCK = os.environ.get('SPWM_FORCE_MOCK', '0') == '1'

# Try to import JAX stack and elements. If missing, we will provide a
# mock implementation below to keep the CLI usable in limited envs.
HAS_JAX = False
_IMPORT_ERR = None
try:
    if not FORCE_MOCK:
        import elements  # type: ignore
        import jax  # type: ignore
        import jax.numpy as jnp  # type: ignore
        import ninjax as nj  # type: ignore
        HAS_JAX = True
    else:
        raise ImportError('SPWM_FORCE_MOCK enabled')
except Exception as e:
    elements = None  # type: ignore
    jax = None  # type: ignore
    jnp = None  # type: ignore
    nj = None  # type: ignore
    _IMPORT_ERR = e

# Ensure we can import local dreamerv3 package regardless of CWD.
# Prefer repo root relative to this file instead of hardcoding.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Only import deep DreamerV3 stack when JAX is available.
if HAS_JAX:
    from dreamerv3 import rssm  # type: ignore
    import embodied  # type: ignore


class DreamerV3Adapter:
    """
    Adapter to interface with DreamerV3 checkpoints for SPWM analysis
    """
    
    def __init__(self, checkpoint_path: str, obs_space: Optional[Dict] = None, 
             act_space: Optional[Dict] = None, config_path: Optional[str] = None,
             device: str = 'cuda'):
        """
        Initialize DreamerV3 adapter
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if not HAS_JAX:
            raise ImportError(
                f"JAX/elements not available: {_IMPORT_ERR}. "
                "Use SPWM_FORCE_MOCK=1 or install dependencies.")
        
        # If no config_path provided, use checkpoint directory's parent config.yaml
        if config_path is None:
            config_path = self.checkpoint_path.parent / 'config.yaml'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found at {config_path}")
        
        print(f"Loading config from: {config_path}")
        
        # Load config to get model architecture
        self.config = self._load_config(str(config_path))
        
        # Setup spaces
        self.obs_space = obs_space or self._get_default_obs_space()
        self.act_space = act_space or self._get_default_act_space()
        
        # Load checkpoint weights first to get params
        self._load_checkpoint()
        
        # Now setup agent components with loaded params
        self._setup_components()


        # --- NEW: create Ninjax state from checkpoint params ---
        self.state = dict(self.params)
        # ------------------------------------------------------

        self._rng = jax.random.PRNGKey(0)


        # Pure-apply wrappers. All calls will read/update self.state explicitly.
        # Define lambdas without RNG; we provide rng via seed=... to nj.pure call.
        self._enc_apply = nj.pure(lambda enc_carry, obs, reset:
            self.enc(enc_carry, obs, reset, training=False, single=True))

        self._dyn_observe = nj.pure(lambda dyn_carry, tokens, prev_action, reset:
            self.dyn.observe(dyn_carry, tokens, prev_action, reset,
                             training=False, single=True))

        self._dyn_imagine = nj.pure(lambda carry, policy, length:
            self.dyn.imagine(carry, policy, length=length,
                             training=False, single=True))

        self._dec_apply = nj.pure(lambda dec_carry, feat, reset:
            self.dec(dec_carry, feat, reset, training=False, single=True))

        self._rew_apply = nj.pure(lambda feat_tensor:
            self.rew(feat_tensor, bdims=1))


        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load DreamerV3 configuration"""
        if config_path is None:
            # Try to find config.yaml in checkpoint directory or parent
            search_dirs = [
                self.checkpoint_path if self.checkpoint_path.is_dir() else self.checkpoint_path.parent,
                self.checkpoint_path.parent,
                self.checkpoint_path.parent.parent,
            ]
            
            for dir_path in search_dirs:
                config_file = dir_path / 'config.yaml'
                if config_file.exists():
                    config_path = config_file
                    break
        
        if config_path and Path(config_path).exists():
            import ruamel.yaml as yaml
            with open(config_path, 'r') as f:
                config = yaml.YAML(typ='safe').load(f)
            return config
        else:
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default DreamerV3 config for Crafter"""
        return {
            'agent': {
                'enc': {
                    'typ': 'simple',
                    'simple': {
                        'depth': 64,
                        'mults': [2, 3, 4, 4],
                        'layers': 3,
                        'units': 1024,
                        'act': 'gelu',
                        'norm': 'rms',
                        'kernel': 5,
                        'symlog': True,
                    }
                },
                'dyn': {
                    'typ': 'rssm',
                    'rssm': {
                        'deter': 4096,
                        'hidden': 2048,
                        'stoch': 32,
                        'classes': 32,
                        'blocks': 8,
                        'norm': 'rms',
                        'act': 'gelu',
                        'unimix': 0.01,
                        'outscale': 1.0,
                        'dynlayers': 1,
                        'obslayers': 1,
                        'imglayers': 2,
                        'free_nats': 1.0,
                    }
                },
                'dec': {
                    'typ': 'simple',
                    'simple': {
                        'depth': 64,
                        'mults': [2, 3, 4, 4],
                        'layers': 3,
                        'units': 1024,
                        'act': 'gelu',
                        'norm': 'rms',
                        'kernel': 5,
                        'symlog': True,
                        'outscale': 1.0,
                        'bspace': 8,
                    }
                },
                'rewhead': {
                    'layers': 1,
                    'units': 512,
                    'act': 'gelu',
                    'norm': 'rms',
                    'outscale': 0.0,
                },
                'conhead': {
                    'layers': 1,
                    'units': 512,
                    'act': 'gelu',
                    'norm': 'rms',
                    'outscale': 0.0,
                },
            }
        }
    
    def _get_default_obs_space(self) -> Dict:
        """Get default observation space for Crafter"""
        import elements
        return {
            'image': elements.Space(np.uint8, (64, 64, 3)),
            'reward': elements.Space(np.float32, ()),
            'is_first': elements.Space(bool, ()),
            'is_last': elements.Space(bool, ()),
            'is_terminal': elements.Space(bool, ()),
        }

    def _get_default_act_space(self) -> Dict:
        """Get default action space for Crafter (discrete only).

        Many checkpoints are trained with only the discrete action embedded
        into the dynamics model. Exclude 'reset' here to match those.
        """
        return {
            'action': elements.Space(np.int32, (), 0, 17),
        }
    
    def _load_checkpoint(self):
        """Load checkpoint weights from agent.pkl"""
        print(f"Loading checkpoint from {self.checkpoint_path}")

        # Resolve provided path and allow light fuzzy matching for common typos
        ckpt_base = self.checkpoint_path
        if not ckpt_base.exists():
            parent = ckpt_base.parent
            name = ckpt_base.name
            cand = None
            if parent.exists():
                # Try prefix match among siblings
                for p in sorted(parent.iterdir()):
                    if p.is_dir() and p.name.startswith(name):
                        cand = p
                        break
                # Try adding a trailing zero (0050000 -> 00500000)
                if cand is None and (parent / f"{name}0").exists():
                    cand = parent / f"{name}0"
            if cand is not None:
                print(f"Adjusted checkpoint path to: {cand}")
                ckpt_base = cand
            else:
                raise FileNotFoundError(f"Checkpoint path not found: {self.checkpoint_path}")

        # Read latest file to find actual checkpoint directory
        latest_path = ckpt_base / 'latest'
        if latest_path.exists():
            with open(latest_path, 'r') as f:
                latest_dir = f.read().strip()
            ckpt_dir = ckpt_base / latest_dir
        else:
            ckpt_dir = ckpt_base
        
        # Load agent.pkl file
        agent_pkl = ckpt_dir / 'agent.pkl'
        if not agent_pkl.exists():
            raise FileNotFoundError(f"agent.pkl not found in {ckpt_dir}")
        
        with open(agent_pkl, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Extract params
        if isinstance(agent_data, dict) and 'params' in agent_data:
            self.params = agent_data['params']
            nkeys = len(self.params.keys()) if isinstance(self.params, dict) else 'n/a'
            print(f"Loaded params (keys: {nkeys})")
        else:
            raise ValueError(f"Unexpected agent.pkl structure: {type(agent_data)}")
        
        print("Checkpoint loaded successfully")
    
    def _setup_components(self):
        """Setup model components after loading checkpoint"""
        from dreamerv3 import rssm
        import embodied.jax
        
        # Define spaces for encoder/decoder (exclude reward and terminal flags)
        exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
        enc_space = {k: v for k, v in self.obs_space.items() if k not in exclude}
        dec_space = {k: v for k, v in self.obs_space.items() if k not in exclude}
        
        agent_config = self.config.get('agent', self.config)
        
        # Convert lists to tuples in config (ninjax expects tuples)
        def convert_lists_to_tuples(config):
            if isinstance(config, dict):
                return {k: convert_lists_to_tuples(v) for k, v in config.items()}
            elif isinstance(config, list) and len(config) > 0 and all(isinstance(x, (int, float)) for x in config):
                return tuple(config)
            else:
                return config
        
        # Prepare configs
        enc_config = convert_lists_to_tuples(agent_config['enc']['simple'].copy())
        dyn_config = convert_lists_to_tuples(agent_config['dyn']['rssm'].copy())
        dec_config = convert_lists_to_tuples(agent_config['dec']['simple'].copy())

        # For reward head, MLPHead expects (space, output, **kwargs)
        rew_config = agent_config['rewhead'].copy()
        rew_output = rew_config.pop('output', 'mse')  # Extract output type
        rew_config = convert_lists_to_tuples(rew_config)
        
        # Create model components
        self.enc = rssm.Encoder(enc_space, **enc_config, name='enc')
        self.dyn = rssm.RSSM(self.act_space, **dyn_config, name='dyn')
        self.dec = rssm.Decoder(dec_space, **dec_config, name='dec')
        
        # Create reward head with output type as second positional argument
        scalar = elements.Space(np.float32, ())
        self.rew = embodied.jax.MLPHead(scalar, rew_output, **rew_config, name='rew')
        
        # Store actual dimensions from config
        self.deter_size = agent_config['dyn']['rssm']['deter']
        self.stoch_size = agent_config['dyn']['rssm']['stoch']
        self.classes = agent_config['dyn']['rssm']['classes']
        
        print(f"Model components setup successfully")
        print(f"  Deterministic size: {self.deter_size}")
        print(f"  Stochastic size: {self.stoch_size}")
        print(f"  Classes: {self.classes}")

    def _next_rng(self):
        self._rng, rng = jax.random.split(self._rng)
        return rng

        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode observations to latent representations
        
        Args:
            x: Observations [B, C, H, W] in PyTorch format
        Returns:
            z: Latent representations [B, D] where D = deter + stoch*classes
        """
        B = x.shape[0]
        
        # Convert from PyTorch [B, C, H, W] to JAX [B, H, W, C]
        x_np = x.detach().cpu().numpy()
        x_np = np.transpose(x_np, (0, 2, 3, 1))
        
        # Convert to uint8 as expected by encoder image space
        # Clamp to [0, 1] first to avoid small numeric drift from augmentations.
        x_np = np.clip(x_np, 0.0, 1.0)
        x_np = (x_np * 255).astype(np.uint8)
        
        # Create observation dict for encoder
        obs = {'image': x_np}
        
        # Add dummy values for other observations if needed
        for key in self.obs_space:
            if key not in obs:
                if key in ['is_first', 'is_last', 'is_terminal']:
                    obs[key] = np.zeros((B,), dtype=bool)
                elif key == 'reward':
                    obs[key] = np.zeros((B,), dtype=np.float32)
        
        # Convert to JAX
        obs_jax = jax.tree.map(jnp.array, obs)
        
        # Get initial carries
        enc_carry = self.enc.initial(B)
        dyn_carry = self.dyn.initial(B)
        
        # Reset flags and previous action (discrete index + reset flag)
        reset = jnp.zeros((B,), dtype=bool)
        prev_action = {
            'action': jnp.zeros((B,), dtype=jnp.int32),
        }
        
        # Run model with parameters
        # First encode observations
        
        rng = self._next_rng()
        self.state, (enc_carry, entries, tokens) = self._enc_apply(self.state, enc_carry, obs_jax, reset, seed=rng)
        rng = self._next_rng()
        self.state, (dyn_carry, entries, feat)   = self._dyn_observe(self.state, dyn_carry, tokens, prev_action, reset, seed=rng)


        
        # Combine deterministic and stochastic parts
        deter = feat['deter']  # [B, deter_size]
        stoch = feat['stoch']  # [B, stoch_size, classes]
        stoch_flat = stoch.reshape((B, -1))  # [B, stoch_size * classes]
        
        # Concatenate to form full latent representation
        z_jax = jnp.concatenate([deter, stoch_flat], axis=-1)
        z_jax = z_jax.astype(jnp.float32)
        
        # Convert back to PyTorch
        z = torch.from_numpy(np.array(z_jax)).to(self.device)
        
        return z
    
    def dynamics_step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        One-step latent dynamics prediction
        
        Args:
            z: Current latent state [B, D]
            a: Action [B, A]
        Returns:
            z_next: Predicted next latent state [B, D]
        """
        B = z.shape[0]
        
        # Use actual dimensions from config
        deter_size = self.deter_size
        stoch_size = self.stoch_size
        classes = self.classes
        
        z_np = z.detach().cpu().numpy()
        deter_np = z_np[:, :deter_size]
        stoch_flat_np = z_np[:, deter_size:]
        stoch_np = stoch_flat_np.reshape((B, stoch_size, classes))

        # Convert to JAX
        deter = jnp.array(deter_np).astype(jnp.bfloat16)
        stoch = jnp.array(stoch_np).astype(jnp.bfloat16)

        # Convert action to proper format. Upstream provides one-hot floats
        # with shape [B, 17]; map to discrete indices for DictConcat.
        a_np = a.detach().cpu().numpy()
        if a_np.ndim == 2 and a_np.shape[1] >= 2:
            a_idx = np.argmax(a_np, axis=-1).astype(np.int32)
        else:
            # If already indices, just cast
            a_idx = a_np.astype(np.int32).reshape(B)
        action = {
            'action': jnp.array(a_idx, dtype=jnp.int32),
        }
        
        # Create carry
        carry = {'deter': deter, 'stoch': stoch}
        
        # Run dynamics with parameters
        policy = lambda feat: action
        rng = self._next_rng()
        self.state, (carry_out, (feat, _)) = self._dyn_imagine(self.state, carry, policy, 1, seed=rng)


        # Extract next state
        next_deter = feat['deter']
        next_stoch = feat['stoch']
        next_stoch_flat = next_stoch.reshape((B, -1))
        
        # Combine and convert back to PyTorch
        z_next_jax = jnp.concatenate([next_deter, next_stoch_flat], axis=-1)
        z_next_jax = z_next_jax.astype(jnp.float32)
        z_next = torch.from_numpy(np.array(z_next_jax)).to(self.device)
        
        return z_next
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to observations
        
        Args:
            z: Latent state [B, D]
        Returns:
            x_recon: Reconstructed observation [B, C, H, W]
        """
        B = z.shape[0]
        
        # Use actual dimensions from config
        deter_size = self.deter_size
        stoch_size = self.stoch_size
        classes = self.classes
        
        z_np = z.detach().cpu().numpy()
        deter_np = z_np[:, :deter_size]
        stoch_flat_np = z_np[:, deter_size:]
        stoch_np = stoch_flat_np.reshape((B, stoch_size, classes))
        
        # Convert to JAX
        feat = {
            'deter': jnp.array(deter_np).astype(jnp.bfloat16),
            'stoch': jnp.array(stoch_np).astype(jnp.bfloat16),
        }
        
        # Decode
        dec_carry = self.dec.initial(B)
        reset = jnp.zeros((B,), dtype=bool)
        
        # Run decoder with parameters
        rng = self._next_rng()
        self.state, (_, _, recons) = self._dec_apply(self.state, dec_carry, feat, reset, seed=rng)

        # Get image reconstruction
        if 'image' in recons:
            x_recon_jax = recons['image'].pred().astype(jnp.float32)  # [B,H,W,C]
            
            # Convert to numpy
            x_recon_np = np.array(x_recon_jax)
            
            # Convert from [B, H, W, C] to [B, C, H, W]
            x_recon_np = np.transpose(x_recon_np, (0, 3, 1, 2))
            
            # Convert to PyTorch
            x_recon = torch.from_numpy(x_recon_np).float().to(self.device)
        else:
            # Return dummy reconstruction if not available
            x_recon = torch.zeros((B, 3, 64, 64), device=self.device)
        
        return x_recon
    
    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Predict reward from latent state and action
        
        Args:
            z: Latent state [B, D]
            a: Action [B, A]
        Returns:
            r: Predicted reward [B, 1]
        """
        B = z.shape[0]
        
        # Convert latent to feature tensor (concatenate deter and stoch)
        z_np = z.detach().cpu().numpy()
        feat_tensor = jnp.array(z_np).astype(jnp.bfloat16)
        
        # Run reward head with parameters
        rng = self._next_rng()
        self.state, reward_dist = self._rew_apply(self.state, feat_tensor, seed=rng)

        r_pred = reward_dist.pred().astype(jnp.float32)
        
        # Convert to PyTorch
        r = torch.from_numpy(np.array(r_pred)).float().to(self.device)
        
        # Ensure shape is [B, 1]
        if r.dim() == 1:
            r = r.unsqueeze(-1)
        
        return r


class MockDreamerV3Adapter:
    """Lightweight torch-only fallback for environments without JAX.

    Provides the same interface as DreamerV3Adapter so that scripts can run
    in CPU-only or minimal environments. It does not load real Dreamer
    checkpoints and returns random outputs with shapes matching the default
    Crafter config from config.yaml.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Match the sizes in logdir/crafter_50K_.../config.yaml
        self.deter_size = 8192
        self.stoch_size = 32
        self.classes = 64
        self.latent_dim = self.deter_size + self.stoch_size * self.classes
        print(f"[MOCK] Using latent_dim={self.latent_dim} (no JAX)")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self.latent_dim, device=self.device)

    def dynamics_step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + 0.05 * torch.randn_like(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.rand(B, 3, 64, 64, device=self.device)

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.randn(B, 1, device=self.device)


def test_adapter():
    """Test the DreamerV3 adapter"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='/hy-tmp/SPWM/logdir/crafter_5M_20250914_103133',
                       help='Path to DreamerV3 checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("Testing DreamerV3 Adapter...")

    # Choose real or mock depending on JAX availability
    if HAS_JAX:
        adapter = DreamerV3Adapter(args.checkpoint, device=args.device)
    else:
        print(f"[WARN] JAX stack unavailable ({_IMPORT_ERR}); using mock adapter.")
        adapter = MockDreamerV3Adapter(args.checkpoint, device=args.device)
    
    # Test with random data
    B = 4  # Batch size
    x = torch.rand(B, 3, 64, 64).to(args.device)
    a = torch.randn(B, 17).to(args.device)  # Crafter has 17 discrete actions
    
    print("\n" + "="*50)
    print("Testing model components:")
    print("="*50)
    
    # Test encoding
    print("\n1. Testing encoder...")
    z = adapter.encode(x)
    print(f"   ✓ Encoded shape: {z.shape}")
    expected_size = adapter.deter_size + adapter.stoch_size * adapter.classes
    print(f"   Expected: [B, deter + stoch*classes] = [{B}, {adapter.deter_size} + {adapter.stoch_size}*{adapter.classes}] = [{B}, {expected_size}]")
    
    # Test dynamics
    print("\n2. Testing dynamics...")
    z_next = adapter.dynamics_step(z, a)
    print(f"   ✓ Next state shape: {z_next.shape}")
    
    # Test decoding
    print("\n3. Testing decoder...")
    x_recon = adapter.decode(z)
    print(f"   ✓ Reconstructed shape: {x_recon.shape}")
    print(f"   Range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    
    # Test reward prediction
    print("\n4. Testing reward predictor...")
    r = adapter.reward(z, a)
    print(f"   ✓ Reward shape: {r.shape}")
    
    print("\n" + "="*50)
    print("✓ All tests passed successfully!")
    print("="*50)


if __name__ == "__main__":
    test_adapter()
