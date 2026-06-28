#!/usr/bin/env python3
"""
SPWM Metrics Evaluation System for DreamerV3
Complete implementation for analyzing representation quality metrics
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.functional as TF
from scipy import stats
from scipy.stats import spearmanr, bootstrap, kendalltau
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from einops import rearrange
import hashlib
from datetime import datetime
import traceback

# ==================== Module 1: I/O and Data Loading ====================

# Import the DreamerV3 adapter
# Adapter import with safe fallback. When running as `python spwm_analysis/...py`,
# sys.path[0] is the folder of this file so plain import works.
RealDreamerV3Wrapper = None
try:
    from dreamerv3_adapter import DreamerV3Adapter as RealDreamerV3Wrapper
except Exception as e:
    RealDreamerV3Wrapper = None
    print(f"[INFO] dreamerv3_adapter not available, will use mock wrapper. Reason: {e}")

class MockDreamerV3Wrapper:
    """Mock wrapper (identical behavior to previous inline mock)"""
    def __init__(self, ckpt_path: str, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ckpt_path = ckpt_path
        print(f"Warning: Using mock DreamerV3 wrapper.")
        self.latent_dim = 5120
        self.action_dim = 17
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        return torch.randn(B, self.latent_dim, device=self.device)
    def dynamics_step(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + 0.1 * torch.randn_like(z)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.rand(B, 3, 64, 64, device=self.device)
    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        return torch.randn(B, 1, device=self.device)

def get_model_wrapper(adapter_choice: str, ckpt_dir: str, device: str):
    if adapter_choice == 'mock' or RealDreamerV3Wrapper is None:
        return MockDreamerV3Wrapper(ckpt_dir, device=device)
    if adapter_choice == 'dreamer':
        return RealDreamerV3Wrapper(ckpt_dir, device=device)
    # auto
    try:
        return RealDreamerV3Wrapper(ckpt_dir, device=device)
    except Exception as e:
        print(f"[WARN] auto adapter failed, falling back to mock: {e}")
        return MockDreamerV3Wrapper(ckpt_dir, device=device)



def create_eval_set(env_path: str, n_samples: int = 5000, seed: int = 42) -> Dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 17 维 one-hot（Crafter）
    idx = torch.randint(0, 17, (n_samples,))
    a = torch.nn.functional.one_hot(idx, num_classes=17).float()
    eval_set = {
        'x': torch.randn(n_samples, 3, 64, 64).clamp(0, 1),
        'a': a,
        'x_next': torch.randn(n_samples, 3, 64, 64).clamp(0, 1),
        'r': torch.randn(n_samples, 1),
    }
    return eval_set



def load_eval_set(npz_path: str) -> Dict:
    """Load evaluation set from file"""
    data = np.load(npz_path)
    return {
        'x': torch.from_numpy(data['x']).float(),
        'a': torch.from_numpy(data['a']).float(),
        'x_next': torch.from_numpy(data['x_next']).float(),
        'r': torch.from_numpy(data['r']).float(),
    }


def load_scores(jsonl_path: str) -> pd.DataFrame:
    """Load scores from JSONL file"""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


# ==================== Module 2: Image Transformations ====================

def affine_rotate(x: torch.Tensor, degrees: float) -> torch.Tensor:
    """Apply rotation transformation"""
    return TF.affine(x, angle=degrees, translate=[0, 0], scale=1.0, shear=0, 
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)


def translate(x: torch.Tensor, px_x: int, px_y: int) -> torch.Tensor:
    """Apply translation transformation"""
    return TF.affine(x, angle=0, translate=[px_x, px_y], scale=1.0, shear=0,
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)


def isotropic_scale(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Apply isotropic scaling transformation"""
    return TF.affine(x, angle=0, translate=[0, 0], scale=scale_factor, shear=0,
                     interpolation=TF.InterpolationMode.BILINEAR, fill=0)


# ==================== Module 3: Kappa Metric (Condition Number) ====================

def compute_kappa(model_wrapper, x_batch: torch.Tensor,
                  n_directions: int = 128, eps: float = 1e-3) -> Dict:
    """
    Black-box finite-difference estimate of kappa(J_phi).
    - No autograd; works with JAX adapter.
    - sigma_max: max_v ||J v|| over random unit v in input space
    - sigma_min_lb: min_v ||J v|| over same samples (lower bound)
    """
    B = x_batch.shape[0]
    with torch.no_grad():
        z0 = model_wrapper.encode(x_batch)
        sigmas = []
        for _ in range(n_directions):
            v = torch.randn_like(x_batch)
            v = v / (v.view(B, -1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
            z_plus = model_wrapper.encode(x_batch + eps * v)
            jv = (z_plus - z0) / eps
            sigmas.append(jv.norm(dim=1))  # [B]
        sigmas = torch.stack(sigmas)        # [K, B]
        sigma_max = sigmas.max(dim=0)[0]    # [B]
        sigma_min_lb = sigmas.min(dim=0)[0] # [B]
        kappa_hat = sigma_max / (sigma_min_lb + 1e-8)
        return {
            'kappa_mean': kappa_hat.mean().item(),
            'kappa_p50': kappa_hat.median().item(),
            'kappa_p90': kappa_hat.quantile(0.9).item(),
            'sigma_max_mean': sigma_max.mean().item(),
            'sigma_min_lb_mean': sigma_min_lb.mean().item(),
        }

# ==================== Module 4: LEE Metric (Local Equivariance Error) ====================

def compute_lee(model_wrapper, x_batch: torch.Tensor,
                transform_type: str = 'rotation', eps: float = 5.0) -> float:
    """
    Local Equivariance Error using symmetric finite differences.
    Normalize by generator scale:
      - rotation/translation: denom = 2*eps (deg / pixels)
      - scale:                denom = 2*(eps/100) (fractional)
    """
    if transform_type == 'rotation':
        x_plus = affine_rotate(x_batch, eps)
        x_minus = affine_rotate(x_batch, -eps)
        denom = 2 * eps
    elif transform_type == 'translation':
        x_plus = translate(x_batch, int(eps), 0)
        x_minus = translate(x_batch, -int(eps), 0)
        denom = 2 * eps
    elif transform_type == 'scale':
        x_plus = isotropic_scale(x_batch, 1 + eps / 100.0)
        x_minus = isotropic_scale(x_batch, 1 - eps / 100.0)
        denom = 2 * (eps / 100.0)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    z_plus = model_wrapper.encode(x_plus)
    z_minus = model_wrapper.encode(x_minus)
    lee = torch.norm(z_plus - z_minus, dim=-1) / (denom + 1e-8)
    return lee.mean().item()



def compute_all_lee(model_wrapper, x_batch: torch.Tensor) -> Dict:
    """Compute LEE for all transformation types"""
    return {
        'lee_rotation': compute_lee(model_wrapper, x_batch, 'rotation', eps=5.0),
        'lee_translation': compute_lee(model_wrapper, x_batch, 'translation', eps=2.0),
        'lee_scale': compute_lee(model_wrapper, x_batch, 'scale', eps=5.0),
    }


# ==================== Module 5: TC Metric (Total Correlation) ====================

def compute_tc(z_batch: torch.Tensor, method: str = 'gaussian', eps: float = 1e-3) -> float:
    """
    Gaussian TC with shrinkage; stable when D >> N by using the matrix determinant lemma.
    TC ≈ 0.5 * (sum_j log(var_j + eps) - logdet(Cov + eps I))
    where Cov = (1/(N-1)) Z^T Z, Z is centered latent [N, D].
    Using det lemma: det(eps I_D + 1/(N-1) Z^T Z) = eps^(D-N) det(eps I_N + 1/(N-1) Z Z^T).
    """
    with torch.no_grad():
        Z = z_batch - z_batch.mean(dim=0, keepdim=True)   # [N, D]
        N, D = Z.shape
        var = Z.var(dim=0, unbiased=True) + eps
        term1 = torch.log(var).sum()
        ZZt = (Z @ Z.T) / max(N - 1, 1)                   # [N, N]
        I_N = torch.eye(N, device=Z.device, dtype=Z.dtype)
        sign, logdet_small = torch.linalg.slogdet(ZZt + eps * I_N)
        logdet = (D - N) * torch.log(torch.tensor(eps, device=Z.device)) + logdet_small
        tc = 0.5 * (term1 - logdet)
        return float(tc.item())



# ==================== Module 6: L_F Metric (Dynamics Lipschitz) ====================

def compute_dynamics_lipschitz(model_wrapper, z_batch: torch.Tensor, 
                                a_batch: torch.Tensor, iters: int = 20) -> float:
    """
    Compute local Lipschitz constant of dynamics
    """
    device = z_batch.device
    batch_size = z_batch.shape[0]
    z_dim = z_batch.shape[-1]
    
    # Power iteration for largest singular value
    u = torch.randn(batch_size, z_dim, device=device)
    u = u / (u.norm(dim=1, keepdim=True) + 1e-8)
    
    for _ in range(iters):
        # Compute J u via finite differences
        eps = 1e-3
        z_plus = z_batch + eps * u
        
        f_original = model_wrapper.dynamics_step(z_batch, a_batch)
        f_perturbed = model_wrapper.dynamics_step(z_plus, a_batch)
        
        ju = (f_perturbed - f_original) / eps
        u = ju / (ju.norm(dim=1, keepdim=True) + 1e-8)
    
    lip_constants = ju.norm(dim=1)
    
    return {
        'lf_mean': lip_constants.mean().item(),
        'lf_p50': lip_constants.median().item(),
        'lf_p90': lip_constants.quantile(0.9).item(),
        'lf_contractive_ratio': (lip_constants <= 1.0).float().mean().item(),
    }


# ==================== Module 7: Baseline Metrics ====================

def compute_baseline_metrics(model_wrapper, eval_set: Dict) -> Dict:
    """
    Compute baseline metrics (DeepMDP/bisim style)
    """
    x = eval_set['x']
    a = eval_set['a']
    x_next = eval_set['x_next']
    r = eval_set['r']
    
    # Encode current and next observations
    z = model_wrapper.encode(x)
    z_next_true = model_wrapper.encode(x_next)
    
    # One-step prediction error
    z_next_pred = model_wrapper.dynamics_step(z, a)
    one_step_error = F.mse_loss(z_next_pred, z_next_true).item()
    
    # Reconstruction error
    x_recon = model_wrapper.decode(z)
    recon_error = F.mse_loss(x_recon, x).item()
    
    # Reward prediction error
    r_pred = model_wrapper.reward(z, a)
    reward_error = F.l1_loss(r_pred, r).item()
    
    return {
        'one_step_mse': one_step_error,
        'recon_mse': recon_error,
        'reward_mae': reward_error,
    }


# ==================== Module 8: Statistical Analysis ====================

def compute_spearman_with_ci(x: np.ndarray, y: np.ndarray, 
                             n_bootstrap: int = 1000) -> Dict:
    """
    Compute Spearman correlation with bootstrap confidence intervals
    """
    # Point estimate
    rho, p_value = spearmanr(x, y)
    
    # Bootstrap CI
    def bootstrap_rho(indices):
        return spearmanr(x[indices], y[indices])[0]
    
    rng = np.random.RandomState(42)
    bootstrap_rhos = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(x), len(x), replace=True)
        bootstrap_rhos.append(bootstrap_rho(indices))
    
    ci_lower = np.percentile(bootstrap_rhos, 2.5)
    ci_upper = np.percentile(bootstrap_rhos, 97.5)
    
    return {
        'rho': rho,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }


def permutation_test(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, 
                     n_perm: int = 1000) -> float:
    """
    Permutation test for comparing two correlations
    """
    rho1_orig = spearmanr(x, y1)[0]
    rho2_orig = spearmanr(x, y2)[0]
    delta_orig = abs(rho1_orig - rho2_orig)
    
    rng = np.random.RandomState(42)
    delta_perms = []
    
    for _ in range(n_perm):
        # Randomly swap labels
        swap_mask = rng.random(len(x)) > 0.5
        y1_perm = np.where(swap_mask, y2, y1)
        y2_perm = np.where(swap_mask, y1, y2)
        
        rho1_perm = spearmanr(x, y1_perm)[0]
        rho2_perm = spearmanr(x, y2_perm)[0]
        delta_perms.append(abs(rho1_perm - rho2_perm))
    
    p_value = np.mean(np.array(delta_perms) >= delta_orig)
    return p_value


# ==================== Module 9: Main Analysis Pipeline ====================

class SPWMAnalyzer:
    def __init__(self, ckpt_dir: str, scores_path: str, device: str = 'cuda',
                 adapter: str = 'auto', fast: bool = False, join_tol: int = 5000):
        self.ckpt_dir = Path(ckpt_dir)
        self.scores_path = Path(scores_path)
        # Respect user intent but fallback safely if CUDA not available.
        if device == 'cuda' and not torch.cuda.is_available():
            print('[INFO] CUDA not available, falling back to CPU for tensors.')
            self.device = 'cpu'
        else:
            self.device = device
        self.results_cache = {}
        self.adapter_choice = adapter  # 'auto' | 'dreamer' | 'mock'
        self.fast = fast
        self.join_tol = int(join_tol)
        # Speed knobs; auto reduce for CPU runs to keep runtime reasonable
        cpu_mode = (self.device == 'cpu')
        eff_fast = fast or cpu_mode
        self.batch_size = 64 if eff_fast else 256
        self.kappa_dirs = 8 if eff_fast else 128
        self.lf_iters = 3 if eff_fast else 20

        
    def analyze_checkpoint(self, ckpt_path: str, eval_set: Dict) -> Dict:
        """Analyze single checkpoint"""
        
        # Check cache
        ckpt_hash = hashlib.md5(str(ckpt_path).encode()).hexdigest()
        if ckpt_hash in self.results_cache:
            return self.results_cache[ckpt_hash]
        
        print(f"Analyzing checkpoint: {ckpt_path}")
        
        # Load model - pass directory or file path
        if Path(ckpt_path).is_file():
            ckpt_dir = Path(ckpt_path).parent
        else:
            ckpt_dir = Path(ckpt_path)

        # Build adapter with fallback
        try:
            model = get_model_wrapper(self.adapter_choice, str(ckpt_dir), device=self.device)
        except Exception as e:
            print(f"[WARN] Failed to init adapter ({self.adapter_choice}): {e}")
            model = get_model_wrapper('mock', str(ckpt_dir), device=self.device)

        # Prepare batch - 限制batch大小
        batch_size = min(self.batch_size, len(eval_set['x']))
        x_batch = eval_set['x'][:batch_size].to(self.device)
        a_batch = eval_set['a'][:batch_size].to(self.device)
        
        # Compute all metrics
        results = {}
        
        try:
            # 1. Kappa (condition number)
            results.update(compute_kappa(model, x_batch, n_directions=self.kappa_dirs))
            
            # 2. LEE (equivariance error)
            results.update(compute_all_lee(model, x_batch))
            
            # 3. First encode to get z_batch
            z_batch = model.encode(x_batch)
            # Record latent dimension for optional TC normalization later
            try:
                results['latent_dim'] = int(z_batch.shape[-1])
            except Exception:
                results['latent_dim'] = None
            
            # 4. TC (total correlation)
            results['tc'] = compute_tc(z_batch)
            
            # 5. L_F (dynamics Lipschitz)
            results.update(compute_dynamics_lipschitz(model, z_batch, a_batch, iters=self.lf_iters))
            
            # 6. Baseline metrics - 使用小batch以避免内存问题
            eval_subset = {k: v[:batch_size].to(self.device) for k, v in eval_set.items()}
            results.update(compute_baseline_metrics(model, eval_subset))
            
            # 7. Add adapter info
            results['adapter'] = self.adapter_choice
            
        except Exception as e:
            print(f"Error computing metrics: {e}")
            import traceback as _tb
            print(_tb.format_exc())
            # 返回默认值而不是失败
            results = {
                'kappa_mean': 0, 'kappa_p50': 0, 'kappa_p90': 0,
                'lee_rotation': 0, 'lee_translation': 0, 'lee_scale': 0,
                'tc': 0, 'lf_mean': 0, 'lf_p50': 0, 'lf_p90': 0,
                'one_step_mse': 0, 'recon_mse': 0, 'reward_mae': 0,
                'adapter': 'failed'
            }
        
        # Cache results
        self.results_cache[ckpt_hash] = results
        
        return results
    
    def analyze_all_checkpoints(self, eval_set: Dict) -> pd.DataFrame:
        """Analyze all checkpoints in directory"""
        # If a specific checkpoint directory is passed, scan its siblings.
        if self.ckpt_dir.is_dir() and self.ckpt_dir.name.startswith('checkpoint_'):
            parent = self.ckpt_dir.parent
            ckpt_dirs = sorted([d for d in parent.glob('checkpoint_*') if d.is_dir()])
        else:
            # Otherwise scan subdirectories named checkpoint_*
            ckpt_dirs = sorted([d for d in self.ckpt_dir.glob('checkpoint_*') if d.is_dir()])
        print(f"Found {len(ckpt_dirs)} checkpoint directories")
        if not ckpt_dirs:
            # Try treating provided path as a single checkpoint directory anyway
            if self.ckpt_dir.exists():
                ckpt_dirs = [self.ckpt_dir]
                print(f"Treating {self.ckpt_dir} as single checkpoint.")
            else:
                print(f"No checkpoint directories found in {self.ckpt_dir}")
                return pd.DataFrame()
        
        # Analyze each checkpoint
        all_results = []
        for ckpt_dir in tqdm(ckpt_dirs, desc="Analyzing checkpoints"):
            try:
                # Extract step from directory name
                step = self._extract_step(ckpt_dir)
                
                # Analyze checkpoint
                metrics = self.analyze_checkpoint(str(ckpt_dir), eval_set)
                
                # Add metadata
                metrics['checkpoint'] = str(ckpt_dir)
                metrics['step'] = step
                metrics['timestamp'] = ckpt_dir.stat().st_mtime
                
                all_results.append(metrics)
                
            except Exception as e:
                print(f"Error analyzing {ckpt_dir}: {e}")
                continue
        
        return pd.DataFrame(all_results)
    
    def _extract_step(self, ckpt_path: Path) -> int:
        """Extract training step from checkpoint path"""
        # Try to extract step from filename or parent directory
        name = str(ckpt_path)
        
        # Common patterns - 添加checkpoint_XXXXXXXX模式
        patterns = [
            r'checkpoint_(\d+)',  # 这个模式应该能匹配
            r'step_(\d+)',
            r'(\d+)\.pt',
            r'model_(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                step = int(match.group(1))
                print(f"Extracted step {step} from {ckpt_path.name}")  # 调试信息
                return step
        
        print(f"Could not extract step from {ckpt_path}")  # 调试信息
        # Fallback: use modification time order
        return 0
    
    def correlate_with_performance(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Correlate metrics with performance scores"""
        
        # Load scores
        scores_df = load_scores(self.scores_path)
        
        # 加：对 step 做数值化，避免字符串导致无法 merge
        for df in (metrics_df, scores_df):
            if 'step' in df.columns:
                df['step'] = pd.to_numeric(df['step'], errors='coerce')
        
        # 更鲁棒地寻找"回报"列（在 merge 之前就确定）
        candidate_keys = [
            'episode_reward', 'return', 'score',
            'eval/episode_reward', 'train/episode_reward',
            'eval/return', 'train/return',
            'episode/score', 'eval/score',
        ]
        target_key = next((k for k in candidate_keys if k in scores_df.columns), None)
        if target_key is None:
            print(f"Available columns in scores: {list(scores_df.columns)}")
            raise ValueError(f"No reward/return key found in scores. Tried {candidate_keys}")
        
        print(f"Using performance metric: {target_key}")
        
        # Align by nearest step for robust correlation (scores logged at irregular steps).
        metrics_df_sorted = metrics_df.sort_values('step')
        scores_df_sorted = scores_df.sort_values('step')
        merged_df = pd.merge_asof(
            metrics_df_sorted, scores_df_sorted, on='step', direction='nearest')
        # If still too few points, skip correlation gracefully.
        if len(merged_df) < 2:
            print('[INFO] Not enough matched steps for correlation; skipping.')
            return pd.DataFrame(columns=['metric', 'target', 'rho', 'p_value', 'ci_lower', 'ci_upper'])
        
        # Compute correlations for each metric
        correlation_results = []
        
        metric_cols = [col for col in metrics_df.columns 
                    if col not in ['checkpoint', 'step', 'timestamp']]
        
        for metric in metric_cols:
            if metric in merged_df.columns:
                corr_data = compute_spearman_with_ci(
                    merged_df[metric].values,
                    merged_df[target_key].values
                )
                corr_data['metric'] = metric
                corr_data['target'] = target_key
                correlation_results.append(corr_data)
        
        return pd.DataFrame(correlation_results)
    
    def create_plots(self, metrics_df: pd.DataFrame, corr_df: pd.DataFrame, 
                    output_dir: str = './outputs'):
        """Create analysis plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Metrics over training
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        metric_cols = [col for col in metrics_df.columns 
                      if col not in ['checkpoint', 'step', 'timestamp']]
        
        for idx, metric in enumerate(metric_cols[:9]):
            ax = axes[idx]
            ax.plot(metrics_df['step'], metrics_df[metric])
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} over Training')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/metrics_over_training.png', dpi=150)
        plt.close()
        
        # 2. Correlation comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(corr_df):
            x_pos = np.arange(len(corr_df))
            ax.bar(x_pos, corr_df['rho'], yerr=[
                corr_df['rho'] - corr_df['ci_lower'],
                corr_df['ci_upper'] - corr_df['rho']
            ], capsize=5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(corr_df['metric'], rotation=45, ha='right')
            ax.set_ylabel('Spearman Correlation with Performance')
            ax.set_title('SPWM Metrics vs Performance Correlation')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'No correlation results available', ha='center', va='center')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_comparison.png', dpi=150)
        plt.close()
        
        print(f"Plots saved to {output_dir}")

    # ---------- H1: Unit-consistent RHS & Bound-vs-Gap ----------
    def _compute_unit_rhs(self, df: pd.DataFrame, coeffs, tc_norm: str = 'none', tc_norm_c: float = 1.0):
        c = list(coeffs) if isinstance(coeffs, (list, tuple)) else [coeffs]
        # pad/trim to 3
        if len(c) < 3:
            c = c + [0.0] * (3 - len(c))
        if len(c) > 3:
            c = c[:3]
        c1, c2, c3 = [float(x) for x in c]
        k = df['kappa_mean'].values if 'kappa_mean' in df.columns else 0.0
        t = df['tc'].values if 'tc' in df.columns else 0.0
        # Optional lightweight TC normalization
        if isinstance(t, (int, float)):
            t_arr = np.asarray([float(t)], dtype=float)
        else:
            t_arr = np.asarray(t, dtype=float)
        if tc_norm == 'divd':
            if 'latent_dim' in df.columns:
                d = np.asarray(df['latent_dim'].values, dtype=float)
                d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
                t_arr = t_arr / np.nan_to_num(d, nan=1.0)
            # if latent_dim not present, fall back to no-op
        elif tc_norm == 'exp':
            c = float(tc_norm_c) if tc_norm_c is not None else 1.0
            t_arr = 1.0 - np.exp(-t_arr / max(c, 1e-8))
        t = t_arr
        lf = df['lf_mean'].values if 'lf_mean' in df.columns else 0.0
        rhs = c1 * np.asarray(k, dtype=float) + c2 * np.asarray(t, dtype=float) + c3 * np.asarray(lf, dtype=float)
        return rhs

    def _compute_measured_gap(self, perf, how: str = 'range'):
        perf = np.asarray(perf, dtype=float)
        if how == 'max':
            best = np.nanmax(perf)
            denom = abs(best) + 1e-8
            gap = (best - perf) / denom
        else:
            best = np.nanmax(perf); worst = np.nanmin(perf)
            denom = max(1e-8, float(best - worst))
            gap = (best - perf) / denom
        return gap

    def _beta_stats_and_plots(self, rhs, gap, outdir, tag='uncalibrated'):
        import pandas as pd
        rhs = np.asarray(rhs, dtype=float)
        gap = np.asarray(gap, dtype=float)
        eps = 1e-8
        num = float(np.sum(rhs * gap))
        den = float(np.sum(rhs * rhs)) + eps
        beta_fit = num / den
        yhat = beta_fit * rhs
        ss_res = float(np.sum((gap - yhat) ** 2))
        ss_tot = float(np.sum((gap - float(np.mean(gap))) ** 2)) + eps
        r2 = 1.0 - ss_res / ss_tot
        ratio = gap / np.maximum(rhs, eps)
        beta50 = float(np.quantile(ratio, 0.50))
        beta90 = float(np.quantile(ratio, 0.90))
        beta_star = float(np.max(ratio))
        import os
        # filenames depend on tag for appendix variants
        def fn(base, ext):
            suffix = '' if (tag == 'uncalibrated') else f'_{tag}'
            return os.path.join(outdir, f"{base}{suffix}.{ext}")

        pd.DataFrame([{
            'beta_fit_origin': beta_fit, 'r2_origin': r2,
            'beta_50': beta50, 'beta_90': beta90, 'beta_star': beta_star
        }]).to_csv(fn('bound_vs_gap_beta_stats','csv'), index=False)
        with open(f'{outdir}/bound_vs_gap_note.txt', 'w') as f:
            f.write(f'beta_fit_origin={beta_fit:.6f}, r2_origin={r2:.6f}\n')
            f.write(f'beta_50={beta50:.6f}, beta_90={beta90:.6f}, beta_star={beta_star:.6f}\n')
        grid = np.linspace(0.0, max(1.2 * beta_star, 1.0), 200)
        cover = [(ratio <= b + 1e-12).mean() for b in grid]
        pd.DataFrame({'beta': grid, 'coverage': cover}).to_csv(
            fn('bound_vs_gap_coverage_curve','csv'), index=False)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.scatter(rhs, gap, s=40)
        xmax = float(max(np.max(rhs) * 1.05, 1.0)) if rhs.size else 1.0
        xs = np.linspace(0, xmax, 200)
        plt.plot(xs, xs, '--', label='y=x (no-calibration)')
        plt.plot(xs, beta_fit * xs, label=f'y={beta_fit:.2f} x (fit), R2={r2:.2f}')
        plt.plot(xs, beta50 * xs, label=f'y={beta50:.2f} x (beta50%)')
        plt.plot(xs, beta90 * xs, label=f'y={beta90:.2f} x (beta90%)')
        plt.plot(xs, beta_star * xs, label=f'y={beta_star:.2f} x (beta*)')
        plt.xlabel('Theoretical RHS (unit-consistent)')
        plt.ylabel('Measured performance gap (normalized)')
        plt.title('Bound vs. Measured Performance Gap')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fn('bound_vs_gap_uncalibrated','png'), dpi=200)
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.plot(grid, cover)
        for val in [beta50, beta90, beta_star]:
            yv = float(np.interp(val, grid, cover))
            plt.scatter([val], [yv], s=60)
            plt.text(val, yv + 0.02, f'β≈{val:.2f}', ha='center')
        plt.xlabel('β')
        plt.ylabel('coverage fraction')
        plt.title('Coverage vs β (appendix)')
        plt.tight_layout()
        plt.savefig(fn('bound_vs_gap_coverage_curve','png'), dpi=200)
        plt.close()
        # write a small note for reproducibility
        with open(fn('bound_vs_gap_note','txt'), 'w') as f:
            f.write(f'beta_fit_origin={beta_fit:.6f}, r2_origin={r2:.6f}\n')
            f.write(f'beta_50={beta50:.6f}, beta_90={beta90:.6f}, beta_star={beta_star:.6f}\n')

    # ---------- H2: Rank consistency (tau/rho) with block bootstrap ----------
    def _rank_consistency_and_bootstrap(self, rhs, perf, B=1000, block=2, seed=42, outdir='./results'):
        import pandas as pd
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(seed)
        rhs = np.asarray(rhs, float)
        perf = np.asarray(perf, float)
        tau0 = kendalltau(rhs, perf, nan_policy='omit').correlation
        rho0 = spearmanr(rhs, perf, nan_policy='omit').correlation
        def _block_resample(arr):
            n = len(arr)
            idx = []
            while len(idx) < n:
                start = int(rng.integers(0, max(1, n - block + 1)))
                idx.extend(list(range(start, min(start + block, n))))
            return arr[np.array(idx[:n])]
        boots_tau, boots_rho = [], []
        for _ in range(int(B)):
            rb = _block_resample(rhs)
            pb = _block_resample(perf)
            boots_tau.append(kendalltau(rb, pb, nan_policy='omit').correlation)
            boots_rho.append(spearmanr(rb, pb, nan_policy='omit').correlation)
        tau_ci = (float(np.nanpercentile(boots_tau, 2.5)), float(np.nanpercentile(boots_tau, 97.5)))
        rho_ci = (float(np.nanpercentile(boots_rho, 2.5)), float(np.nanpercentile(boots_rho, 97.5)))
        pd.DataFrame([{
            'B': int(B), 'block': int(block), 'seed': int(seed),
            'tau': float(tau0), 'tau_ci_low': tau_ci[0], 'tau_ci_high': tau_ci[1],
            'rho': float(rho0), 'rho_ci_low': rho_ci[0], 'rho_ci_high': rho_ci[1]
        }]).to_csv(f'{outdir}/rank_consistency_ci.csv', index=False)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(boots_tau, bins=30)
        plt.axvline(0, color='k'); plt.title('Kendall tau bootstrap'); plt.xlabel('tau')
        plt.subplot(1, 2, 2)
        plt.hist(boots_rho, bins=30)
        plt.axvline(0, color='k'); plt.title('Spearman rho bootstrap'); plt.xlabel('rho')
        plt.tight_layout()
        plt.savefig(f'{outdir}/rank_consistency_boot_hist.png', dpi=200)
        plt.close()


# ==================== Module 10: CLI Interface ====================

def main():
    parser = argparse.ArgumentParser(description='SPWM Metrics Analysis for DreamerV3')
    parser.add_argument('--adapter', type=str, default='dreamer',
                    choices=['auto', 'dreamer', 'mock'],
                    help='Adapter backend: use mock to guarantee quick runs')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: use smaller batches/iters to speed up')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['create_eval', 'analyze', 'full'],
                       help='Mode of operation')
    parser.add_argument('--ckpt_dir', type=str, 
                       default='/hy-tmp/SPWM/logdir/crafter_5M_20250914_103133',
                       help='Directory containing checkpoints')
    parser.add_argument('--scores_path', type=str,
                       default='/hy-tmp/SPWM/logdir/crafter_5M_20250914_103133/scores.jsonl',
                       help='Path to scores JSONL file')
    parser.add_argument('--eval_set_path', type=str, default='./eval_set.npz',
                       help='Path to evaluation set')
    parser.add_argument('--n_eval_samples', type=int, default=5000,
                       help='Number of samples in evaluation set')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    # Join tolerance for aligning steps (in env steps)
    parser.add_argument('--join_tol', type=int, default=5000,
                        help='Tolerance for nearest step alignment (abs diff <= tol)')
    # M1: Fixed-effects / partial analysis controls
    parser.add_argument('--controlled', action='store_true',
                        help='Run fixed-effects/partial analysis controlling step & baseline')
    # M2: Early->Late lag analysis
    parser.add_argument('--lag', action='store_true',
                        help='Run early->late (t->t+Δ) Spearman with bootstrap CI')
    parser.add_argument('--lag_early_max', type=int, default=200000)
    parser.add_argument('--lag_final_min', type=int, default=500000)
    parser.add_argument('--lag_delta', type=int, default=-1,
                        help='Explicit lag Δ; if <0, use (final_min - early_max)')
    parser.add_argument('--lag_avg_window', type=int, default=0,
                        help='Window half-width around t+Δ to average scores (0=off)')
    parser.add_argument('--lag_block', type=int, default=2,
                        help='Block length for block bootstrap (>=1)')
    # N1: Dependent correlation compare + FDR
    parser.add_argument('--compare', action='store_true',
                        help='Compare dependent correlations (SPWM vs Baseline) and control FDR')
    # N2: Early checkpoint selection utility
    parser.add_argument('--select', action='store_true',
                        help='Run early checkpoint selection utility (Top-K regret)')
    parser.add_argument('--select_metric', type=str, default='kappa_mean')
    parser.add_argument('--select_topk', type=int, default=3)
    parser.add_argument('--select_early_max', type=int, default=200000)
    # N3: Toy MDP demo
    parser.add_argument('--toy', action='store_true',
                        help='Run N3 toy MDP demo and save plots')
    parser.add_argument('--toy_n', type=int, default=5000)
    parser.add_argument('--toy_noise', type=float, default=0.05)
    parser.add_argument('--toy_seed', type=int, default=0)
    # H1/H2: Hard no-calibration bound and rank consistency
    parser.add_argument('--hard1_bound_plot', action='store_true',
                        help='Plot uncalibrated bound (beta=1) vs measured performance gap')
    parser.add_argument('--hard2_rank_consistency', action='store_true',
                        help='Report Kendall tau / Spearman rho with block-bootstrap CI')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Discount factor; if None, try to read from config.yaml, else fallback to 0.99')
    parser.add_argument('--reward_scale', type=float, default=1.0,
                        help='Reward range normalization (max-min). If scores already normalized, keep 1.0')
    parser.add_argument('--lp_mode', type=str, default='conservative',
                        choices=['conservative','from_lf'],
                        help='How to set L_P: conservative=1.0; from_lf=normalize using lf_mean')
    # ---------- Bound vs Gap / rank consistency ----------
    parser.add_argument('--rhs_coeffs', type=str, default='1.0,1.0,1.0',
                        help='Unit-consistent RHS coefficients for [kappa_mean, tc, lf_mean], e.g., "1.0,1.0,1.0"')
    parser.add_argument('--gap_norm', type=str, default='range',
                        choices=['range', 'max'], help='How to normalize measured performance gap.')
    parser.add_argument('--bootstrap_B', type=int, default=1000,
                        help='Number of bootstrap resamples for rank consistency.')
    parser.add_argument('--bootstrap_block', type=int, default=2,
                        help='Block length for block bootstrap (temporal dependence).')
    parser.add_argument('--bootstrap_seed', type=int, default=42,
                        help='Random seed for bootstrap.')
    # TC lightweight normalization for appendix plot
    parser.add_argument('--tc_norm', type=str, default='none',
                        choices=['none','divd','exp'],
                        help='Optional TC normalization in RHS for appendix plot: none, divide by latent dim (TC/d), or 1-exp(-TC/c).')
    parser.add_argument('--tc_norm_c', type=float, default=1.0,
                        help='c for TC normalization when tc_norm=exp (1-exp(-TC/c)).')
    # Ablations and null distribution
    parser.add_argument('--ablation_alpha', type=float, default=1.0,
                        help='Alpha coefficient for TC in ablation RHS.')
    parser.add_argument('--ablation_beta', type=float, default=1.0,
                        help='Beta coefficient for LEE in ablation RHS.')
    parser.add_argument('--rhs_permute', type=int, default=0,
                        help='Number of random pseudo-RHS draws for null beta* distribution (0 to disable).')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'create_eval':
        # Create evaluation set
        print("Creating evaluation set...")
        eval_set = create_eval_set(args.ckpt_dir, args.n_eval_samples)
        
        # Save to file
        np.savez(args.eval_set_path, 
                 x=eval_set['x'].numpy(),
                 a=eval_set['a'].numpy(),
                 x_next=eval_set['x_next'].numpy(),
                 r=eval_set['r'].numpy())
        print(f"Evaluation set saved to {args.eval_set_path}")
        
    elif args.mode == 'analyze':
        # Load evaluation set
        if not os.path.exists(args.eval_set_path):
            print(f"[INFO] Eval set not found at {args.eval_set_path}. Creating a new one...")
            eval_set = create_eval_set(args.ckpt_dir, args.n_eval_samples)
            np.savez(args.eval_set_path,
                     x=eval_set['x'].numpy(),
                     a=eval_set['a'].numpy(),
                     x_next=eval_set['x_next'].numpy(),
                     r=eval_set['r'].numpy())
        print("Loading evaluation set...")
        eval_set = load_eval_set(args.eval_set_path)
        
        # Create analyzer
        analyzer = SPWMAnalyzer(args.ckpt_dir, args.scores_path, args.device,
                        adapter=args.adapter, fast=args.fast, join_tol=args.join_tol)
        
        # Analyze all checkpoints
        print("Analyzing checkpoints...")
        metrics_df = analyzer.analyze_all_checkpoints(eval_set)
        
        # Save metrics
        metrics_path = f'{args.output_dir}/metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")
        
        # Compute correlations
        print("Computing correlations with performance...")
        corr_df = analyzer.correlate_with_performance(metrics_df)
        
        # Save correlations
        corr_path = f'{args.output_dir}/correlations.csv'
        corr_df.to_csv(corr_path, index=False)
        print(f"Correlations saved to {corr_path}")
        
        # Create plots
        print("Creating visualization plots...")
        analyzer.create_plots(metrics_df, corr_df, args.output_dir)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Checkpoints analyzed: {len(metrics_df)}")
        print(f"\nTop correlated metrics with performance:")
        print(corr_df.sort_values('rho', ascending=False)[['metric', 'rho', 'p_value']].head())
        # M1: Fixed effects
        if args.controlled:
            try:
                metrics_path = f'{args.output_dir}/metrics.csv'
                summ, delta_r2 = run_fixed_effects(metrics_path, args.scores_path, tol=args.join_tol)
                print("\n=== FIXED EFFECTS (HC3) ===\n", summ)
                print(f"\nΔR^2 (add kappa+tc+lf over baseline+step): {delta_r2:.4f}")
                # Save results
                with open(f"{args.output_dir}/fixed_effects_summary.txt", 'w') as f:
                    f.write(summ)
                with open(f"{args.output_dir}/fixed_effects_delta_r2.csv", 'w') as f:
                    f.write('delta_r2\n')
                    f.write(f'{delta_r2:.6f}\n')
            except Exception as e:
                print('[WARN] Fixed-effects analysis failed:', e)
                print(traceback.format_exc())
        # M2: Lagged correlation
        if args.lag:
            try:
                # New: evaluate multiple deltas [150k, 200k] and early<=250k using K=3 windowed targets
                df_lag = compute_lagged_correlation(
                    f'{args.output_dir}/metrics.csv', args.scores_path,
                    early_max=250000,
                    deltas=[150000, 200000],
                    tol=args.join_tol,
                    K=3,
                    block=args.lag_block,
                )
                print("\n=== EARLY -> FINAL (Lag) ===\n", df_lag)
                df_lag.to_csv(f'{args.output_dir}/lagged_corr.csv', index=False)
            except Exception as e:
                print('[WARN] Lagged correlation failed:', e)
                print(traceback.format_exc())
        # N1: Compare dependent correlations + FDR
        if args.compare:
            try:
                import pandas as pd
                m = pd.read_csv(f'{args.output_dir}/metrics.csv')
                s = load_scores(args.scores_path)
                ykey = next(k for k in ['episode_reward','return','score','eval/episode_reward',
                                        'train/episode_reward','eval/return','train/return',
                                        'episode/score','eval/score'] if k in s.columns)
                ms = m.copy(); ss = s[['step', ykey]].copy()
                ms['step'] = ms['step'].astype(int); ss['step'] = ss['step'].astype(int)
                ss = ss.sort_values('step').assign(s_step=ss['step'])
                df = pd.merge_asof(ms.sort_values('step'), ss, on='step', direction='nearest')
                df = df[(df['s_step'] - df['step']).abs() <= args.join_tol].dropna()
                tests = []
                pairs = [('kappa_mean','one_step_mse'), ('tc','one_step_mse'), ('lf_mean','one_step_mse')]
                for a, b in pairs:
                    try:
                        # Steiger/Williams dependent correlation test
                        from scipy.stats import pearsonr
                        r_xy = pearsonr(df[a].values, df[ykey].values)[0]
                        r_xz = pearsonr(df[b].values, df[ykey].values)[0]
                        r_yz = pearsonr(df[a].values, df[b].values)[0]
                        z, p = compare_dependent_corr(r_xy, r_xz, r_yz, n=len(df))
                        tests.append({'metric':a, 'baseline':b, 'z':z, 'p_value':p,
                                      'r_xy': float(r_xy), 'r_xz': float(r_xz), 'r_yz': float(r_yz), 'n': int(len(df))})
                    except Exception:
                        tests.append({'metric':a, 'baseline':b, 'z':float('nan'), 'p_value':1.0,
                                      'r_xy': float('nan'), 'r_xz': float('nan'), 'r_yz': float('nan'), 'n': int(len(df))})
                res = pd.DataFrame(tests)
                mask, cutoff = benjamini_hochberg(res['p_value'].values, alpha=0.05)
                res['fdr_pass'] = mask
                res.to_csv(f'{args.output_dir}/corr_compare_fdr.csv', index=False)
                print("\n=== DEPENDENT CORR COMPARE (with BH-FDR) ===\n", res)
            except Exception as e:
                print('[WARN] Correlation compare failed:', e)
                print(traceback.format_exc())
        # N2: Early checkpoint selection utility
        if args.select:
            try:
                out = checkpoint_selection_utility(
                    f'{args.output_dir}/metrics.csv', args.scores_path,
                    select_metric=args.select_metric,
                    topk=args.select_topk,
                    early_max=250000,
                    early_min=50000,
                    K=3,
                )
                print("\n=== EARLY SELECTION (Top-K Regret) ===\n", out)
                # Save selection utility results
                import pandas as pd
                pd.DataFrame([out]).to_csv(f"{args.output_dir}/selection_utility.csv", index=False)
            except Exception as e:
                print('[WARN] Selection utility failed:', e)
                print(traceback.format_exc())
        # N3: Toy MDP demo plots
        if args.toy:
            try:
                alphas, kappas, perf = toy_mdp_demo(n=args.toy_n, noise=args.toy_noise, seed=args.toy_seed)
                # Plot performance vs kappa
                plt.figure(figsize=(6,4))
                plt.plot(kappas, perf, '-o')
                plt.xlabel('kappa (condition number)')
                plt.ylabel('toy performance (-MSE)')
                plt.title('Toy MDP: Performance vs Kappa')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/toy_kappa_perf.png", dpi=150)
                plt.close()
                # Plot alpha vs kappa
                plt.figure(figsize=(6,4))
                plt.plot(alphas, kappas, '-o')
                plt.xlabel('alpha (scaling parameter)')
                plt.ylabel('kappa (condition number)')
                plt.title('Toy MDP: Alpha -> Kappa')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/toy_alpha_kappa.png", dpi=150)
                plt.close()
                # Plot alpha vs performance
                plt.figure(figsize=(6,4))
                plt.plot(alphas, perf, '-o')
                plt.xlabel('alpha (scaling parameter)')
                plt.ylabel('toy performance (-MSE)')
                plt.title('Toy MDP: Alpha -> Performance')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/toy_alpha_perf.png", dpi=150)
                plt.close()
                print(f"Toy MDP plots saved to {args.output_dir}")
            except Exception as e:
                print('[WARN] Toy MDP demo failed:', e)
                print(traceback.format_exc())

        # ---------- New: Bound-vs-Gap + Rank Consistency (always run) ----------
        try:
            import pandas as pd
            coeffs = [float(x) for x in str(args.rhs_coeffs).split(',') if x.strip()]
            # Align performance to metrics steps via nearest-asof with tolerance
            s = load_scores(args.scores_path)
            ykey = next(k for k in ['episode_reward','return','score','eval/episode_reward',
                                    'train/episode_reward','eval/return','train/return',
                                    'episode/score','eval/score'] if k in s.columns)
            ms = metrics_df.copy(); ss = s[['step', ykey]].copy()
            ms['step'] = pd.to_numeric(ms['step'], errors='coerce').astype('Int64')
            ss['step'] = pd.to_numeric(ss['step'], errors='coerce').astype('Int64')
            ms = ms.dropna(subset=['step']).astype({'step': int})
            ss = ss.dropna(subset=['step']).astype({'step': int})
            ss = ss.sort_values('step').assign(s_step=ss['step'])
            dfh = pd.merge_asof(ms.sort_values('step'), ss, on='step', direction='nearest')
            dfh = dfh[(dfh['s_step'] - dfh['step']).abs() <= args.join_tol].dropna(subset=[ykey])
            # Compute RHS and GAP
            rhs_vec = analyzer._compute_unit_rhs(dfh, coeffs)
            perf_vec = dfh[ykey].values.astype(float)
            gap_vec = analyzer._compute_measured_gap(perf_vec, how=args.gap_norm)
            # Outputs
            analyzer._beta_stats_and_plots(rhs_vec, gap_vec, args.output_dir, tag='uncalibrated')
            analyzer._rank_consistency_and_bootstrap(
                rhs_vec, perf_vec,
                B=args.bootstrap_B, block=args.bootstrap_block, seed=args.bootstrap_seed,
                outdir=args.output_dir,
            )
            # Appendix: Bound–Gap with normalized TC (optional)
            if args.tc_norm != 'none':
                rhs_tc = analyzer._compute_unit_rhs(dfh, coeffs, tc_norm=args.tc_norm, tc_norm_c=args.tc_norm_c)
                analyzer._beta_stats_and_plots(rhs_tc, gap_vec, args.output_dir, tag='tc_norm')

            # Ablation study: remove one component each among {kappa, TC, LEE}
            try:
                import numpy as _np
                kappa = dfh['kappa_mean'].values.astype(float) if 'kappa_mean' in dfh.columns else _np.zeros(len(dfh))
                tc = dfh['tc'].values.astype(float) if 'tc' in dfh.columns else _np.zeros(len(dfh))
                # Aggregate LEE
                lee_cols = [c for c in ['lee_rotation','lee_translation','lee_scale'] if c in dfh.columns]
                if lee_cols:
                    lee = _np.nanmean(dfh[lee_cols].values.astype(float), axis=1)
                else:
                    lee = _np.zeros(len(dfh))
                a, b = float(args.ablation_alpha), float(args.ablation_beta)
                eps = 1e-12
                rows = []
                def beta_star_and_cov(x_rhs, y_gap):
                    x = _np.asarray(x_rhs, float)
                    y = _np.asarray(y_gap, float)
                    ratio = y / _np.clip(x, eps, None)
                    beta_star = float(_np.max(ratio))
                    cov1 = float(_np.mean(y <= x + eps))  # coverage at beta=1
                    return beta_star, cov1
                # RHS-κ = a*TC + b*LEE
                rhs_mk = a * tc + b * lee
                bs_mk, cov_mk = beta_star_and_cov(rhs_mk, gap_vec)
                rows.append({'variant':'RHS-minus-kappa', 'beta_star': bs_mk, 'coverage_at_1': cov_mk, 'alpha': a, 'beta': b})
                # RHS-TC = κ + b*LEE
                rhs_mt = kappa + b * lee
                bs_mt, cov_mt = beta_star_and_cov(rhs_mt, gap_vec)
                rows.append({'variant':'RHS-minus-tc', 'beta_star': bs_mt, 'coverage_at_1': cov_mt, 'alpha': a, 'beta': b})
                # RHS-LEE = κ + a*TC
                rhs_ml = kappa + a * tc
                bs_ml, cov_ml = beta_star_and_cov(rhs_ml, gap_vec)
                rows.append({'variant':'RHS-minus-lee', 'beta_star': bs_ml, 'coverage_at_1': cov_ml, 'alpha': a, 'beta': b})
                # Save ablation stats
                pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'beta_stats_ablation.csv'), index=False)
            except Exception as e2:
                print('[WARN] Ablation stats failed:', e2)

            # Null distribution by random pseudo-RHS
            try:
                N = int(max(0, args.rhs_permute))
                if N > 0:
                    rng = np.random.default_rng(args.bootstrap_seed)
                    kappa = dfh['kappa_mean'].values.astype(float) if 'kappa_mean' in dfh.columns else np.zeros(len(dfh))
                    tc = dfh['tc'].values.astype(float) if 'tc' in dfh.columns else np.zeros(len(dfh))
                    lee_cols = [c for c in ['lee_rotation','lee_translation','lee_scale'] if c in dfh.columns]
                    if lee_cols:
                        lee = np.nanmean(dfh[lee_cols].values.astype(float), axis=1)
                    else:
                        lee = np.zeros(len(dfh))
                    # true RHS combining kappa + a*TC + b*LEE
                    a, b = float(args.ablation_alpha), float(args.ablation_beta)
                    rhs_true = kappa + a * tc + b * lee
                    eps = 1e-12
                    beta_star_true = float(np.max(gap_vec / np.clip(rhs_true, eps, None)))
                    null = []
                    for _ in range(N):
                        w = rng.random(3)
                        w = w / max(w.sum(), 1e-8)
                        kshuf = rng.permutation(kappa)
                        tcshuf = rng.permutation(tc)
                        leeshuf = rng.permutation(lee)
                        rhs_null = w[0]*kshuf + w[1]*tcshuf + w[2]*leeshuf
                        beta_star_null = float(np.max(gap_vec / np.clip(rhs_null, eps, None)))
                        null.append({'beta_star': beta_star_null, 'w_kappa': float(w[0]), 'w_tc': float(w[1]), 'w_lee': float(w[2])})
                    nd = pd.DataFrame(null)
                    nd.to_csv(os.path.join(args.output_dir, 'beta_star_null_dist.csv'), index=False)
                    # violin plot comparing null dist and true beta*
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(5,4))
                    parts = plt.violinplot([nd['beta_star'].values], showmeans=True, showextrema=True)
                    for pc in parts['bodies']:
                        pc.set_facecolor('#1f77b4'); pc.set_alpha(0.6)
                    plt.axhline(beta_star_true, color='r', linestyle='--', label=f'true beta* = {beta_star_true:.2f}')
                    plt.ylabel('beta*')
                    plt.xticks([1], ['null'])
                    plt.title('Null distribution of beta* (random pseudo-RHS)')
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'beta_star_null_dist.png'), dpi=200)
                    plt.close()
            except Exception as e3:
                print('[WARN] RHS null distribution failed:', e3)
        except Exception as e:
            print('[WARN] Auto Bound/Rank generation failed:', e)
            print(traceback.format_exc())

        # H1/H2: Bound-vs-Error scatter + Rank consistency with block bootstrap
        if args.hard1_bound_plot or args.hard2_rank_consistency:
            try:
                import pandas as pd
                from scipy import stats as _stats
                # Helper: load config.yaml and read discount if not provided
                def _load_yaml_discount(ckpt_dir: str) -> float | None:
                    paths = []
                    p = Path(ckpt_dir)
                    paths.append(p / 'config.yaml')
                    paths.append(p.parent / 'config.yaml')
                    for q in paths:
                        if q.exists():
                            try:
                                import yaml  # type: ignore
                                with open(q, 'r') as f:
                                    cfg = yaml.safe_load(f) or {}
                                val = cfg.get('discount', None)
                                if val is None and 'agent' in cfg and isinstance(cfg['agent'], dict):
                                    val = cfg['agent'].get('discount', None)
                                return float(val) if val is not None else None
                            except Exception:
                                continue
                    return None
                # Determine gamma
                gamma = args.gamma
                if gamma is None:
                    g = _load_yaml_discount(args.ckpt_dir)
                    gamma = g if g is not None else 0.99
                gamma = float(gamma)
                print(f"[H1/H2] Using gamma={gamma:.4f}")
                # Load metrics and scores
                m = metrics_df.copy()
                s = load_scores(args.scores_path)
                ykey = next(k for k in ['episode_reward','return','score','eval/episode_reward',
                                        'train/episode_reward','eval/return','train/return',
                                        'episode/score','eval/score'] if k in s.columns)
                ms = m.copy(); ss = s[['step', ykey]].copy()
                ms['step'] = pd.to_numeric(ms['step'], errors='coerce').astype('Int64')
                ss['step'] = pd.to_numeric(ss['step'], errors='coerce').astype('Int64')
                ms = ms.dropna(subset=['step']).astype({'step': int})
                ss = ss.dropna(subset=['step']).astype({'step': int})
                ss = ss.sort_values('step').assign(s_step=ss['step'])
                df = pd.merge_asof(ms.sort_values('step'), ss, on='step', direction='nearest')
                df = df[(df['s_step'] - df['step']).abs() <= args.join_tol].dropna()
                # Require key columns
                for col in ['kappa_mean','tc','lf_mean', ykey]:
                    if col not in df.columns:
                        raise ValueError(f"[H1/H2] Missing column: {col}")
                # Robust min-max
                def robust_minmax(x: np.ndarray) -> np.ndarray:
                    lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
                    return np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
                kappa_n = robust_minmax(df['kappa_mean'].values.astype(float))
                tc_n    = robust_minmax(df['tc'].values.astype(float))
                lf_n    = robust_minmax(df['lf_mean'].values.astype(float))
                # L_P selection
                if args.lp_mode == 'from_lf':
                    lp_hat = float(np.clip(np.nanmean(lf_n), 0.5, 1.0))
                else:
                    lp_hat = 1.0
                C1 = float(args.reward_scale) / max(1e-8, (1.0 - gamma * lp_hat))
                rhs = C1 * kappa_n + 1.0 * tc_n + 1.0 * lf_n
                # Measured performance and normalized gap
                perf = df[ykey].values.astype(float)
                best = float(np.nanmax(perf))
                worst = float(np.nanmin(perf))
                denom = max(best - worst, 1e-8)
                gap = (best - perf) / denom
                # Mask invalid
                mask = np.isfinite(rhs) & np.isfinite(gap)
                x = rhs[mask]
                y = gap[mask]
                if args.hard1_bound_plot and len(x) >= 2:
                    # Through-origin slope and R^2
                    beta = float(np.dot(x, y) / max(np.dot(x, x), 1e-8))
                    y_hat = beta * x
                    ss_res = float(np.sum((y - y_hat) ** 2))
                    ss_tot = float(np.sum((y - y.mean()) ** 2) + 1e-8)
                    r2 = 1.0 - ss_res / ss_tot
                    # Compute beta statistics on the exact same (x,y)
                    eps = 1e-12
                    ratio = y / np.clip(x, eps, None)
                    beta_star = float(np.max(ratio))
                    beta_50   = float(np.quantile(ratio, 0.50))
                    beta_90   = float(np.quantile(ratio, 0.90))
                    # Plot with coverage lines and uncalibrated line
                    plt.figure(figsize=(6, 6))
                    plt.scatter(x, y, alpha=0.7)
                    xmax = float(np.max(x)) if np.isfinite(np.max(x)) and np.max(x) > 0 else 1.0
                    grid = np.linspace(0, xmax, 200)
                    plt.plot(grid, grid, '--', label='y=x (no-calibration)')
                    plt.plot(grid, beta * grid, '-', label=f'y={beta:.2f} x (fit through origin), R2={r2:.2f}')
                    # Coverage lines based on data: beta50, beta90, beta*
                    beta_list = [
                        (beta_50,  'beta50%'),
                        (beta_90,  'beta90%'),
                        (beta_star,'beta* (cover all)'),
                    ]
                    cover_rows = []
                    for b, blabel in beta_list:
                        plt.plot(grid, b * grid, '-', alpha=0.8, label=f'y={b:.2f} x ({blabel})')
                        cov = float(np.mean(y <= b * x + 1e-12)) if len(x) else float('nan')
                        cover_rows.append({'beta': b, 'label': blabel, 'coverage_frac': cov})
                    plt.xlabel('Theoretical RHS (unit-consistent)')
                    plt.ylabel('Measured performance gap (normalized)')
                    plt.legend()
                    plt.title('Bound vs. Measured Performance Gap (uncalibrated)')
                    plt.tight_layout()
                    out_png = os.path.join(args.output_dir, 'bound_vs_gap_uncalibrated.png')
                    plt.savefig(out_png, dpi=200)
                    plt.close()
                    # CSV
                    pd.DataFrame({'rhs': x, 'gap': y, 'yhat': y_hat}).to_csv(
                        os.path.join(args.output_dir, 'bound_vs_gap_uncalibrated.csv'), index=False)
                    # Coverage summary CSV and note
                    cov_df = pd.DataFrame(cover_rows)
                    cov_df['beta_hat_fit'] = beta
                    cov_df['r2_fit'] = r2
                    cov_df.to_csv(os.path.join(args.output_dir, 'bound_vs_gap_coverage.csv'), index=False)
                    with open(os.path.join(args.output_dir, 'bound_vs_gap_note.txt'), 'w') as f:
                        f.write(f'beta* (100% 覆盖) = {beta_star:.4f}; '
                                f'beta50% = {beta_50:.4f}; beta90% = {beta_90:.4f}.\n')
                        f.write(f'过原点拟合斜率 beta_fit={beta:.4f}, R2={r2:.4f}.\n')
                    # Save beta statistics for audit
                    pd.DataFrame([{
                        'beta_fit': beta,
                        'R2': r2,
                        'beta_50': beta_50,
                        'beta_90': beta_90,
                        'beta_star': beta_star,
                        'n': int(len(x)),
                        'x_min': float(np.min(x)), 'x_max': float(np.max(x)),
                        'y_min': float(np.min(y)), 'y_max': float(np.max(y)),
                    }]).to_csv(os.path.join(args.output_dir, 'bound_vs_gap_beta_stats.csv'), index=False)
                    print(f"[H1] Saved plot and CSV. beta={beta:.4f}, R2={r2:.4f}")
                    # Coverage-vs-beta curve for appendix (same x,y)
                    beta_grid = np.linspace(0.0, max(1.2, 1.2*beta_star), 200)
                    cov_curve = [float(np.mean(y <= b * x + 1e-12)) for b in beta_grid]
                    curve_df = pd.DataFrame({'beta': beta_grid, 'coverage': cov_curve})
                    curve_df.to_csv(os.path.join(args.output_dir, 'bound_vs_gap_coverage_curve.csv'), index=False)
                    # Use quantile-based betas for 50%, 90%, and coverage beta* for 100%.
                    b50, b90, b100 = beta_50, beta_90, beta_star
                    # Appendix plot
                    plt.figure(figsize=(5, 4))
                    plt.plot(beta_grid, cov_curve, label='coverage vs beta')
                    for bx, ty in [(b50, 0.5), (b90, 0.9), (b100, 1.0)]:
                        if np.isfinite(bx):
                            plt.scatter([bx], [ty], zorder=3)
                            plt.text(bx, ty, f'  β≈{bx:.2f}', va='bottom')
                    plt.xlabel('β')
                    plt.ylabel('coverage fraction')
                    plt.ylim(-0.02, 1.02)
                    plt.title('Coverage vs β (appendix)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'bound_vs_gap_coverage_curve.png'), dpi=200)
                    plt.close()
                if args.hard2_rank_consistency and len(x) >= 3:
                    # Rank consistency of -rhs (lower better) vs perf (higher better)
                    tau, tau_p = _stats.kendalltau(-x, perf[mask])
                    rho, rho_p = _stats.spearmanr(-x, perf[mask])
                    # Moving Block Bootstrap (block=2)
                    def mbb_indices(n: int, block: int = 2, rng=None) -> np.ndarray:
                        idx = []
                        while len(idx) < n:
                            if rng is None:
                                start = np.random.randint(0, max(n - block + 1, 1))
                            else:
                                start = rng.randint(0, max(n - block + 1, 1))
                            idx.extend(range(start, min(start + block, n)))
                        return np.array(idx[:n])
                    B = 2000
                    seed = 42
                    rng = np.random.RandomState(seed)
                    taus, rhos = [], []
                    n = len(x)
                    for _ in range(B):
                        ii = mbb_indices(n, block=2, rng=rng)
                        taus.append(_stats.kendalltau(-x[ii], perf[mask][ii]).correlation)
                        rhos.append(_stats.spearmanr(-x[ii], perf[mask][ii]).correlation)
                    def ci(a, alpha=0.05):
                        lo, hi = np.nanpercentile(a, 2.5), np.nanpercentile(a, 97.5)
                        return float(lo), float(hi)
                    tau_lo, tau_hi = ci(taus)
                    rho_lo, rho_hi = ci(rhos)
                    res_ci_df = pd.DataFrame({
                        'tau': [float(tau)], 'tau_lo': [tau_lo], 'tau_hi': [tau_hi],
                        'rho': [float(rho)], 'rho_lo': [rho_lo], 'rho_hi': [rho_hi],
                        'block': [2], 'B': [B], 'seed': [seed]
                    })
                    res_ci_df.to_csv(os.path.join(args.output_dir, 'rank_consistency_ci.csv'), index=False)
                    # Bar plot with error bars and caption text
                    plt.figure(figsize=(6, 4))
                    vals = np.array([float(tau), float(rho)])
                    lowers = np.array([float(tau) - tau_lo, float(rho) - rho_lo])
                    uppers = np.array([tau_hi - float(tau), rho_hi - float(rho)])
                    yerr = np.vstack([lowers, uppers])
                    xs = np.arange(2)
                    # Pastel colors for tau / rho bars.
                    plt.bar(xs, vals, yerr=yerr, capsize=6, color=['#BDE4FF', '#DEF2D9'])
                    # Use mathtext for tau / rho while keeping style identical, and enlarged tick fonts.
                    plt.xticks(xs, ['Kendall $\\tau$', 'Spearman $\\rho$'], fontsize=16)
                    plt.yticks(fontsize=16)
                    plt.axhline(0.0, color='k', lw=0.8, alpha=0.6)
                    # Main CI plot title at 16pt.
                    plt.title('Rank Consistency (block bootstrap CI, block=2)', fontsize=16)
                    caption = (
                        f"$\\tau$={float(tau):.3f} [{tau_lo:.3f},{tau_hi:.3f}], "
                        f"$\\rho$={float(rho):.3f} [{rho_lo:.3f},{rho_hi:.3f}]"
                    )
                    plt.legend([caption], loc='lower right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'rank_consistency_ci.png'), dpi=200)
                    plt.close()
                    # Bootstrap histogram of taus and rhos (appendix)
                    plt.figure(figsize=(8, 3.5))
                    ax1 = plt.subplot(1, 2, 1)
                    ax1.hist(taus, bins=30, color='tab:blue', alpha=0.8)
                    ax1.axvline(0.0, color='k', lw=0.8)
                    ax1.set_title('Kendall $\\tau$ bootstrap', fontsize=14)
                    ax1.set_xlabel(r"$\tau$", fontsize=14); ax1.set_ylabel('count', fontsize=14)
                    ax1.tick_params(axis='both', labelsize=12)
                    ax2 = plt.subplot(1, 2, 2)
                    ax2.hist(rhos, bins=30, color='tab:orange', alpha=0.8)
                    ax2.axvline(0.0, color='k', lw=0.8)
                    ax2.set_title('Spearman $\\rho$ bootstrap', fontsize=14)
                    ax2.set_xlabel(r"$\rho$", fontsize=14)
                    ax2.tick_params(axis='both', labelsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.output_dir, 'rank_consistency_boot_hist.png'), dpi=200)
                    plt.close()
                    # Note text with methodological sentence
                    note = (
                        '我们遵循依赖数据的重抽样规范，采用 block-bootstrap（block length=2）；'
                        'τ、ρ 的正方向与理论一致，但由于样本量与依赖结构，CI 仍覆盖 0。\n'
                        f"数值：{caption}\n"
                    )
                    with open(os.path.join(args.output_dir, 'rank_consistency_note.txt'), 'w') as f:
                        f.write(note)
                    print(f"[H2] {caption}")
            except Exception as e:
                print('[WARN] H1/H2 computation failed:', e)
                print(traceback.format_exc())
        
    elif args.mode == 'full':
        # Run complete pipeline
        print("Running full analysis pipeline...")
        
        # Step 1: Create eval set if not exists
        if not os.path.exists(args.eval_set_path):
            print("Creating evaluation set...")
            eval_set = create_eval_set(args.ckpt_dir, args.n_eval_samples)
            np.savez(args.eval_set_path,
                     x=eval_set['x'].numpy(),
                     a=eval_set['a'].numpy(),
                     x_next=eval_set['x_next'].numpy(),
                     r=eval_set['r'].numpy())
        else:
            print("Loading existing evaluation set...")
            eval_set = load_eval_set(args.eval_set_path)
        
        # Step 2: Analyze
        analyzer = SPWMAnalyzer(args.ckpt_dir, args.scores_path, args.device,
                        adapter=args.adapter, fast=args.fast, join_tol=args.join_tol)

        metrics_df = analyzer.analyze_all_checkpoints(eval_set)
        metrics_df.to_csv(f'{args.output_dir}/metrics.csv', index=False)
        
        # Step 3: Correlate
        corr_df = analyzer.correlate_with_performance(metrics_df)
        corr_df.to_csv(f'{args.output_dir}/correlations.csv', index=False)
        
        # Step 4: Plot
        analyzer.create_plots(metrics_df, corr_df, args.output_dir)
        
        print("\nAnalysis complete! Check the outputs directory for results.")
    
    else:
        parser.print_help()


# __main__ moved to end of file after advanced analysis helpers.


# ==================== Module 11: Advanced Analyses (M1, M2, N1, N2, N3) ====================

def run_fixed_effects(metrics_csv: str, scores_jsonl: str, target_key: str = None, tol: int = 5000):
    """M1: Control for step and baseline (one_step_mse) using OLS with HC3 robust SE.
    Returns (summary_text, delta_r2)."""
    import pandas as pd
    try:
        import statsmodels.formula.api as smf
    except Exception as e:
        raise RuntimeError('statsmodels is required for --controlled') from e

    m = pd.read_csv(metrics_csv)
    s = load_scores(scores_jsonl)
    for df in (m, s):
        if 'step' in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
    if target_key is None:
        cand = ['episode_reward','return','score','eval/episode_reward',
                'train/episode_reward','eval/return','train/return','episode/score','eval/score']
        target_key = next((k for k in cand if k in s.columns), None)
        if target_key is None:
            raise ValueError(f'No score key in {list(s.columns)}')
    # Nearest alignment with tolerance
    ms = m.copy(); ss = s.copy()
    if 'step' in ms.columns: ms['step'] = pd.to_numeric(ms['step'], errors='coerce').astype('Int64')
    if 'step' in ss.columns: ss['step'] = pd.to_numeric(ss['step'], errors='coerce').astype('Int64')
    ms = ms.dropna(subset=['step']).astype({'step': int})
    ss = ss.dropna(subset=['step']).astype({'step': int})
    ss = ss.sort_values('step').assign(s_step=ss['step'])
    df = pd.merge_asof(ms.sort_values('step'), ss, on='step', direction='nearest')
    df = df[(df['s_step'] - df['step']).abs() <= tol].dropna()
    # Ensure columns exist
    for col in ['one_step_mse','kappa_mean','tc','lf_mean','step',target_key]:
        if col not in df.columns:
            raise ValueError(f'Missing column for fixed-effects: {col}')
    # Quote target if it contains special characters (e.g., 'episode/score')
    from patsy import EvalEnvironment
    if not target_key.replace('_','').isalnum():
        lhs = f'Q("{target_key}")'
    else:
        lhs = target_key
    formula = f"{lhs} ~ one_step_mse + kappa_mean + tc + lf_mean + step"
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    base_lhs = lhs  # use the same quoted/cleaned lhs for the base model
    base = smf.ols(f"{base_lhs} ~ one_step_mse + step", data=df).fit(cov_type="HC3")
    delta_r2 = (model.rsquared_adj or 0) - (base.rsquared_adj or 0)
    # Some summaries (e.g., Jarque-Bera) require at least 2 residuals.
    try:
        summ_text = model.summary().as_text()
    except Exception:
        # Fallback minimal summary for very small N
        import io
        buf = io.StringIO()
        buf.write('Insufficient observations for full summary.\n')
        buf.write(f'N = {int(getattr(model.nobs, "__int__", lambda: model.nobs)())}\n')
        buf.write('Params (HC3):\n')
        try:
            tbl = {
                'param': model.params.index.tolist(),
                'coef': model.params.values.tolist(),
                'std_err': getattr(model, 'bse', model.params*0).values.tolist(),
                'p_value': getattr(model, 'pvalues', model.params*0+1).values.tolist(),
            }
            import pandas as pd
            pd.DataFrame(tbl).to_string(buf, index=False)
        except Exception:
            buf.write(str(model.params))
        summ_text = buf.getvalue()
    return summ_text, float(delta_r2)


def _score_key(scores_df: pd.DataFrame) -> str:
    cand = ['episode_reward','return','score','eval/episode_reward','train/episode_reward',
            'eval/return','train/return','episode/score','eval/score']
    return next((k for k in cand if k in scores_df.columns), None)


def _perf_final_win_mean(scores_df: pd.DataFrame, K: int = 3) -> Tuple[float, List[int]]:
    s_sorted = scores_df.sort_values('step')
    ykey = _score_key(scores_df)
    if ykey is None:
        raise ValueError('No score key found for final window mean.')
    tail = s_sorted.tail(K)
    return float(tail[ykey].mean()), tail['step'].astype(int).tolist()


def _perf_future_win_mean_for_steps(scores_df: pd.DataFrame, steps: np.ndarray, delta: int, K: int = 3) -> np.ndarray:
    """For each step t, compute mean of K nearest score points around t+delta."""
    s_sorted = scores_df[['step', _score_key(scores_df)]].dropna().sort_values('step')
    ykey = _score_key(scores_df)
    st = s_sorted['step'].values.astype(int)
    y = s_sorted[ykey].values.astype(float)
    out = []
    for t in steps.astype(int):
        target = t + int(delta)
        # argsort by absolute distance to target
        order = np.argsort(np.abs(st - target))
        k = min(K, len(order))
        if k == 0:
            out.append(np.nan)
        else:
            out.append(float(np.mean(y[order[:k]])))
    return np.asarray(out, dtype=float)


def lagged_correlation(metrics_csv: str, scores_jsonl: str,
                       early_max: int = 200000, final_min: int = 500000,
                       metrics=('kappa_mean','tc','lf_mean','one_step_mse'),
                       n_bootstrap: int = 1000,
                       tol: int = 5000,
                       delta: int | None = None,
                       avg_window: int = 0,
                       block: int = 2):
    """[Legacy] Kept for backward compat. Prefer compute_lagged_correlation()."""
    return compute_lagged_correlation(
        metrics_csv, scores_jsonl,
        early_max=early_max if early_max is not None else 250000,
        deltas=[delta if (delta is not None and delta >= 0) else max(0, final_min - early_max)],
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        tol=tol,
        K=3,
        block=block,
    )


def compute_lagged_correlation(metrics_csv: str, scores_jsonl: str,
                               early_max: int = 250000,
                               deltas: List[int] = [150000, 200000],
                               metrics: Tuple[str, ...] = ('kappa_mean','tc','lf_mean','one_step_mse'),
                               n_bootstrap: int = 1000,
                               tol: int = 5000,
                               K: int = 3,
                               block: int = 2) -> pd.DataFrame:
    """M2: Early -> (t + Δ) predictive Spearman using K-nearest window mean.

    - Uses early_max <= 250k.
    - Evaluates for multiple deltas, default [150k, 200k].
    - Targets are perf_future_win_mean(Δ) computed as mean of K nearest
      score points to each target step t+Δ.
    """
    import pandas as pd, numpy as np
    from scipy.stats import spearmanr
    m = pd.read_csv(metrics_csv)
    s = load_scores(scores_jsonl)
    for df in (m, s):
        if 'step' in df.columns:
            df['step'] = pd.to_numeric(df['step'], errors='coerce')
    ykey = _score_key(s)
    if ykey is None:
        raise ValueError('No score key found.')
    m_sorted = m.sort_values('step').dropna(subset=['step'])
    s_sorted = s[['step', ykey]].dropna().sort_values('step')
    early_m = m_sorted[m_sorted['step'] <= int(early_max)].copy()
    if early_m.empty:
        raise ValueError('No early metric points available (<= early_max).')
    out_rows = []
    rng = np.random.RandomState(42)
    for D in deltas:
        # Compute K-NN mean y around target t+D for each early step t
        steps = early_m['step'].values.astype(int)
        y_future = _perf_future_win_mean_for_steps(s_sorted, steps, int(D), K=K)
        pairs = early_m.copy()
        pairs['_y'] = y_future
        pairs = pairs.dropna(subset=['_y'])
        for met in metrics:
            if met not in pairs.columns:
                continue
            x = pairs[met].values
            yv = pairs['_y'].values
            n_pairs = len(x)
            if n_pairs < 2 or np.allclose(yv, yv[0]):
                rho, p, lo, hi = (float('nan'),) * 4
            else:
                rho, p = spearmanr(x, yv)
                if n_pairs < max(4, block):
                    lo, hi = float('nan'), float('nan')
                else:
                    L = int(max(1, block))
                    boots = []
                    for _ in range(n_bootstrap):
                        idxs = []
                        while len(idxs) < n_pairs:
                            start = rng.randint(0, n_pairs)
                            idxs.extend([(start + k) % n_pairs for k in range(L)])
                        idx = np.array(idxs[:n_pairs])
                        boots.append(spearmanr(x[idx], yv[idx])[0])
                    lo, hi = np.percentile(boots, [2.5, 97.5])
            out_rows.append({
                'metric': met,
                'rho': float(rho),
                'ci_lower': float(lo),
                'ci_upper': float(hi),
                'p_value': float(p),
                'n_pairs': int(n_pairs),
                'delta': int(D),
                'K': int(K),
                'early_max': int(early_max),
            })
    return pd.DataFrame(out_rows)


def compare_correlations_same_sample(x, y1, y2, r12=None):
    """Legacy MRR z-approx; kept for backward compatibility."""
    from scipy.stats import norm, pearsonr
    import numpy as np
    n = len(x)
    r1 = pearsonr(x, y1)[0]
    r2 = pearsonr(x, y2)[0]
    if r12 is None:
        r12 = pearsonr(y1, y2)[0]
    z1, z2 = np.arctanh(max(min(r1, 0.999999), -0.999999)), np.arctanh(max(min(r2, 0.999999), -0.999999))
    denom = np.sqrt((2 * (1 - r12)) / max(n - 3, 1))
    if denom == 0:
        return float('nan'), 1.0
    z = (z1 - z2) / denom
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def compare_dependent_corr(r_xy: float, r_xz: float, r_yz: float, n: int) -> Tuple[float, float]:
    """N1: Steiger/Williams test for dependent correlations sharing X.

    Tests H0: r(X,Y) == r(X,Z) given r(Y,Z) with sample size n.
    Returns (z_stat_like, p_value). We report the t-based statistic as 'z'.
    """
    import numpy as np
    from scipy.stats import t as student_t
    # Clamp correlations to open interval to avoid numerical issues
    r12 = float(np.clip(r_xy, -0.999999, 0.999999))
    r13 = float(np.clip(r_xz, -0.999999, 0.999999))
    r23 = float(np.clip(r_yz, -0.999999, 0.999999))
    df = max(n - 3, 1)
    num = (r12 - r13) * np.sqrt((df) * (1 + r23))
    den_inner = 2 * (1 - r23) * (1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23)
    den_correction = (r12 - r13)**2 * (1 - r23)**3
    den = np.sqrt(max(den_inner + den_correction, 1e-12))
    if den == 0:
        return float('nan'), 1.0
    t_stat = num / den
    p = 2 * (1 - student_t.cdf(np.abs(t_stat), df=df))
    return float(t_stat), float(p)


def benjamini_hochberg(pvals, alpha=0.05):
    import numpy as np
    p = np.asarray(pvals)
    m = len(p)
    idx = np.argsort(p)
    q = p[idx]
    thresh = alpha * np.arange(1, m + 1) / m
    passed = q <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    cutoff = q[k - 1] if k > 0 else 0
    mask = p <= cutoff
    return mask, float(cutoff)


def checkpoint_selection_utility(metrics_csv: str, scores_jsonl: str,
                                 select_metric: str = 'kappa_mean', topk: int = 3,
                                 early_max: int = 200000,
                                 early_min: int = 50000,
                                 K: int = 3):
    """N2: Early checkpoint selection Top-K regret.

    - Target value uses perf_final_win_mean (last K=3 checkpoint scores).
    - Candidate early steps restricted to [50k, 250k] by default.
    - topk typically 1 or 2.
    """
    import pandas as pd, numpy as np
    m = pd.read_csv(metrics_csv)
    s = load_scores(scores_jsonl)
    ykey = _score_key(s)
    if ykey is None:
        raise ValueError('No score key found in scores.jsonl')
    df = pd.merge(m, s[['step', ykey]], on='step', how='inner').dropna()
    # Compute final window mean target
    y_final_win, used_steps = _perf_final_win_mean(s, K=K)
    # Restrict early window
    lo = int(early_min)
    hi = int(max(early_max, 250000))
    early = df[(df['step'] >= lo) & (df['step'] <= hi)].copy()
    if early.empty:
        return {'error': 'No early candidates in the specified range.'}
    if select_metric not in early.columns:
        raise ValueError(f'Metric {select_metric} not in metrics.csv')
    # Pick top-k by metric (ascending)
    early_sorted = early.sort_values(select_metric, ascending=True)
    picked_steps = early_sorted['step'].head(topk).tolist()
    # Evaluate regrets vs final window mean
    y_pick_best = float(early[early['step'].isin(picked_steps)][ykey].max()) if len(picked_steps) else float('nan')
    base_steps = early.sort_values('one_step_mse', ascending=True).head(topk)['step'].tolist()
    y_base_best = float(early[early['step'].isin(base_steps)][ykey].max()) if len(base_steps) else float('nan')
    y_last = float(df[ykey].iloc[-1]) if len(df) else float('nan')
    reg_pick = float(y_final_win - y_pick_best) if np.isfinite(y_pick_best) else float('nan')
    reg_base = float(y_final_win - y_base_best) if np.isfinite(y_base_best) else float('nan')
    reg_last = float(y_final_win - y_last) if np.isfinite(y_last) else float('nan')
    return {'topk': int(topk), 'metric': select_metric,
            'regret_pick': reg_pick, 'regret_base': reg_base, 'regret_last': reg_last,
            'target_final_win_mean': float(y_final_win),
            'target_window_steps': used_steps,
            'pick_best': float(y_pick_best),
            'base_best': float(y_base_best),
            'last': float(y_last),
            'early_min': lo, 'early_max': hi}


def toy_mdp_demo(n=5000, noise=0.05, seed=0):
    """N3: Simple 2D linear-Gaussian MDP to illustrate kappa-performance relation."""
    import numpy as np
    rng = np.random.RandomState(seed)
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    S = rng.randn(n, 2)
    R = -np.sum(S ** 2, axis=1)
    alphas = np.linspace(0.5, 2.0, 10)
    kappas, perf = [], []
    for a in alphas:
        Phi = np.diag([1.0, a])
        Z = (Phi @ S.T).T
        kappa = np.linalg.cond(Phi)
        R_hat = -np.sum(Z ** 2, axis=1)
        mse = np.mean((R - R_hat) ** 2)
        kappas.append(kappa)
        perf.append(-mse)
    return alphas, np.array(kappas), np.array(perf)


if __name__ == "__main__":
    main()
