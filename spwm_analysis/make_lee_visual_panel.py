#!/usr/bin/env python3
"""
Generate a small visual panel for LEE:
- Sample one Crafter frame from a given logdir replay buffer.
- Apply the exact same rotation transform used in the LEE metric.
- Save both the original and rotated images into a fresh output directory.

This script is intentionally self-contained and does not affect training/eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def _add_repo_root_to_path() -> None:
    """Ensure the repository root is on sys.path so we can import spwm_analysis."""
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_add_repo_root_to_path()

from spwm_analysis.spwm_metrics_system import affine_rotate  # noqa: E402


def pick_frame(images: np.ndarray, frame_index: int) -> np.ndarray:
    """
    Pick a single frame from images[time, H, W, C].

    If frame_index is out of range, use the middle frame as a fallback.
    """
    assert images.ndim == 4, f"Expected images with shape [T, H, W, C], got {images.shape}"
    t = images.shape[0]
    if frame_index < 0 or frame_index >= t:
        frame_index = t // 2
    return images[frame_index]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Extract one Crafter frame from replay and apply the same rotation "
            "transform used for LEE (affine_rotate with bilinear interpolation, fill=0)."
        )
    )
    ap.add_argument(
        "--logdir",
        type=str,
        default="/hy-tmp/SPWM/logdir/crafter_50K_20250917_194237",
        help="DreamerV3 logdir containing the 'replay/' subdirectory.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="spwm_analysis/lee_visual_crafter_50K_20250917",
        help="Directory to save output images (will be created if missing).",
    )
    ap.add_argument(
        "--file_index",
        type=int,
        default=0,
        help="Index of the replay .npz file to use (sorted by filename).",
    )
    ap.add_argument(
        "--frame_index",
        type=int,
        default=512,
        help="Index of the frame inside the chosen .npz file (time dimension).",
    )
    ap.add_argument(
        "--degrees",
        type=float,
        default=5.0,
        help="Rotation in degrees (used for both LEE and this visual).",
    )
    ap.add_argument(
        "--upscale_factor",
        type=int,
        default=16,
        help=(
            "Integer factor to upscale images for visualization only. "
            "The LEE transform still runs at native 64x64 resolution. "
            "If <=1, no extra high-res copies are saved."
        ),
    )
    args = ap.parse_args()

    logdir = Path(args.logdir)
    replay_dir = logdir / "replay"
    if not replay_dir.is_dir():
        raise SystemExit(f"[ERROR] Replay directory not found: {replay_dir}")

    replay_files = sorted(replay_dir.glob("*.npz"))
    if not replay_files:
        raise SystemExit(f"[ERROR] No .npz replay files found in {replay_dir}")

    fi = max(0, min(args.file_index, len(replay_files) - 1))
    replay_path = replay_files[fi]

    print(f"[INFO] Using replay file: {replay_path}")
    data = np.load(replay_path)
    if "image" not in data:
        raise SystemExit(f"[ERROR] 'image' key not found in {replay_path}")

    images = data["image"]  # [T, H, W, C], uint8
    print(f"[INFO] Loaded images with shape {images.shape}, dtype={images.dtype}")

    frame_np = pick_frame(images, args.frame_index)  # [H, W, C]
    print(f"[INFO] Selected frame index {args.frame_index} "
          f"(clipped if out of range); frame shape={frame_np.shape}")

    # Convert NHWC uint8 -> NCHW float32 in [0,1]
    frame_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0
    frame_t = frame_t.clamp(0.0, 1.0)
    x_batch = frame_t.unsqueeze(0)  # [1, 3, H, W]

    # Apply the exact LEE rotation transform (bilinear, fill=0)
    degrees = float(args.degrees)
    x_rot = affine_rotate(x_batch, degrees=degrees)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    orig_path = outdir / "orig.png"
    rot_path = outdir / f"rot{degrees:+.1f}.png"
    panel_path = outdir / "panel_orig_rot.png"

    # Save base-resolution images (64x64) – this matches the data used for LEE.
    save_image(x_batch, orig_path)
    save_image(x_rot, rot_path)

    panel = torch.cat([x_batch, x_rot], dim=0)
    save_image(panel, panel_path, nrow=2)

    # Optionally create higher-resolution copies for visualization only.
    scale = int(args.upscale_factor)
    if scale > 1:
        new_h = x_batch.shape[-2] * scale
        new_w = x_batch.shape[-1] * scale

        def _up(img: torch.Tensor) -> torch.Tensor:
            # Nearest-neighbor upsampling preserves pixel-level artifacts.
            return F.interpolate(img, size=(new_h, new_w), mode="nearest")

        x_hi = _up(x_batch)
        x_rot_hi = _up(x_rot)
        panel_hi = torch.cat([x_hi, x_rot_hi], dim=0)

        orig_hi_path = outdir / f"orig_x{scale}.png"
        rot_hi_path = outdir / f"rot{degrees:+.1f}_x{scale}.png"
        panel_hi_path = outdir / f"panel_orig_rot_x{scale}.png"

        save_image(x_hi, orig_hi_path)
        save_image(x_rot_hi, rot_hi_path)
        save_image(panel_hi, panel_hi_path, nrow=2)
    else:
        orig_hi_path = rot_hi_path = panel_hi_path = None

    # Basic self-check: report stats and difference at native resolution
    diff = torch.mean(torch.abs(x_rot - x_batch)).item()
    print(f"[INFO] Saved original image (native) to: {orig_path}")
    print(f"[INFO] Saved rotated image (native) to: {rot_path}")
    print(f"[INFO] Saved side-by-side panel (native) to: {panel_path}")
    if orig_hi_path is not None:
        print(f"[INFO] Saved upscaled original to: {orig_hi_path}")
        print(f"[INFO] Saved upscaled rotated to: {rot_hi_path}")
        print(f"[INFO] Saved upscaled panel to: {panel_hi_path}")
    print(f"[INFO] Original min/max: {float(x_batch.min()):.4f} / {float(x_batch.max()):.4f}")
    print(f"[INFO] Rotated  min/max: {float(x_rot.min()):.4f} / {float(x_rot.max()):.4f}")
    print(f"[INFO] Mean |rot - orig| over all pixels (native): {diff:.6f}")


if __name__ == "__main__":
    main()
