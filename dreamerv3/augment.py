"""
dreamerv3/augment.py

Symmetry data augmentation for the LEE-intervention experiment.

Applies random rotation / translation / isotropic-scale to image observations
ONLY at the encoder input, so the world model learns an encoder that is robust
(equivariant) to these transforms. The reconstruction TARGET stays the original
image (handled in agent.py), which is what makes LEE decrease.

The transform math mirrors the verified NumPy reference (augment_reference.py):
inverse-warp affine grid + bilinear sampling + zero-fill boundary, matching the
LEE measurement protocol in spwm_analysis (torchvision affine, fill=0).

Image format here (DreamerV3 / JAX): (..., H, W, C), dtype uint8 in [0,255].
We augment in float, then cast back to uint8 so the encoder/decoder pipeline is
byte-for-byte compatible with the un-augmented path.
"""
import jax
import jax.numpy as jnp


def _affine_grid(angle_rad, tx, ty, scale, H, W):
    """Inverse-warp grid. For each OUTPUT pixel, where to sample in INPUT.
    Normalized coords in [-1, 1]. tx, ty in normalized units. Convention:
    tx>0 moves content toward +x (documented/verified in augment_reference.py).
    """
    cos, sin = jnp.cos(angle_rad), jnp.sin(angle_rad)
    ys, xs = jnp.meshgrid(jnp.linspace(-1, 1, H),
                          jnp.linspace(-1, 1, W), indexing='ij')
    xr = (cos * xs + sin * ys) / scale - tx
    yr = (-sin * xs + cos * ys) / scale - ty
    return xr, yr


def _bilinear_sample(img, xr, yr):
    """Bilinear-sample img (H,W,C) at normalized coords; out-of-bounds -> 0."""
    H, W = img.shape[0], img.shape[1]
    xpix = (xr + 1) * (W - 1) / 2
    ypix = (yr + 1) * (H - 1) / 2
    x0 = jnp.floor(xpix).astype(jnp.int32); x1 = x0 + 1
    y0 = jnp.floor(ypix).astype(jnp.int32); y1 = y0 + 1
    wx = (xpix - x0)[..., None]
    wy = (ypix - y0)[..., None]

    def gather(yy, xx):
        valid = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        xc = jnp.clip(xx, 0, W - 1); yc = jnp.clip(yy, 0, H - 1)
        val = img[yc, xc]
        return jnp.where(valid[..., None], val, 0.0)

    Ia = gather(y0, x0); Ib = gather(y0, x1)
    Ic = gather(y1, x0); Id = gather(y1, x1)
    top = Ia * (1 - wx) + Ib * wx
    bot = Ic * (1 - wx) + Id * wx
    return top * (1 - wy) + bot * wy


def _augment_one(img_f, key, cfg):
    """Augment a single float image (H,W,C) in [0,1]. Returns float in [0,1]."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    H, W = img_f.shape[0], img_f.shape[1]
    angle = jax.random.uniform(
        k1, (), minval=-cfg['rotation_deg'], maxval=cfg['rotation_deg'])
    angle = angle * jnp.pi / 180.0
    # pixel translation -> normalized (divide by half-extent)
    tx = jax.random.uniform(
        k2, (), minval=-cfg['translation_px'], maxval=cfg['translation_px'])
    tx = tx / (W / 2.0)
    ty = jax.random.uniform(
        k3, (), minval=-cfg['translation_px'], maxval=cfg['translation_px'])
    ty = ty / (H / 2.0)
    scale = 1.0 + jax.random.uniform(
        k4, (), minval=-cfg['scale_range'], maxval=cfg['scale_range'])
    xr, yr = _affine_grid(angle, tx, ty, scale, H, W)
    return _bilinear_sample(img_f, xr, yr)


def augment_images(images, rng, cfg):
    """
    Apply per-sample symmetry augmentation to a batch of uint8 images.

    Args:
      images: uint8 array, shape (..., H, W, C), values in [0,255].
      rng:    jax PRNGKey.
      cfg:    dict-like with keys:
                enabled (bool), rotation_deg, translation_px,
                scale_range, apply_prob.

    Returns:
      uint8 array, same shape. Each image is independently augmented with
      probability apply_prob; otherwise returned unchanged.
    """
    if not cfg.get('enabled', False):
        return images

    orig_shape = images.shape
    H, W, C = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    flat = images.reshape((-1, H, W, C))
    N = flat.shape[0]
    flat_f = flat.astype(jnp.float32) / 255.0

    keys = jax.random.split(rng, N + 1)
    sample_keys = keys[:N]
    gate_key = keys[N]

    def per_image(img_f, k):
        return _augment_one(img_f, k, cfg)

    aug_f = jax.vmap(per_image)(flat_f, sample_keys)

    # Apply with probability apply_prob, per image.
    gates = jax.random.uniform(gate_key, (N,)) < cfg['apply_prob']
    out_f = jnp.where(gates[:, None, None, None], aug_f, flat_f)

    out = jnp.clip(jnp.round(out_f * 255.0), 0, 255).astype(jnp.uint8)
    return out.reshape(orig_shape)