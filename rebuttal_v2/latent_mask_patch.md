# T4 (optional): inference-time latent clamping in the policy

Purpose: create checkpoints with a **known, dose-controlled representation defect**
(information removed from k of the 32 categorical groups) without any training,
then check (a) how much return drops and (b) whether the audit channels and the
frozen bound respond. Only policy rollouts (inference) are needed.

## 1. `dreamerv3/configs.yaml`, inside `agent:` (next to `symmetry_aug`)

```yaml
    latent_mask_k: 0      # T4: clamp the first k stoch groups at inference (0 = off)
```

## 2. `dreamerv3/agent.py`, in `Agent.policy()` right after `self.dyn.observe(...)`

```python
    dyn_carry, dyn_entry, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    # ---- T4: inference-time latent clamping (off when latent_mask_k == 0) ----
    k = int(self.config.latent_mask_k)
    if k > 0:
      st = feat['stoch']                                   # [B, G, C]
      keep = (jnp.arange(st.shape[-2]) >= k)[:, None]       # [G, 1]
      fixed = jax.nn.one_hot(0, st.shape[-1], dtype=st.dtype)
      st = jnp.where(keep, st, fixed)
      feat = {**feat, 'stoch': st}
      dyn_carry = {**dyn_carry, 'stoch': st}
    # -------------------------------------------------------------------------
```

## 3. Run (per k, same late checkpoint, e.g. the main run at 1.1M or 3.5M)

```bash
for k in 0 4 8 16 24; do
  python dreamerv3/main.py --script eval_only --configs crafter \
    --run.from_checkpoint logdir/<main>/checkpoint_03500000/<latest>/ \
    --logdir logdir/t4_mask_k$k --agent.latent_mask_k $k \
    --run.steps 60000 --run.envs 4
done
# audit the same masked representation:
for k in 0 4 8 16 24; do
  python spwm_analysis/audit_v2.py --run_dir logdir/<main> --run_name t4_k$k \
    --group t4 --mask_k $k --steps 3500000 --eval_set shared_eval_set_info.npz \
    --out results_v2/t4_k$k.csv
done
```

Notes
- `--run.from_checkpoint` must point to the directory that contains `agent.pkl`
  (check the `latest` file inside the checkpoint folder).
- `run.steps 60000` gives roughly 20-60 Crafter episodes; report n and SEM.
- The world-model training path is untouched; only the acting policy sees the clamp.
- Untested here (no JAX in the drafting environment): run `k=0` first and check
  that its return matches the unpatched checkpoint.
