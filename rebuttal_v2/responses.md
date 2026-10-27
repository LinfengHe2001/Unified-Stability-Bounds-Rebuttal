## R-2q75

Thank you. New analyses use saved checkpoints only, under a protocol fixed before running.

**Q1.** [A] Regret of 3.5M is {X} (J={X}, J_max={X}); it {is/is not} the best checkpoint. [B] The plotted point was the 500k calibration checkpoint (legend "late 500K"); 3.5M has regret {X}. Either way, that test could not fail.

**W1.** Redone so it can fail: β and channel scales frozen on the main run (50k-500k), J* = best return over all runs. Tests: all {16} later checkpoints of that run and the 6 augmentation-study runs{ + {m} TC-sweep runs}. Frozen β* covers {X}% of {N} (median slack {X}); step-matched AUROC {X} [{CI}].

**W2/Q2.** Agreed: as defined, TC and LEE rise with return on Crafter. [A] In {n}/{N} over-compressed runs κ deviated {X}k steps before the drop (false alarms on healthy seeds: {X}). [B] No channel leads a drop at 50k resolution; we remove "flag failures before return drops". [C] The sweep runs were not kept; we withdraw that experiment.

**W4.** One-step KL alone: {X}%, AUROC {X}.{ Not beaten: we drop the causes-vs-symptoms framing.}

**Q3.** Checking this, we found our κ/LEE probes encoded a sampled latent with a new RNG per call, and uint8 casting turned the ε-probe into one-level steps (~300ε), so κ/LEE are dominated by resampling noise. [A] On deterministic posteriors, κ w.r.t. a semantic proxy for d_S (position, inventory, local map) has ρ={X} with pixel κ. [B] We have no evidence that pixel κ tracks d_S.

**Q4.** Z was [h; sampled one-hot], h constant per observe step. Gaussian TC was dominated by the one-hot sum-to-one constraint and ridge. We now use between-group TC on posterior probabilities.

Theorem 5 becomes motivation; the audit, a heuristic.

## R-bajJ

Thank you.

**Q1.** Theorem 5 bounds only the lift of the abstract-optimal policy π̃*; Assumption 4 will say so. For the trained agent, J(π*)−J(π) = [J(π*)−J(π̃*∘φ)] + [J(π̃*∘φ)−J(π)]. The audit bounds the first (representation) term. The second holds learned-model and planning error, and is nonzero also because Dreamer's policy uses recurrent state; bounding it by a simulation lemma needs TV model error and Lipschitz Ṽ^π, stated as extra assumptions. Coverage of the measured gap is thus a heuristic test, not verification.

**Q2.** Not by themselves. A constant encoder has TC=0 yet loses reward information, so the MMSE bridge needs the information term ε_I and is stated as an assumption. With ρ_g=0, LEE measures invariance and penalises task-relevant transformations (our augmentation result shows this). New: residual after fitting a linear ρ_g, {X}; aliasing rate (latent 10-NN with a different semantic state), {X}, {rising/flat} in failing runs.

**Q3.** With β frozen on one run: {X}% coverage on {N} checkpoints of {R} other runs, step-matched AUROC {X}. With β refit after removing κ/TC/LEE: {X}/{X}/{X} vs {X}. Other agents and distribution shift remain future work.

κ/LEE were dominated by resampling noise (2q75 Q3); Crafter numbers use corrected channels.

## R-Niwv

Thank you.

As stated, Theorem 5 does not account for learned-dynamics error or policy suboptimality; extra assumptions are needed. The revision states: (A1) the bounded policy is the lift of the abstract-optimal policy π̃*; (A2) abstract reward/transition are pushforwards of the true MDP, not Dreamer's learned heads; (A3) concentrability between μ and the policies' occupancies; (A4) both proxy bridges are assumptions.

For the trained agent we split J(π*)−J(π) = [J(π*)−J(π̃*∘φ)] + [J(π̃*∘φ)−J(π)]. Eq. (5) bounds only the first. The second (learned-model and planning error) can be bounded by a simulation lemma under TV model error and Lipschitz Ṽ^π (A5). The measured gap contains both, so coverage is a heuristic test, not verification.

**W2.** Agreed. Fixed-β ablation is uninformative; with β refit, held-out AUROC is {X}/{X}/{X} vs {X}. β frozen on one run: {X}% coverage on {R} other runs, median slack {X}; one-step KL alone: AUROC {X}. We keep the low correlation explicit and narrow claims to Crafter and Pong.

κ/LEE were dominated by resampling noise (2q75 Q3); Crafter numbers use corrected channels.

## R-hQQj

Thank you.

**W1.** Your concern was right. κ/LEE probes encoded a sampled latent with a new RNG per call, and uint8 casting turned the ε-probe into one-level steps, so κ/LEE were dominated by resampling noise; Gaussian TC was dominated by the one-hot constraint and ridge. All channels now use deterministic posterior probabilities. Held-out AUROC changes by {X} across TC ridge 1e-4 to 1e-2, {X} for κ quantiles (q90/q10 vs q99/q01), {X} for 1 vs 2 px translation.

**W2.** With β frozen on one run, the bound covers {X}% of {N} checkpoints from {R} other runs. We add split-conformal calibration over seeds: for a new seed of the same configuration, P(all checkpoints covered) ≥ 1−α, α ≥ 1/(n+1) = {X} for n={n}. It does not remove per-task calibration or cover failing runs, and we say so.

**W3.** Agreed. Recomputed on probabilities, augmentation changes LEE by {X}% (paper: −22.7%) while lowering return. A flagged channel can be functional; we rename LEE "local transformation sensitivity".

**Other architectures, continuous control.** Not feasible in this window; claims are narrowed to DreamerV3 on Crafter and Pong, listed as a limitation.

## R-drHq

Thank you.

**Q1.** Not sufficient. We define δ_id as the L2(μ) reward error and assume χ²-type concentrability C of the discounted occupancies of π and π* w.r.t. μ, so E_d|e| ≤ √C‖e‖_L2(μ). With μ from a random policy, C is unverifiable and may be large; we say so.

**Q2.** None we know of. MMSE(F|Z) ≤ C(TC+ε_I) is an assumed bridge and will be labelled so.

**Q3.** Agreed; we rename it and add, as an equivariance proxy, the residual after fitting a linear latent action ρ_g: {X}.

**Q4.** Crafter: one trained agent (10 calibration checkpoints and one {3.5M/500k, see 2q75 Q1} checkpoint). Pong: {n} run(s), old estimator. New tests use {R} independent runs{ + {m} TC-sweep}; runs are the bootstrap units.

**Q5.** Eq. (5) itself cannot be falsified from checkpoints. The frozen transfer can: with β and scales fixed on one run, any checkpoint of another run above β·RHS falsifies it. Result: {X}% covered, {V} violations ({runs}).

**W3/W4.** We remove causal wording ("TC drives performance") and call the measured quantity regret to the best observed checkpoint, not the optimality gap.

**Correction.** Fig. 3/Table 2 used min-max-clipped (1/(1−γ))κ̂+TC+L_F with γ defaulting to 0.99, not κ+TC+LEE; L_F and κ/LEE contained resampling noise. These results will be replaced.

## R-AI (optional)

Thank you; we accept these points.

- **Permutation test.** Its null used a different RHS and scale than the observed β*, so the test is invalid; we withdraw it.
- **Ablation.** Fixed-β ablation lowers coverage by construction; replaced by a refit-β held-out ablation.
- **Fig. 3.** β* was calibrated on one scale and tested on another, and the RHS was not κ+TC+LEE (see drHq). We recompute on one stated scale. The legend {correctly shows the 500k calibration checkpoint (2q75 Q1) / is corrected to 3.5M}. Some labels were set by hand; all will be generated from computed values.
- **Theorem 5 and proxies.** Stated for the lift of the abstract-optimal policy, with bridge assumptions explicit and a table mapping each quantity to its estimator.
- **LEE intervention.** Recomputed: {X}% (paper −22.7%), with per-seed returns and CIs.
- **Smaller fixes.** Add Kemertas & Aumentado-Armstrong (2021); narrow novelty to the checkpoint-only combination; restore ½ in Eq. 9; call Eq. 8 checkpoint-relative regret in the Intro; fix the Fig. 2 label and Fig. 5 "monotone".

## Confidential comment to SPC/AC

Dear SPC/AC,

Re-auditing our code for the responses, we found errors that affect the reported results:

1. κ and LEE were computed on a sampled categorical latent with a fresh RNG per call, and uint8 casting turned the ε=1e-3 probe into one-level steps; reported κ/LEE are dominated by resampling noise. Gaussian TC was dominated by the one-hot constraint and the ridge.
2. Fig. 3/Table 2 used min-max-clipped (1/(1−γ))κ̂+TC+L_F (γ defaulting to 0.99), not the stated κ+TC+LEE. Fig. 4 used a third RHS, and the permutation null used a different RHS and scale than the observed β*.
3. Plotting scripts set some displayed values by hand (β* slope 0.11, β90 label 0.09, R²=0.07), and Figs. 3 and 4 come from different runs.
4. {If the note says 500000: the "held-out" point in Fig. 3 was the 500k calibration checkpoint.}
5. {If no sweep logs: the supplement's TC-penalty experiment cannot be reproduced from our code; we withdraw it.}
6. The code used ±2 px translation (paper: ±1 px) and 256 frames for TC (paper: 5,000).

All channels are recomputed on deterministic posterior probabilities, and the responses report held-out tests under a protocol fixed beforehand. We will replace the affected figures and claims, and understand this may change the assessment.
