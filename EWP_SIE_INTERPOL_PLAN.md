# EWP on SIE: Interpolation-Based Plan

## Purpose
This repo is intended to become a new EWP variant built on top of the validated `SIE_3DIE` setup.

The current goal is not to reproduce SIE again. The goal is to keep the same benchmark/training scaffold where it is already validated, but replace the known train-time rotation prior with a learned geometric conditioning mechanism.

Core research question:
- Can we learn an explicit rotation-like conditioning variable from the views themselves, without using the known relative rotation during training, while still supporting the same downstream rotation objective?

## Current status
Validated control:
- `SIE_3DIE` ResNet baseline is behaving in line with the paper.
- On the current setup, the ResNet run reached about `0.7 R^2` by about `600` epochs, which is close to the paper's reported regime (`0.72` by `2000` epochs).
- This removes the major uncertainty around the dataset, evaluation, and general SIE training path.

Implication:
- The next step should stay on the ResNet path first.
- ViT can remain a secondary branch, but it should not be the main control for the first EWP design.

## Problem framing
SIE:
- Uses the true relative rotation during training as the conditioning variable for the predictor.

EWP:
- Must not use the true relative rotation during training.
- May still use the true relative rotation for evaluation only.
- Needs to learn a conditioning variable from the views/embeddings themselves.

Desired relationship to SIE:
- Same base structure where possible.
- Same benchmark.
- Same evaluation.
- Main change: replace known train-time conditioning with learned conditioning.

## Grounded facts from the data
Known facts from 3DIEBench generation:
- Rotations are sampled as Tait-Bryan angles with extrinsic rotations.
- Each of the X/Y/Z object rotation angles is sampled uniformly in `[-pi/2, pi/2]`.

Implications:
- The train-time transformation space is bounded.
- The missing transformation is low-dimensional.
- A small continuous transform code is plausible.
- Inverse structure is strongly grounded.
- Interpolation between two poses is well-defined.
- Full unrestricted `SO(3)` capacity is not necessary for v1.

## High-confidence assumptions
These are the assumptions we currently consider most grounded.

1. The conditioning variable should be geometric.
- It should live in a real rotation representation, not an unconstrained free latent.

2. The conditioning variable should be low-capacity.
- No extra dimensions beyond the chosen rotation representation.

3. The conditioning variable should support an analytic inverse.
- If it acts as a rotation, reverse transport should come from the known inverse formula, not a learned inverse head.

4. The condition should only enter through the predictor/transport path.
- It should not be injected directly into the backbone representation used for downstream probing.

5. The task should be driven by alignment of transported embeddings.
- Pose-only self-consistency is too close to tautology.
- The actual learning pressure should be on whether the transported embedding matches another independently produced embedding.

## Representation choice
Current choice for v1:
- Unit quaternion (`4D`) as the learned pose/conditioning representation.

Why quaternion for v1:
- Exact inverse by conjugation for unit quaternions.
- Exact composition.
- Exact interpolation via slerp.
- No extra latent dimensions beyond the representation itself.
- Cleaner research instrument than a freer representation.

Why not a fully free latent:
- Too much leakage capacity.
- Too weakly tied to true geometry.
- Harder to interpret failures.

Why not use ground-truth quaternion at train time:
- That would collapse the problem back toward SIE.
- EWP must remain a no-prior training setup.

## Important distinction: no data prior vs model prior
This design is still considered "no prior" in the sense relevant for EWP.

Not allowed:
- Feeding the true relative rotation into the model during training.
- Supervising the learned conditioning to match the true rotation directly during training.

Allowed:
- Using rotation algebra as a structural prior.
- Constraining the learned code to be a valid quaternion.
- Using analytic inverse/composition/interpolation on the learned code.

Interpretation:
- This uses a geometric model prior, not a train-time data prior.

## Main design decision: merged training is feasible
We considered a 2-stage setup:
- Stage 1: learn pose estimator.
- Stage 2: freeze or reuse it for predictor conditioning.

Main concern:
- Jointly learning pose estimator and predictor from the same loss can create messy gradients and co-adaptation.
- A strict stage split introduces new problems if the backbone later changes and breaks the pose estimator.

Current conclusion:
- A merged setup is feasible if the losses are placed on transported embeddings, not on pose alignment alone.
- Pose-only alignment is too self-referential.
- Embedding alignment removes the trivial "treat predicted pose as ground truth" behavior.

## Core variables
For a pair of views `x`, `y`:
- `x_equi`, `y_equi`: equivariant branch embeddings
- `q_x = normalize(C(x_equi))`
- `q_y = normalize(C(y_equi))`
- `q_xy = quat_inv(q_x) * q_y`
- `q_yx = quat_inv(q_y) * q_x`

Where:
- `C` is the learned pose estimator
- `P` is the predictor / transport operator
- `quat_inv` is the analytic quaternion inverse

## Chosen v1 objective set
We currently want three distinct signals.

### 1. Symmetric endpoint transport
Main objective.

Predict:
- `y_pred = P(x_equi, q_xy)`
- `x_pred = P(y_equi, q_yx)`

Loss:
- align `y_pred` with `stopgrad(y_equi)`
- align `x_pred` with `stopgrad(x_equi)`

Purpose:
- enforce useful forward and reverse transport using the learned geometric delta.

### 2. Symmetric interpolation consistency
Path-shaping objective.

Sample:
- `p ~ Uniform(0, 1)`

Define interpolated pose:
- `q_t = slerp(q_x, q_y, p)`

Define endpoint-to-point deltas:
- `q_{x->t} = quat_inv(q_x) * q_t`
- `q_{y->t} = quat_inv(q_y) * q_t`

Predict the interpolated embedding from both ends:
- `z_x = P(x_equi, q_{x->t})`
- `z_y = P(y_equi, q_{y->t})`

Loss:
- align `z_x` and `z_y`

Purpose:
- enforce that the learned transport is coherent along the path, not only at the endpoints.

### 3. Interpolation-position prediction (`p` prediction)
Anti-collapse auxiliary.

Idea:
- If the path collapses toward invariance, the position along the path should become unrecoverable.
- Therefore the intermediate state should contain enough ordered structure to recover `p`.

Example:
- use `x_equi`, `y_equi`, and the predicted interpolated state(s) to predict `p`
- regress the known sampled scalar `p in [0,1]`

Purpose:
- discourage trivial collapse where all intermediate states become indistinguishable.

## Why 2 and 3 are both needed
Interpolation consistency alone:
- gives path coherence
- but is a weak anti-collapse signal by itself

`p` prediction alone:
- gives anti-collapse pressure
- but needs the interpolation construction to define a meaningful target

Interpretation:
- `2` = path coherence
- `3` = path identifiability

## Why not use extrapolation in v1
Extrapolation is geometrically valid, but we are excluding it from v1.

Reason:
- Interpolation stays inside the interval between two observed endpoints.
- Extrapolation goes beyond observed support.
- Interpolation is easier to justify and debug.

Conclusion:
- v1 uses interpolation only.
- Extrapolation can be a later ablation.

## Why we do not use pose-only alignment losses in v1
Pose-only alignment is too easy to satisfy by construction.

Example problem:
- If both branches are computed from the same `q_x`, `q_y` using exact geometry, many pose consistency equalities become tautological.

Therefore:
- the learning signal should act on transported embeddings
- not on pose objects alone

Pose still matters, but only as the geometric scaffold used to generate the conditioning deltas.

## Failure modes to watch
1. Constant pose code collapse.
- `C(x)` outputs nearly the same quaternion for all views.

2. Object identity encoded instead of pose.
- `C(x)` captures object semantics, not view transformation.

3. No shared frame across objects.
- pose codes become object-private.

4. Predictor overcompensation.
- `P` carries the task even when pose codes are poor.

5. Weak path identifiability.
- interpolation states agree but do not encode ordered position.

These failure modes motivate the inclusion of:
- low-capacity quaternion codes
- symmetric endpoint transport
- interpolation consistency
- `p` prediction auxiliary

## Why the assumptions are testable
This setup is not purely speculative. The assumptions can be validated with diagnostics and downstream checks.

Diagnostics during training:
- variance of `q_x` across batch
- variance of `q_xy` across batch
- endpoint transport loss
- interpolation consistency loss
- `p` prediction error

Downstream checks:
- frozen rotation probe (`R^2`)
- class probe
- ablations with/without interpolation and `p` prediction

Interpretation:
- if the setup fails, we should be able to say whether the failure is due to code collapse, poor path structure, or predictor misuse.

## v1 implementation constraints
1. Stay on the validated ResNet/SIE path first.
2. Keep the backbone/data/eval scaffold as close as possible to working `SIE_3DIE`.
3. Use quaternion-conditioned transport only.
4. Keep predictor modest.
5. Do not add extra free latent dimensions.
6. Do not use true rotation as train-time input.
7. Do not add many extra losses at once beyond the three current signals.

## Immediate next step
Implementation should start by copying in the validated `SIE_3DIE` base and then changing only what is necessary for:
- learned per-view quaternion estimator `C`
- learned transport conditioned on `q_xy`
- symmetric endpoint transport loss
- symmetric interpolation consistency loss
- `p` prediction auxiliary

This repo should be treated as a controlled research branch built on top of a validated SIE baseline, not as an unrelated fresh training codebase.
