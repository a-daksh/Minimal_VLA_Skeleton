# Minimal VLA Skeleton

A minimal Vision-Language-Action (VLA) policy
**Inputs:** single RGB image, proprioceptive state, language instruction string
**Output:** continuous action vector (Δxyz + Δrpy + grippe)

---

## Architecture

```
[Image]   → VisionEncoder   (frozen SigLIP)         → (B, 768)
[Text]    → LanguageEncoder (frozen SigLIP)         → (B, 768)
[Proprio] → ProprioEncoder  (learned Linear + SiLU) → (B,   9)
                  │
                  ▼
          concat → fused (B, 1545)
                  │
                  ▼
          FlowHead - flow matching MLP
                  │
                  ▼
          action (B, 7)
```

---

## Key Decisions

### Backbone - SigLIP (`google/siglip-base-patch16-224`)

Chose SigLIP over CLIP for the vision encoder. CLIP uses global pooling which loses spatial relationships - if the task is "pick up the mug on your left", the notion of left/right is gone after pooling. SigLIP also uses global pooling by default, but it allows bypassing this to extract patch tokens `(B, N_patches, D)` instead, preserving spatial info. 

- [ ] TODO: Currently using global (single vector), but the structure is set up to swap to patch tokens later with a spatial adapter before fusion.

### Language Encoder

SigLIP/CLIP text encoders were trained on general image captions, not robot instructions. "Pick up the red block" is a different distribution. 
- [ ] TODO: If conditioning turns out weak, switch to an instruction-tuned LLM

For now using the SigLIP text encoder directly, no projection. Also: SigLIP tokenizer hard-caps at 64 tokens which is fine for short instructions.

Both encoders share the same SigLIP model - loaded once, weights split, everything frozen. No gradients flow through the backbone.

### Action Head - Flow Matching

Model predicts a velocity field u(x_t, t | cond), not the action directly.

- **Training:** Sample noise x_0 ~ N(0, I) and GT action x_1, draw t ~ U(0,1), compute x_t = t·x_1 + (1-t)·x_0. Loss = MSE between predicted and true velocity (x_1 - x_0).
- **Inference:** Start from x_0 ~ N(0, I), Euler-integrate dx/dt = u_θ(x, t, cond) from t=0 to t=1. x at t=1 is the action.

Time is embedded via a learnable Fourier encoder, concatenated with x_t and the fused conditioning before the MLP. `FourierEncoder` and the Euler integration logic are adapted from [Flow_Matching_and_Diffusion_Models](https://github.com/a-daksh/Flow_Matching_and_Diffusion_Models/tree/torch).

### Fusion

Concatenation of all three modalities: vision (768) + language (768) + proprio (9) = 1545-d conditioning vector into the flow head. 
- [ ] TODO: Could potentially look into cross-attention between modalities at this stage.

### Training

| | |
|---|---|
| Precision | bf16 AMP (`torch.autocast`), CUDA only - no-op on CPU so the script runs anywhere |
| Loop | Step-based, not epoch-based |
| LR schedule | Cosine annealing over `num_steps` |
| Validation | Every `val_every` steps; computes flow matching loss + action MSE using 10 Euler steps (faster, same signal) |
| Checkpoints | `latest.pt` every val interval (for resuming), `best.pt` when val loss improves (for eval) |

---

## Limitations and Future Work

### Multi-frame, Multi-camera

Right now the model takes 1 frame from 1 camera. In real settings that doesn't work - physics capture requires video, and multiple cameras add views that solve occlusion. All paths lead to running the VLM on multiple images and getting multiple feature vectors.

If you add time (T frames) and cameras (C views), each feature vector of dimension d gives you a `(T, C, d)` tensor. Naively concatenating everything into one big vector and dumping it into the flow head MLP is a bad idea - you'd be asking an MLP to learn temporal relationships from a flat vector, which is the same argument that led to CNNs.

The cleaner approach: add a small per-camera transformer that takes `(T, D)` → `(1, D)`, using self-attention to mix temporal information into a single latent that captures the "physics" of that view. This gives `(C, D)` as the final conditioning, which is manageable since C is usually 2–3. Whether to then collapse C further is less clear - mixing views might require more training data than collapsing time does.

There's another downstream problem: if input frames come at a stride (200ms is reasonable - 20ms is useless since consecutive frames are nearly identical, and you might want ~1s of context, i.e. 5 frames at 200ms), then the feature vector is being computed at ~1 Hz. That means the model outputs actions at 1 Hz, which is too slow for robot control. This is exactly why Pi did action chunking - predict a short sequence of actions at once, execute them open-loop, then re-plan.

### Other TODOs

- [ ] TODO: Action normalization - currently raw; tanh clamping or per-dim dataset normalization needed for real data
- [ ] TODO: Policy head is hardcoded to flow matching; pluggable diffusion / deterministic MLP would be easy to add

---

## Usage

### Install

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py                        # fresh run
python train.py --resume               # resume from latest checkpoint
python train.py --run_id experiment_1  # custom run name (default: timestamp)
```

Checkpoints saved to `checkpoints/<run_id>/`.

### Eval

```bash
python eval.py --checkpoint checkpoints/<run_id>/best.pt
python eval.py --checkpoint checkpoints/<run_id>/best.pt --num_steps 50
```

Prints action shape, MSE vs ground truth, and first 3 predicted vs GT actions.

### Shape check

```bash
python test_shapes.py
```

Runs a forward pass and validates all intermediate tensor shapes without training.

---

## Data

`SyntheticVLADataset` generates random tensors on first use and caches them to disk as `.pt` files. The interface (`__getitem__` returning a dict of named tensors) is designed to be a drop-in swap for real datasets.
