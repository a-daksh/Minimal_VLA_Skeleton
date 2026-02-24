# VLA Policy — Requirements & Decisions

**Purpose:** Single source of truth for implementation. **Read this first** when continuing the project — a new chat can read this and have full context.

---

## Project Context

- **What:** VLA (Vision-Language-Action) policy for a course assignment
- **Inputs:** Vision (single image for toy), proprioception, language/task embedding
- **Output:** Continuous action vector
- **Scope:** Toy/synthetic data; shape-correct forward pass; trainable; industry-ready structure
- **Not required:** Actually training to convergence; real robot datasets

**Repo structure:**
```
Minimal_VLA_Skeleton/
├── config.py           # (or config loading from YAML)
├── train.py            # Training loop
├── eval.py             # Inference: load checkpoint, run forward
├── README.md
├── models/
│   ├── vision_encoder.py   # Image → embedding
│   ├── policy_head.py     # Flow matching action head
│   └── vla_model.py       # Wires everything together
└── data/
    └── dataset.py         # Synthetic/toy dataset (lazy, on-disk)
```

---

## Build Order

Build in this sequence — each step locks in shapes/contracts that the next depends on:

1. **Vision encoder** (`models/vision_encoder.py`) — wraps frozen SigLIP image backbone; establishes `D_vision=768` and required image resolution
2. **Language encoder** (`models/vision_encoder.py` or separate) — wraps frozen SigLIP text encoder from the same model; establishes `D_lang=768`
3. **Dataset** (`data/dataset.py`) — now that encoder input formats are known (image size, tokenizer), define `(image, proprio, language_string, action)` with real dims; everything downstream conforms to this
4. **Policy head** (`models/policy_head.py`) — flow matching MLP; by this point `action_dim`, `proprio_dim`, and fused embedding dim are all known from data
5. **VLA model** (`models/vla_model.py`) — wires encoders + fusion + policy head
6. **Train loop** (`train.py`) — bf16 AMP, step-based, checkpointing
7. **Eval** (`eval.py`) — load checkpoint, Euler integration, output action

**Config:** grows organically alongside each step — add fields to `config.py`/YAML as each component needs them, rather than as a separate upfront task.

**Note:** Vision and language encoders share the same SigLIP model — load it once, use for both modalities to avoid double memory cost.

---

## Architecture Overview

```
[Image]        → Vision Encoder (SigLIP) → vision_emb (B, D_vision)
[Proprio]      → Proprio encoder        → proprio_emb
[Language]     → Language encoder      → task_emb
                      │
                      ▼
              Fusion (concat or similar)
                      │
                      ▼
              fused_emb (B, D_fused)
                      │
                      ▼
              Policy Head (Flow Matching MLP)
                      │
                      ▼
              action (B, action_dim)
```

---

## Training Infrastructure

| Item | Decision |
|------|----------|
| Hardware | Single GPU (4090-class); design scalable to better single GPUs |
| Checkpointing | Full: model + optimizer + LR schedule + step |
| Resumability | Resume from latest checkpoint without losing progress |
| Logging | Simple for now; easy to add Wandb later |

---

## Data & Efficiency

| Item | Decision |
|------|----------|
| Precision | bf16 AMP (`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`) |
| Batch size | Fixed in config, easy CLI override; sensible default (32 or 64) |
| Data loading | Lazy / on-disk from day one — `__getitem__` reads from disk; works unchanged for real data |
| Gradient accumulation | Not needed |

---

## Validation & Checkpoints

| Item | Decision |
|------|----------|
| Validation | Every few steps; val loss + action MSE (predicted vs ground truth) |
| Checkpoint dir | `checkpoints/` organized by run |
| Run ID | Timestamp |
| What to keep | Both best and latest checkpoints |

---

## Configuration

| Item | Decision |
|------|----------|
| Format | YAML |
| Source of truth | All model dims (proprio_dim, action_dim, etc.), paths, training params defined in YAML |
| Loader | `config.py` loads from YAML |
| Overrides | CLI overrides for key fields |
| Validation | None — run and crash if wrong |

---

## Model & Inference

| Item | Decision |
|------|----------|
| Inference entry | `eval.py` — load checkpoint, run forward on new inputs |

---

## Reproducibility & Dependencies

| Item | Decision |
|------|----------|
| Seeds | Basic: `torch.manual_seed`, numpy, `random` — reasonable reproducibility |
| CuDNN | Default non-deterministic (bit-for-bit not required, avoids slowdown) |
| Dependencies | `requirements.txt` with pinned versions |
| Env | Conda |

---

## Training Loop

| Item | Decision |
|------|----------|
| Termination | Steps (not epochs) |

---

## Action Head (Flow Matching)

| Item | Decision |
|------|----------|
| Architecture | Conditional MLP (no CFG) |
| Action dim | Config-driven |
| Conditioning | Always condition — no classifier-free guidance |
| CFG note | Could add CFG later if model doesn't respond strongly to conditioning; unlikely bottleneck |
| Inference steps | In config |

**How flow matching works**
- Model predicts **velocity field** u(x_t, t | cond), not x_t or the action directly.
- **Inputs:** x_t (current position on interpolant), t (time ∈ [0,1]), cond (fused embedding).
- **Output:** velocity vector same shape as x_t.
- **Training:** Sample (x_0=noise, x_1=gt action), t~Uniform(0,1); x_t = t·x_1 + (1-t)·x_0; loss = MSE(predicted u, true u from linear flow formula).
- **Inference:** Sample x_0~N(0,I), integrate ODE dx/dt = u_θ(x,t,cond) from t=0→1 (Euler steps), x at t=1 is the action.
- Linear interpolant: α(t)=t, β(t)=1-t; true velocity formula in flow-matching literature.

**MLP architecture (fixed, not config):** Time embedder (Fourier) → concat [x_t, t_embed, cond] → MLP (2–4 layers, 256–512 hidden, SiLU) → output to action_dim.

**Flow repo reference:** https://github.com/a-daksh/Flow_Matching_and_Diffusion_Models/tree/torch — reuse probability_paths math, FourierEncoder, Euler integration; adapt for vectors and continuous conditioning; write new MLP (no U-Net).

**TODO**
- Action range / normalization: decide later (tanh, dataset normalization, or raw)

---

## Vision Encoder

**Current implementation:** 1 camera, 1 frame.

| Item | Decision |
|------|----------|
| Cameras | 1 (for now) |
| Frames | 1 (for now) |
| Image size | Match backbone native resolution (e.g. 224×224); resize/center-crop |
| Backbone | SigLIP; pretrained |
| Frozen | Yes (for toy) |
| Output | Vision embedding `(B, D_vision)` |

**NOTE:** 
I was looking into what to use as vision encoder and CLIP was the first choice, but given the choice of SigLIP in Pi i was exploring the cons of these. CLIP uses global pooling which loses spatial relationship i.e. if given a task like 'pick up the mug on your left' it will perform poor given the notion of left an right has gone in the pooling operation. Although SigLIP also has global pooling it also allows bypassing global pooling to extract patch tokens instead, preserving spatial info. Currently we will use global (single vector) but in **Future** bypass pooling, extract patch tokens `(B, N_patches, D)` → add spatial adapter before fusion.

**Pretrained constraints**
- Image size must match backbone’s trained resolution — don’t treat it as free config.
- Single-image backbones (SigLIP, CLIP, ViT) expect one image. For multi-frame, run per frame then aggregate.

---

## Language Encoder

| Item | Decision |
|------|----------|
| Backbone | Frozen SigLIP text encoder (same model as vision) |
| Projection | None — use encoder output directly |
| Output | `(B, D_lang)` |

**NOTE:** SigLIP/CLIP text encoders were trained on general captions, not robot instructions. Robot-specific text ("pick up the red block") may not be ideal for this prior. If conditioning is weak we have a few options
- add a trainable projection after the encoder, but this means more learning to be done i.e. more data need
- or maybe we can switch to an instruction-tuned LM (T5, Flan-T5) for stronger language prior.

---

**NOTE — :**
Hmm so this took a lot of thought and i will jsut dump all out here
1. Right now we are giving 1 frame and 1 camera. In real settings this cannot be used. Physics capturing requires video, and number of cameras depend on the robot and can add more 'views' solving things like occlusion. So surely we need to handle that
2. The problem- VLMs dont take in video or multiple views, they take in image and return feature vector. 
3. There are multiple ways to handle this but at its heart all lead to apssing multiple images one by one through VLM and getting multiple feature vectors.
4. Assume one feature vector is of dimension `d` . Now adding time we get T more of these, and adding cameras add C more of these. So in total `T,C,d`
Idea -
Concatenate all together to get TXCXD- bad idea as increases the input dim of flow head by a lot, expects the action head architecture to learn relation between it. and if action head is MLP we expect temporal relation to be drawn in a single vector, which is wrong (The argument that lead to CNNs)
Solution -
Since time vectors per camera carry a lot of redundant info, maybe we add a small transformer that takes in TXD and gives 1XD. Why transformer? Because it can relate time dimension through self attention and give one feature vector corresponding to each camera which encapsulates the 'physics' from that view. It would also result in final feature vector to be C X d which is fine as C is usually 2-3.

Maybe we can also epxlore similar thing for C dim and get a only d dim feature vector? But im not sure if that is a good idea. Because this transformer (like the time one) needs to be trained on robot data. Time dimension has a lot of simialr information and trainign a block to mix similar info into latent micht be easy, but mixing views might increase data requirement. 

Maybe some other thing inplace of transformers can be used? But given how everything runs on transformers, i dont see an issue here unless compute becomes a bottleneck. 

Another issue i see, is control, if we look at our input to action head i.e. the above caluclated feature vector, it takes in multiple frames across time, which is usually done with a stride. Practically a 20ms stride will be of no use as it is going to be very similar to previous one. 200ms sounds decent and we also might need a 1s input. i.e. 5 frames at 200ms stride. This means our action head input is coming at 1hz, thereby giving actions at 1hz, which is bad for robots. THAT IS EXACTLY WHY pi GUYS DID ACTION CHUNKING. So can think of that
---

## Dataset

**Strategy:** Use random/synthetic values for now — verify shapes, forward pass, and training loop. Replace with real dataset (e.g. Language-Table, CALVIN, Bridge) later; same `Dataset` interface, swap implementation.

---

## Fusion

| Item | Decision |
|------|----------|
| Method | Concat (stack) all modalities |
| Vision | 768 (SigLIP Base) |
| Language | 768 (SigLIP Base) |
| Proprio | Config-driven. Proprio — includes position, velocity, etc. |
| Total | 1536 + proprio_dim |

---

## Open Questions (not yet decided)

| Topic | Status |
|-------|--------|
| Toy dataset layout | On-disk format, shapes; synthetic first |

---

## Deferred / TBD

- Policy head modularity: pluggable Flow vs Diffusion vs deterministic MLP (future)

---