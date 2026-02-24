"""
Shape validation: dataset → vision encoder → language encoder.
Run this before building the policy head to confirm all dims are correct.

    python test_shapes.py
"""

import torch
from torch.utils.data import DataLoader

from data.dataset import SyntheticVLADataset
from models.vision_encoder import VisionEncoder
from models.language_encoder import LanguageEncoder

BATCH_SIZE  = 4
DATA_ROOT   = "/tmp/vla_test_data"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def check(name: str, tensor: torch.Tensor, expected: tuple) -> None:
    status = "OK" if tuple(tensor.shape) == expected else f"FAIL — expected {expected}"
    print(f"  {name:30s} {str(tuple(tensor.shape)):25s} {status}")


def main():
    print(f"Device: {DEVICE}\n")

    # --- Dataset ---
    print("[ Dataset ]")
    dataset = SyntheticVLADataset(root=DATA_ROOT, split="train", num_samples=16)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch   = next(iter(loader))

    check("pixel_values",   batch["pixel_values"],   (BATCH_SIZE, 3, 224, 224))
    check("input_ids",      batch["input_ids"],       (BATCH_SIZE, 64))
    check("attention_mask", batch["attention_mask"],  (BATCH_SIZE, 64))
    check("proprio",        batch["proprio"],         (BATCH_SIZE, 9))
    check("action",         batch["action"],          (BATCH_SIZE, 7))

    # --- Vision encoder ---
    print("\n[ Vision Encoder ]")
    vision_enc = VisionEncoder().to(DEVICE)
    pixel_values = batch["pixel_values"].to(DEVICE)
    vision_emb = vision_enc(pixel_values)
    check("vision_emb", vision_emb, (BATCH_SIZE, 768))

    # --- Language encoder ---
    print("\n[ Language Encoder ]")
    lang_enc = LanguageEncoder().to(DEVICE)
    input_ids      = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    lang_emb = lang_enc(input_ids, attention_mask)
    check("lang_emb", lang_emb, (BATCH_SIZE, 768))

    # --- Fusion preview ---
    print("\n[ Fusion (concat) ]")
    proprio = batch["proprio"].to(DEVICE)
    fused = torch.cat([vision_emb, lang_emb, proprio], dim=-1)
    expected_fused = 768 + 768 + 9  # 1545
    check("fused_emb", fused, (BATCH_SIZE, expected_fused))

    print(f"\nPolicy head will receive conditioning of dim: {expected_fused}")
    print(f"Action dim (policy head output):              {batch['action'].shape[-1]}")


if __name__ == "__main__":
    main()
