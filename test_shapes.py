import torch
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import SyntheticVLADataset
from models.vla_model import VLAModel

BATCH_SIZE = 4
DATA_ROOT  = "/tmp/vla_test_data"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def check(name: str, tensor: torch.Tensor, expected: tuple) -> None:
    status = "OK" if tuple(tensor.shape) == expected else f"FAIL: expected {expected}"
    print(f"  {name:30s} {str(tuple(tensor.shape)):25s} {status}")


def main():
    print(f"Device: {DEVICE}\n")

    # --- Dataset ---
    print("[ Dataset ]")
    dataset = SyntheticVLADataset(root=DATA_ROOT, split="train", num_samples=16)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)
    batch   = next(iter(loader))

    check("pixel_values",   batch["pixel_values"],   (BATCH_SIZE, 3, 224, 224))
    check("input_ids",      batch["input_ids"],       (BATCH_SIZE, cfg.backbone.seq_len))
    check("attention_mask", batch["attention_mask"],  (BATCH_SIZE, cfg.backbone.seq_len))
    check("proprio",        batch["proprio"],         (BATCH_SIZE, cfg.robot.proprio_dim))
    check("action",         batch["action"],          (BATCH_SIZE, cfg.robot.action_dim))

    # --- VLAModel (loads SiglipModel once, splits internally) ---
    print("\n[ VLAModel ]")
    model = VLAModel().to(DEVICE)
    pv   = batch["pixel_values"].to(DEVICE)
    ids  = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)
    prop = batch["proprio"].to(DEVICE)
    act  = batch["action"].to(DEVICE)

    vision_emb  = model.vision_encoder(pv)
    lang_emb    = model.language_encoder(ids, mask)
    proprio_emb = model.proprio_encoder(prop)
    check("vision_emb",  vision_emb,  (BATCH_SIZE, cfg.backbone.vision_dim))
    check("lang_emb",    lang_emb,    (BATCH_SIZE, cfg.backbone.lang_dim))
    check("proprio_emb", proprio_emb, (BATCH_SIZE, cfg.robot.proprio_dim))

    fused = torch.cat([vision_emb, lang_emb, proprio_emb], dim=-1)
    check("fused_emb", fused, (BATCH_SIZE, cfg.backbone.vision_dim + cfg.backbone.lang_dim + cfg.robot.proprio_dim))

    loss = model.loss(pv, ids, mask, prop, act)
    print(f"loss {loss.item():.4f} OK" if loss.ndim == 0 else "loss FAIL. Not a scalar")

    print("\n[ VLAModel: infer() ]")
    pred = model.infer(pv, ids, mask, prop, num_steps=10)
    check("predicted action", pred, (BATCH_SIZE, cfg.robot.action_dim))


if __name__ == "__main__":
    main()
