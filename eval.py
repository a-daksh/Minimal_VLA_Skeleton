import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import SyntheticVLADataset
from models.vla_model import VLAModel


def eval(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Trained for {ckpt['step']} steps  |  best val loss: {ckpt['best_val_loss']:.4f}")

    model = VLAModel().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = SyntheticVLADataset(root=cfg.train.data_root, split="val", num_samples=200)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    batch   = next(iter(loader))

    pv   = batch["pixel_values"].to(device)
    ids  = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    prop = batch["proprio"].to(device)
    act  = batch["action"].to(device)

    with torch.no_grad():
        pred = model.infer(pv, ids, mask, prop, num_steps=args.num_steps)

    mse = torch.nn.functional.mse_loss(pred, act).item()

    print(f"\nBatch size:    {pv.shape[0]}")
    print(f"Euler steps:   {args.num_steps}")
    print(f"Action shape:  {pred.shape}")
    print(f"MSE vs GT:     {mse:.4f}")
    print(f"\nPredicted actions (first 3):\n{pred[:3].cpu()}")
    print(f"\nGround truth  (first 3):\n{act[:3].cpu()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps",  type=int, default=cfg.policy.num_inference_steps)
    eval(parser.parse_args())
