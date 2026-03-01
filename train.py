"""
python train.py
python train.py --resume
python train.py --run_id my_run
"""

import time
import argparse
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data.dataset import SyntheticVLADataset
from models.vla_model import VLAModel
from utils import set_seeds, infinite, save_checkpoint, load_checkpoint


@torch.no_grad()
def validate(model: VLAModel, loader: DataLoader, device: torch.device) -> dict:
    """
    Returns val_loss (flow matching) and val_mse (infer vs gt action).
    """
    model.eval()
    total_loss = total_mse = 0.0
    n = 0

    use_amp = device.type == "cuda"
    for batch in loader:
        pv   = batch["pixel_values"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        prop = batch["proprio"].to(device)
        act  = batch["action"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model.loss(pv, ids, mask, prop, act)
            pred = model.infer(pv, ids, mask, prop, num_steps=10)   # fewer steps for speed

        total_loss += loss.item()
        total_mse  += nn.functional.mse_loss(pred, act).item()
        n += 1

    model.train()
    return {"val_loss": total_loss / n, "val_mse": total_mse / n}


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    set_seeds()

    run_id      = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir    = Path(cfg.train.checkpoint_dir) / run_id
    latest_path = ckpt_dir / "latest.pt"
    best_path   = ckpt_dir / "best.pt"

    print(f"Run: {run_id}  |  Device: {device}")

    train_ds = SyntheticVLADataset(root=cfg.train.data_root, split="train", num_samples=800)
    val_ds   = SyntheticVLADataset(root=cfg.train.data_root, split="val",   num_samples=200)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=2, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size, shuffle=False,
                              num_workers=2, pin_memory=use_amp)

    model     = VLAModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_steps)

    start_step    = 0
    best_val_loss = float("inf")

    if args.resume and latest_path.exists():
        start_step, best_val_loss, run_id = load_checkpoint(
            latest_path, model, optimizer, scheduler, device
        )
        print(f"Resumed from step {start_step}  |  best val loss so far: {best_val_loss:.4f}")

    model.train()
    data_iter = infinite(train_loader)
    pbar = tqdm(range(start_step, cfg.train.num_steps), initial=start_step, total=cfg.train.num_steps, desc="Training")

    for step in pbar:
        batch = next(data_iter)
        pv   = batch["pixel_values"].to(device)
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        prop = batch["proprio"].to(device)
        act  = batch["action"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss = model.loss(pv, ids, mask, prop, act)

        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Validate + checkpoint
        if (step + 1) % cfg.train.val_every == 0 or step == cfg.train.num_steps - 1:
            metrics = validate(model, val_loader, device)
            tqdm.write(
                f"Step {step+1:>6d}  |  "
                f"val_loss: {metrics['val_loss']:.4f}  |  "
                f"val_mse: {metrics['val_mse']:.4f}"
            )

            save_checkpoint(latest_path, step + 1, model, optimizer, scheduler, best_val_loss, run_id)

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(best_path, step + 1, model, optimizer, scheduler, best_val_loss, run_id)
                tqdm.write(f"  ↳ New best ({best_val_loss:.4f}) — saved to {best_path}")

    print(f"\nDone. Checkpoints at {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--run_id", type=str, default=None, help="Run ID (default: timestamp)")
    
    train(parser.parse_args())