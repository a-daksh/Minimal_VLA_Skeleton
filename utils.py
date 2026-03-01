import itertools
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vla_model import VLAModel


def set_seeds(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def infinite(loader: DataLoader):
    return itertools.cycle(loader)

def save_checkpoint(
    path: Path,
    step: int,
    model: VLAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_val_loss: float,
    run_id: str,
) -> None:
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":          step,
        "run_id":        run_id,
        "best_val_loss": best_val_loss,
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
    }, path)


def load_checkpoint(
    path: Path,
    model: VLAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float, str]:
    
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["step"], ckpt["best_val_loss"], ckpt["run_id"]
