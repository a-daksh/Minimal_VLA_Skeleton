import torch
from torch.utils.data import Dataset
from pathlib import Path
from config import cfg


class SyntheticVLADataset(Dataset):
    """
    Toy dataset with random tensors. Lazy / on-disk: each sample is saved as an
    individual .pt file so __getitem__ reads from disk without loading everything
    into memory. Swap implementation for real data later — interface stays the same.

    Each sample (all float32 unless noted):
        pixel_values   (3, 224, 224)   dummy image — already in encoder-ready form
        input_ids      (64,)  int64    dummy token ids
        attention_mask (64,)  int64    all-ones (full attention, no padding)
        proprio        (proprio_dim,)  joint positions / velocities / gripper state
        action         (action_dim,)   ground truth action
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_samples: int = 1000,
        proprio_dim: int = cfg.robot.proprio_dim,
        action_dim:  int = cfg.robot.action_dim,
    ):
        self.root = Path(root) / split
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.num_samples = num_samples

        existing = list(self.root.glob("*.pt"))
        if len(existing) < num_samples:
            self._generate(num_samples)

    def _generate(self, n: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            sample = {
                "pixel_values":   torch.randn(3, cfg.backbone.image_size, cfg.backbone.image_size),
                "input_ids":      torch.randint(0, 32000, (cfg.backbone.seq_len,), dtype=torch.long),
                "attention_mask": torch.ones(cfg.backbone.seq_len, dtype=torch.long),
                "proprio":        torch.randn(self.proprio_dim),
                "action":         torch.randn(self.action_dim),
            }
            torch.save(sample, self.root / f"sample_{i:06d}.pt")
        print(f"Generated {n} synthetic samples → {self.root}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.root / f"sample_{idx:06d}.pt", weights_only=True)
