import torch
import torch.nn as nn


class ProprioEncoder(nn.Module):
    def __init__(self, proprio_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, proprio_dim),
            nn.SiLU(),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio)
