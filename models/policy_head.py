import math
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. FourierEncoder
#    Learnable random Fourier features: sin/cos projections of t.
# ---------------------------------------------------------------------------

class FourierEncoder(nn.Module):
    """Lifted from Flow_Matching_and_Diffusion_Models/models.py"""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B, 1) or (B,) — time in [0, 1]
        Returns:
            t_emb: (B, dim)
        """
        t = t.view(-1, 1)                               # (B, 1)
        freqs = t * self.weights * 2 * math.pi          # (B, half_dim)
        return torch.cat([torch.sin(freqs),
                          torch.cos(freqs)], dim=-1) * math.sqrt(2)  # (B, dim)


# ---------------------------------------------------------------------------
# 2. VelocityMLP
#    Predicts the velocity field u_θ(x_t, t, cond).
#    Input:  concat[x_t (action_dim) | t_emb (t_embed_dim) | cond (cond_dim)]
#    Output: velocity (action_dim) — same shape as x_t
# ---------------------------------------------------------------------------

class VelocityMLP(nn.Module):
    def __init__(self, action_dim: int, t_embed_dim: int, cond_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        in_dim = action_dim + t_embed_dim + cond_dim

        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, action_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:   (B, action_dim) — noisy action at time t
            t_emb: (B, t_embed_dim) — from FourierEncoder
            cond:  (B, cond_dim)   — fused vision+language+proprio embedding
        Returns:
            velocity: (B, action_dim)
        """
        return self.net(torch.cat([x_t, t_emb, cond], dim=-1))


# ---------------------------------------------------------------------------
# 3. FlowPolicyHead
#    Owns FourierEncoder + VelocityMLP.
#    Exposes:
#      .loss(x1, cond)       — for the training loop
#      .infer(cond, steps)   — Euler integration at inference
#
#    Flow matching recap (linear interpolant):
#      x_t = t * x1 + (1-t) * x0,  x0 ~ N(0,I)
#      true velocity = x1 - x0   (d/dt of the above, constant in t)
#      training loss  = MSE(model_velocity, true_velocity)
#      inference      = Euler: x += model_velocity * dt, from t=0 to t=1
# ---------------------------------------------------------------------------

class FlowPolicyHead(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, t_embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.action_dim = action_dim
        self.time_embedder = FourierEncoder(t_embed_dim)
        self.mlp = VelocityMLP(action_dim, t_embed_dim, cond_dim, hidden_dim, num_layers)

    def loss(self, x1: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Computes flow matching training loss.

        Args:
            x1:   (B, action_dim) — ground truth action
            cond: (B, cond_dim)   — fused conditioning embedding
        Returns:
            scalar MSE loss
        """
        B = x1.shape[0]
        x0 = torch.randn_like(x1)                              # noise
        t  = torch.rand(B, 1, device=x1.device)                # t ~ Uniform(0,1)
        x_t = t * x1 + (1 - t) * x0                           # linear interpolant
        true_velocity = x1 - x0                                # d/dt of interpolant

        t_emb = self.time_embedder(t)
        pred_velocity = self.mlp(x_t, t_emb, cond)

        return nn.functional.mse_loss(pred_velocity, true_velocity)

    @torch.no_grad()
    def infer(self, cond: torch.Tensor, num_steps: int = 50) -> torch.Tensor:
        """
        Euler integration from x0~N(0,I) at t=0 to action at t=1.

        Args:
            cond:      (B, cond_dim) — fused conditioning embedding
            num_steps: number of Euler steps (more = smoother, slower)
        Returns:
            action: (B, action_dim)
        """
        B = cond.shape[0]
        x = torch.randn(B, self.action_dim, device=cond.device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((B, 1), t_val, device=cond.device)
            t_emb = self.time_embedder(t)
            velocity = self.mlp(x, t_emb, cond)
            x = x + velocity * dt

        return x
