import torch
import torch.nn as nn
from transformers import SiglipModel
from models.vision_encoder import VisionEncoder, MODEL_ID
from models.language_encoder import LanguageEncoder
from models.proprio_encoder import ProprioEncoder
from models.policy_head import FlowPolicyHead
from config import cfg


class VLAModel(nn.Module):
    def __init__(
        self,
        proprio_dim: int = cfg.robot.proprio_dim,
        action_dim:  int = cfg.robot.action_dim,
        t_embed_dim: int = cfg.policy.t_embed_dim,
        hidden_dim:  int = cfg.policy.hidden_dim,
        num_layers:  int = cfg.policy.num_layers,
    ):
        super().__init__()

        # SigLIP is loaded once and split between the two encoders
        siglip = SiglipModel.from_pretrained(MODEL_ID)
        self.vision_encoder   = VisionEncoder(backbone=siglip.vision_model)
        self.language_encoder = LanguageEncoder(backbone=siglip.text_model)
        self.proprio_encoder = ProprioEncoder(proprio_dim)

        cond_dim = cfg.backbone.vision_dim + cfg.backbone.lang_dim + proprio_dim

        self.policy_head = FlowPolicyHead(
            action_dim=action_dim,
            cond_dim=cond_dim,
            t_embed_dim=t_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def _fuse(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        proprio:        torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            conditioning (B, VISION_DIM + LANG_DIM + proprio_dim)
        """
        vision_emb  = self.vision_encoder(pixel_values)
        lang_emb    = self.language_encoder(input_ids, attention_mask)
        proprio_emb = self.proprio_encoder(proprio)
        return torch.cat([vision_emb, lang_emb, proprio_emb], dim=-1)

    def loss(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        proprio:        torch.Tensor,
        action:         torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values:   (B, 3, 224, 224)
            input_ids:      (B, 64)
            attention_mask: (B, 64)
            proprio:        (B, proprio_dim)
            ground truth action:         (B, action_dim)
        Returns:
            scalar flow matching loss
        """
        cond = self._fuse(pixel_values, input_ids, attention_mask, proprio)
        return self.policy_head.loss(action, cond)

    @torch.no_grad()
    def infer(
        self,
        pixel_values:   torch.Tensor,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        proprio:        torch.Tensor,
        num_steps:      int = 50,       # Euler integration steps
    ) -> torch.Tensor:
        """
        Returns:
            action: (B, action_dim)
        """
        cond = self._fuse(pixel_values, input_ids, attention_mask, proprio)
        return self.policy_head.infer(cond, num_steps)