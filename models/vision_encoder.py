import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor
from config import cfg

MODEL_ID = cfg.backbone.model_id


class VisionEncoder(nn.Module):

    def __init__(self, backbone: SiglipVisionModel = None):
        """
        Args:
            backbone: optional pre-loaded SiglipVisionModel.
        """
        super().__init__()
        self.model = backbone if backbone is not None else SiglipVisionModel.from_pretrained(MODEL_ID)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            vision_emb: (B, 768)
        """
        return self.model(pixel_values=pixel_values).pooler_output


def get_image_processor() -> SiglipImageProcessor:
    """converts raw PIL/numpy images to pixel_values tensors."""
    return SiglipImageProcessor.from_pretrained(MODEL_ID)
