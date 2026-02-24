import torch
import torch.nn as nn
from transformers import SiglipVisionModel, SiglipImageProcessor

# We are currently using the base version
MODEL_ID = "google/siglip-base-patch16-224"


class VisionEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(MODEL_ID)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224) preprocessed by SiglipImageProcessor
        Returns:
            vision_emb: (B, 768)
        """
        return self.model(pixel_values=pixel_values).pooler_output


def get_image_processor() -> SiglipImageProcessor:
    """converts raw PIL/numpy images to pixel_values tensors."""
    return SiglipImageProcessor.from_pretrained(MODEL_ID)
