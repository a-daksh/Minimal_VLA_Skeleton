import torch
import torch.nn as nn
from transformers import SiglipTextModel, AutoTokenizer
from config import cfg

# load SiglipModel at the VLAModel level and pass backbone here to avoid downloading twice
MODEL_ID = cfg.backbone.model_id


class LanguageEncoder(nn.Module):
    # NOTE: SigLIP tokenizer hard-caps at 64 tokens
    # fine for short robot instructions but keep in mind for future

    def __init__(self, backbone: SiglipTextModel = None):
        super().__init__()
        self.model = backbone if backbone is not None else SiglipTextModel.from_pretrained(MODEL_ID)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask:(B, seq_len)
        Returns:
            lang_emb: (B, 768)
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output


def get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_ID)
