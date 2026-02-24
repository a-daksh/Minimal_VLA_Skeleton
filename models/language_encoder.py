import torch
import torch.nn as nn
from transformers import SiglipTextModel, AutoTokenizer

# Same model as vision encoder, weights are shared at the SiglipModel level.
# When loading both encoders together in vla_model.py, load SiglipModel once
# and pass .vision_model / .text_model to avoid loading the checkpoint twice.
MODEL_ID = "google/siglip-base-patch16-224"


class LanguageEncoder(nn.Module):
    # NOTE: SiglipTextModel has a fixed max sequence length of 64 tokens. 
    # That's fine for short robot instructions but anything longer gets
    # truncated by the tokenizer by default

    def __init__(self):
        super().__init__()
        self.model = SiglipTextModel.from_pretrained(MODEL_ID)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len) — from get_tokenizer()
            attention_mask: (B, seq_len) — from get_tokenizer()
        Returns:
            lang_emb: (B, 768)
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).pooler_output


def get_tokenizer() -> AutoTokenizer:
    """converts raw strings to input_ids/attention_mask tensors"""
    return AutoTokenizer.from_pretrained(MODEL_ID)
