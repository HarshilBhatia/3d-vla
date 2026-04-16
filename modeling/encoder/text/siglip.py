import torch
from torch import nn
from transformers import AutoTokenizer, SiglipTextModel

SIGLIP2_MODEL = "google/siglip2-base-patch16-256"
SIGLIP2_MAX_LENGTH = 64  # SigLIP2 text sequence length


class SigLIP2Tokenizer:

    def __init__(self, model_name=SIGLIP2_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            padding="max_length",
            max_length=SIGLIP2_MAX_LENGTH,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]


class SigLIP2TextEncoder(nn.Module):

    def __init__(self, model_name=SIGLIP2_MODEL):
        super().__init__()
        self.model = SiglipTextModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
