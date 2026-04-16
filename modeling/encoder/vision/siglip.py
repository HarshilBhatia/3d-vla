import torch
from torch import nn
from transformers import SiglipVisionModel

SIGLIP2_MODEL = "google/siglip2-base-patch16-256"


class SigLIP2Transform(nn.Module):
    """SigLIP2 image normalization: maps [0, 1] -> [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        )

    def forward(self, img):
        return (img - self.mean) / self.std


class SigLIP2ViTFeatures(nn.Module):
    """
    SigLIP2 ViT backbone.
    Returns a 2D spatial feature map (B, hidden_size, h, w) where
    h = w = image_size // patch_size.
    """

    def __init__(self, model_name=SIGLIP2_MODEL):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(model_name)
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.h = cfg.image_size // cfg.patch_size
        self.w = cfg.image_size // cfg.patch_size

    def forward(self, x):
        # x: (B, 3, H, W) where H == W == model image_size
        feats = self.model(pixel_values=x).last_hidden_state  # (B, h*w, hidden_size)
        B = feats.shape[0]
        feats = feats.reshape(B, self.h, self.w, self.hidden_size)
        feats = feats.permute(0, 3, 1, 2)  # (B, hidden_size, h, w)
        return feats


def load_siglip2(model_name=SIGLIP2_MODEL):
    backbone = SigLIP2ViTFeatures(model_name)
    normalize = SigLIP2Transform()
    return backbone, normalize
