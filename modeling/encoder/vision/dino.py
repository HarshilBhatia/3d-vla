import torch
from torch import nn
from transformers import Dinov2Model

DINOV2_MODEL = "facebook/dinov2-base"


class DiNOv2Transform(nn.Module):
    """DINOv2 image normalization: ImageNet mean/std."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        )

    def forward(self, img):
        return (img - self.mean) / self.std


class DiNOv2Features(nn.Module):
    """
    DINOv2 ViT backbone.
    Returns a 2D spatial feature map (B, hidden_size, h, w).
    Works with any square input resolution; h = w = sqrt(num_patches).
    """

    def __init__(self, model_name=DINOV2_MODEL):
        super().__init__()
        self.model = Dinov2Model.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size

    def forward(self, x):
        # x: (B, 3, H, W)
        if not torch.is_autocast_enabled():
            model_dtype = next(self.model.parameters()).dtype
            if x.dtype != model_dtype:
                x = x.to(model_dtype)
        tokens = self.model(pixel_values=x).last_hidden_state  # (B, 1+h*w, C)
        tokens = tokens[:, 1:]                                   # strip CLS: (B, h*w, C)
        B, N, C = tokens.shape
        h = w = int(N ** 0.5)
        tokens = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)
        return tokens


def load_dinov2(model_name=DINOV2_MODEL):
    backbone = DiNOv2Features(model_name)
    normalize = DiNOv2Transform()
    return backbone, normalize
