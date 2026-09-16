import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation

from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


class Encoder(BaseEncoder):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 scene_sampling='image_space',
                 finetune_backbone=False,
                 finetune_text_encoder=False,
                 rot_dim=3):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            scene_sampling=scene_sampling,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Postprocess scene features
        if self._backbone_name == 'clip':
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res4"
            )
            self.rgb2d_proj = nn.Conv2d(2048, embedding_dim, 1)
        elif self._backbone_name == 'siglip2':
            self.siglip2_proj = nn.Conv2d(self.backbone.hidden_size, embedding_dim, 1)

        # Camera ids
        self.camera_ids = nn.Embedding(5, embedding_dim)

        # Proprioception learnable projection if no 3D is used
        self.rot_dim = rot_dim
        self.proprio_feat = nn.Linear(3 + rot_dim, embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+)
            - context_feats: (B, npt, C)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        return self.proprio_feat(proprio[..., :3 + self.rot_dim])

    def encode_clip(self, rgb3d, rgb2d, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb3d: (B, ncam3d, 3, H, W), rgb obs of 3D cameras
            - rgb2d: (B, ncam2d, 3, H, W), rgb obs of 2D cameras
            - pcd: (B, ncam3d, 3, H, W) or None
            - text: [str] of len=B, text instruction

        Returns:
            - rgb3d_feats: (B, Np, F)
            - rgb2d_feats: (B, ncam2d, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        has_history = rgb3d.ndim == 6
        instr_base = None
        if has_history:
            batch, nhist, ncam = rgb3d.shape[:3]
            rgb3d = rgb3d.reshape(batch * nhist, ncam, *rgb3d.shape[3:])

        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)
        instr_feats = self.maybe_drop_lang(instr_feats)
        if has_history:
            instr_base = instr_feats
            instr_feats = instr_feats.unsqueeze(1).expand(
                -1, nhist, -1, -1
            ).reshape(batch * nhist, instr_feats.shape[1], instr_feats.shape[2])

        # 3D camera features (not 3D, we just keep the naming convention)
        rgb3d_feats = None
        if rgb3d is not None:
            num_cameras = rgb3d.shape[1]
            _bt = rgb3d.shape[0]
            # Pass each view independently through backbone
            rgb3d = rgb3d.reshape(-1, *rgb3d.shape[2:])
            rgb3d = self.normalize(rgb3d)
            rgb3d_feats = self.backbone(rgb3d)
            if self._backbone_name == 'clip':
                rgb3d_feats = self.feature_pyramid(rgb3d_feats)["res4"]
            elif self._backbone_name == 'siglip2':
                rgb3d_feats = self.siglip2_proj(rgb3d_feats)
            else:
                raise ValueError(f"2D encoder does not support backbone={self._backbone_name}")
            # Add camera id embeddings
            _c, _fh, _fw = rgb3d_feats.shape[1], rgb3d_feats.shape[2], rgb3d_feats.shape[3]
            rgb3d_feats = rgb3d_feats.reshape(_bt, num_cameras, _c, _fh, _fw)
            rgb3d_feats = rgb3d_feats + self.camera_ids.weight[:num_cameras][
                None, :, :, None, None
            ]
            # Merge different cameras
            rgb3d_feats = rgb3d_feats.permute(0, 1, 3, 4, 2).reshape(_bt, num_cameras * _fh * _fw, _c)
            # Attention from vision to language
            rgb3d_feats = self.vl_attention(seq1=rgb3d_feats, seq2=instr_feats)[-1]

        # 2D camera features
        rgb2d_feats = None

        pcd_out = torch.zeros(
            rgb3d_feats.shape[0], rgb3d_feats.shape[1], 3,
            device=rgb3d_feats.device, dtype=rgb3d_feats.dtype
        )
        if has_history:
            rgb3d_feats = rgb3d_feats.reshape(batch, nhist, *rgb3d_feats.shape[1:])
            pcd_out = pcd_out.reshape(batch, nhist, *pcd_out.shape[1:])
        return rgb3d_feats, rgb2d_feats, pcd_out, instr_base if has_history else instr_feats

    # The base encoder dispatches by backbone name. This implementation is
    # shared by CLIP and SigLIP2; the visual-backbone branch above handles the
    # feature-map difference while the rest of the 2D pipeline is identical.
    def encode_siglip2(self, rgb3d, rgb2d, pcd, text):
        return self.encode_clip(rgb3d, rgb2d, pcd, text)
