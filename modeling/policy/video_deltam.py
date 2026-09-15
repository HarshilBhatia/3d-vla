"""Sparse causal Video-DeltaM attention over per-camera frame tokens."""

import torch
from torch import nn


class _Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key_value=None, attn_mask=None):
        key_value = query if key_value is None else key_value
        out = self.attn(query, key_value, key_value, attn_mask=attn_mask, need_weights=False)[0]
        x = self.norm1(query + out)
        return self.norm2(x + self.ffn(x))


class VideoDeltaM(nn.Module):
    """Video-DeltaM sparse attention stack.

    The legacy input is one pooled visual token per ``(history time, camera)``.
    The full-image path instead appends a learned per-image token to every
    image's visual-token sequence. The per-image token is excluded from
    same-time cross-camera attention so it retains a camera-specific identity;
    it participates only in its own camera's causal temporal attention. Each
    depth performs exactly two operations:

    1. visual patch tokens mix only with other cameras at their own timestamp;
    2. each camera attends causally to its own earlier timestamps;
    There is intentionally no frame-to-future or arbitrary cross-camera,
    cross-time attention, and no global camera-readout attention.
    """

    def __init__(self, embedding_dim, num_heads, depth, max_history=32, max_cameras=8,
                 dropout=0.1, full_image=False):
        super().__init__()
        if depth < 1:
            raise ValueError("video_deltam_depth must be at least one")
        self.max_history = max_history
        self.max_cameras = max_cameras
        self.full_image = full_image
        self.time_embedding = nn.Embedding(max_history, embedding_dim)
        self.camera_embedding = nn.Embedding(max_cameras, embedding_dim)
        # Instantiated once per (history time, camera) image and appended last
        # to that image's visual sequence in the full-image path.
        if full_image:
            self.image_token = nn.Parameter(torch.empty(1, 1, 1, 1, embedding_dim))
            nn.init.normal_(self.image_token, std=0.02)
        self.same_time = nn.ModuleList([_Block(embedding_dim, num_heads, dropout) for _ in range(depth)])
        self.temporal = nn.ModuleList([_Block(embedding_dim, num_heads, dropout) for _ in range(depth)])

    def forward(self, frame_tokens):
        """Return refined per-image frame tokens.

        Args:
            frame_tokens: legacy ``(B, K, M, C)`` pooled frame tokens, or,
                with ``full_image=True``, ``(B, K, M, P, C)`` full visual
                token sequences. Time is ordered oldest to current.
        """
        if self.full_image:
            if frame_tokens.ndim != 5:
                raise ValueError(
                    "full-image Video-DeltaM expects frame tokens shaped (B, K, M, P, C), "
                    f"got {tuple(frame_tokens.shape)}"
                )
            bsz, history, ncam, npatch, channels = frame_tokens.shape
            if npatch < 1:
                raise ValueError("full-image Video-DeltaM requires at least one visual token per image")
            # (B, K, M, P+1, C): the last token is a learned summary for this
            # particular image. It is deliberately held out of same-time
            # cross-camera SA, and enters only temporal SA below.
            frame_tokens = torch.cat(
                [frame_tokens, self.image_token.expand(bsz, history, ncam, 1, channels)], dim=3
            )
        elif frame_tokens.ndim == 4:
            bsz, history, ncam, _ = frame_tokens.shape
        else:
            raise ValueError(
                "pooled Video-DeltaM expects frame tokens shaped (B, K, M, C), "
                f"got {tuple(frame_tokens.shape)}"
            )
        if history > self.max_history or ncam > self.max_cameras:
            raise ValueError(
                f"Video-DeltaM got K={history}, cameras={ncam}; configured maxima are "
                f"{self.max_history}, {self.max_cameras}"
            )
        time_ids = torch.arange(history, device=frame_tokens.device)
        cam_ids = torch.arange(ncam, device=frame_tokens.device)
        if self.full_image:
            # Add time/camera identity to every patch and its image token.
            frames = frame_tokens + self.time_embedding(time_ids)[None, :, None, None] \
                + self.camera_embedding(cam_ids)[None, None, :, None]
            tokens_per_image = frames.shape[3]
        else:
            frames = frame_tokens + self.time_embedding(time_ids)[None, :, None] + self.camera_embedding(cam_ids)[None, None]
            tokens_per_image = 1
        # True above means a query cannot see a later key. The time axis is
        # oldest -> current, therefore this is strictly causal. In the full
        # image path this is a block mask: every token at t can see every token
        # from its own and earlier images, never a future image.
        time_mask = torch.triu(
            torch.ones(history, history, dtype=torch.bool, device=frames.device), diagonal=1
        )
        causal_mask = (
            time_mask.repeat_interleave(tokens_per_image, dim=0).repeat_interleave(tokens_per_image, dim=1)
            if self.full_image else time_mask
        )
        for same_time, temporal in zip(self.same_time, self.temporal):
            # At a timestamp, patches from all cameras may mix, but learned
            # per-image tokens are excluded so they do not lose their camera
            # identity. No token crosses time in this operation.
            if self.full_image:
                patches, image_tokens = frames[..., :-1, :], frames[..., -1:, :]
                patches = same_time(
                    patches.reshape(bsz * history, ncam * (tokens_per_image - 1), -1)
                ).reshape(bsz, history, ncam, tokens_per_image - 1, -1)
                frames = torch.cat([patches, image_tokens], dim=3)
                # (B*M, K*(P+1), C): for one camera, each token can attend to
                # all patches and its per-image token from that camera's
                # causal image history.
                temporal_frames = frames.transpose(1, 2).reshape(
                    bsz * ncam, history * tokens_per_image, -1
                )
                frames = temporal(temporal_frames, attn_mask=causal_mask).reshape(
                    bsz, ncam, history, tokens_per_image, -1
                ).transpose(1, 2)
            else:
                frames = same_time(frames.reshape(bsz * history, ncam, -1)).reshape(
                    bsz, history, ncam, -1
                )
                # (B*M, K, C): each camera sees only its own causal temporal path.
                temporal_frames = frames.transpose(1, 2).reshape(bsz * ncam, history, -1)
                frames = temporal(temporal_frames, attn_mask=causal_mask).reshape(bsz, ncam, history, -1).transpose(1, 2)
        # Keep the old downstream interface: the learned final token is the
        # refined per-image feature used to predict one Delta-M per camera.
        return frames[..., -1, :] if self.full_image else frames
