"""HistoryFeatureExtractor: sparse causal attention over per-camera frame tokens.

Historically "Video-DeltaM". The module name, the config keys and the
``prediction_head.video_deltam.*`` state_dict paths keep the old spelling for
checkpoint compatibility; the class name does not, because class names never
appear in a state_dict.

Two independent axes (see ``docs/architecture.md``):

* ``predict_delta_m`` -- what the extractor hands the policy. False ("refine")
  emits history-refined camera summaries plus a global history register, and the
  policy head still predicts delta_M from them, layer-wise. True ("direct")
  emits a delta_M and nothing else; the policy's scene tokens are untouched and
  its own delta_M head is never constructed.
* ``full_image`` -- how much detail the extractor sees. False pools each
  (timestep, camera) image to one token; True keeps its whole patch grid.
"""

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


class HistoryFeatureExtractor(nn.Module):
    """Sparse causal attention stack with a global causal readout.

    The legacy input is one pooled visual token per ``(history time, camera)``.
    The full-image path instead appends a learned per-image token to every
    image's visual-token sequence. The per-image token is excluded from
    same-time cross-camera attention so it retains a camera-specific identity;
    it participates only in its own camera's causal temporal attention. Each
    depth performs exactly two operations:

    1. visual patch tokens mix only with other cameras at their own timestamp;
    2. each camera attends causally to its own earlier timestamps;
    3. the policy camera register reads the complete history-by-camera bank.

    ``camera_readout`` is part of the serialized architecture of the released
    Video-DeltaM checkpoints.  Keep both its name and its global-token output
    contract stable: changing either silently invalidates old checkpoints when
    loaded with ``strict=False``.
    """

    def __init__(self, embedding_dim, num_heads, depth, max_history=32, max_cameras=8,
                 dropout=0.1, full_image=False, predict_delta_m=False):
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
            self.image_readout = _Block(embedding_dim, num_heads, dropout)
        self.same_time = nn.ModuleList([_Block(embedding_dim, num_heads, dropout) for _ in range(depth)])
        self.temporal = nn.ModuleList([_Block(embedding_dim, num_heads, dropout) for _ in range(depth)])
        # When this stack owns delta_M it has no other output, so the global
        # register readout is not built at all.
        self.predict_delta_m = predict_delta_m
        if predict_delta_m:
            self.delta_m_head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim), nn.ReLU(),
                nn.Linear(embedding_dim, 36),
            )
            # Zero-init: delta_M = exp(0) = I, so the arm starts from no correction.
            nn.init.zeros_(self.delta_m_head[-1].weight)
            nn.init.zeros_(self.delta_m_head[-1].bias)
        else:
            # Legacy checkpoint-compatible global camera-register readout.
            self.camera_readout = nn.ModuleList(
                [_Block(embedding_dim, num_heads, dropout) for _ in range(depth)]
            )
        # Set by the checkpoint loader when evaluating a checkpoint written by
        # the pre-camera-readout implementation.  That implementation used the
        # learned image token directly (rather than image_readout) and returned
        # refined per-camera frames only.
        self._legacy_checkpoint_compat = False

    def enable_legacy_checkpoint_compat(self):
        """Use the pre-refactor full-patch computation for old checkpoints."""
        self._legacy_checkpoint_compat = True

    def _forward_legacy(self, frame_tokens):
        """Exact forward contract of the checkpoint-era HistoryFeatureExtractor stack."""
        if self.full_image:
            if frame_tokens.ndim != 5:
                raise ValueError(
                    "full-image Video-DeltaM expects frame tokens shaped (B, K, M, P, C), "
                    f"got {tuple(frame_tokens.shape)}"
                )
            bsz, history, ncam, npatch, channels = frame_tokens.shape
            if npatch < 1:
                raise ValueError("full-image Video-DeltaM requires at least one visual token per image")
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
            frames = frame_tokens + self.time_embedding(time_ids)[None, :, None, None] \
                + self.camera_embedding(cam_ids)[None, None, :, None]
            tokens_per_image = frames.shape[3]
        else:
            frames = frame_tokens + self.time_embedding(time_ids)[None, :, None] \
                + self.camera_embedding(cam_ids)[None, None]
            tokens_per_image = 1
        time_mask = torch.triu(
            torch.ones(history, history, dtype=torch.bool, device=frames.device), diagonal=1
        )
        causal_mask = (
            time_mask.repeat_interleave(tokens_per_image, dim=0).repeat_interleave(tokens_per_image, dim=1)
            if self.full_image else time_mask
        )
        for same_time, temporal in zip(self.same_time, self.temporal):
            if self.full_image:
                patches, image_tokens = frames[..., :-1, :], frames[..., -1:, :]
                patches = same_time(
                    patches.reshape(bsz * history, ncam * (tokens_per_image - 1), -1)
                ).reshape(bsz, history, ncam, tokens_per_image - 1, -1)
                frames = torch.cat([patches, image_tokens], dim=3)
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
                temporal_frames = frames.transpose(1, 2).reshape(bsz * ncam, history, -1)
                frames = temporal(temporal_frames, attn_mask=causal_mask).reshape(
                    bsz, ncam, history, -1
                ).transpose(1, 2)
        return frames[..., -1, :] if self.full_image else frames

    def _delta_m_from(self, camera_summaries):
        """(B, ncam, C) -> (B, ncam, 6, 6) orthogonal, same parameterisation as
        the head's predictor: skew-symmetrise, clamp Frobenius norm, exponentiate."""
        A_skew = self.delta_m_head(camera_summaries).reshape(*camera_summaries.shape[:-1], 6, 6)
        A = A_skew - A_skew.transpose(-1, -2)
        norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        A = A * (norm.clamp(max=3.0) / norm)
        return torch.linalg.matrix_exp(A)

    def forward(self, frame_tokens, fixed_camera_token):
        """Refine frame tokens and read history into a camera register.

        Args:
            frame_tokens: legacy ``(B, K, M, C)`` pooled frame tokens, or,
                with ``full_image=True``, ``(B, K, M, P, C)`` full visual
                token sequences. Time is ordered oldest to current.
        """
        # Full-image Video-DeltaM keeps the complete patch sequence through
        # both attention stages.  Do not collapse each image through a
        # learned-token ``image_readout`` before temporal modeling: that loses
        # the per-patch history the caller explicitly supplied.  The
        # checkpoint-era implementation already has the desired computation
        # (patches mix across cameras at each time, then patches + image token
        # mix causally across time), so use it for all full-image checkpoints.
        if self.full_image:
            refined = self._forward_legacy(frame_tokens)
            if self._legacy_checkpoint_compat or self.predict_delta_m:
                return refined, None, (
                    self._delta_m_from(refined) if self.predict_delta_m else None
                )
            cam = fixed_camera_token
            for readout in self.camera_readout:
                cam = readout(cam, refined[:, -1])
            return refined, cam, None
        if self._legacy_checkpoint_compat:
            return self._forward_legacy(frame_tokens), None, None
        if self.full_image:
            if frame_tokens.ndim != 5:
                raise ValueError(
                    "full-image Video-DeltaM expects frame tokens shaped (B, K, M, P, C), "
                    f"got {tuple(frame_tokens.shape)}"
                )
            bsz, history, ncam, npatch, channels = frame_tokens.shape
            if npatch < 1:
                raise ValueError("full-image Video-DeltaM requires at least one visual token per image")
            image_query = self.image_token.expand(bsz, history, ncam, -1, -1)
            frame_tokens = self.image_readout(
                image_query.reshape(bsz * history * ncam, 1, channels),
                frame_tokens.reshape(bsz * history * ncam, npatch, channels),
            ).reshape(bsz, history, ncam, channels)
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
        frames = frame_tokens + self.time_embedding(time_ids)[None, :, None] + self.camera_embedding(cam_ids)[None, None]
        cam = fixed_camera_token
        # True above means a query cannot see a later key. The time axis is
        # oldest -> current, therefore this is strictly causal. In the full
        # image path this is a block mask: every token at t can see every token
        # from its own and earlier images, never a future image.
        time_mask = torch.triu(
            torch.ones(history, history, dtype=torch.bool, device=frames.device), diagonal=1
        )
        readouts = [None] * len(self.same_time) if self.predict_delta_m else self.camera_readout
        for same_time, temporal, readout in zip(self.same_time, self.temporal, readouts):
            frames = same_time(frames.reshape(bsz * history, ncam, -1)).reshape(
                bsz, history, ncam, -1
            )
            # Each camera sees only its own causal temporal path.
            temporal_frames = frames.transpose(1, 2).reshape(bsz * ncam, history, -1)
            frames = temporal(temporal_frames, attn_mask=time_mask).reshape(
                bsz, ncam, history, -1
            ).transpose(1, 2)
            if readout is not None:
                # The learned policy camera register reads all cameras and history
                # at this depth.  This intentionally has no temporal mask: the
                # register is emitted only for the current action decision.
                cam = readout(cam, frames.reshape(bsz, history * ncam, -1))
        if self.predict_delta_m:
            # Sole output: one orthogonal correction per camera, from the
            # history-refined summary of the current frame.
            return frames, None, self._delta_m_from(frames[:, -1])
        return frames, cam, None
