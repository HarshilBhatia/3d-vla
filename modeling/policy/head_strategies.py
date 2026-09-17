"""
Helpers for TransformerHead: view-alignment prediction and output attention.

View alignment corrects camera-extrinsic error during cross-view fusion. Two
mechanisms exist: a RoPE mixing matrix (``delta_M``, representation space) and a
physical SE(3) correction (``R,T``). See ``docs/terminology.md``.
"""
from torch import nn


# ---- View alignment ----

class ViewAlignPredictor(nn.Module):
    """Base: no view alignment (``view_align_mode=none``)."""

    def forward(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        return None, None, None


class SE3ViewAlignPredictor(ViewAlignPredictor):
    """``view_align_mode=physical_se3``: predict a physical R,T (6D) from the
    camera token. Stores head as non-module ref to avoid circular module graph."""

    def __init__(self, head):
        super().__init__()
        object.__setattr__(self, "_head", head)  # do not register as submodule (would create cycle)

    def forward(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        # R,T comes from the camera token alone, so the per-camera scene features
        # are accepted and ignored. They are still in the signature because the
        # single call site passes them for every predictor.
        rt = self._head._predict_rt(batch_size, device)
        return rt, None, rt.detach()


class RopeViewAlignPredictor(ViewAlignPredictor):
    """``view_align_mode=rope_6d``/``rope_full``: predict per-camera delta_M RoPE
    corrections from pooled camera features.

    delta_M is a representation-space mixing matrix, NOT an estimate of a
    physical camera extrinsic -- hence "Rope", not "Extrinsics", in the name.
    Unused when the HistoryFeatureExtractor owns view alignment
    (``video_deltam_role=predict_delta_m``); see ``modeling/policy/video_deltam.py``.
    """

    def __init__(self, head):
        super().__init__()
        object.__setattr__(self, "_head", head)  # do not register as submodule (would create cycle)

    def forward(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        delta_M = self._head._predict_delta_M(batch_size, device, fps_scene_feats, fps_cam_ids)
        return None, delta_M, delta_M.detach()


def make_view_align_predictor(head, predict_extrinsics, extrinsics_prediction_mode):
    if not predict_extrinsics:
        return ViewAlignPredictor()
    mode = extrinsics_prediction_mode.lower()
    if mode == 'rt':
        return SE3ViewAlignPredictor(head)
    if mode in ('delta_m', 'delta_m_full'):
        return RopeViewAlignPredictor(head)
    raise ValueError(f"extrinsics_prediction_mode must be 'rt', 'delta_m', or 'delta_m_full', got {extrinsics_prediction_mode}")


# ---- Output head self-attn (single call site for com / standard / none) ----

def run_output_attn(attn_module, features, rel_pos, time_embs, rope_mode):
    """
    Call position_self_attn or rotation_self_attn with the right args for rope_mode.
    Returns the last layer output (B, S, C).
    rope_mode: "standard" | "learned_abs" | "none"
    """
    if rope_mode == "standard":
        out = attn_module(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs,
        )[-1]
    else:
        out = attn_module(
            seq1=features,
            seq2=features,
            seq1_pos=None,
            seq2_pos=None,
            ada_sgnl=time_embs,
        )[-1]
    return out
