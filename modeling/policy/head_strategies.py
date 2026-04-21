"""
Helpers for TransformerHead: extrinsics predictor and run_output_attn for position/rotation heads.
"""
from torch import nn


# ---- Extrinsics ----

class ExtrinsicsPredictor(nn.Module):
    """Base: no extrinsics prediction."""

    def forward(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        return None, None, None


class RTExtrinsicsPredictor(ExtrinsicsPredictor):
    """Predict R,T (6D) from camera token. Stores head as non-module ref to avoid circular module graph."""

    def __init__(self, head):
        super().__init__()
        object.__setattr__(self, "_head", head)  # do not register as submodule (would create cycle)

    def forward(self, batch_size, device):
        rt = self._head._predict_rt(batch_size, device)
        return rt, None, rt.detach()


class DeltaMExtrinsicsPredictor(ExtrinsicsPredictor):
    """Predict delta_M (6x6) from camera token. Stores head as non-module ref to avoid circular module graph."""

    def __init__(self, head):
        super().__init__()
        object.__setattr__(self, "_head", head)  # do not register as submodule (would create cycle)

    def forward(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        delta_M = self._head._predict_delta_M(batch_size, device, fps_scene_feats, fps_cam_ids)
        return None, delta_M, delta_M.detach()


def make_extrinsics_predictor(head, predict_extrinsics, extrinsics_prediction_mode):
    if not predict_extrinsics:
        return ExtrinsicsPredictor()
    mode = extrinsics_prediction_mode.lower()
    if mode == 'rt':
        return RTExtrinsicsPredictor(head)
    if mode in ('delta_m', 'delta_m_full'):
        return DeltaMExtrinsicsPredictor(head)
    raise ValueError(f"extrinsics_prediction_mode must be 'rt', 'delta_m', or 'delta_m_full', got {extrinsics_prediction_mode}")


# ---- Output head self-attn (single call site for com / standard / none) ----

def run_output_attn(attn_module, features, rel_pos, time_embs, rope_mode):
    """
    Call position_self_attn or rotation_self_attn with the right args for rope_mode.
    Returns the last layer output (B, S, C).
    rope_mode: "standard" | "none"
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
