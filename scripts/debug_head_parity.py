"""Compare checkpoint-loaded Video-DeltaM head output on fixed tensors."""
import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from modeling.policy import fetch_model_class
try:
    from utils.config_migrations import migrate_config
except ImportError:
    migrate_config = lambda cfg: cfg


CKPT = Path("/home/harshilb/3dfa_unified/train_logs/PerAct2/peract2_orbital_video_deltam_external_warmstart_k5v_k3p_a5000_resume/best.pth")


def build():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    legacy_unwired = "dynamic_rope_from_camtoken" in cfg and "delta_m_camera_ids" in cfg
    try:
        cfg = migrate_config(cfg)
    except Exception:
        pass
    if legacy_unwired and cfg.get("view_align_mode") != "none":
        cfg["view_align_cameras"] = None
    print("CFGIDS", cfg.get("view_align_cameras"), cfg.get("delta_m_camera_ids"))
    cls = fetch_model_class(cfg.get("model_type", "denoise3d"))
    params = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in cfg.items() if k in params}
    kwargs["nhand"] = 2
    model = cls(**kwargs)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt["weight"].items()}
    incompatible = model.load_state_dict(state, strict=False)
    print("MISSING", len(incompatible.missing_keys), "UNEXPECTED", len(incompatible.unexpected_keys))
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(incompatible.missing_keys[:8], incompatible.unexpected_keys[:8])
    model.eval()
    return model


def main():
    torch.manual_seed(1234)
    model = build()
    # Keep the synthetic inputs identical despite architecture-specific init RNG draws.
    torch.manual_seed(999)
    head = model.prediction_head
    print("DMIDS", getattr(head, "delta_m_camera_ids", "missing"))
    # Compare the static RoPE branch first; this isolates config/refactor wiring
    # from the layerwise refinement loop.
    if hasattr(head, "layerwise_view_align"):
        head.layerwise_view_align = False
    if hasattr(head, "dynamic_rope_from_camtoken"):
        head.dynamic_rope_from_camtoken = False
    if hasattr(head, "get_positional_embeddings"):
        original_pe = head.get_positional_embeddings
        def capture_pe(*args, **kwargs):
            result = original_pe(*args, **kwargs)
            torch.save(tuple(x.cpu() if x is not None else None for x in result), "head_parity_pe.pt")
            return result
        head.get_positional_embeddings = capture_pe
    b, t, c, ncam, npatch = 1, 1, 120, 4, 4
    traj = torch.randn(b, t, 2, 9)
    xyz = torch.randn(b, t, 2, 3)
    rgb = torch.randn(b, ncam * npatch, c)
    pcd = torch.randn(b, ncam * npatch, 3)
    instr = torch.randn(b, 1, c)
    proprio = torch.randn(b, 6, c)
    fps = torch.randn(b, 8 + ncam, c)
    fps_pos = torch.randn(b, 8 + ncam, 3)
    cam_ids = torch.randint(0, ncam, (b, 8))
    video = torch.randn(b, 5, ncam, c)
    with torch.inference_mode():
        history_register = None
        if getattr(head, "video_deltam", None) is not None:
            fixed = head.camera_token.unsqueeze(0).expand(b, -1, -1)
            vd_out = head.video_deltam(video, fixed)
            if len(vd_out) == 2:
                refined, history_register = vd_out
            else:
                refined, history_register, _ = vd_out
            torch.save((refined.cpu(), history_register.cpu() if history_register is not None else None), "head_parity_video.pt")
            fps[:, -ncam:] = refined[:, -1]
        if hasattr(head, "_predict_from_cam_feat"):
            try:
                torch.save(fps[:, -ncam:].cpu(), "head_parity_camfeat.pt")
                torch.save(head.camera_proj(fps[:, -ncam:]).cpu(), "head_parity_proj.pt")
                torch.save(head.camera_trunk(head.camera_proj(fps[:, -ncam:])).cpu(), "head_parity_trunk.pt")
                torch.save(head.camera_predictor(head.camera_trunk(head.camera_proj(fps[:, -ncam:]))).cpu(), "head_parity_raw.pt")
                _, dm = head._predict_from_cam_feat(fps[:, -ncam:])
                torch.save(dm.cpu() if dm is not None else None, "head_parity_dm.pt")
            except Exception:
                pass
        kwargs = dict(fps_cam_ids=cam_ids)
        sig = inspect.signature(head.forward).parameters
        # The kwarg was renamed video_camera -> history_register; this script
        # compares heads across that boundary, so probe for both.
        for name in ("history_register", "video_camera"):
            if name in sig:
                kwargs[name] = history_register
                break
        if "video_frame_feats" in sig:
            kwargs["video_frame_feats"] = video
        out = head(
            model.traj_encoder(traj), xyz, torch.zeros(b, dtype=torch.long), rgb, pcd, None, None,
            instr, torch.zeros(b, 1, 3), proprio, fps, fps_pos, **kwargs
        )
    torch.save(out[0][-1].cpu(), "head_parity_output.pt")
    print(out[0][-1].flatten()[:8])


if __name__ == "__main__":
    main()
