import torch
from torch import nn
from torch.nn import functional as F

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from .head_strategies import make_extrinsics_predictor, run_output_attn
from .video_deltam import VideoDeltaM
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


# Public name -> (internal rope mode, traj tokens anchored at current gripper).
# Encoder RoPE is separate and always on.
# Public name -> (whether to predict, which mechanism).
VIEW_ALIGN_MODES = {
    "none":         (False, "delta_m"),
    "rope_6d":      (True, "delta_m"),
    "rope_full":    (True, "delta_m_full"),
    "physical_se3": (True, "rt"),
}

HEAD_POSITIONAL_ENCODINGS = {
    "rope3d":         ("standard", False),
    "rope3d_proprio": ("standard", True),
    "learned_abs":    ("learned_abs", False),
    "none":           ("none", False),
}


class DenoiseActor(nn.Module):

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Training arguments
                 lv2_batch_size=1,
                 head_positional_encoding='rope3d',
                 view_align_mode='none'):
        super().__init__()
        if view_align_mode not in VIEW_ALIGN_MODES:
            raise ValueError(f"view_align_mode must be one of {sorted(VIEW_ALIGN_MODES)}, got {view_align_mode!r}")
        predict_extrinsics, extrinsics_prediction_mode = VIEW_ALIGN_MODES[view_align_mode]
        # View alignment mixes/transforms a 3D RoPE basis, so it needs one.
        if predict_extrinsics and HEAD_POSITIONAL_ENCODINGS.get(
                head_positional_encoding, (None, None))[0] != "standard":
            raise ValueError(
                f"head_positional_encoding={head_positional_encoding!r} disables 3D RoPE in "
                f"the head, leaving view_align_mode={view_align_mode!r} nothing to correct; "
                "set view_align_mode=none or use a rope3d* encoding."
            )
        # Arguments to be accessed by the main class
        self._rotation_format = rotation_format
        self._relative = relative
        self._lv2_batch_size = lv2_batch_size

        self.embedding_dim = embedding_dim

        # Vision-language encoder, runs only once
        self.encoder = None  # Implement this!

        # Action decoder, runs at every denoising timestep
        self.traj_encoder = nn.Linear(
            6 if rotation_format == 'euler' else 9,  # XYZ + Euler or 6D
            embedding_dim
        )
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format == 'euler' else 6,
            # Subclasses replace this head. Both values are pinned to the old
            # defaults: the module set built here consumes RNG draws, so changing
            # it shifts from-scratch weight init.
            head_positional_encoding=(
                'none' if head_positional_encoding == 'none' else 'rope3d'
            ),
            view_align_mode='rope_6d',
        )

        # Noise/denoise schedulers and hyperparameters
        self.position_scheduler, self.rotation_scheduler = fetch_schedulers(
            denoise_model, denoise_timesteps
        )
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process
        if rotation_format == 'euler':  # normalize pos+rot
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0, 0, 0, 0, 0], [1., 1, 1, 1, 1, 1]]),
                requires_grad=False
            )
        else:
            self.workspace_normalizer = nn.Parameter(
                torch.Tensor([[0., 0., 0.], [1., 1., 1.]]),
                requires_grad=False
            )
        self.nrm_dim = int(self.workspace_normalizer.size(-1))

    def encode_inputs(self, rgb3d, rgb2d, pcd, instruction, proprio):
        (rgb3d_feats, pcd_out, rgb2d_feats, rgb2d_pos, instr_feats, instr_pos,
         proprio_feats, fps_scene_feats, fps_scene_pos, fps_cam_ids,
         video_frame_feats, camera_summaries, camera_summary_pos) = self.encoder(
            rgb3d, rgb2d, pcd, instruction, proprio.flatten(1, 2),
        )

        # Video-DeltaM runs here, not in the head: its inputs do not change
        # across denoising steps, so the head would recompute it n_steps times.
        video_camera = None
        head = self.prediction_head
        if getattr(head, 'video_deltam', None) is not None:
            if fps_cam_ids is None:
                raise ValueError("video_deltam=True requires camera-indexed FPS tokens")
            fixed_camera_token = head.camera_token.unsqueeze(0).expand(rgb3d.shape[0], -1, -1)
            refined_frames, video_camera = head.video_deltam(video_frame_feats, fixed_camera_token)
            camera_summaries = refined_frames[:, -1]

        fps_scene_feats = torch.cat([fps_scene_feats, camera_summaries], dim=1)
        fps_scene_pos = torch.cat([fps_scene_pos, camera_summary_pos], dim=1)

        # Query trajectory (for relative trajectory prediction)
        query_trajectory = proprio[:, -1:]
        return (query_trajectory, rgb3d_feats, pcd_out, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats, fps_scene_feats,
                fps_scene_pos, fps_cam_ids, video_camera)

    def policy_forward_pass(self, trajectory, timestep, fixed_inputs):
        # Parse inputs
        (
            query_trajectory,
            rgb3d_feats, pcd,
            rgb2d_feats, rgb2d_pos,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos,
            fps_cam_ids,
            video_camera,
        ) = fixed_inputs

        # Get features from normalized (relative) trajectory
        trajectory_feats = self.traj_encoder(trajectory)

        

        # But use positions from unnormalized absolute trajectory
        traj_xyz = self.unnormalize_pos(trajectory)[..., :3]
        if self._relative:  # relative to absolute
            traj_xyz = (
                query_trajectory[..., :3]
                + torch.cumsum(traj_xyz, dim=1)
            )

        # Hook for an upstream module to own delta_M and bypass the head's
        # predictor. Nothing sets it today.
        precomputed_delta_M = None
        if rgb3d_feats.ndim == 4:
            # Visual history > 1: the decoder consumes the current (latest) frame.
            rgb3d_feats = rgb3d_feats[:, -1]
            pcd = pcd[:, -1]

        # Returns (traj_list, ee_stacked) — ee_stacked is empty tensor when predict_ee_aux=False
        return self.prediction_head(
            trajectory_feats,
            traj_xyz,
            timestep,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            rgb2d_feats=rgb2d_feats,
            rgb2d_pos=rgb2d_pos,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos,
            fps_cam_ids=fps_cam_ids,
            precomputed_delta_M=precomputed_delta_M,
            video_camera=video_camera,
        )

    def conditional_sample(self, trajectory, device, fixed_inputs):
        # Set schedulers
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            traj_preds, _ = self.policy_forward_pass(
                trajectory,
                t * torch.ones(len(trajectory), device=device, dtype=torch.long),
                fixed_inputs,
            )
            out = traj_preds[-1]  # keep only last layer's output
            pos = self.position_scheduler.step(
                out[..., :3],
                t_ind, trajectory[..., :3]
            ).prev_sample
            rot = self.rotation_scheduler.step(
                out[..., 3:-1],
                t_ind, trajectory[..., 3:]
            ).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return torch.cat((trajectory, out[..., -1:]), -1)

    def conditional_sample_cfg(self, trajectory, device, cond_fixed_inputs, uncond_fixed_inputs,
                               cfg_scale):
        self.position_scheduler.set_timesteps(self.n_steps, device=device)
        self.rotation_scheduler.set_timesteps(self.n_steps, device=device)

        timesteps = self.position_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            t_batch = t * torch.ones(len(trajectory), device=device, dtype=torch.long)

            out_uncond = self.policy_forward_pass(trajectory, t_batch, uncond_fixed_inputs)[0][-1]

            if cfg_scale == 0:
                out = out_uncond
            else:
                out_cond = self.policy_forward_pass(trajectory, t_batch, cond_fixed_inputs)[0][-1]

                diff = (out_cond - out_uncond).norm(dim=-1).mean().item()
                print(f"[CFG t={t_ind}] cfg_scale={cfg_scale}  |out_cond - out_uncond|={diff:.4f}", flush=True)

                # CFG combination: v_cfg = v_uncond + scale * (v_cond - v_uncond)
                out = out_uncond + cfg_scale * (out_cond - out_uncond)

            pos = self.position_scheduler.step(out[..., :3], t_ind, trajectory[..., :3]).prev_sample
            rot = self.rotation_scheduler.step(out[..., 3:-1], t_ind, trajectory[..., 3:]).prev_sample
            trajectory = torch.cat((pos, rot), -1)

        return torch.cat((trajectory, out[..., -1:]), -1)

    def compute_trajectory_cfg(self, trajectory_mask,
                               rgb3d, rgb2d, pcd, instruction, proprio,
                               cfg_scale=2.0):
                               
        uncond_fixed_inputs = self.encode_inputs(rgb3d, rgb2d, pcd, None, proprio)

        if cfg_scale == 0:
            cond_fixed_inputs = uncond_fixed_inputs  # cond pass is skipped in the loop
        else:
            cond_fixed_inputs = self.encode_inputs(rgb3d, rgb2d, pcd, instruction, proprio)

            # Verify cond and uncond scene features actually differ (catches any lang leak)
            fps_diff   = (cond_fixed_inputs[8] - uncond_fixed_inputs[8]).norm(dim=-1).mean().item()
            instr_diff = (cond_fixed_inputs[5] - uncond_fixed_inputs[5]).norm(dim=-1).mean().item()
            print(f"[CFG encode] cfg_scale={cfg_scale}  "
                  f"|cond_fps - uncond_fps|={fps_diff:.4f}  "
                  f"|cond_instr - uncond_instr|={instr_diff:.4f}", flush=True)

        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=tuple(trajectory_mask.shape) + (out_dim,),
            device=trajectory_mask.device
        )
        trajectory = self.conditional_sample_cfg(
            trajectory,
            device=trajectory_mask.device,
            cond_fixed_inputs=cond_fixed_inputs,
            uncond_fixed_inputs=uncond_fixed_inputs,
            cfg_scale=cfg_scale,
        )

        _, traj_len, nhand, _ = trajectory.shape
        trajectory = self.unconvert_rot(trajectory.flatten(1, 2)).unflatten(1, (traj_len, nhand))
        trajectory = self.unnormalize_pos(trajectory)
        trajectory[..., -1] = trajectory[..., -1].sigmoid()

        return trajectory

    def compute_trajectory(self, trajectory_mask,
                           rgb3d, rgb2d, pcd, instruction, proprio):
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
        )

        # Sample from learned model starting from noise
        out_dim = 6 if self._rotation_format == 'euler' else 9
        trajectory = torch.randn(
            size=tuple(trajectory_mask.shape) + (out_dim,),
            device=trajectory_mask.device
        )
        trajectory = self.conditional_sample(
            trajectory,
            device=trajectory_mask.device,
            fixed_inputs=fixed_inputs,
        )

        # Back to quaternion
        _, traj_len, nhand, _ = trajectory.shape
        trajectory = self.unconvert_rot(
            trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))
        # unnormalize position
        trajectory = self.unnormalize_pos(trajectory)
        # Convert gripper status to probaility
        trajectory[..., -1] = trajectory[..., -1].sigmoid()

        return trajectory

    def compute_loss(self, gt_trajectory,
                     rgb3d, rgb2d, pcd, instruction, proprio):
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(
            rgb3d, rgb2d, pcd, instruction, proprio,
        )

        # Process gt_trajectory
        gt_openess = gt_trajectory[..., -1:]
        gt_trajectory = gt_trajectory[..., :-1]
        # Normalize all pos
        gt_trajectory = self.normalize_pos(gt_trajectory)
        # Convert rotation parametrization
        _, traj_len, nhand, _ = gt_trajectory.shape
        gt_trajectory = self.convert_rot(
            gt_trajectory.flatten(1, 2)
        ).unflatten(1, (traj_len, nhand))

        # GT EE XYZ for aux loss: first keypose, mean over hands, in normalized coords
        head = self.prediction_head
        if head.predict_ee_aux:
            gt_ee_xyz = gt_trajectory[:, 0, :, :3].mean(dim=1).detach()  # (B, 3)

        # Loop lv2_batch_size times and sample different noises with same input
        # Trick to effectively increase the batch size without re-encoding
        # It speeds up training but may decrease performance a bit
        total_loss = 0
        self._last_ee_aux_loss = None
        for _ in range(self._lv2_batch_size):
            # Sample noise
            noise = torch.randn(gt_trajectory.shape, device=gt_trajectory.device)

            # Sample a random timestep
            timesteps = self.position_scheduler.sample_noise_step(
                num_noise=len(noise), device=noise.device
            )

            # Add noise to the clean trajectories
            pos = self.position_scheduler.add_noise(
                gt_trajectory[..., :3], noise[..., :3],
                timesteps
            )
            rot = self.rotation_scheduler.add_noise(
                gt_trajectory[..., 3:], noise[..., 3:],
                timesteps
            )

            # Q: uhhm, why add noise seperately?

            noisy_trajectory = torch.cat((pos, rot), -1)

            # Predict the noise residual; returns (traj_list, ee_stacked)
            pred, ee_stacked = self.policy_forward_pass(
                noisy_trajectory,
                timesteps, fixed_inputs,
            )

            # Compute flow-matching loss
            for layer_pred in pred:
                pos = layer_pred[..., :3]
                rot = layer_pred[..., 3:-1]
                openess = layer_pred[..., -1:]
                denoise_target = self.position_scheduler.prepare_target(
                    noise, gt_trajectory
                )
                loss = (
                    30 * F.l1_loss(pos, denoise_target[..., :3], reduction='mean')
                    + 10 * F.l1_loss(rot, denoise_target[..., 3:], reduction='mean')
                    + F.binary_cross_entropy_with_logits(openess, gt_openess)
                )
                total_loss = total_loss + loss

            # EE aux loss: ee_stacked is (n_layers, B, n_ext_cams, 3) when predict_ee_aux=True
            if head.predict_ee_aux:
                n_ext = len(head.ee_aux_cam_ids)
                gt_ee_exp = gt_ee_xyz[:, None, :].expand(-1, n_ext, -1)  # (B, n_ext, 3)
                # mean over SA layers
                loss_aux = F.mse_loss(ee_stacked, gt_ee_exp.unsqueeze(0).expand_as(ee_stacked))
                total_loss = total_loss + head.lambda_aux * loss_aux
                self._last_ee_aux_loss = loss_aux.detach()

        return total_loss / self._lv2_batch_size

    def normalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = (_max - _min).clamp(min=1e-6)  # avoid div by zero -> NaN

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] - _min) / diff * 2.0
            - 1.0
        )
        return out

    def unnormalize_pos(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = (_max - _min).clamp(min=1e-6)

        out = signal.clone()
        out[..., :self.nrm_dim] = (
            (signal[..., :self.nrm_dim] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    def convert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        rot = normalise_quat(signal[..., 3:7])
        res = signal[..., 7:] if signal.size(-1) > 7 else None
        # The following code expects wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            rot = rot[..., (3, 0, 1, 2)]
        # Convert to rotation matrix
        rot = quaternion_to_matrix(rot)
        # Convert to 6D
        if len(rot.shape) == 4:
            B, L, D1, D2 = rot.shape
            rot = rot.reshape(B * L, D1, D2)
            rot = get_ortho6d_from_rotation_matrix(rot)
            rot = rot.reshape(B, L, 6)
        else:
            rot = get_ortho6d_from_rotation_matrix(rot)
        # Concatenate pos, rot, other state info
        signal = torch.cat([signal[..., :3], rot], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def unconvert_rot(self, signal):
        # If Euler then no conversion
        if self._rotation_format == 'euler':
            return signal
        # Else assume quaternion
        res = signal[..., 9:] if signal.size(-1) > 9 else None
        if len(signal.shape) == 3:
            B, L, _ = signal.shape
            rot = signal[..., 3:9].reshape(B * L, 6)
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
            quat = quat.reshape(B, L, 4)
        else:
            rot = signal[..., 3:9]
            mat = compute_rotation_matrix_from_ortho6d(rot)
            quat = matrix_to_quaternion(mat)
        # The above code handled wxyz quaternion format!
        if self._rotation_format == 'quat_xyzw':
            quat = quat[..., (1, 2, 3, 0)]
        signal = torch.cat([signal[..., :3], quat], dim=-1)
        if res is not None:
            signal = torch.cat((signal, res), -1)
        return signal

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False,
        cfg_scale=None,
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, nhand, 3+4+X)
            trajectory_mask: (B, trajectory_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            rgb2d: (B, num_2d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)
            cfg_scale: if set, use classifier-free guidance with this scale (inference only)

        Note:
            The input rotation is expressed either as:
                a) quaternion (4D), then the model converts it to 6D internally.
                b) Euler angles (3D).

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, nhand, 3+rot+1), at inference
        """
        # Inference, don't use gt_trajectory
        if run_inference:
            if cfg_scale is not None:
                return self.compute_trajectory_cfg(
                    trajectory_mask,
                    rgb3d, rgb2d, pcd, instruction, proprio,
                    cfg_scale=cfg_scale,
                )
            return self.compute_trajectory(
                trajectory_mask,
                rgb3d, rgb2d, pcd, instruction, proprio,
            )

        # Training, use gt_trajectory to compute loss
        return self.compute_loss(
            gt_trajectory,
            rgb3d, rgb2d, pcd, instruction, proprio,
        )


class TransformerHead(nn.Module):
    """
    Action decoder head (trajectory + rotation + openness). RoPE usage is set by
    the single ``head_positional_encoding`` knob (see HEAD_POSITIONAL_ENCODINGS):

    - 'rope3d' / 'rope3d_proprio': shared self-attn uses standard RoPE; cross_attn and
      position/rotation output heads get RoPE positions. 'rope3d_proprio' additionally
      anchors every trajectory token at the current gripper XYZ.
    - 'learned_abs' / 'none': RoPE is disabled everywhere in this head:
      cross_attn (rotary_pe=False), traj_self_attn / scene_self_attn / traj_scene_attn
      (all rotary_pe=False), and position_self_attn / rotation_self_attn (rope_mode='none').
      'learned_abs' adds a learned per-index PE table on trajectory tokens.
    - Encoder (vision-language) RoPE is independent and controlled by the encoder config.

    delta_M contract: predicted from the per-camera summary tokens (not camera_token),
    applied to scene/FPS tokens only -- traj tokens keep raw positions. Those same
    tokens are also SA members and the EE-aux input, which is why ee_aux requires
    view alignment. layerwise_view_align re-predicts per block with shared weights.

    RoPE usage map (all places RoPE can be applied):
    - Encoder (encoder3d): relative_pe_layer (RotaryPositionEncoding3D) + gripper_context_head(rotary_pe=True).
    - Decoder: get_positional_embeddings uses relative_pe_layer (3D); cross_attn/self_attn/position_self_attn/
      rotation_self_attn use those positions only when _rope_mode != "none". Actual rotation is in
      multihead_custom_attention.py (embed_rotary).
    """

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 rot_dim=6,
                 head_positional_encoding='rope3d',
                 view_align_mode='none',
                 view_align_cameras=None,
                 layerwise_view_align=False,
                 video_deltam=False,
                 video_deltam_depth=4,
                 video_deltam_max_history=32,
                 video_deltam_max_cameras=8,
                 video_deltam_full_image=False,
                 ee_aux=False,
                 ee_aux_weight=1.0,
                 ee_aux_cameras=(0, 1)):
        super().__init__()

        if view_align_mode not in VIEW_ALIGN_MODES:
            raise ValueError(f"view_align_mode must be one of {sorted(VIEW_ALIGN_MODES)}, got {view_align_mode!r}")
        self.view_align_mode = view_align_mode
        # Derived so the rest of the head keeps its existing vocabulary.
        predict_extrinsics, self.extrinsics_prediction_mode = VIEW_ALIGN_MODES[view_align_mode]
        delta_m_camera_ids = view_align_cameras
        predict_ee_aux, lambda_aux, ee_aux_cam_ids = ee_aux, ee_aux_weight, ee_aux_cameras
        print(f" ************** view alignment: {view_align_mode} **************")

        self.predict_extrinsics = predict_extrinsics
        self.delta_m_camera_ids = (
            None if delta_m_camera_ids is None else tuple(sorted(set(delta_m_camera_ids)))
        )
        if self.delta_m_camera_ids is not None and any(i < 0 for i in self.delta_m_camera_ids):
            raise ValueError(f"delta_m_camera_ids must be non-negative, got {self.delta_m_camera_ids}")
        self.layerwise_view_align = layerwise_view_align
        self.video_deltam = (
            VideoDeltaM(
                embedding_dim, num_attn_heads, video_deltam_depth,
                max_history=video_deltam_max_history,
                max_cameras=video_deltam_max_cameras,
                full_image=video_deltam_full_image,
            ) if video_deltam else None
        )
        if head_positional_encoding not in HEAD_POSITIONAL_ENCODINGS:
            raise ValueError(
                f"head_positional_encoding must be one of "
                f"{sorted(HEAD_POSITIONAL_ENCODINGS)}, got {head_positional_encoding!r}"
            )
        self.head_positional_encoding = head_positional_encoding
        self._rope_mode, self.use_proprio_rope = HEAD_POSITIONAL_ENCODINGS[head_positional_encoding]

        # In the 'none' and 'learned_abs' modes RoPE is disabled in this TransformerHead in:
        #   - cross_attn (traj-to-scene), shared self-attn branch, position/rotation output heads.
        # Encoder RoPE (encoder3d) is unchanged and controlled by the encoder config only.

        # Learned absolute positional encoding for trajectory tokens only.
        # Uses a learnable parameter table of shape (1, max_traj_tokens, embedding_dim),
        # sliced to the actual traj_len * nhand at forward time.
        if self._rope_mode == "learned_abs":
            self.traj_abs_pe = nn.Parameter(torch.zeros(1, 32, embedding_dim))
            nn.init.trunc_normal_(self.traj_abs_pe, std=0.02)
            print('Using learned absolute positional encoding for traj tokens (abs_pe)')

        # _use_rope_in_attn: False when RoPE is replaced by abs PE or disabled entirely
        _use_rope_in_attn = rotary_pe and self._rope_mode == "standard"

        # Different embeddings
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Learnable tokens
        self.register_tokens = nn.Parameter(torch.randn(4, embedding_dim))
        # One learned global register/context token, not a per-camera alignment
        # token. Keep this parameter name for checkpoint compatibility.
        self.camera_token = nn.Parameter(torch.randn(1, embedding_dim))
        self.embedding_dim = embedding_dim

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Estimate attends to context (no subsampling). RoPE off unless _rope_mode == "standard".
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=_use_rope_in_attn,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers

        if self._rope_mode != "none":
            print(f'Head positional encoding: {head_positional_encoding}')
            self.self_attn = AttentionModule(
                    num_layers=num_shared_attn_layers,
                    d_model=embedding_dim,
                    dim_fw=embedding_dim,
                    dropout=0.1,
                    n_heads=num_attn_heads,
                    pre_norm=False,
                    rotary_pe=_use_rope_in_attn,
                    use_adaln=True,
                    is_self=True
                )
        else:
            # head_positional_encoding='none': no RoPE in traj/scene/traj_scene self-attn or cross_attn
            print('Head positional encoding: none (no RoPE)')
            self.traj_self_attn = AttentionModule(
                num_layers=num_shared_attn_layers // 2,
                d_model=embedding_dim,
                dim_fw=embedding_dim,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=False,
                use_adaln=True,
                is_self=True
            )
            self.scene_self_attn = AttentionModule(
                num_layers=num_shared_attn_layers // 2,
                d_model=embedding_dim,
                dim_fw=embedding_dim,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=False,
                use_adaln=True,
                is_self=True
            )
            self.traj_scene_attn = AttentionModule(
                num_layers=num_shared_attn_layers // 2,
                d_model=embedding_dim,
                dim_fw=embedding_dim,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=False,
                use_adaln=True,
                is_self=True,
            )
       
        # Specific (non-shared) Output layers:
        # 1. Rotation
        self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        self.rotation_self_attn = AttentionModule(
                num_layers=2,
                d_model=embedding_dim,
                dim_fw=embedding_dim,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=_use_rope_in_attn,
                use_adaln=True,
                is_self=True
            )
        self.rotation_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, rot_dim)
        )

        # 2. Position
        self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        self.position_self_attn = AttentionModule(
                num_layers=2,
                d_model=embedding_dim,
                dim_fw=embedding_dim,
                dropout=0.1,
                n_heads=num_attn_heads,
                pre_norm=False,
                rotary_pe=_use_rope_in_attn,
                use_adaln=True,
                is_self=True
            )
        self.position_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3)
        )

        # 3. Openess
        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        # 4. Predict either physical R,T or a representation-level RoPE
        # correction. ``camera_predictor`` is retained for checkpoint compatibility.
        if predict_extrinsics:
            self.camera_proj = nn.Linear(embedding_dim, embedding_dim)
            # Shared trunk (Linear+ReLU) — used by both the extrinsics and EE aux heads
            self.camera_trunk = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU()
            )
            if self.extrinsics_prediction_mode == 'rt':
                self.camera_predictor = nn.Linear(embedding_dim, 6)  # axis_angle (3) + translation (3)
                # Init so output ≈ 0 -> axis_angle=0, t=0 = identity transform at start
                nn.init.normal_(self.camera_predictor.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.camera_predictor.bias)
            elif self.extrinsics_prediction_mode == 'delta_m_full':
                D = (embedding_dim // 6) * 6
                self._delta_m_full_dim = D
                self.camera_predictor = nn.Linear(embedding_dim, D * D)  # D×D A_skew for full delta_M
                # Init so A_skew ≈ 0 -> delta_M = exp(A) ≈ I at start
                nn.init.normal_(self.camera_predictor.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.camera_predictor.bias)
            else:  # delta_m
                self.camera_predictor = nn.Linear(embedding_dim, 36)  # 6x6 A_skew for delta_M
                # Init so A_skew ≈ 0 -> delta_M = exp(A) ≈ I (small delta_M at start)
                nn.init.normal_(self.camera_predictor.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.camera_predictor.bias)

        # 5. EE aux prediction head — branches from camera_trunk (shared with extrinsics predictor)
        self.predict_ee_aux = predict_ee_aux
        self.lambda_aux = lambda_aux
        self.ee_aux_cam_ids = list(ee_aux_cam_ids)
        if predict_ee_aux:
            assert predict_extrinsics, "predict_ee_aux requires predict_extrinsics=True"
            self.ee_predictor = nn.Linear(embedding_dim, 3)
            nn.init.normal_(self.ee_predictor.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.ee_predictor.bias)

        self.extrinsics_predictor = make_extrinsics_predictor(
            self, predict_extrinsics, self.extrinsics_prediction_mode
        )

    def _predict_from_cam_feat(self, cam_feat):
        """
        Predict delta_M or (R,T) from an arbitrary camera feature tensor.

        Args:
            cam_feat: (B, C) or (B, ncam, C) — camera feature(s)

        Returns:
            (cam_params_rt, delta_M): one is non-None depending on extrinsics_prediction_mode.
            Shape mirrors input: (B, 6, 6) or (B, ncam, 6, 6) for delta_m mode.
        """
        h = self.camera_proj(cam_feat)
        trunk = self.camera_trunk(h)
        if self.extrinsics_prediction_mode == 'delta_m':
            A_skew = self.camera_predictor(trunk).reshape(*cam_feat.shape[:-1], 6, 6)
            A = A_skew - A_skew.transpose(-1, -2)
            max_norm = 3.0
            norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            A = A * (norm.clamp(max=max_norm) / norm)
            delta_M = self._mask_delta_m_camera_ids(torch.linalg.matrix_exp(A))
            return None, delta_M
        elif self.extrinsics_prediction_mode == 'delta_m_full':
            D = self._delta_m_full_dim
            A_skew = self.camera_predictor(trunk).reshape(*cam_feat.shape[:-1], D, D)
            A = A_skew - A_skew.transpose(-1, -2)
            max_norm = 3.0
            norm = torch.linalg.norm(A, ord='fro', dim=(-2, -1), keepdim=True).clamp(min=1e-8)
            A = A * (norm.clamp(max=max_norm) / norm)
            delta_M = self._mask_delta_m_camera_ids(torch.linalg.matrix_exp(A))
            return None, delta_M
        else:  # rt
            return self.camera_predictor(trunk), None

    def _mask_delta_m_camera_ids(self, delta_M):
        """Keep DeltaM identity for cameras outside the configured correction set."""
        if self.delta_m_camera_ids is None or delta_M.ndim != 4:
            return delta_M
        ncam = delta_M.shape[1]
        invalid = [i for i in self.delta_m_camera_ids if i >= ncam]
        if invalid:
            raise ValueError(
                f"delta_m_camera_ids={invalid} outside ncam={ncam}; "
                "camera order is [orbital_left, orbital_right, wrist_left, wrist_right]"
            )
        enabled = torch.zeros(ncam, dtype=torch.bool, device=delta_M.device)
        enabled[list(self.delta_m_camera_ids)] = True
        identity = torch.eye(delta_M.shape[-1], dtype=delta_M.dtype, device=delta_M.device)
        identity = identity.view(1, 1, *identity.shape).expand_as(delta_M)
        return torch.where(enabled.view(1, ncam, 1, 1), delta_M, identity)

    def _predict_rt(self, batch_size, device):
        """Predict axis-angle (3) + translation (3) from cam token. Returns (B, 6)."""
        cam_feat = self._expand_camera_token(batch_size)
        rt, _ = self._predict_from_cam_feat(cam_feat)
        return rt

    def _predict_delta_M(self, batch_size, device, fps_scene_feats=None, fps_cam_ids=None):
        """
        Predict delta_M from pooled per-camera image features (one per camera).

        If fps_scene_feats/fps_cam_ids are provided, sources from the camera summaries
        (fps_scene_feats[:, M:, :] where M = fps_cam_ids.shape[1]).
        Returns delta_M: (B, ncam, 6, 6) — one orthogonal matrix per camera.

        Falls back to (B, 6, 6) from the learnable camera_token if not provided.
        """
        if fps_scene_feats is not None and fps_cam_ids is not None:
            M = fps_cam_ids.shape[1]
            camera_summaries = fps_scene_feats[:, M:, :]  # (B, ncam, C)
            _, delta_M = self._predict_from_cam_feat(camera_summaries)  # (B, ncam, 6, 6)
        else:
            cam_feat = self._expand_camera_token(batch_size)
            _, delta_M = self._predict_from_cam_feat(cam_feat)  # (B, 6, 6)
        return delta_M

    def _predict_ee_from_camera_summaries(self, camera_summaries):
        """Predict EE XYZ for external cameras from their summary tokens.

        Args:
            camera_summaries: (B, ncam, C)
        Returns:
            (B, n_ext_cams, 3) predicted EE XYZ in normalized workspace coords
        """
        cam_feats = camera_summaries[:, self.ee_aux_cam_ids, :]  # (B, n_ext_cams, C)
        h = self.camera_proj(cam_feats)
        trunk = self.camera_trunk(h)
        return self.ee_predictor(trunk)

    def _expand_camera_token(self, batch_size):
        """Expand (1, C) camera_token to (B, C)."""
        return self.camera_token.unsqueeze(0).expand(batch_size, -1, -1).squeeze(1)

    def _recompute_rope(self, traj_xyz, orig_rgb3d_pos, orig_fps_scene_pos,
                        bases=None):
        """Base-class stub; overridden in the 3D head."""
        return None, None, None, None

    def transform_pcd_with_extrinsics(self, pcd, cam_params):

        return pcd  # Base class does nothing


    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos, fps_cam_ids=None,
                precomputed_delta_M=None, video_camera=None):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F) 
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape
        _ee_layer_preds = []  # local accumulator — avoids module-attribute mutation for torch.compile

        # Trajectory features
        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None] # bimanual support.

        # noisy
        traj_feats = traj_feats.reshape(traj_feats.shape[0], -1, traj_feats.shape[-1])
        trajectory = trajectory.reshape(trajectory.shape[0], -1, trajectory.shape[-1])

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = traj_time_pos.reshape(traj_time_pos.shape[0], -1, traj_time_pos.shape[-1])
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]
        if self.use_proprio_rope:
            # Use current gripper position (same as instr_pos anchor) for all traj token RoPE
            traj_xyz = instr_pos[:, :1, :].expand(-1, traj_xyz.shape[1], -1)


        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats
        )

        batch_size, device = trajectory.shape[0], trajectory.device
        # Video-DeltaM already ran in encode_inputs; the camera summaries it
        # refined are in fps_scene_feats and its history register arrives here.
        if video_camera is not None:
            traj_feats = traj_feats + video_camera
        if precomputed_delta_M is not None:
            # An upstream module already produced delta_M; skip the head's predictor.
            cam_params_rt, delta_M = None, precomputed_delta_M
            self._last_predicted_cam_params = precomputed_delta_M.detach()
        else:
            cam_params_rt, delta_M, self._last_predicted_cam_params = self.extrinsics_predictor(
                batch_size, device, fps_scene_feats=fps_scene_feats, fps_cam_ids=fps_cam_ids
            )

        if self.layerwise_view_align and self._rope_mode == "standard" and self.predict_extrinsics \
                and precomputed_delta_M is None:
            # Re-predict delta_M before every block. The camera summaries live in
            # the SA sequence, so each SA block sees a refined delta_M. Originals
            # are kept so RT transforms always start from a clean base.
            orig_rgb3d_pos, orig_fps_scene_pos = rgb3d_pos, fps_scene_pos

            # Per-camera alignment features (evolve each SA layer); shape (B, ncam, C)
            assert fps_cam_ids is not None, "layerwise_view_align requires fps_cam_ids"
            M = fps_cam_ids.shape[1]
            current_camera_summaries = fps_scene_feats[:, M:, :]

            # Pre-compute sin/cos bases once; reuse across all blocks (delta_M mode only)
            precomputed_bases = (
                self._precompute_rope_bases(traj_xyz, rgb3d_pos, fps_scene_pos)
                if self.extrinsics_prediction_mode in ('delta_m', 'delta_m_full') else None
            )

            # Camera summaries are static until SA updates them, so every CA
            # layer would get the same delta_M: predict once.
            rel_traj_pos, rel_scene_pos, rel_pos, _ = self._recompute_rope(
                traj_xyz, orig_rgb3d_pos, orig_fps_scene_pos,
                bases=precomputed_bases, fps_cam_ids=fps_cam_ids,
                camera_summaries=current_camera_summaries)
            for i in range(self.cross_attn.num_layers):
                traj_feats = self.cross_attn.attn_layers[i](
                    traj_feats, rgb3d_feats,
                    seq1_pos=rel_traj_pos, seq2_pos=rel_scene_pos, ada_sgnl=time_embs)
                traj_feats = self.cross_attn.ffw_layers[i](traj_feats, time_embs)

            # Build the shared SA sequence (camera_token is last token at index -1)
            features = self.get_sa_feature_sequence(
                traj_feats, fps_scene_feats,
                rgb3d_feats, rgb2d_feats, instr_feats
            )
            traj_seq_len = traj_feats.shape[1]

            # SA updates the camera summaries in place, so delta_M refines per layer.
            for i in range(self.self_attn.num_layers):
                rel_traj_pos, rel_scene_pos, rel_pos, _ = self._recompute_rope(
                    traj_xyz, orig_rgb3d_pos, orig_fps_scene_pos,
                    bases=precomputed_bases, fps_cam_ids=fps_cam_ids,
                    camera_summaries=current_camera_summaries)
                sa_pos = rel_pos
                features = self.self_attn.attn_layers[i](
                    features, features,
                    seq1_pos=sa_pos, seq2_pos=sa_pos, ada_sgnl=time_embs)
                features = self.self_attn.ffw_layers[i](features, time_embs)
                ncam = fps_scene_feats.shape[1] - M
                current_camera_summaries = features[:, traj_seq_len + M:traj_seq_len + M + ncam, :]
                if self.predict_ee_aux:
                    _ee_layer_preds.append(
                        self._predict_ee_from_camera_summaries(current_camera_summaries)
                    )

            rotation = self.predict_rot(features, rel_pos, time_embs, traj_feats.shape[1])
            position, position_features = self.predict_pos(features, rel_pos, time_embs, traj_feats.shape[1])
        elif self._rope_mode == "standard":
            # Static RoPE path: compute positional embeddings once and use them everywhere
            rel_traj_pos, rel_scene_pos, rel_pos, rel_fps_pos = self.get_positional_embeddings(
                traj_xyz, traj_feats,
                rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
                timesteps, proprio_feats,
                fps_scene_feats, fps_scene_pos,
                instr_feats, instr_pos,
                delta_M=delta_M,
                cam_params_rt=cam_params_rt,
                fps_cam_ids=fps_cam_ids,
            )
            traj_feats = self.cross_attn(
                seq1=traj_feats,
                seq2=rgb3d_feats,
                seq1_pos=rel_traj_pos,
                seq2_pos=rel_scene_pos,
                ada_sgnl=time_embs
            )[-1]
            features = self.get_sa_feature_sequence(
                traj_feats, fps_scene_feats,
                rgb3d_feats, rgb2d_feats, instr_feats
            )
            features = self.self_attn(
                seq1=features,
                seq2=features,
                seq1_pos=rel_pos,
                seq2_pos=rel_pos,
                ada_sgnl=time_embs,
            )[-1]
            if self.predict_ee_aux and fps_cam_ids is not None:
                traj_seq_len = traj_feats.shape[1]
                M = fps_cam_ids.shape[1]
                ncam = fps_scene_feats.shape[1] - M
                final_camera_summaries = features[:, traj_seq_len + M:traj_seq_len + M + ncam, :]
                _ee_layer_preds.append(self._predict_ee_from_camera_summaries(final_camera_summaries))
            rotation = self.predict_rot(features, rel_pos, time_embs, traj_feats.shape[1])
            position, position_features = self.predict_pos(features, rel_pos, time_embs, traj_feats.shape[1])
        elif self._rope_mode == "learned_abs":
            # Learned absolute PE: add per-index learned embedding to traj tokens only, no RoPE anywhere.
            traj_feats = traj_feats + self.traj_abs_pe[:, :traj_feats.shape[1], :]
            traj_feats = self.cross_attn(
                seq1=traj_feats,
                seq2=rgb3d_feats,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs
            )[-1]
            features = self.get_sa_feature_sequence(
                traj_feats, fps_scene_feats,
                rgb3d_feats, rgb2d_feats, instr_feats
            )
            features = self.self_attn(
                seq1=features,
                seq2=features,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs,
            )[-1]
            rotation = self.predict_rot(features, None, time_embs, traj_feats.shape[1])
            position, position_features = self.predict_pos(features, None, time_embs, traj_feats.shape[1])
        else:
            # No RoPE: skip get_positional_embeddings; no position args to attention
            traj_feats = self.cross_attn(
                seq1=traj_feats,
                seq2=rgb3d_feats,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs
            )[-1]
            traj_feats = self.traj_self_attn(
                seq1=traj_feats,
                seq2=traj_feats,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs,
            )[-1]
            fps_scene_feats = self.scene_self_attn(
                seq1=fps_scene_feats,
                seq2=fps_scene_feats,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs,
            )[-1]
            features = self.get_sa_feature_sequence(
                traj_feats, fps_scene_feats,
                rgb3d_feats, rgb2d_feats, instr_feats
            )
            features = self.traj_scene_attn(
                seq1=features,
                seq2=features,
                seq1_pos=None,
                seq2_pos=None,
                ada_sgnl=time_embs,
            )[-1]
            rotation = self.predict_rot(features, None, time_embs, traj_feats.shape[1])
            position, position_features = self.predict_pos(features, None, time_embs, traj_feats.shape[1])

        # Openess head from position head
        openess = self.openess_predictor(position_features[:, :self.embedding_dim]) # don't use camera and register tokens here.

        traj_pred = torch.cat((position, rotation, openess), -1).unflatten(1, (traj_len, nhand))

        # Stack per-layer EE predictions into a single tensor for compile-safe return
        if self.predict_ee_aux and _ee_layer_preds:
            ee_stacked = torch.stack(_ee_layer_preds, dim=0)  # (n_layers, B, n_ext_cams, 3)
        else:
            ee_stacked = traj_pred.new_empty(0)  # empty sentinel when unused

        return [traj_pred], ee_stacked

    def encode_denoising_timestep(self, timestep, proprio_feats):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        proprio_feats = proprio_feats.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
        return time_feats + curr_gripper_feats

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos,
        delta_M=None,
        cam_params_rt=None,
    ):
        return None, None, None

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        batch_size = traj_feats.shape[0]
        register_tokens = self.register_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        camera_token = self.camera_token.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([traj_feats, fps_scene_feats, register_tokens, camera_token], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = run_output_attn(
            self.position_self_attn, features, pos, time_embs,
            self._rope_mode
        )
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = run_output_attn(
            self.rotation_self_attn, features, pos, time_embs,
            self._rope_mode
        )
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
