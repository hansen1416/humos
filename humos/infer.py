"""Manual inference (Option 2): iterate over a dataloader and save predictions.

This script is meant to replace the previous `trainer.validate(...)`-based infer.
It loads a trained HUMOS CYCLIC_TMR checkpoint and runs motion->motion generation
for every sample in the chosen split, saving one *.pt per input keyid.

Notes
-----
- We set `drop_last=False` so you don't silently lose the final partial batch.
- File writing is guarded to be deterministic in single-process runs; if you
  later run this under DDP, you should gate writes by global rank.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from loguru import logger
import numpy as np

from aitviewer.models.smpl import SMPLLayer
from humos.src.data.text_motion import TextMotionDataset
from humos.src.initialize import initialize_dataloaders, initialize_model
from humos.utils.config import parse_args, run_grid_search_experiments
from humos.utils.mesh_utils import smplh_breakdown


# ======================================================================================
# ASE-matching velocity computation for HUMOS outputs
# --------------------------------------------------------------------------------------
# You said your HUMOS outputs have:
#   root_orient: (T, 3)   axis-angle (rotation vector) of the root
#   pose_body:   (T, 63)  axis-angle for 21 body joints (21*3)
#   trans:       (T, 3)   global root translation
#
# ASE reference (your uploaded /mnt/data/motion_lib_base.py):
#   - dof_vel is computed from *local joint quaternions* with:
#       diff_q = quat_mul(quat_conjugate(q_t), q_{t+1})
#       angle, axis = quat_to_angle_axis(diff_q)
#       omega = axis * angle / dt
#       drop root joint (index 0), flatten
#     See: local_rotation_to_dof_vel() and compute_motion_dof_vels()
#
#   - root_vel and root_ang_vel (in get_motion_state) are taken as the
#     rigid-body-0 entries of global velocity / global angular velocity arrays.
#     For the root joint, local==global rotation, so we can compute root_ang_vel
#     the same way from root quaternions (forward-diff + last-frame pad).
#
# This function produces:
#   root_vel:      (T, 3)
#   root_ang_vel:  (T, 3)
#   dof_vel:       (T, (J-1)*3)  where J = 1 + pose_body_joints = 22, so (J-1)*3 = 63
# ======================================================================================


def _axis_angle_to_quat_xyzw(rotvec: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle (rotation vector) to quaternion in (x, y, z, w) order.

    rotvec: (..., 3) where magnitude is the rotation angle in radians.
    return: (..., 4) quaternion (x,y,z,w)

    Notes:
    - This matches IsaacGym / ASE torch_utils convention (x,y,z,w).
    - For very small angles, returns identity-like quaternions smoothly.
    """
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)  # (...,1)
    half = 0.5 * angle
    # Avoid division by 0 for tiny angles:
    axis = rotvec / angle.clamp_min(1e-8)
    sin_half = torch.sin(half)
    xyz = axis * sin_half
    w = torch.cos(half)
    return torch.cat([xyz, w], dim=-1)


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate for (x,y,z,w)."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul_xyzw(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions in (x,y,z,w).
    """
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _quat_to_angle_axis_xyzw(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert quaternion (x,y,z,w) to (angle, axis).

    Returns:
      angle: (...,)
      axis:  (..., 3)

    ASE uses torch_utils.quat_to_angle_axis(); this is a compatible implementation:
    - Normalize q
    - Enforce shortest-arc (w>=0) so angle is in [0, pi]
    """
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)

    # Enforce shortest arc: if w<0, flip quaternion sign (represents same rotation).
    # This ensures the angle we compute is <= pi (matches typical IsaacGym utilities).
    w = q[..., 3:4]
    sign = torch.where(w < 0.0, -torch.ones_like(w), torch.ones_like(w))
    q = q * sign

    xyz = q[..., :3]
    w = q[..., 3:4].clamp(-1.0, 1.0)

    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)  # (...,1)
    angle = 2.0 * torch.atan2(sin_half, w)  # (...,1)

    # axis = xyz / sin_half; handle tiny sin_half
    axis = xyz / sin_half.clamp_min(1e-8)

    # If angle is ~0, axis is arbitrary; return 0 axis for stability.
    small = sin_half < 1e-8
    axis = torch.where(small.expand_as(axis), torch.zeros_like(axis), axis)

    return angle.squeeze(-1), axis


def _forward_diff_pad_last(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Forward finite difference with last-frame padding:
      v[t] = (x[t+1]-x[t]) / dt for t=0..T-2
      v[T-1] = v[T-2]
    This matches the "append last" convention used in ASE compute_motion_dof_vels().
    """
    if x.shape[0] < 2:
        return torch.zeros_like(x)
    v = (x[1:] - x[:-1]) / dt
    return torch.cat([v, v[-1:]], dim=0)


def humos_to_ase_velocities(
    root_orient_aa: torch.Tensor,  # (T,3) axis-angle
    pose_body_aa: torch.Tensor,  # (T,63) axis-angle
    trans: torch.Tensor,  # (T,3)
    fps: float = 20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ASE-style velocities from HUMOS outputs.

    Outputs:
      root_vel:     (T,3)
      root_ang_vel: (T,3)
      dof_vel:      (T,(J-1)*3)  (here J=22 => 63)

    Key ASE correspondence:
      - ASE dof_vel uses local joint quats:
          diff_q = conj(q_t) * q_{t+1}
          omega  = axis(diff_q) * angle(diff_q) / dt
        drop root joint, flatten. (see /mnt/data/motion_lib_base.py)
      - For root_ang_vel, root is joint 0; local==global rotation, so we compute
        it via the same quaternion differencing on root quats.
      - root_vel is computed from root position; in ASE it’s body_vel[...,0,:].
    """
    assert root_orient_aa.ndim == 2 and root_orient_aa.shape[-1] == 3
    assert pose_body_aa.ndim == 2 and pose_body_aa.shape[-1] % 3 == 0
    assert trans.ndim == 2 and trans.shape[-1] == 3
    assert root_orient_aa.shape[0] == pose_body_aa.shape[0] == trans.shape[0]

    T = trans.shape[0]
    dt = 1.0 / float(fps)

    # -----------------------------
    # 1) root_vel (ASE: body_vel[...,0,:])
    # -----------------------------
    # In ASE the root linear velocity is taken from "global_velocity" at body 0.
    # For the root, that corresponds to finite-diff of the root translation.
    root_vel = _forward_diff_pad_last(trans, dt)  # (T,3)

    # -----------------------------
    # 2) root_ang_vel (ASE: body_ang_vel[...,0,:])
    # -----------------------------
    # Compute from root orientation via quaternion differencing:
    #   diff_q = conj(q_t) * q_{t+1}
    #   omega  = axis*angle/dt
    #
    # This matches the style used in ASE's local_rotation_to_dof_vel().
    q_root = _axis_angle_to_quat_xyzw(root_orient_aa)  # (T,4)

    if T < 2:
        root_ang_vel = torch.zeros((T, 3), device=trans.device, dtype=trans.dtype)
    else:
        diff_q = _quat_mul_xyzw(
            _quat_conjugate_xyzw(q_root[:-1]), q_root[1:]
        )  # (T-1,4)
        diff_angle, diff_axis = _quat_to_angle_axis_xyzw(diff_q)  # (T-1,), (T-1,3)
        root_ang_vel_step = (diff_axis * diff_angle.unsqueeze(-1)) / dt  # (T-1,3)
        root_ang_vel = torch.cat(
            [root_ang_vel_step, root_ang_vel_step[-1:]], dim=0
        )  # (T,3)

    # -----------------------------
    # 3) dof_vel (ASE: dvs then dof_vel.view(B,-1))
    # -----------------------------
    # HUMOS pose_body is 21 joints (63 dims) in axis-angle.
    # ASE expects local rotations array including root at index 0.
    # We'll build local_rot_quat[t] = [root_quat, body_joint_quats...]
    J_body = pose_body_aa.shape[1] // 3  # 21
    J = 1 + J_body  # 22 total in this simplified view
    pose_body_aa_reshaped = pose_body_aa.view(T, J_body, 3)  # (T,21,3)
    q_body = _axis_angle_to_quat_xyzw(pose_body_aa_reshaped)  # (T,21,4)

    local_rot = torch.cat([q_root.unsqueeze(1), q_body], dim=1)  # (T,22,4)

    if T < 2:
        dof_vel = torch.zeros((T, (J - 1) * 3), device=trans.device, dtype=trans.dtype)
    else:
        # Frame-wise forward diff, identical to ASE compute_motion_dof_vels():
        #   for f in range(T-1):
        #       diff_q = conj(local_rot[f]) * local_rot[f+1]
        #       omega  = axis*angle/dt
        #       take omega[1:] (drop root), flatten
        diff_q = _quat_mul_xyzw(
            _quat_conjugate_xyzw(local_rot[:-1]), local_rot[1:]
        )  # (T-1,J,4)
        diff_angle, diff_axis = _quat_to_angle_axis_xyzw(diff_q)  # (T-1,J), (T-1,J,3)
        omega = (diff_axis * diff_angle.unsqueeze(-1)) / dt  # (T-1,J,3)

        omega_no_root = omega[:, 1:, :]  # (T-1,J-1,3)
        dof_vel_step = omega_no_root.reshape(T - 1, (J - 1) * 3)  # (T-1,(J-1)*3)

        # ASE appends last frame to keep length T:
        dof_vel = torch.cat([dof_vel_step, dof_vel_step[-1:]], dim=0)  # (T,(J-1)*3)

    return root_vel, root_ang_vel, dof_vel


# -----------------------
# Example usage (your shapes):
# -----------------------
# fps = 20  # HUMOS pipeline typically uses 20Hz
# root_vel, root_ang_vel, dof_vel = humos_to_ase_velocities(
#     root_orient_aa=root_orient,   # (200,3)
#     pose_body_aa=pose_body,       # (200,63)
#     trans=trans,                  # (200,3)
#     fps=fps,
# )
# root_vel:     (200,3)
# root_ang_vel: (200,3)
# dof_vel:      (200,63)


def _to_device(x: Any, device: torch.device) -> Any:
    """Recursively move tensors in nested structures to `device`."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_device(v, device) for v in x]
        return type(x)(t)
    return x


def get_text_motion_dataset(hparams, split: str):
    motion_loader, text_to_sent_emb, text_to_token_emb = initialize_dataloaders(hparams)

    return TextMotionDataset(
        path=hparams.TM_LOADER.PATH,
        motion_loader=motion_loader,
        text_to_sent_emb=text_to_sent_emb,
        text_to_token_emb=text_to_token_emb,
        split=split,
        preload=hparams.TM_LOADER.PRELOAD,
        demo=hparams.DEMO,
    )


def sample_betas_energy_uniform(
    batch_size: int,
    num_betas: int = 10,
    energy_max: float = 20.25,  # E_max = 4.5^2
    energy_min: float = 0.0,  # set = energy_max for fixed energy
    per_dim_clip: float = 3.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Sample SMPL betas with:
        - beta in R^{num_betas}
        - each component in [-per_dim_clip, per_dim_clip]
        - energy E = ||beta||^2 in [energy_min, energy_max]
        - directions uniform on the sphere
        - energy approximately uniform in [energy_min, energy_max]

    Args:
        batch_size: number of vectors to sample.
        num_betas: dimensionality (e.g. 10 or 16).
        energy_max: maximum energy (e.g. 20.25).
        energy_min: minimum energy (0 for full range; set equal to energy_max
                    if you want fixed energy).
        per_dim_clip: per-component bound (e.g. 3.0).
        rng: optional np.random.Generator.

    Returns:
        betas: (batch_size, num_betas) array.
    """
    if rng is None:
        rng = np.random.default_rng()

    collected = []

    def enough():
        return sum(x.shape[0] for x in collected) >= batch_size

    while not enough():
        # oversample to reduce the number of while-loop iterations
        n_try = max(batch_size - sum(x.shape[0] for x in collected), 1) * 8

        # 1) directions ~ uniform on sphere
        z = rng.standard_normal(size=(n_try, num_betas))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0.0] = 1.0
        dirs = z / norms

        # 2) energies: uniform in [energy_min, energy_max]
        E = rng.uniform(energy_min, energy_max, size=(n_try, 1))
        r = np.sqrt(E)

        # 3) construct candidates
        cand = dirs * r  # shape (n_try, num_betas)

        # 4) enforce per-dim constraint via rejection
        mask = np.all(np.abs(cand) <= per_dim_clip, axis=1)
        cand = cand[mask]

        if cand.size > 0:
            collected.append(cand)

    betas = np.concatenate(collected, axis=0)[:batch_size]
    # for smplsim, 4 decimal is good enough
    betas = np.round(betas, 4)
    return betas


def normalize_betas_np(betas_raw_10: np.ndarray, eps: float = 1e-8) -> np.ndarray:

    mu = np.array(
        [
            0.10722565,
            0.07524276,
            0.38341999,
            -0.00355983,
            0.1628108,
            -0.07072387,
            0.49694822,
            -0.20991106,
            -0.18934217,
            -0.63121307,
        ],
        dtype=np.float32,
    )

    sigma = np.array(
        [
            0.95840472,
            0.86099731,
            1.06210745,
            1.52516376,
            1.02516258,
            1.3749677,
            0.90751153,
            1.2583065,
            1.15225398,
            1.15355528,
        ],
        dtype=np.float32,
    )

    betas = np.asarray(betas_raw_10, dtype=np.float32)

    if betas.ndim == 1:
        assert betas.shape[0] == 10
        return (betas - mu) / (sigma + eps)

    # supports [n, 10] and more generally [..., 10]
    assert betas.shape[-1] == 10
    return (betas - mu[None, :]) / (sigma[None, :] + eps)


def build_betas_gender_table(
    all_betas_norm: np.ndarray,
    genders: Sequence[float] = (-1.0, 0.0, 1.0),
    dtype=np.float32,
) -> np.ndarray:
    """
    Input:  all_betas_norm shape [N, 10]
    Output: table shape [N * G, 11] where last column is gender, and each beta is repeated for each gender.
    """
    betas = np.asarray(all_betas_norm, dtype=dtype)
    assert (
        betas.ndim == 2 and betas.shape[1] == 10
    ), f"Expected [N,10], got {betas.shape}"

    genders_arr = np.asarray(genders, dtype=dtype).reshape(-1)
    G = genders_arr.shape[0]
    N = betas.shape[0]

    betas_rep = np.repeat(betas, G, axis=0)  # [N*G, 10]
    gender_col = np.tile(genders_arr, N).reshape(N * G, 1)  # [N*G, 1]

    return np.concatenate([betas_rep, gender_col], axis=1)  # [N*G, 11]


# ---------------------------------------------------------------------
# Grounding: compute a static (frame-0) ground offset from SMPL-H vertices
# ---------------------------------------------------------------------
def _gender_value_to_str(g: float) -> str:
    """Map identity gender value to SMPLLayer gender string."""
    if g == 1 or g == 1.0:
        return "male"
    if g == -1 or g == -1.0:
        return "female"
    # HUMOS also uses 0 for neutral
    return "neutral"


@torch.no_grad()
def compute_static_ground_offset_height(
    bm: SMPLLayer,
    *,
    betas: torch.Tensor,  # [1, 10]
    pose_body: torch.Tensor,  # [1, 63]
    root_orient: torch.Tensor,  # [1, 3]
    trans: torch.Tensor,  # [1, 3]
    up_axis: int = 2,  # SMPL/AMASS are typically Z-up
    safety_margin: float = 0.002,
) -> torch.Tensor:
    """
    Compute a scalar offset that would lift the first frame so the
    lowest vertex lies at `ground_level + safety_margin`.

    This mirrors the static grounding logic used in the visualizer, but returns
    the offset only (does not modify the motion).

    joints shape is [1, 73, 3]
    0-21: Body (Pelvis, Spine, ..., Ankles)
    22-36: Left Hand
    37-51: Right Hand
    52-72: Extras (Nose, Eyes, Heels, Toes, etc.)
    """
    verts, joints = bm(
        poses_body=pose_body,
        betas=betas,
        poses_root=root_orient,
        trans=trans,
    )  # verts: [1, V, 3] in world coordinates

    verts_first_frame = verts[0]

    h0_min = verts_first_frame[:, up_axis].amin()
    target = verts_first_frame.new_tensor(0.0 + safety_margin)
    offset = target - h0_min
    return offset, joints


@torch.no_grad()
def run_inference(hparams, all_betas_dict: Dict[str, np.ndarray]) -> None:
    ckpt = hparams.RESUME_CKPT
    if ckpt is None:
        raise ValueError("No checkpoint provided: set RESUME_CKPT in the config")

    ckpt_path = os.path.join(hparams.OUTPUT_DIR, ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {ckpt_path}")

    # Dataset / dataloader
    datasets = [
        get_text_motion_dataset(hparams, split=s) for s in ["train", "val", "test"]
    ]
    dataset = ConcatDataset(datasets)

    collate_fn = getattr(datasets[0], "collate_fn", None)

    dataloader = DataLoader(
        dataset,
        # batch_size=hparams.DATASET.BATCH_SIZE,
        batch_size=1,
        num_workers=hparams.DATASET.NUM_WORKERS,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    # renderer is not needed for pure generation
    model = initialize_model(hparams, ckpt_path, renderer=None)
    model.to(device)
    model.eval()

    # Cache SMPL-H layers for grounding (male/female/neutral)
    smpl_cache: Dict[str, SMPLLayer] = {}

    if not hasattr(model, "forward_cycle"):
        raise AttributeError(
            "Model does not expose `forward_cycle`; expected CYCLIC_TMR"
        )

    out_root = os.path.join(".", "output")

    os.makedirs(out_root, exist_ok=True)

    logger.info(f"Running inference on all with {len(dataset)} samples")
    logger.info(f"Saving outputs to: {out_root}")

    batch_idx = 0

    for _, batch in enumerate(tqdm(dataloader, desc="infer", dynamic_ncols=True)):
        # we are setting the btach size = 1
        batch = _to_device(batch, device)

        keyids_A = batch["keyid"]  # list-like, length = bs
        motion_x_dict_A = model.construct_input(batch["motion_x_dict"])
        mask_A = motion_x_dict_A["mask"]
        identity_A = motion_x_dict_A["identity"]

        # bs = identity_A.shape[0]
        T = identity_A.shape[1]

        # NEW: one output file per motion keyid
        motion_out = {"male": {}, "neutral": {}, "female": {}, "text": batch["text"]}

        for gender in [-1, 0, 1]:

            gender_str = _gender_value_to_str(gender)

            if gender_str not in smpl_cache:
                smpl_cache[gender_str] = SMPLLayer(
                    model_type="smplh", gender=gender_str, device=device
                )

            bm = smpl_cache[gender_str]

            for beta_key, beta_norm in all_betas_dict.items():

                beta_norm = beta_norm.astype(np.float32)  # [10]
                # we don't have to keep this, pred_dict_un["betas"] will get the unormalized value, equal to this one.
                # beta_raw = all_betas[i].astype(np.float32)  # [10]
                g = np.float32(gender)

                # beta_norm + gender
                identity_B11 = np.concatenate(
                    [beta_norm, np.array([g], dtype=np.float32)], axis=0
                )

                identity_B11 = torch.from_numpy(identity_B11).to(
                    device=device, dtype=torch.float32
                )

                identity_B = identity_B11.view(1, 1, 11).expand(1, T, 11).contiguous()

                # Forward cycle: A content, B identity
                outputs = model.forward_cycle(
                    motion_x_dict_A,
                    identity_A,
                    identity_B,
                    mask_A=mask_A,
                    return_all=True,
                    # make inference deterministic unless you explicitly want sampling
                    sample_mean=True,
                )
                motions_identityB_giv_contentA = outputs[1]

                # Decode back into feature dict, then unnormalize
                pred_dict_norm = model.deconstruct_input(
                    motions_identityB_giv_contentA[:, :, :-11],
                    identity_B,
                )

                pred_dict_un = model.normalizer.inverse(pred_dict_norm)

                # Convert to SMPLH parameters (batched)
                # fk_male is OK for kinematic decomposition; gender-specific mesh can be handled downstream.
                # betas: torch.Size([1, 200, 10])
                # gender: torch.Size([1, 200, 1])
                # root_orient: torch.Size([1, 200, 3])
                # pose_body: torch.Size([1, 200, 63])
                # trans: torch.Size([1, 200, 3])
                smpl_params_batched = smplh_breakdown(pred_dict_un, fk=model.fk_male)

                # Static grounding offset (frame 0) for this (beta, gender) pair.
                # We compute it from the SMPL-H mesh in world coordinates (trans applied),
                # then store it as metadata without altering the motion itself.
                offset_h, joints = compute_static_ground_offset_height(
                    bm,
                    betas=smpl_params_batched["betas"][0],
                    pose_body=smpl_params_batched["pose_body"][0],
                    root_orient=smpl_params_batched["root_orient"][0],
                    trans=smpl_params_batched["trans"][0],
                    up_axis=2,
                    safety_margin=0.002,
                )  # scalar tensor on `device`

                root_orient = smpl_params_batched["root_orient"][0].detach().cpu()
                pose_body = smpl_params_batched["pose_body"][0].detach().cpu()
                trans = smpl_params_batched["trans"][0].detach().cpu()

                root_vel, root_ang_vel, dof_vel = humos_to_ase_velocities(
                    root_orient_aa=root_orient,  # (200,3)
                    pose_body_aa=pose_body,  # (200,63)
                    trans=trans,  # (200,3)
                )

                joints_pos = joints.detach().cpu().squeeze(0)
                joints_pos = joints_pos[:, :22, :].contiguous()

                pose_body_21 = pose_body.reshape(T, 21, 3).contiguous()
                dof_vel_21 = dof_vel.reshape(T, 21, 3).contiguous()

                # todo, we need to pad joints_pos, pose_body_21, dof_vel_21
                T = pose_body_21.shape[0]  # or len(trans), etc.

                # 1. joints_pos: keep 22 (root + 21 body), or pad → 24 if downstream strictly wants SMPL 24-joint FK
                #    → most motion diffusion / ASE / HumanML3D-style use 22 → no padding needed here
                #    If you really need 24:
                joints_pos_padded = torch.cat(
                    [
                        joints_pos,  # (T, 22, 3)
                        torch.zeros(
                            T, 2, 3, dtype=joints_pos.dtype, device=joints_pos.device
                        ),
                    ],
                    dim=1,
                )  # → (T, 24, 3)

                # 2. pose_body: from 21 → 23 joints (add zero rotation for left & right hand roots)
                pose_body_23 = torch.cat(
                    [
                        pose_body_21,  # (T, 21, 3)
                        torch.zeros(
                            T,
                            2,
                            3,
                            dtype=pose_body_21.dtype,
                            device=pose_body_21.device,
                        ),
                    ],
                    dim=1,
                )  # → (T, 23, 3)

                # 3. dof_vel: same as pose_body
                dof_vel_23 = torch.cat(
                    [
                        dof_vel_21,  # (T, 21, 3)
                        torch.zeros(
                            T, 2, 3, dtype=dof_vel_21.dtype, device=dof_vel_21.device
                        ),
                    ],
                    dim=1,
                )  # → (T, 23, 3)

                motion_out[gender_str][beta_key] = {
                    "betas": smpl_params_batched["betas"].detach().cpu().squeeze(0),
                    "gender": smpl_params_batched["gender"].detach().cpu().squeeze(0),
                    "root_orient": root_orient,
                    "pose_body": pose_body_23,
                    "trans": trans,
                    "offset_height": offset_h.detach().cpu().squeeze(0),
                    "joints_pos": joints_pos_padded,
                    "root_vel": root_vel,
                    "root_ang_vel": root_ang_vel,
                    "dof_vel": dof_vel_23,
                }

        # # betas:        torch.Size([200, 10])
        # # gender:       torch.Size([200, 1])
        # # root_orient:  torch.Size([200, 3])
        # # pose_body:    torch.Size([200, 23, 3])
        # # trans:        torch.Size([200, 3])
        # # offset_height:torch.Size([])
        # # joints_pos:   torch.Size([200, 22, 3])
        # # root_vel:     torch.Size([200, 3])
        # # root_ang_vel: torch.Size([200, 3])
        # # dof_vel:      torch.Size([200, 23, 3])
        # for k, v in motion_out.items():
        #     # gender key
        #     print(k)
        #     # print(v.keys())

        #     if k != "text":

        #         for k1, v1 in v.items():
        #             # beta key
        #             print(k1)
        #             print(v1.keys())

        #             for k2, v2 in v1.items():
        #                 # features "betas", "gender", "root_orient", "pose_body", "trans", "offset_height", "joints_pos"
        #                 print(k2)
        #                 print(v2.shape)
        # exit()

        # when set batch_szie=1, keyids_A[0] is fine
        save_path = os.path.join(out_root, f"{keyids_A[0]}.pt")
        torch.save(motion_out, save_path)
        print(f"Saved: {save_path}")

        batch_idx += 1

        if batch_idx > 10:
            break

    logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    # Keep the same grid-search / config loading pathway as training for compatibility.
    hparams = run_grid_search_experiments(args, script="train.py")

    def load_all_betas_dict(pt_path: str) -> Dict[str, np.ndarray]:
        """
        Load betas dict saved as {betas_key: tensor/array shape [10]} and return
        {betas_key: np.ndarray float32 shape [10]}.
        """
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"all_betas.pt not found: {pt_path}")

        raw = torch.load(pt_path, map_location="cpu")
        if not isinstance(raw, dict):
            raise TypeError(f"Expected a dict in {pt_path}, got: {type(raw)}")

        betas_dict: Dict[str, np.ndarray] = {}
        for k, v in raw.items():
            key = str(k)
            if torch.is_tensor(v):
                v = v.detach().cpu().numpy()
            v = np.asarray(v, dtype=np.float32)
            if v.shape != (10,):
                raise ValueError(
                    f"Betas for key={key} must be shape (10,), got {v.shape}"
                )
            betas_dict[key] = v

        return betas_dict

    def normalize_all_betas_dict(
        all_betas_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Convert {betas_key: beta_raw[10]} -> {betas_key: beta_norm[10]} using your
        existing normalize_betas_np(...) function.

        Returns float32 arrays of shape (10,).
        """
        out: Dict[str, np.ndarray] = {}
        for betas_key, beta_raw in all_betas_dict.items():
            beta_raw = np.asarray(beta_raw, dtype=np.float32)
            if beta_raw.shape != (10,):
                raise ValueError(
                    f"Betas for key={betas_key} must be shape (10,), got {beta_raw.shape}"
                )

            beta_norm = normalize_betas_np(beta_raw)  # <- existing function in infer.py
            beta_norm = np.asarray(beta_norm, dtype=np.float32)

            if beta_norm.shape != (10,):
                raise ValueError(
                    f"Normalized betas for key={betas_key} must be shape (10,), got {beta_norm.shape}"
                )

            out[betas_key] = beta_norm

        return out

    betas_pt_path = os.path.join(os.path.dirname(__file__), "all_betas.pt")
    all_betas_dict = load_all_betas_dict(betas_pt_path)

    all_betas_dict = normalize_all_betas_dict(all_betas_dict)

    run_inference(hparams, all_betas_dict)
