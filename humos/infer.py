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

from humos.src.data.text_motion import TextMotionDataset
from humos.src.initialize import initialize_dataloaders, initialize_model
from humos.utils.config import parse_args, run_grid_search_experiments
from humos.utils.mesh_utils import smplh_breakdown


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


def _default_identity_B(device: torch.device) -> torch.Tensor:
    """Default target identity (10 betas + gender). Gender: +1 male, -1 female."""
    return torch.tensor(
        [
            1.0472344,
            -1.3409365,
            0.97568285,
            -0.4312587,
            -1.2148422,
            -1.4349254,
            0.7715073,
            1.0130371,
            0.8836092,
            2.6459184,
            -1.0,
        ],
        dtype=torch.float32,
        device=device,
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


@torch.no_grad()
def run_inference(hparams, all_betas) -> None:
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

    if not hasattr(model, "forward_cycle"):
        raise AttributeError(
            "Model does not expose `forward_cycle`; expected CYCLIC_TMR"
        )

    out_root = os.path.join(".", "output")

    os.makedirs(out_root, exist_ok=True)

    logger.info(f"Running inference on all with {len(dataset)} samples")
    logger.info(f"Saving outputs to: {out_root}")

    all_betas_norm = normalize_betas_np(all_betas)

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

        for gender in [-1, 0, 1]:

            # accumulate 64 retargeted results (one per beta) for this gender
            acc = {
                "betas": [],
                "gender": [],
                "root_orient": [],
                "pose_body": [],
                "trans": [],
            }

            for i in range(all_betas_norm.shape[0]):

                beta_norm = all_betas_norm[i].astype(np.float32)  # [10]
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

                # append (drop bs dim=1) -> [T, D]
                for k in acc.keys():
                    v = smpl_params_batched[k]
                    acc[k].append(v.detach().cpu().squeeze(0))

            # stack 64 betas for this gender: [64, T, D]
            # betas: torch.Size([64, 200, 10]); gender: torch.Size([64, 200, 1]); root_orient: torch.Size([64, 200, 3]); pose_body: torch.Size([64, 200, 63]); trans: torch.Size([64, 200, 3])
            stacked = {k: torch.stack(vlist, dim=0) for k, vlist in acc.items()}

            stacked["text"] = batch["text"]

            # when set batch_szie=1, keyids_A[0] is fine
            save_path = os.path.join(out_root, f"{keyids_A[0]}_{gender}.pt")
            torch.save(stacked, save_path)
            print(f"Saved: {save_path}")

        batch_idx += 1

        if batch_idx > 10:
            break

    logger.info("Done.")


if __name__ == "__main__":
    args = parse_args()
    # Keep the same grid-search / config loading pathway as training for compatibility.
    hparams = run_grid_search_experiments(args, script="train.py")

    num_betas: int = 10
    batch_size: int = 64
    per_dim_clip: float = 3.0
    energy_max: float = 20.25
    energy_min: float = 0.0

    rng = np.random.default_rng(46)

    all_betas = sample_betas_energy_uniform(
        batch_size=batch_size,
        num_betas=num_betas,
        per_dim_clip=per_dim_clip,
        energy_max=energy_max,
        energy_min=energy_min,
        rng=rng,
    )

    run_inference(hparams, all_betas)
