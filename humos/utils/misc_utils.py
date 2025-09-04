from typing import List
import wandb

import numpy as np
import torch
from torch import Tensor
import trimesh
from pathlib import Path


def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def length_to_mask(length: List[int], device: torch.device = None) -> Tensor:
    if device is None:
        device = "cpu"

    if isinstance(length, list):
        length = torch.tensor(length, device=device)

    max_len = max(length)
    mask = torch.arange(max_len, device=device).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    return mask


def get_rgba_colors(color='grey'):
    if color == 'grey':
        rgba = (160 / 255, 160 / 255, 160 / 255, 1.0)
    if color == 'red':
        rgba = (160 / 255, 0 / 255, 0 / 255, 1.0)
    if color == 'green':
        rgba = (0 / 255, 160 / 255, 0 / 255, 1.0)
    if color == 'blue':
        rgba = (0 / 255, 0 / 255, 160 / 255, 1.0)
    if color == 'purple':
        rgba = (160 / 255, 0 / 255, 160 / 255, 1.0)
    return rgba


def visualize_joints(joints, name='joints'):
    # Plot the 3D SMPL model
    import matplotlib.pyplot as plt
    import os
    num_frames = joints.shape[0]
    for frame_idx in range(num_frames):
        # Get the vertices of the SMPL model
        frame_joints = joints[frame_idx, :, :].cpu().numpy()

        # Plot the 3D SMPL model
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the joints on the 3D model
        ax.scatter(frame_joints[:, 0], frame_joints[:, 1], frame_joints[:, 2], c='r', marker='x', label='Joints')

        # Plot the ground plane at z=0
        ground_size = 2.0  # Adjust as needed
        ax.plot_surface(np.array([[-ground_size, ground_size]]),
                        np.array([[-ground_size, ground_size]]),
                        np.array([[0, 0]]), alpha=0.3, color='g', label='Ground Plane')

        # Set labels and legend
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(f'Frame {frame_idx}; root joint height {frame_joints[0, 2]:.3f}')

        ax.legend()

        # Save plot
        out_dir = f'./smpl_{name}_vis'
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f'{out_dir}/{frame_idx}.png')
        # close
        plt.close()

def save_demo_meshes(m_verts_B_giv_A, faces, keyids, target_keyids, ckpt_name="DEMO", num_seqs=20):
    # This function saves the motion objs created for the demo
    out_dir = Path(
        f"./demo/humos_{ckpt_name}_fit_objs/pred")
    # create out_dir if doesn't exist
    for i, (keyid, target_keyid) in enumerate(zip(keyids, target_keyids)):
        if i>num_seqs:
            break
        vertices = m_verts_B_giv_A[i].cpu().numpy()
        out_path = out_dir / f"reconstruction_{keyid}_{target_keyid}_objs"
        out_path.mkdir(exist_ok=True, parents=True)
        for seq_id, vert in enumerate(vertices):
            mesh = trimesh.Trimesh(vertices=vert, faces=faces, process=False)
            mesh.export(out_path / f"frame_{seq_id:04d}.obj")
        print(f"Saved {out_path}")

def update_best_metrics():
    down_val_keys = ["val_step/loss_epoch", "val_step/motion_prior_epoch", "val_step/joint_recons_epoch",
                "val_step/physics_float_epoch", "val_step/physics_penetrate_epoch",
                "val_step/physics_skate_epoch", "val_step/cycle_latent_epoch", "val_step/recons_epoch",
                "val_step/kl_epoch", "val_step/latent_epoch",
                "metrics/pose_body_epoch", "metrics/root_orient_epoch", "metrics/trans_epoch", "metrics/net_recons_epoch",
                "metrics/mean_dyn_positive_distance_epoch",
                "metrics/physics_penetrate_epoch", "metrics/physics_skate_epoch", "metrics/physics_float_epoch",
                "metrics/physics_skate_perc_epoch", "metrics/physics_skate_sum_epoch", "metrics/physics_skate_std_epoch",
                "metrics/fid_epoch"]

    up_val_keys = ["metrics/perc_dyn_stable_epoch", "metrics/gt_diversity_epoch", "metrics/pred_diversity_epoch"]

    for key in down_val_keys:
        try:
            new_key = key.replace("val_step", "valBest").replace("metrics", "metricsBest")
            if wandb.run.summary[new_key] > wandb.run.summary[key]:
                wandb.run.summary[new_key] = wandb.run.summary[key]
                new_key_epoch_num = new_key + "_num"
                wandb.run.summary[new_key_epoch_num] = wandb.run.summary["current_epoch"]
        except KeyError:
            if key in wandb.run.summary.keys():
                new_key = key.replace("val_step", "valBest").replace("metrics", "metricsBest")
                wandb.run.summary[new_key] = wandb.run.summary[key]
                new_key_epoch_num = new_key + "_num"
                wandb.run.summary[new_key_epoch_num] = wandb.run.summary["current_epoch"]

    for key in up_val_keys:
        try:
            new_key = key.replace("val_step", "valBest").replace("metrics", "metricsBest")
            if wandb.run.summary[new_key] < wandb.run.summary[key]:
                wandb.run.summary[new_key] = wandb.run.summary[key]
                new_key_epoch_num = new_key + "_num"
                wandb.run.summary[new_key_epoch_num] = wandb.run.summary["current_epoch"]
        except KeyError:
            if key in wandb.run.summary.keys():
                new_key = key.replace("val_step", "valBest").replace("metrics", "metricsBest")
                wandb.run.summary[new_key] = wandb.run.summary[key]
                new_key_epoch_num = new_key + "_num"
                wandb.run.summary[new_key_epoch_num] = wandb.run.summary["current_epoch"]
