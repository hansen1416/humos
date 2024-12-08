import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import wandb

from humos.utils.mesh_utils import smplh_breakdown
from humos.utils.misc_utils import copy2cpu as c2c
from humos.utils import constants

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import ViewerCamera
from .losses import InfoNCE_with_filtering
from .metrics import all_contrastive_metrics, all_physics_metrics
from .temos import TEMOS


# x.T will be deprecated in pytorch
def transpose(x):
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))


def get_sim_matrix(x, y):
    x_logits = torch.nn.functional.normalize(x, dim=-1)
    y_logits = torch.nn.functional.normalize(y, dim=-1)
    sim_matrix = x_logits @ transpose(y_logits)
    return sim_matrix


# Scores are between 0 and 1
def get_score_matrix(x, y):
    sim_matrix = get_sim_matrix(x, y)
    scores = sim_matrix / 2 + 0.5
    return scores


class TMR(TEMOS):
    r"""TMR: Text-to-Motion Retrieval
    Using Contrastive 3D Human Motion Synthesis
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/tmr

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
        temperature: temperature of the softmax in the contrastive loss (optional).
        threshold_selfsim: threshold used to filter wrong negatives for the contrastive loss (optional).
        threshold_selfsim_metrics: threshold used to filter wrong negatives for the metrics (optional).
    """

    def __init__(
            self,
            motion_encoder: nn.Module,
            motion_decoder: nn.Module,
            normalizer: nn.Module,
            vae: bool,
            fact: Optional[float] = None,
            sample_mean: Optional[bool] = False,
            lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5, "contrastive": 0.1},
            lr: float = 1e-4,
            fps: float =20,
            temperature: float = 0.7,
            threshold_selfsim: float = 0.80,
            threshold_selfsim_metrics: float = 0.95,
            num_val_videos: int = 3,
            val_video_log_freq: int = 10,
    ) -> None:
        # Initialize module like TEMOS
        super().__init__(
            motion_encoder=motion_encoder,
            motion_decoder=motion_decoder,
            normalizer=normalizer,
            vae=vae,
            fact=fact,
            sample_mean=sample_mean,
            lmd=lmd,
            lr=lr,
            fps=fps,
        )

        self.fps = fps

        self.num_val_videos = num_val_videos
        self.val_video_log_freq = val_video_log_freq

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        self.threshold_selfsim_metrics = threshold_selfsim_metrics

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_m_latents = []
        self.validation_step_val_videos = []
        self.validation_step_ref_videos = []
        self.validation_step_metrics = []

        # Set AITviewer Renderer

        # Set a custom camera saved from the viewer
        self.camera = ViewerCamera()
        # cam_dir = C.export_dir + "/camera_params/"
        # cam_dict = joblib.load(cam_dir + "cam_params.pkl")
        self.camera.load_cam()
        self.renderer = HeadlessRenderer(size=(320, 240))
        self.renderer.set_temp_camera(self.camera)

        # Get smpl body models
        self.bm_male = SMPLLayer(model_type="smplh", gender="male", device=self.device)
        self.bm_female = SMPLLayer(model_type="smplh", gender="female", device=self.device)

    def run_smpl_fk(self, motions, betas, gender):
        smpl_params = smplh_breakdown(motions, betas, gender, self.fps)

        m_idx = torch.nonzero(gender == 1).squeeze()
        f_idx = torch.nonzero(gender == -1).squeeze()

        m_bs, f_bs = len(m_idx) if m_idx.dim() > 0 else 0, len(f_idx) if f_idx.dim() > 0 else 0
        m_fr, f_fr = motions.shape[1], motions.shape[1]

        if m_bs > 0:
            # split male
            male_params = {k: v[m_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            male_params = {k: v.view(m_bs * m_fr, -1) for k, v in male_params.items() if k != 'gender'}
            # run smpl fk
            m_verts, m_joints = self.bm_male(poses_body=male_params['pose_body'],
                                             betas=male_params['betas'],
                                             poses_root=male_params['root_orient'],
                                             trans=male_params['transl'])
            # exclude hand joints
            m_joints = m_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            m_verts, m_joints = m_verts.view(m_bs, m_fr, -1, 3), m_joints.view(m_bs, m_fr, -1, 3)

        if f_bs > 0:
            # split female
            female_params = {k: v[f_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            female_params = {k: v.view(f_bs * f_fr, -1) for k, v in female_params.items() if k != 'gender'}
            # run smpl fk
            f_verts, f_joints = self.bm_female(poses_body=female_params['pose_body'],
                                               betas=female_params['betas'],
                                               poses_root=female_params['root_orient'],
                                               trans=female_params['transl'])
            # exclude hand joints
            f_joints = f_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            f_verts, f_joints = f_verts.view(f_bs, f_fr, -1, 3), f_joints.view(f_bs, f_fr, -1, 3)

        if m_bs > 0 and f_bs > 0:
            # join the f_verts and m_verts according to m_idx and f_idx
            verts = torch.zeros_like(torch.concatenate((m_verts, f_verts), dim=0))
            verts[m_idx] = m_verts
            verts[f_idx] = f_verts

            joints = torch.zeros_like(torch.concatenate((m_joints, f_joints), dim=0))
            joints[m_idx] = m_joints
            joints[f_idx] = f_joints
        elif m_bs > 0:
            verts = m_verts
            joints = m_joints
        elif f_bs > 0:
            verts = f_verts
            joints = f_joints

        # # Visualize the meshes
        # # save the mesh as an obj file
        # for i, vert in enumerate(verts[0]):
        #     import trimesh
        #     body_mesh = trimesh.Trimesh(vertices=c2c(vert), faces=c2c(self.bm_male.faces),
        #                                 vertex_colors=np.tile([255, 200, 200, 255], (6890, 1)))
        #     out_folder = f'./debug_mesh/0/'
        #     os.makedirs(out_folder, exist_ok=True)
        #     body_mesh.export(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        #     print(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        # print('------------------')
        #
        # # Visualize the meshes
        # # save the mesh as an obj file
        # for i, vert in enumerate(verts[1]):
        #     import trimesh
        #     body_mesh = trimesh.Trimesh(vertices=c2c(vert), faces=c2c(self.bm_male.faces),
        #                                 vertex_colors=np.tile([255, 200, 200, 255], (6890, 1)))
        #     out_folder = f'./debug_mesh/1/'
        #     os.makedirs(out_folder, exist_ok=True)
        #     body_mesh.export(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        #     print(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        # print('------------------')
        #
        # import ipdb; ipdb.set_trace()

        return verts, joints

    def compute_loss(self, batch: Dict, return_all=False) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # Get shape parameters
        betas = ref_motions[:, 0, -11:-1]

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, betas, mask=mask, return_all=True)

        # Store all losses and metrics
        losses = {}
        metrics = {k: 0.0 for k in ["penetrate", "float", "skate"]}

        # Reconstructions losses
        # fmt: off

        recons_type = self.lmd["recons_type"]

        # unnormalize the motion features
        ref_motions_un = self.normalizer.inverse(ref_motions)
        m_motions_un = self.normalizer.inverse(m_motions)

        # Get GT shape and gender
        betas = ref_motions[:, :, -11:-1]
        gender = ref_motions[:, 0, -1]

        # Run SMPLH FK to get the joints predicted
        m_verts, m_joints = self.run_smpl_fk(m_motions_un, betas, gender)

        if recons_type == "rotations":
            losses["recons"] = (
                + self.reconstruction_loss_fn(m_motions[:, :, :-11],
                                              ref_motions[:, :, :-11])  # motion -> motion
            )
        elif recons_type == "joints":
            # Run SMPLH FK to get the GT joints
            ref_verts, ref_joints = self.run_smpl_fk(ref_motions_un, betas, gender)

            losses["recons"] = (
                + self.reconstruction_loss_fn(m_joints,
                                              ref_joints)  # motion -> motion
            )
        else:
            raise ValueError(f"recons_type {recons_type} not recognized")
        # fmt: on

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            ref_mus = torch.zeros_like(m_dists[0])
            ref_logvar = torch.zeros_like(m_dists[1])
            ref_dists = (ref_mus, ref_logvar)

            losses["kl"] = (
                self.kl_loss_fn(m_dists, ref_dists)  # motion
            )

        # Latent manifold loss
        losses["latent"] = torch.zeros_like(losses["recons"])

        # TMR: adding the contrastive loss
        losses["contrastive"] = torch.zeros_like(losses["recons"])

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Weight all loss values with config weight
        losses = {x: self.lmd[x] * val if x in self.lmd else val for x, val in losses.items()}

        # Used for the validation step
        if return_all:
            # Get physical plausability metrics
            metrics["penetrate"], metrics["float"], metrics["skate"] = all_physics_metrics(m_verts, m_joints,
                                                                                           device=self.device)
            return losses, None, m_latents, metrics

        return losses

    def render_frames(self, motions, betas, gender, tag="pred"):
        pred_params = smplh_breakdown(motions, betas, gender, self.fps)

        # get num_videos equally spaced frames
        select_indices = torch.linspace(0, motions.shape[0] - 1, self.num_val_videos).long()
        # get the corresponding SMPL parameters
        pred_params = {k: v[select_indices] for k, v in pred_params.items()}

        # loop over filtered frames
        all_frames = []
        for i in range(self.num_val_videos):
            gender = pred_params['gender'][i]
            if gender == 1:
                smpl_layer = SMPLLayer(model_type="smplh", gender="male", device=self.device)
            elif gender == -1:
                smpl_layer = SMPLLayer(model_type="smplh", gender="female", device=self.device)
            else:
                assert gender == 1 or gender == -1
            smpl_seq = SMPLSequence(
                poses_body=pred_params["pose_body"][i, :, :],
                poses_root=pred_params["root_orient"][i, :, :],
                betas=pred_params["betas"][i, :, :],
                trans=pred_params["transl"][i, :, :],
                smpl_layer=smpl_layer,
                z_up=True,
            )
            # Create the headless renderer and add the sequence.
            self.renderer.scene.add(smpl_seq)
            frames = self.renderer.save_video(video_dir=os.path.join(C.export_dir, f"{tag}_vis/{tag}_{i}.mp4"),
                                              save_on_disk=False)

            # Convert frames to array
            frames = np.array(frames)
            all_frames.append(frames)

            self.renderer.scene.remove(smpl_seq)
            # save frames as a log in wandb

        # join all frames into one video horizontally
        all_frames = np.concatenate(all_frames, axis=2)
        return all_frames

    def visualize_reconstructions(self, batch: Dict, batch_idx: int):

        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # Get shape parameters
        betas = ref_motions[:, 0, -11:-1]

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, betas, mask=mask, return_all=True)

        ref_motions = self.normalizer.inverse(ref_motions)
        betas = ref_motions[:, :, -11:-1]
        gender = ref_motions[:, 0, -1]

        all_ref_frames = self.render_frames(ref_motions, betas, gender, tag="ref")

        m_motions = self.normalizer.inverse(m_motions)
        all_pred_frames = self.render_frames(m_motions, betas, gender, tag="pred")

        return all_pred_frames, all_ref_frames

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])

        losses, t_latents, m_latents, metrics = self.compute_loss(batch, return_all=True)

        # Store the metrics
        self.validation_step_metrics.append(metrics)

        # Store the latent vectors
        self.validation_step_m_latents.append(m_latents)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_step/val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        if batch_idx % self.val_video_log_freq == 0:
            val_videos, ref_videos = self.visualize_reconstructions(batch, batch_idx)
            self.validation_step_val_videos.append(val_videos)
            self.validation_step_ref_videos.append(ref_videos)

        return losses["loss"]

    def on_validation_epoch_end(self):
        # Compute contrastive metrics on the whole batch
        m_latents = torch.cat(self.validation_step_m_latents)

        # join all frames into one video vertically
        val_videos = np.concatenate(self.validation_step_val_videos, axis=1)
        ref_videos = np.concatenate(self.validation_step_ref_videos, axis=1)

        # reorder the dimensions to be (T, C, H, W)
        val_videos = np.transpose(val_videos, (0, 3, 1, 2))
        ref_videos = np.transpose(ref_videos, (0, 3, 1, 2))

        wandb.log({"vis/pred_vis": [wandb.Video(val_videos, fps=20, format="mp4")]})
        wandb.log({"vis/ref_vis": [wandb.Video(ref_videos, fps=20, format="mp4")]})

        # Combine all metrics from the batches
        all_metrics = {}
        for metric_name in self.validation_step_metrics[0].keys():
            # Get the mean metrics in cms
            all_metrics[metric_name] = np.mean(
                [x[metric_name] * 100 for x in self.validation_step_metrics]
            )
            self.log(
                f"val_{metric_name}_epoch",
                all_metrics[metric_name],
                on_epoch=True,
                on_step=False,
            )

        self.validation_step_m_latents.clear()
        self.validation_step_val_videos.clear()
        self.validation_step_ref_videos.clear()
        self.validation_step_metrics.clear()
