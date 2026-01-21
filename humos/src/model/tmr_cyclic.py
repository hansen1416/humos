import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
import pickle as pkl
from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import ViewerCamera
from torch import Tensor

from bosRegressor.core.biomechanics import (
    BiomechanicalEvaluator,
    setup_biomechanical_evaluator,
)
from humos.utils import constants
from humos.utils.fk import ForwardKinematicsLayer
from humos.utils.mesh_utils import smplh_breakdown
from humos.utils.misc_utils import (
    get_rgba_colors,
    update_best_metrics,
    save_demo_meshes,
)
from .losses import InfoNCE_with_filtering, MotionPriorLoss, DynStabilityLoss
from .metrics import (
    all_physics_metrics,
    calculate_recons_metrics,
    calculate_dyn_stability_metric,
    MotionPriorMetric,
)
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


class CYCLIC_TMR(TEMOS):
    r"""HUMOS: Motion-to-Motion Model
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
        pretrained_motion_encoder: nn.Module,
        normalizer: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        train_feats: Optional[List[str]] = None,
        lmd: Dict = {
            "recons": 1.0,
            "joint_recons": 1.0,
            "latent": 1.0e-5,
            "cycle_latent": 1.0e-5,
            "kl": 1.0e-5,
            "contrastive": 0.1,
            "motion_prior": 1.0,
            "physics_float": 1.0,
            "physics_penetrate": 1.0,
            "physics_skate": 1.0,
            "dyn_stability": 1.0,
        },
        compute_metrics: Dict = {
            "dyn_stability": False,
            "recons": True,
            "physics": True,
            "motion_prior": True,
        },
        bs: int = 14,
        lr: float = 1e-4,
        fps: float = 20.0,
        temperature: float = 0.7,
        threshold_selfsim: float = 0.80,
        threshold_selfsim_metrics: float = 0.95,
        cop_w=30.0,
        cop_k=100.0,
        num_val_videos: int = 3,
        max_vid_rows: int = 3,
        run_cycle: bool = True,
        demo: bool = False,
        renderer: HeadlessRenderer = None,
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
            compute_metrics=compute_metrics,
            fps=fps,
            lr=lr,
        )

        self.fps = fps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # adding the contrastive loss
        self.contrastive_loss_fn = InfoNCE_with_filtering(
            temperature=temperature, threshold_selfsim=threshold_selfsim
        )
        # adding the physics losses
        self.physics_loss_fn = all_physics_metrics
        self.motion_prior_loss_fn = MotionPriorLoss(pretrained_motion_encoder)
        self.dyn_stability_loss_fn = DynStabilityLoss()
        self.threshold_selfsim_metrics = threshold_selfsim_metrics
        self.motion_prior_metrics = MotionPriorMetric(pretrained_motion_encoder)

        # store validation values to compute retrieval metrics
        # on the whole validation set
        self.validation_step_m_latents_A = []
        self.validation_step_m_latents_B = []
        self.validation_step_ref_videos_A = []
        self.validation_step_val_videos_A_giv_A = []
        self.validation_step_val_videos_B_giv_A = []
        self.validation_step_val_videos_A_giv_B = []
        self.validation_step_metrics = []
        self.validation_step_pred_embeddings = []
        self.validation_step_gt_embeddings = []

        self.run_cycle = run_cycle

        # Get smpl body models
        self.bm_male = SMPLLayer(model_type="smplh", gender="male", device=device)
        self.bm_female = SMPLLayer(model_type="smplh", gender="female", device=device)

        # Get fk models
        self.fk_male = ForwardKinematicsLayer(
            constants.SMPL_PATH,
            gender="male",
            num_joints=constants.SMPLH_BODY_JOINTS,
            device=device,
        )
        # self.fk_female = ForwardKinematicsLayer(constants.SMPL_PATH, gender="female", num_joints=constants.SMPLH_BODY_JOINTS)

        # Specify training features
        self.train_feats = train_feats

        # # Sample the random body shapes for validation run
        # sampled_idx_path = './stats/random_body_shapes.json'
        # if os.path.exists(sampled_idx_path):
        #     with open(sampled_idx_path, 'r') as f:
        #         sampled_dict = json.load(f)
        #         self.sampled_idx = sampled_dict[str(bs)]
        # else:
        #     assert False, "Please run the script to sample random body shapes idx first"

        self.sampled_idx = torch.randperm(bs)

        faces = torch.tensor(self.bm_male.faces, dtype=torch.int64).to(device)
        self.biomechanical_evaluator = BiomechanicalEvaluator(
            faces=faces,
            fps=self.fps,
            cop_w=cop_w,
            cop_k=cop_k,
            stencil_size=3,
            device=device,
        )

        # args = Arguments('./configs', filename='generative.yaml')
        # self.nemf_model = NemfArchitecture(args, ngpu=1)

    def forward_cycle(
        self,
        inputs_A,
        identity_A,
        identity_B,
        lengths_A: Optional[List[int]] = None,
        mask_A: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:
        # Encoding the inputs and sampling if needed
        # `latent_vectors_A` is the **actual latent code** used by the decoder to generate motion. Shape is typically `[B, Z]` (batch × latent_dim) or similar. If you want to generate retargeted motions for many body shapes, this is the thing you reuse: encode once → decode many times with different `identity_B`.
        # `distributions_A`
        # This is the **parameterization of the latent distribution** produced by the encoder (only meaningful if the model is VAE-like). Usually a tuple like `(mu, logvar)` (or `(mu, sigma)`), representing a Gaussian posterior \(q(z \mid x, \text{identity})\). It is used for: - computing **KL divergence** regularization (`kl_loss_fn(...)` later uses `m_dists_A`), and/or - deciding how to sample `latent_vectors_A` (sample vs mean).
        latent_vectors_A, distributions_A = self.encode(
            inputs_A,
            identity_A,
            sample_mean=sample_mean,
            fact=fact,
            return_distribution=True,
        )
        # Decoding the latent vector: generating motions
        motions_identityA_giv_contentA = self.decode(
            latent_vectors_A, identity_A, lengths_A, mask_A
        )

        if self.run_cycle:
            # Decoding the latent vector: generating motions
            # Note beta_B includes gender
            motions_identityB_giv_contentA = self.decode(
                latent_vectors_A, identity_B, lengths_A, mask_A
            )

            # Do not output the betas and gender, use GT
            inputs_motion_identityB_giv_contentA = inputs_A.copy()
            inputs_motion_identityB_giv_contentA["x"] = motions_identityB_giv_contentA[
                :, :, :-11
            ]
            inputs_motion_identityB_giv_contentA["identity"] = identity_B

            # Encoding in the backward cycle
            latent_vectors_B, distributions_B = self.encode(
                inputs_motion_identityB_giv_contentA,
                identity_B,
                sample_mean=sample_mean,
                fact=fact,
                return_distribution=True,
            )

            # Decoding in the backward cycle
            motions_identityB_giv_contentB = self.decode(
                latent_vectors_B, identity_B, lengths_A, mask_A
            )

            # Decoding in the backward cycle
            motions_identityA_giv_contentB = self.decode(
                latent_vectors_B, identity_A, lengths_A, mask_A
            )
        else:
            (
                motions_identityB_giv_contentA,
                motions_identityA_giv_contentB,
                motions_identityB_giv_contentB,
                latent_vectors_B,
                distributions_B,
            ) = (None, None, None, None, None)

        if return_all:
            return (
                motions_identityA_giv_contentA,
                motions_identityB_giv_contentA,
                latent_vectors_A,
                distributions_A,
                motions_identityB_giv_contentB,
                motions_identityA_giv_contentB,
                latent_vectors_B,
                distributions_B,
            )

        return (
            motions_identityA_giv_contentA,
            motions_identityB_giv_contentA,
            motions_identityB_giv_contentB,
            motions_identityA_giv_contentB,
        )

    def run_smpl_fk(self, data, skinning=True):
        gender = data["identity"][:, 0, -1]
        smpl_params = smplh_breakdown(data, fk=self.fk_male)

        m_idx = torch.nonzero(gender == 1).squeeze(-1)
        f_idx = torch.nonzero(gender == -1).squeeze(-1)

        m_bs, f_bs = len(m_idx) if m_idx.dim() > 0 else 0, (
            len(f_idx) if f_idx.dim() > 0 else 0
        )
        m_fr, f_fr = data["betas"].shape[1], data["betas"].shape[1]

        if m_bs > 0:
            # split male
            male_params = {k: v[m_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            male_params = {
                k: v.view(m_bs * m_fr, -1)
                for k, v in male_params.items()
                if k != "gender"
            }
            # run smpl fk
            m_verts, m_joints = self.bm_male(
                poses_body=male_params["pose_body"],
                betas=male_params["betas"],
                poses_root=male_params["root_orient"],
                trans=male_params["trans"],
                # skinning=skinning,
            )
            # exclude hand joints
            m_joints = m_joints[:, : constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            m_verts, m_joints = (
                m_verts.view(m_bs, m_fr, -1, 3) if m_verts is not None else m_verts,
                m_joints.view(m_bs, m_fr, -1, 3),
            )

        if f_bs > 0:
            # split female
            female_params = {k: v[f_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            female_params = {
                k: v.view(f_bs * f_fr, -1)
                for k, v in female_params.items()
                if k != "gender"
            }
            # run smpl fk
            f_verts, f_joints = self.bm_female(
                poses_body=female_params["pose_body"],
                betas=female_params["betas"],
                poses_root=female_params["root_orient"],
                trans=female_params["trans"],
                # skinning=skinning,
            )
            # exclude hand joints
            f_joints = f_joints[:, : constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            f_verts, f_joints = (
                f_verts.view(f_bs, f_fr, -1, 3) if f_verts is not None else f_verts,
                f_joints.view(f_bs, f_fr, -1, 3),
            )

        if m_bs > 0 and f_bs > 0:
            # join the f_verts and m_verts according to m_idx and f_idx
            if m_verts is not None and f_verts is not None:
                verts = torch.zeros_like(torch.concatenate((m_verts, f_verts), dim=0))
                verts[m_idx] = m_verts
                verts[f_idx] = f_verts
            else:
                verts = None

            joints = torch.zeros_like(torch.concatenate((m_joints, f_joints), dim=0))
            joints[m_idx] = m_joints
            joints[f_idx] = f_joints
        elif m_bs > 0:
            verts = m_verts
            joints = m_joints
        elif f_bs > 0:
            verts = f_verts
            joints = f_joints
        else:
            print("CHECK WHY ERROR HERE")
            import ipdb

            ipdb.set_trace()

        return verts, joints

    def construct_input(self, data):
        """
        Flatten the input data and create motion_features "x" and identity_features "identity"
        """
        # get unflatten sizes for each features
        self.unflat_feat_sizes = {
            k: v.shape if isinstance(v, torch.Tensor) else -1 for k, v in data.items()
        }
        # flatten each feature
        data = {
            k: (
                v.view(v.shape[0], v.shape[1], -1)
                if isinstance(v, torch.Tensor) and len(v.shape) > 2
                else v
            )
            for k, v in data.items()
        }
        # get sizes of each feature
        self.flat_feat_sizes = {
            k: v.shape[-1] if isinstance(v, torch.Tensor) else -1
            for k, v in data.items()
        }
        features = torch.cat([data[k].float() for k in self.train_feats], dim=-1)
        data["x"] = features

        identity_feats = ["betas", "gender"]
        identity_features = torch.cat([data[k].float() for k in identity_feats], dim=-1)
        data["identity"] = identity_features
        return data

    def deconstruct_input(self, data, identity):
        """
        Get original features from "x" and "identity" and unflatten them
        """
        # split features according to sequentially
        split_indxs = [self.flat_feat_sizes[k] for k in self.train_feats]

        # split the features
        data = torch.split(data, split_indxs, dim=-1)
        # make dict again
        data = {k: v for k, v in zip(self.train_feats, data)}

        data["betas"] = identity[:, :, :-1]
        data["gender"] = identity[:, :, -1]

        # unflatten each feature
        data = {
            k: v.view(*self.unflat_feat_sizes[k]) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        return data
