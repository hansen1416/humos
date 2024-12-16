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

from bosRegressor.core.biomechanics import BiomechanicalEvaluator, setup_biomechanical_evaluator
from humos.utils import constants
from humos.utils.fk import ForwardKinematicsLayer
from humos.utils.mesh_utils import smplh_breakdown
from humos.utils.misc_utils import get_rgba_colors, update_best_metrics
from .losses import InfoNCE_with_filtering, MotionPriorLoss, DynStabilityLoss
from .metrics import all_physics_metrics, calculate_recons_metrics, calculate_dyn_stability_metric, MotionPriorMetric
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
            lmd: Dict = {"recons": 1.0, "joint_recons": 1.0, "latent": 1.0e-5, "cycle_latent": 1.0e-5, "kl": 1.0e-5,
                         "contrastive": 0.1, "motion_prior": 1.0, "physics_float": 1.0, "physics_penetrate": 1.0,
                         "physics_skate": 1.0, "dyn_stability": 1.0},
            compute_metrics: Dict = {"dyn_stability": False, "recons": True, "physics": True, "motion_prior": True},
            bs: int = 14,
            lr: float = 1e-4,
            fps: float = 20.,
            temperature: float = 0.7,
            threshold_selfsim: float = 0.80,
            threshold_selfsim_metrics: float = 0.95,
            cop_w=30.0,
            cop_k=100.0,
            num_val_videos: int = 3,
            max_vid_rows: int = 3,
            run_cycle: bool = True,
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

        self.num_val_videos = num_val_videos
        self.max_vid_rows = max_vid_rows

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

        # Set AITviewer Renderer
        self.renderer = renderer

        # Set a custom camera saved from the viewer
        self.camera = ViewerCamera()
        # cam_dir = C.export_dir + "/camera_params/"
        # cam_dict = joblib.load(cam_dir + "cam_params.pkl")
        self.camera.load_cam()
        self.renderer.scene.camera = self.camera

        # # set custom lights
        # self.light = Light.facing_origin(
        #         light_color=(1.0, 1.0, 1.0),
        #         name="Extra Light",
        #         position=(0.0, 5.0, 10.0) if C.z_up else (0.0, 5.0, 15.0),
        #     )
        # self.renderer.scene.add_light(self.light)

        self.run_cycle = run_cycle

        # Get smpl body models
        self.bm_male = SMPLLayer(model_type="smplh", gender="male", device=device)
        self.bm_female = SMPLLayer(model_type="smplh", gender="female", device=device)

        # Get fk models
        self.fk_male = ForwardKinematicsLayer(constants.SMPL_PATH, gender="male",
                                              num_joints=constants.SMPLH_BODY_JOINTS, device=device)
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
        self.biomechanical_evaluator = BiomechanicalEvaluator(faces=faces, fps=self.fps,
                                                              cop_w=cop_w, cop_k=cop_k,
                                                              stencil_size=3,
                                                              device=device)


        identity_pkl = "./datasets/splits/identity_dict_test_split_smpl.pkl"
        with open(identity_pkl, "rb") as f:
            self.identity_dict_smpl = pkl.load(f)

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
        latent_vectors_A, distributions_A = self.encode(
            inputs_A, identity_A, sample_mean=sample_mean, fact=fact, return_distribution=True
        )
        # Decoding the latent vector: generating motions
        motions_A_giv_A = self.decode(latent_vectors_A, identity_A, lengths_A, mask_A)

        if self.run_cycle:
            # Decoding the latent vector: generating motions
            # Note beta_B includes gender
            motions_B_giv_A = self.decode(latent_vectors_A, identity_B, lengths_A, mask_A)

            # Do not output the betas and gender, use GT
            inputs_B_giv_A = inputs_A.copy()
            inputs_B_giv_A["x"] = motions_B_giv_A[:, :, :-11]
            inputs_B_giv_A["identity"] = identity_B

            # Encoding in the backward cycle
            latent_vectors_B, distributions_B = self.encode(
                inputs_B_giv_A, identity_B, sample_mean=sample_mean, fact=fact, return_distribution=True
            )

            # Decoding in the backward cycle
            motions_B_giv_B = self.decode(latent_vectors_B, identity_B, lengths_A, mask_A)

            # Decoding in the backward cycle
            motions_A_giv_B = self.decode(latent_vectors_B, identity_A, lengths_A, mask_A)
        else:
            motions_B_giv_A, motions_A_giv_B, motions_B_giv_B, latent_vectors_B, distributions_B = (
                None, None, None, None, None)

        if return_all:
            return motions_A_giv_A, motions_B_giv_A, latent_vectors_A, distributions_A, motions_B_giv_B, motions_A_giv_B, latent_vectors_B, distributions_B

        return motions_A_giv_A, motions_B_giv_A, motions_B_giv_B, motions_A_giv_B

    def run_smpl_fk(self, data, skinning=True):
        gender = data["identity"][:, 0, -1]
        smpl_params = smplh_breakdown(data, fk=self.fk_male)

        m_idx = torch.nonzero(gender == 1).squeeze(-1)
        f_idx = torch.nonzero(gender == -1).squeeze(-1)

        m_bs, f_bs = len(m_idx) if m_idx.dim() > 0 else 0, len(f_idx) if f_idx.dim() > 0 else 0
        m_fr, f_fr = data["betas"].shape[1], data["betas"].shape[1]

        if m_bs > 0:
            # split male
            male_params = {k: v[m_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            male_params = {k: v.view(m_bs * m_fr, -1) for k, v in male_params.items() if k != 'gender'}
            # run smpl fk
            m_verts, m_joints = self.bm_male(poses_body=male_params['pose_body'],
                                             betas=male_params['betas'],
                                             poses_root=male_params['root_orient'],
                                             trans=male_params['trans'],
                                             skinning=skinning)
            # exclude hand joints
            m_joints = m_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            m_verts, m_joints = (m_verts.view(m_bs, m_fr, -1, 3) if m_verts is not None else m_verts,
                                 m_joints.view(m_bs, m_fr, -1, 3))

        if f_bs > 0:
            # split female
            female_params = {k: v[f_idx] for k, v in smpl_params.items()}
            # squeeze parameter values in batch dimension
            female_params = {k: v.view(f_bs * f_fr, -1) for k, v in female_params.items() if k != 'gender'}
            # run smpl fk
            f_verts, f_joints = self.bm_female(poses_body=female_params['pose_body'],
                                               betas=female_params['betas'],
                                               poses_root=female_params['root_orient'],
                                               trans=female_params['trans'],
                                               skinning=skinning)
            # exclude hand joints
            f_joints = f_joints[:, :constants.SMPLH_BODY_JOINTS, :]
            # unsqueeze back the batch dimension
            f_verts, f_joints = (f_verts.view(f_bs, f_fr, -1, 3) if f_verts is not None else f_verts,
                                 f_joints.view(f_bs, f_fr, -1, 3))

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
            import ipdb; ipdb.set_trace()

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

    def construct_input(self, data):
        """
        Flatten the input data and create motion_features "x" and identity_features "identity"
        """
        # get unflatten sizes for each features
        self.unflat_feat_sizes = {k: v.shape if isinstance(v, torch.Tensor) else -1 for k, v in data.items()}
        # flatten each feature
        data = {k: v.view(v.shape[0], v.shape[1], -1) if isinstance(v, torch.Tensor) and len(v.shape) > 2 else v for
                k, v
                in data.items()}
        # get sizes of each feature
        self.flat_feat_sizes = {k: v.shape[-1] if isinstance(v, torch.Tensor) else -1 for k, v in data.items()}
        features = torch.cat([data[k].float() for k in self.train_feats], dim=-1)
        data['x'] = features

        identity_feats = ["betas", "gender"]
        identity_features = torch.cat([data[k].float() for k in identity_feats], dim=-1)
        data['identity'] = identity_features
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
        data = {k: v.view(*self.unflat_feat_sizes[k]) if isinstance(v, torch.Tensor) else v for k, v in
                data.items()}
        return data

    def compute_loss(self, batch: Dict, shuffle_idx=None, skinning=True, return_all=False, visualize=False) -> Dict:
        keyids_A = batch["keyid"]
        motion_x_dict_A = batch["motion_x_dict"]
        motion_x_dict_A = self.construct_input(motion_x_dict_A)

        # shuffle the A batch to get the B batch
        if shuffle_idx is None:
            shuffle_idx = torch.randperm(len(motion_x_dict_A["x"]))
        motion_x_dict_B = {k: v[shuffle_idx] if isinstance(v, torch.Tensor) else [v[i] for i in shuffle_idx] for k, v in
                           motion_x_dict_A.items()}

        mask_A = motion_x_dict_A["mask"]
        ref_motions_A = motion_x_dict_A["x"]
        identity_A = motion_x_dict_A["identity"]  # these include betas + gender

        # mask_B = motion_x_dict_B["mask"]
        # ref_motions_B = motion_x_dict_B["x"]
        if return_all:
            # Get target shape
            identity_B = torch.zeros_like(identity_A)
            for i, keyid_A in enumerate(keyids_A):
                identity = self.identity_dict_smpl[keyid_A]
                identity_B[i] = torch.FloatTensor(identity["identity_B_norm"]).to(self.device)
        else:
            identity_B = motion_x_dict_B["identity"]  # these include betas + gender

        # # sentence embeddings
        # sent_emb_A = batch["sent_emb"]
        # sent_emb_B = batch["sent_emb"][shuffle_idx]

        # motion -> motion (input to cycle is A)
        (m_motions_A_giv_A, m_motions_B_giv_A, m_latents_A, m_dists_A,
         m_motions_B_giv_B, m_motions_A_giv_B, m_latents_B, m_dists_B) = self.forward_cycle(
            motion_x_dict_A,
            identity_A,
            identity_B,
            mask_A=mask_A,
            return_all=True
        )

        # unnormalize the motion features
        ref_motions_un_A = self.normalizer.inverse(self.deconstruct_input(ref_motions_A, identity_A))
        ref_motions_un_A = self.construct_input(ref_motions_un_A)
        m_motions_un_A_giv_A = self.normalizer.inverse(
            self.deconstruct_input(m_motions_A_giv_A[:, :, :-11], identity_A))
        m_motions_un_A_giv_A = self.construct_input(m_motions_un_A_giv_A)
        if self.run_cycle:
            m_motions_un_B_giv_A = self.normalizer.inverse(
                self.deconstruct_input(m_motions_B_giv_A[:, :, :-11], identity_B))
            m_motions_un_B_giv_A = self.construct_input(m_motions_un_B_giv_A)
            m_motions_un_B_giv_B = self.normalizer.inverse(
                self.deconstruct_input(m_motions_B_giv_B[:, :, :-11], identity_B))
            m_motions_un_B_giv_B = self.construct_input(m_motions_un_B_giv_B)
            m_motions_un_A_giv_B = self.normalizer.inverse(
                self.deconstruct_input(m_motions_A_giv_B[:, :, :-11], identity_A))
            m_motions_un_A_giv_B = self.construct_input(m_motions_un_A_giv_B)

        # Store all losses and metrics
        losses = {}
        compute_physics_losses = self.lmd["physics_penetrate"] > 0 or self.lmd["physics_float"] > 0 or self.lmd[
            "physics_skate"] > 0 or return_all
        compute_dyn_stability_loss = self.lmd["dyn_stability"] > 0


        m_verts_A_giv_A, m_verts_B_giv_A, m_verts_B_giv_B, m_verts_A_giv_B = (None, None, None, None)
        m_joints_A_giv_A, m_joints_B_giv_A, m_joints_B_giv_B, m_joints_A_giv_B = (None, None, None, None)

        # compute the motion prior loss
        if self.run_cycle:
            inputs_B_giv_A = motion_x_dict_A.copy()
            inputs_B_giv_A["x"] = m_motions_B_giv_A[:, :, :-11]
            inputs_B_giv_A["identity"] = identity_B
            losses["motion_prior"] = (
                self.motion_prior_loss_fn(motion_x_dict_A, inputs_B_giv_A)
            )

        # For physics-losses
        if compute_physics_losses:
            if self.run_cycle:
                m_verts_B_giv_A, m_joints_B_giv_A = self.run_smpl_fk(m_motions_un_B_giv_A,
                                                                     skinning=True)  # skinning required for physics losses
                losses["physics_penetrate"], losses["physics_float"], losses["physics_skate"], skate_std, skate_sum, skate_perc = self.physics_loss_fn(
                    m_verts_B_giv_A,
                    m_joints_B_giv_A,
                    device=self.device)
            else:
                m_verts_A_giv_A, m_joints_A_giv_A = self.run_smpl_fk(m_motions_un_A_giv_A,
                                                                     skinning=True)  # skinning required for physics losses
                losses["physics_penetrate"], losses["physics_float"], losses["physics_skate"], skate_std, skate_sum, skate_perc  = self.physics_loss_fn(
                    m_verts_A_giv_A,
                    m_joints_A_giv_A,
                    device=self.device)

        # For dynamic stability loss
        if compute_dyn_stability_loss:
            if self.run_cycle:
                if m_verts_B_giv_A is None:
                    m_verts_B_giv_A, m_joints_B_giv_A = self.run_smpl_fk(m_motions_un_B_giv_A, skinning=True)
                # compute biomechanics losses
                setup_biomechanical_evaluator(self.biomechanical_evaluator,
                                                joints=m_joints_B_giv_A,
                                                verts=m_verts_B_giv_A)

            else:
                if m_verts_A_giv_A is None:
                    m_verts_A_giv_A, m_joints_A_giv_A = self.run_smpl_fk(m_motions_un_A_giv_A, skinning=True)
                    setup_biomechanical_evaluator(self.biomechanical_evaluator,
                                                  joints=m_joints_A_giv_A,
                                                  verts=m_verts_A_giv_A)
            cops = self.biomechanical_evaluator.cops
            # take the center frames
            center_frame = self.biomechanical_evaluator.center_frame
            cops = cops[:, center_frame:-center_frame, :, :]
            zmps = self.biomechanical_evaluator.zmps
            # check if the z coordinate is all zero for both cops and zmps
            losses["dyn_stability"] = self.dyn_stability_loss_fn(cops, zmps)


        # Reconstructions losses
        # fmt: off

        recons_type = self.lmd["recons_type"]

        # Calculate loss on rotations and transl
        losses["recons"] = (
            + self.reconstruction_loss_fn(
                self.deconstruct_input(m_motions_A_giv_A[:, :, :-11], m_motions_A_giv_A[:, :, -11:]),
                self.deconstruct_input(ref_motions_A, identity_A))  # motion -> motion
        )
        if self.run_cycle:
            losses["recons"] += (
                    # + self.reconstruction_loss_fn(
                    #     self.deconstruct_input(m_motions_B_giv_A[:, :, :-11], m_motions_B_giv_A[:, :, -11:]),
                    #     self.deconstruct_input(m_motions_B_giv_B[:, :, :-11], m_motions_B_giv_B[:, :, -11:]))
                    + self.reconstruction_loss_fn(
                self.deconstruct_input(m_motions_A_giv_B[:, :, :-11], m_motions_A_giv_B[:, :, -11:]),
                self.deconstruct_input(ref_motions_A, identity_A))  # motion -> motion
            )

        if recons_type == "joints":
            # Run SMPLH FK to get the joints
            ref_verts_A, ref_joints_A = self.run_smpl_fk(ref_motions_un_A, skinning)
            if m_joints_A_giv_A is None:
                m_verts_A_giv_A, m_joints_A_giv_A = self.run_smpl_fk(m_motions_un_A_giv_A, skinning)
            if self.run_cycle:
                if m_joints_A_giv_B is None:
                # if m_joints_B_giv_A is None or m_joints_B_giv_B is None or m_joints_A_giv_B is None:
                    m_verts_A_giv_B, m_joints_A_giv_B = self.run_smpl_fk(m_motions_un_A_giv_B, skinning)
                    # m_verts_B_giv_A, m_joints_B_giv_A = self.run_smpl_fk(m_motions_un_B_giv_A, skinning)
                    # m_verts_B_giv_B, m_joints_B_giv_B = self.run_smpl_fk(m_motions_un_B_giv_B, skinning)

            losses["joint_recons"] = (
                + self.joint_reconstruction_loss_fn(m_joints_A_giv_A,
                                                    ref_joints_A)  # motion -> motion
            )
            if self.run_cycle:
                losses["joint_recons"] += (
                        # + self.joint_reconstruction_loss_fn(m_joints_B_giv_A,
                        #                                     m_joints_B_giv_B)  # motion -> motion
                        + self.joint_reconstruction_loss_fn(m_joints_A_giv_B,
                                                            ref_joints_A)  # motion -> motion
                )

        # VAE losses
        if self.vae:
            # Create a centred normal distribution to compare with
            # logvar = 0 -> std = 1
            # Get reference for A distributions
            ref_mus_A = torch.zeros_like(m_dists_A[0])
            ref_logvar_A = torch.zeros_like(m_dists_A[1])
            ref_dists_A = (ref_mus_A, ref_logvar_A)

            # Get reference for B distributions
            if self.run_cycle:
                ref_mus_B = torch.zeros_like(m_dists_B[0])
                ref_logvar_B = torch.zeros_like(m_dists_B[1])
                ref_dists_B = (ref_mus_B, ref_logvar_B)

            losses["kl"] = (
                + self.kl_loss_fn(m_dists_A, ref_dists_A)  # motion for A
            )
            if self.run_cycle:
                losses["kl"] += (
                    + self.kl_loss_fn(m_dists_B, ref_dists_B)  # motion for B
                )

        # Latent manifold loss in cycle between latent_A and latent_B
        if self.run_cycle:
            losses["cycle_latent"] = (
                self.latent_loss_fn(m_latents_A, m_latents_B)
            )

        # Latent manifold loss
        losses["latent"] = torch.zeros_like(losses["recons"])

        # TMR: adding the contrastive loss
        losses["contrastive"] = torch.zeros_like(losses["recons"])

        # Used for the validation step
        if return_all:
            # Metrics
            metrics = {}
            if self.compute_metrics["recons"]:
                if self.run_cycle:
                    recons_metrics = calculate_recons_metrics(fk=self.fk_male,
                                                              pred_data=m_motions_un_A_giv_B,
                                                              gt_data=ref_motions_un_A)
                else:
                    recons_metrics = calculate_recons_metrics(fk=self.fk_male,
                                                              pred_data=m_motions_un_A_giv_A,
                                                              gt_data=ref_motions_un_A)
                metrics.update(recons_metrics)

            if self.compute_metrics["dyn_stability"]:
                if self.run_cycle:
                    if m_verts_B_giv_A is None:
                        m_verts_B_giv_A, m_joints_B_giv_A = self.run_smpl_fk(m_motions_un_B_giv_A, skinning=True)
                    # compute biomechanics losses
                    setup_biomechanical_evaluator(self.biomechanical_evaluator,
                                                  joints=m_joints_B_giv_A,
                                                  verts=m_verts_B_giv_A)

                else:
                    if m_verts_A_giv_A is None:
                        m_verts_A_giv_A, m_joints_A_giv_A = self.run_smpl_fk(m_motions_un_A_giv_A, skinning=True)
                        setup_biomechanical_evaluator(self.biomechanical_evaluator,
                                                      joints=m_joints_A_giv_A,
                                                      verts=m_verts_A_giv_A)
                perc_dyn_stable, mean_min_positive_distance, min_pos_distance, min_signed_distance = calculate_dyn_stability_metric(
                    self.biomechanical_evaluator,
                    joints=m_joints_B_giv_A if self.run_cycle else m_joints_A_giv_A,
                    verts=m_verts_B_giv_A if self.run_cycle else m_verts_A_giv_A)

                cops = self.biomechanical_evaluator.cops
                coms = self.biomechanical_evaluator.coms
                zmps = self.biomechanical_evaluator.zmps
                zmps = torch.cat([zmps[:, 0:1], zmps, zmps[:, -1:]], dim=1)

                hull_verts = self.biomechanical_evaluator.hull_verts
                bs, nf, _, _ = coms.shape
                hull_verts = [hull_verts[i:i + nf] for i in range(0, len(hull_verts), nf)]


                # # check if dyn_stability has nans
                # import ipdb;
                # ipdb.set_trace()
                # out_dir = f'./debug_dyn_stability/dyn'
                # os.makedirs(out_dir, exist_ok=True)
                #
                # # plot pred meshes
                # for i in range(cops.shape[0]):
                #     cop = cops[i]
                #     com = coms[i]
                #     zmp = zmps[i]
                #     min_pos = min_pos_distance[i]
                #     min_sign = min_signed_distance[i]
                #     hull_vert = hull_verts[i]
                #
                #     # repeat the last frame in zmps
                #     debug_dict = {}
                #     debug_dict['vertices'] = m_verts_B_giv_A[i].detach().cpu().numpy()
                #     debug_dict['faces'] = self.bm_male.faces.detach().cpu().numpy()
                #     debug_dict['cops'] = cop.detach().cpu().numpy()
                #     debug_dict['zmps_raw'] = zmp.detach().cpu().numpy()
                #     debug_dict['coms'] = com.detach().cpu().numpy()
                #     debug_dict[
                #         'mass_per_vert_init'] = self.biomechanical_evaluator.init_mass_per_vert[0].detach().cpu().numpy()
                #     debug_dict[
                #         'per_part_multiplier'] = self.biomechanical_evaluator.per_part_multiplier[0].detach().cpu().numpy()
                #     debug_dict['min_pos'] = min_pos.detach().cpu().numpy()
                #     debug_dict['min_sign'] = min_sign.detach().cpu().numpy()
                #     debug_dict['hull_verts'] = [item.detach().cpu().numpy() if item is not None else item for item in hull_vert]
                #     torch.save(debug_dict, os.path.join(out_dir, f'{i:04d}.pt'))
                #     print(f"Saved metadata to {os.path.join(out_dir, f'{i:04d}.npz')}")

                metrics["perc_dyn_stable"] = perc_dyn_stable.detach().cpu().numpy()
                metrics["mean_dyn_positive_distance"] = mean_min_positive_distance.detach().cpu().numpy()

            if self.compute_metrics["physics"]:
                if compute_physics_losses:
                    metrics["physics_penetrate"] = losses["physics_penetrate"].detach().cpu().numpy()
                    metrics["physics_float"] = losses["physics_float"].detach().cpu().numpy()
                    metrics["physics_skate"] = losses["physics_skate"].detach().cpu().numpy() * self.fps # convert from m/frame to m / sec
                    metrics["physics_skate_perc"] = skate_perc.detach().cpu().numpy()
                    metrics["physics_skate_sum"] = skate_sum.detach().cpu().numpy() * self.fps # convert from m/frame to m / sec
                    metrics["physics_skate_std"] = skate_std.detach().cpu().numpy() * self.fps # convert from m/frame to m / sec

            if self.compute_metrics["motion_prior"]:
                if self.run_cycle:
                    inputs_B_giv_A = motion_x_dict_A.copy()
                    inputs_B_giv_A["x"] = m_motions_B_giv_A[:, :, :-11]
                    inputs_B_giv_A["identity"] = identity_B
                    pred_embedding = self.motion_prior_metrics.encode(inputs_B_giv_A, sample_mean=None, fact=None)
                    gt_embedding = self.motion_prior_metrics.encode(motion_x_dict_A, sample_mean=None, fact=None)
                    self.validation_step_pred_embeddings.append(pred_embedding)
                    self.validation_step_gt_embeddings.append(gt_embedding)

            # Generate all visualizations
            if visualize:
                all_ref_frames_A = self.render_frames(ref_motions_un_A, color='green', tag="ref")
                if self.run_cycle:
                    all_pred_frames_B_giv_A = self.render_frames(m_motions_un_B_giv_A, color='red',
                                                                 tag="pred_B_giv_A")
                    all_pred_frames_A_giv_B = self.render_frames(m_motions_un_A_giv_B, color='purple',
                                                                 tag="pred_A_giv_B")
                    all_pred_frames_A_giv_A = None
                else:
                    all_pred_frames_A_giv_A = self.render_frames(m_motions_un_A_giv_A, color='blue',
                                                                 tag="pred_A_giv_A")
                    all_pred_frames_B_giv_A = None
                    all_pred_frames_A_giv_B = None

            else:
                all_ref_frames_A = None
                all_pred_frames_A_giv_A = None
                all_pred_frames_B_giv_A = None
                all_pred_frames_A_giv_B = None

            all_vis = [all_ref_frames_A, all_pred_frames_A_giv_A, all_pred_frames_B_giv_A, all_pred_frames_A_giv_B]


        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Weight all loss values with config weight
        losses = {x: self.lmd[x] * val if x in self.lmd else val for x, val in losses.items()}

        if torch.isnan(losses["loss"]).any():
            print("************ NAN LOSS ************")
            # print which loss was nan
            for k, v in losses.items():
                if torch.isnan(v).any():
                    print(f"{k} contains nan values")

        # # check if dyn_stability has nans
        # for k, v in losses.items():
        #     if torch.isnan(v).any():
        #         print('NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN! NAN!')
        #         import ipdb;
        #         ipdb.set_trace()
        #         out_dir = f'./debug_dyn_stability/pred'
        #         os.makedirs(out_dir, exist_ok=True)
        #         coms = self.biomechanical_evaluator.coms
        #
        #         # plot pred meshes
        #         for i in range(cops.shape[0]):
        #             com = coms[i]
        #             cop = cops[i]
        #             zmp = zmps[i]
        #             # repeat the last frame in zmps
        #             zmp = torch.cat([zmp[0].unsqueeze(0), zmp, zmp[-1].unsqueeze(0)], dim=0)
        #             cop = torch.cat([cop[0].unsqueeze(0), cop, cop[-1].unsqueeze(0)], dim=0)
        #             debug_dict = {}
        #             debug_dict['vertices'] = m_verts_B_giv_A[i].detach().cpu().numpy()
        #             debug_dict['faces'] = self.bm_male.faces.detach().cpu().numpy()
        #             debug_dict['cops'] = cop.detach().cpu().numpy()
        #             debug_dict['zmps_raw'] = zmp.detach().cpu().numpy()
        #             debug_dict['coms'] = com.detach().cpu().numpy()
        #             debug_dict['mass_per_vert_init'] = self.biomechanical_evaluator.init_mass_per_vert[
        #                 0].detach().cpu().numpy()
        #             debug_dict['per_part_multiplier'] = self.biomechanical_evaluator.per_part_multiplier[
        #                 0].detach().cpu().numpy()
        #             torch.save(debug_dict, os.path.join(out_dir, f'{i:04d}.pt'))
        #             print(f"Saved metadata to {os.path.join(out_dir, f'{i:04d}.npz')}")
        #
        #         # plot gt meshes
        #         ref_verts_A, ref_joints_A = self.run_smpl_fk(ref_motions_un_A, skinning=True)
        #         setup_biomechanical_evaluator(self.biomechanical_evaluator,
        #                                       joints=ref_joints_A,
        #                                       verts=ref_verts_A)
        #         cops = self.biomechanical_evaluator.cops
        #         # take the center frames
        #         center_frame = self.biomechanical_evaluator.center_frame
        #         cops = cops[:, center_frame:-center_frame, :, :]
        #         zmps = self.biomechanical_evaluator.zmps
        #
        #         # plot pred meshes
        #         out_dir = f'./debug_dyn_stability/gt'
        #         os.makedirs(out_dir, exist_ok=True)
        #         for i in range(cops.shape[0]):
        #             cop = cops[i]
        #             zmp = zmps[i]
        #             # repeat the last frame in zmps
        #             zmp = torch.cat([zmp[0].unsqueeze(0), zmp, zmp[-1].unsqueeze(0)], dim=0)
        #             cop = torch.cat([cop[0].unsqueeze(0), cop, cop[-1].unsqueeze(0)], dim=0)
        #             debug_dict = {}
        #             debug_dict['vertices'] = ref_verts_A[i].detach().cpu().numpy()
        #             debug_dict['faces'] = self.bm_male.faces.detach().cpu().numpy()
        #             debug_dict['cops'] = cop.detach().cpu().numpy()
        #             debug_dict['zmps_raw'] = zmp.detach().cpu().numpy()
        #             debug_dict['zmps_smoothed'] = zmp.detach().cpu().numpy()
        #             debug_dict[
        #                 'mass_per_vert_init'] = self.biomechanical_evaluator.init_mass_per_vert.detach().cpu().numpy()
        #             debug_dict[
        #                 'mass_per_vert_optim'] = self.biomechanical_evaluator.mass_per_vert.detach().cpu().numpy()
        #             debug_dict[
        #                 'per_part_multiplier'] = self.biomechanical_evaluator.per_part_multiplier.detach().cpu().numpy()
        #             torch.save(debug_dict, os.path.join(out_dir, f'{i:04d}.pt'))
        #             print(f"Saved metadata to {os.path.join(out_dir, f'{i:04d}.npz')}")

        if return_all:
            return losses, None, None, m_latents_A, m_latents_B, metrics, all_vis
        else:
            return losses

    def render_frames(self, data, color='grey', tag="pred"):
        color = get_rgba_colors(color)

        pred_params = smplh_breakdown(data, fk=self.fk_male)

        # get num_videos equally spaced frames
        pred_params["gender"] = pred_params["gender"][:, 0, -1]
        select_indices = torch.linspace(0, pred_params["gender"].shape[0] - 1, self.num_val_videos).long()
        # get the corresponding SMPL parameters
        pred_params = {k: v[select_indices] for k, v in pred_params.items()}

        # loop over filtered frames
        all_frames = []
        for i in range(self.num_val_videos):
            gender = int(pred_params['gender'][i])
            if gender == 1:
                smpl_layer = self.bm_male
            elif gender == -1:
                smpl_layer = self.bm_female
            else:
                assert gender == 1 or gender == -1
            smpl_seq = SMPLSequence(
                poses_body=pred_params["pose_body"][i, :, :],
                poses_root=pred_params["root_orient"][i, :, :],
                betas=pred_params["betas"][i, :, :],
                trans=pred_params["trans"][i, :, :],
                smpl_layer=smpl_layer,
                z_up=True,
                color=color,
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

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["trans"])
        shuffle_idx = self.sampled_idx

        visualize = False

        if len(self.validation_step_ref_videos_A) < self.max_vid_rows:
            # skip every other batch
            if batch_idx % 2 == 0:
                visualize = True

        losses, t_latents_A, t_latents_B, m_latents_A, m_latents_B, metrics, all_vis = self.compute_loss(batch,
                                                                                                         shuffle_idx,
                                                                                                         skinning=True,
                                                                                                         return_all=True,
                                                                                                         visualize=visualize)
        if visualize:
            ref_videos_A, val_videos_A_giv_A, val_videos_B_giv_A, val_videos_A_giv_B = all_vis
            self.validation_step_ref_videos_A.append(ref_videos_A)
            if self.run_cycle:
                self.validation_step_val_videos_B_giv_A.append(val_videos_B_giv_A)
                self.validation_step_val_videos_A_giv_B.append(val_videos_A_giv_B)
            else:
                self.validation_step_val_videos_A_giv_A.append(val_videos_A_giv_A)

        # Store the metrics
        self.validation_step_metrics.append(metrics)

        # Store the latent vectors
        self.validation_step_m_latents_A.append(m_latents_A)
        self.validation_step_m_latents_B.append(m_latents_B)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_step/{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )

        return losses["loss"]

    def overlay_videos(self, video_A, video_B):
        # overlay video A on video B
        alpha = 0.5
        combined_video = alpha * video_A + (1 - alpha) * video_B
        return combined_video

    def on_validation_epoch_end(self):
        # join all frames into one video vertically
        ref_videos_A = np.concatenate(self.validation_step_ref_videos_A, axis=1)
        ref_videos_A = np.transpose(ref_videos_A, (0, 3, 1, 2))
        wandb.log({"vis/ref_videos_A": [wandb.Video(ref_videos_A, fps=20, format="mp4")]})
        if self.run_cycle:
            val_videos_B_giv_A = np.concatenate(self.validation_step_val_videos_B_giv_A, axis=1)
            val_videos_B_giv_A = np.transpose(val_videos_B_giv_A, (0, 3, 1, 2))
            olay_videos_B_giv_A = self.overlay_videos(val_videos_B_giv_A, ref_videos_A)
            wandb.log({"vis/val_videos_B_giv_A": [wandb.Video(val_videos_B_giv_A, fps=20, format="mp4")]})
            wandb.log({"vis/olay_videos_B_giv_A": [wandb.Video(olay_videos_B_giv_A, fps=20, format="mp4")]})
            val_videos_A_giv_B = np.concatenate(self.validation_step_val_videos_A_giv_B, axis=1)
            val_videos_A_giv_B = np.transpose(val_videos_A_giv_B, (0, 3, 1, 2))
            olay_videos_A_giv_B = self.overlay_videos(val_videos_A_giv_B, ref_videos_A)
            wandb.log({"vis/val_videos_A_giv_B": [wandb.Video(val_videos_A_giv_B, fps=20, format="mp4")]})
            wandb.log({"vis/olay_videos_A_giv_B": [wandb.Video(olay_videos_A_giv_B, fps=20, format="mp4")]})
        else:
            val_videos_A_giv_A = np.concatenate(self.validation_step_val_videos_A_giv_A, axis=1)
            val_videos_A_giv_A = np.transpose(val_videos_A_giv_A, (0, 3, 1, 2))
            # overlay val_videos_a_giv_a on ref_videos_A
            olay_videos_A_giv_A = self.overlay_videos(val_videos_A_giv_A, ref_videos_A)
            wandb.log({"vis/val_videos_A_giv_A": [wandb.Video(val_videos_A_giv_A, fps=20, format="mp4")]})
            wandb.log({"vis/olay_videos_A_giv_A": [wandb.Video(olay_videos_A_giv_A, fps=20, format="mp4")]})

        # Combine all metrics from the batches
        all_metrics = {}
        for metric_name in self.validation_step_metrics[0].keys():
            if metric_name == "in_hull_label":
                in_hull_label = np.concatenate([x[metric_name] for x in self.validation_step_metrics])
                # in_hull_label contains, 1, 0 and -1. Find the percentage of 1s in the in_hull_label
                all_metrics["in_bos"] = np.mean(in_hull_label == 1)
            # Get the mean metrics in cms
            all_metrics[metric_name] = np.mean(
                [x[metric_name] * 100 for x in self.validation_step_metrics]
            )
            wandb.log({f"metrics/{metric_name}_epoch": all_metrics[metric_name]})

        # Calculation motion prior metrics
        if self.compute_metrics["motion_prior"]:
            if self.run_cycle:
                fid, pred_diversity, gt_diversity = self.motion_prior_metrics.compute_fid_diversity(self.validation_step_pred_embeddings, self.validation_step_gt_embeddings)
                all_metrics['fid'] = np.mean(fid)
                all_metrics['pred_diversity'] = np.mean(pred_diversity)
                all_metrics['gt_diversity'] = np.mean(gt_diversity)
                wandb.log({"metrics/fid_epoch": all_metrics['fid']})
                wandb.log({"metrics/pred_diversity_epoch": all_metrics['pred_diversity']})
                wandb.log({"metrics/gt_diversity_epoch": all_metrics['gt_diversity']})

        # save epoch
        wandb.log({"current_epoch": self.current_epoch})
        update_best_metrics()

        # Record best metrics on wandb
        self.validation_step_m_latents_A.clear()
        self.validation_step_m_latents_B.clear()
        self.validation_step_ref_videos_A.clear()
        self.validation_step_val_videos_A_giv_A.clear()
        self.validation_step_val_videos_B_giv_A.clear()
        self.validation_step_val_videos_A_giv_B.clear()
        self.validation_step_metrics.clear()
        self.validation_step_pred_embeddings.clear()
        self.validation_step_gt_embeddings.clear()

