import copy
import os.path as osp

import numpy as np
import smplx
import torch
import torch.nn as nn
import trimesh
import json

from bosRegressor.utils.constants import ESSENTIALS_DIR
from bosRegressor.utils.geometry import matrix_to_axis_angle, ea2rm, axis_angle_to_matrix

def sparse_batch_mm(m1, m2):
    """
    https://github.com/pytorch/pytorch/issues/14489

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """

    batch_size = m2.shape[0]
    # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
               .transpose(1, 0)
    return result

class HDfier(nn.Module):
    def __init__(self, model_type='smplx'):
        super().__init__()

        hd_operator_path = osp.join(ESSENTIALS_DIR, 'hd_model', model_type,
                                    f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
        hd_operator = np.load(hd_operator_path)
        self.hd_operator = torch.sparse.FloatTensor(
            torch.tensor(hd_operator['index_row_col']),
            torch.tensor(hd_operator['values']),
            torch.Size(hd_operator['size']))

    def hdfy_mesh(self, vertices, model_type='smplx'):
        """
        Applies a regressor that maps SMPL vertices to uniformly distributed vertices
        """
        # device = body.vertices.device
        # check if vertices ndim are 3, if not , add a new axis
        if vertices.ndim != 3:
            # batchify the vertices
            vertices = vertices[None, :, :]

        # check if vertices are an ndarry, if yes, make pytorch tensor
        if isinstance(vertices, np.ndarray):
            vertices = torch.from_numpy(vertices).to(self.device)

        vertices = vertices.to(torch.double)
        if self.hd_operator.device != vertices.device:
            self.hd_operator = self.hd_operator.to(vertices.device)
        hd_verts = sparse_batch_mm(self.hd_operator, vertices).to(torch.float)
        return hd_verts


def get_trans_offset(pelvis, smplx_params, trans, body_model):
    '''
    SOMA fits are in AMASS format (xy plane is ground) but SMPLify-XMC uses Pyrender format (xz plane is ground). global_orient is corrected by 270 rotation along x-axis to match the two formats.
    While the ground plane is z=0 in soma, since global_orient is wrt pelvis, this would mean the ground plane is not at y=0 after rotation.
    Fix: distance of pelvis to ground in preserved. Use it to push the mesh up after rotation such that the ground plane is at y=0.
    Args:
        pelvis: pelvis joint before rotation
    '''
    bs = trans.shape[0]
    init_root_orient = smplx_params['global_orient']
    pelvis_height = pelvis[:, 2]  # this is always conserved

    new_smplx_params = copy.deepcopy(smplx_params)
    # Rotate AMASS root orient to smplify-xmc format
    R_init = axis_angle_to_matrix(init_root_orient)
    R1 = ea2rm(torch.tensor([[np.radians(270)]]), torch.tensor([[np.radians(0)]]),
               torch.tensor([[np.radians(0)]])).float().to(R_init.device)
    R = torch.bmm(R1.expand(bs, -1, -1), R_init)
    new_smplx_params['global_orient'] = matrix_to_axis_angle(R)

    # posed body with hand, with global orient
    body_model_output = body_model(
        global_orient=new_smplx_params['global_orient'],
        body_pose=new_smplx_params['body_pose'])

    new_pelvis = body_model_output.joints[:, 0]
    new_ground_plane_height = new_pelvis[:, 1] - pelvis_height
    trans_offset = -new_ground_plane_height
    return trans_offset


def smplx_breakdown(bdata, device):
    if 'poses' not in bdata.keys():
        bdata['poses'] = bdata['fullpose']

    global_orient = torch.from_numpy(bdata['poses'][:, :3]).float().to(device)
    body_pose = torch.from_numpy(bdata['poses'][:, 3:66]).float().to(device)
    jaw_pose = torch.from_numpy(bdata['poses'][:, 66:69]).float().to(device)
    leye_pose = torch.from_numpy(bdata['poses'][:, 69:72]).float().to(device)
    reye_pose = torch.from_numpy(bdata['poses'][:, 72:75]).float().to(device)
    left_hand_pose = torch.from_numpy(bdata['poses'][:, 75:120]).float().to(device)
    right_hand_pose = torch.from_numpy(bdata['poses'][:, 120:]).float().to(device)

    trans = torch.from_numpy(bdata['trans']).float().to(device)
    betas = torch.from_numpy(bdata['betas']).float().to(device).unsqueeze(0)
    # check if attr exists
    if 'num_betas' in bdata.keys():
        num_betas = bdata['num_betas']
    else:
        num_betas = 10
    betas = betas[:, :num_betas]

    body_params = {'betas': betas, 'transl': trans, 'global_orient': global_orient, 'body_pose': body_pose,
                   'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                   'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                   }
    return body_params


def smplx_to_mesh(body_params, model_folder, model_type, gender='neutral', device='cuda'):
    # TODO: here there was torch.no_grad() earlier, but I removed it. Check if it is needed
    # param_type: either pkl from mosh or AMASS style npz

    smplx_params = smplx_breakdown(body_params, device)

    # # get overweight caesar betas
    # caesar_json = '/efs/shashank.tripathi/vsc-remote/tipman/tipman/data/caesar/smplx_fits/caesar_male_betas.json'
    # with open(caesar_json, 'r') as f:
    #     caesar_betas = json.load(f)
    #     overweight_betas = caesar_betas['us_1259']['betas_male']
    #     overweight_betas = torch.tensor(overweight_betas).float().to(device).unsqueeze(0)
    #
    # smplx_params['betas'] = overweight_betas
    # gender = 'male'

    body_model_params = dict(model_path=model_folder,
                             model_type=model_type,
                             gender=gender,
                             # joint_mapper=joint_mapper,
                             batch_size=smplx_params['transl'].shape[0],
                             create_global_orient=True,
                             create_body_pose=True,
                             create_betas=True,
                             num_betas=smplx_params['betas'].shape[-1],
                             create_left_hand_pose=True,
                             create_right_hand_pose=True,
                             create_expression=True,
                             create_jaw_pose=True,
                             create_leye_pose=True,
                             create_reye_pose=True,
                             create_transl=True,
                             use_pca=False,
                             flat_hand_mean=True,
                             dtype=torch.float32)

    body_model = smplx.create(**body_model_params).to(device)
    body_model_output = body_model(betas = smplx_params['betas'],
                                   transl=smplx_params['transl'],
                                   global_orient=smplx_params['global_orient'],
                                   body_pose=smplx_params['body_pose'],
                                   left_hand_pose=smplx_params['left_hand_pose'],
                                   right_hand_pose=smplx_params['right_hand_pose'])

    pelvis = body_model_output.joints[:, 0]
    trans_offset = get_trans_offset(pelvis, smplx_params, smplx_params['transl'], body_model)
    zero_vec = torch.zeros_like(trans_offset)
    transl = torch.stack([zero_vec, trans_offset, zero_vec], dim=-1)

    return body_model, body_model_output, smplx_params, body_model.faces
