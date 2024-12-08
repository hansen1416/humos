# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import sys

import torch
import torch.nn as nn
import pickle as pkl
import os.path as osp
import numpy as np
from bosRegressor.utils.mesh_utils import HDfier
from bosRegressor.utils.constants import ESSENTIALS_DIR
from bosRegressor.core.part_volumes import PartVolume

SMPL_PART_BOUNDS = osp.join(ESSENTIALS_DIR, 'yogi_segments/smpl/part_meshes_ply/smpl_segments_bounds.pkl')
FID_TO_PART = osp.join(ESSENTIALS_DIR,'yogi_segments/smpl/part_meshes_ply/fid_to_part.pkl')
PART_VID_FID = osp.join(ESSENTIALS_DIR,'yogi_segments/smpl/part_meshes_ply/smpl_part_vid_fid.pkl')
HD_SMPL_MAP  = osp.join(ESSENTIALS_DIR,'hd_model/smpl/smpl_neutral_hd_sample_from_mesh_out.pkl')

class StabilityLossCoP(nn.Module):
    def __init__(self,
                 faces,
                 cos_w = 10,
                 cos_k = 100,
                 model_type='smpl',
    ):
        super().__init__()
        """
        Loss that ensures that the COM of the SMPL mesh is close to the center of support 
        """
        if model_type == 'smplx':
            self.num_faces = 20908
            self.num_verts = 10475
        if model_type == 'smpl' or model_type == 'smplh':
            self.num_faces = 13776
            self.num_verts = 6890
        self.num_verts_hd = 20000


        assert faces is not None, 'Faces tensor is none'
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

        # self.gmof_rho = gmof_rho
        self.cos_w = cos_w
        self.cos_k = cos_k
        # self.contact_thresh = contact_thresh

        self.hdfy_op = HDfier()

        with open(SMPL_PART_BOUNDS, 'rb') as f:
            d = pkl.load(f)
            self.part_bounds = {k: d[k] for k in sorted(d)}
        self.part_order = sorted(self.part_bounds)

        with open(PART_VID_FID, 'rb') as f:
            self.part_vid_fid = pkl.load(f)

        # mapping between vid and part
        self.vid_in_part = self.vert_id_to_part_mapping()

        # mapping between vid_hd and fid
        with open(HD_SMPL_MAP, 'rb') as f:
            faces_vert_is_sampled_from = pkl.load(f)['faces_vert_is_sampled_from']
        index_row_col = torch.stack(
            [torch.LongTensor(np.arange(0, self.num_verts_hd)), torch.LongTensor(faces_vert_is_sampled_from)], dim=0)
        values = torch.ones(self.num_verts_hd, dtype=torch.float)
        size = torch.Size([self.num_verts_hd, self.num_faces])
        hd_vert_on_fid = torch.sparse.FloatTensor(index_row_col, values, size)

        # mapping between fid and part label
        with open(FID_TO_PART, 'rb') as f:
            fid_to_part_dict = pkl.load(f)
        fid_to_part = torch.zeros([len(fid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in fid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            fid_to_part[fid, part_idx] = 1.

        # mapping between vid_hd and part label
        self.hd_vid_in_part = self.vert_hd_id_to_part_mapping(hd_vert_on_fid, fid_to_part)

    def compute_triangle_area(self, triangles):
        ### Compute the area of each triangle in the mesh
        # Compute the cross product of the two vectors of each triangle
        # Then compute the length of the cross product
        # Finally, divide by 2 to get the area of each triangle

        vectors = torch.diff(triangles, dim=2)
        crosses = torch.cross(vectors[:, :, 0], vectors[:, :, 1])
        area = torch.norm(crosses, dim=2) / 2
        return area

    def compute_per_part_volume(self, vertices):
        """
        Compute the volume of each part in the reposed mesh
        """
        part_volume = []
        for part_name, part_bounds in self.part_bounds.items():
            # get part vid and fid
            part_vid = torch.LongTensor(self.part_vid_fid[part_name]['vert_id']).to(vertices.device)
            part_fid = torch.LongTensor(self.part_vid_fid[part_name]['face_id']).to(vertices.device)
            pv = PartVolume(part_name, vertices, self.faces)
            for bound_name, bound_vids in part_bounds.items():
                bound_vids = torch.LongTensor(bound_vids).to(vertices.device)
                pv.close_mesh(bound_vids)
            # add extra vids and fids to original part ids
            new_vert_ids = torch.LongTensor(pv.new_vert_ids).to(vertices.device)
            new_face_ids = torch.LongTensor(pv.new_face_ids).to(vertices.device)
            part_vid = torch.cat((part_vid, new_vert_ids), dim=0)
            part_fid = torch.cat((part_fid, new_face_ids), dim=0)
            pv.extract_part_triangles(part_vid, part_fid)
            part_volume.append(pv.part_volume())
        return torch.vstack(part_volume).permute(1,0).to(vertices.device)

    def vert_id_to_part_mapping(self):
        vid_to_part_dict = dict({k: [] for k in range(self.num_verts)})
        for part_label, part_vid_fid in self.part_vid_fid.items():
            for vid in part_vid_fid['vert_id']:
                if part_label not in vid_to_part_dict[vid]:
                    vid_to_part_dict[vid].append(part_label)
        # check if no multiple part names per face and weight each vertex volume by the number of vertices in that part
        for vid, part_names in vid_to_part_dict.items():
            if len(part_names) > 2:
                print('Warning: more than two part labels for vertex {}'.format(vid))
                import ipdb; ipdb.set_trace()
            if len(part_names) <= 2:
                vid_to_part_dict[vid] = part_names[0]

        # convert dict to N x 10 matrix
        vid_to_part = torch.zeros([len(vid_to_part_dict.keys()), len(self.part_order)], dtype=torch.float32)
        for fid, partname in vid_to_part_dict.items():
            part_idx = self.part_order.index(partname)
            vid_to_part[fid, part_idx] = 1.
        return vid_to_part

    def vert_id_to_part_volume_mapping(self, per_part_volume, device):
        # Note: here the vertex volumes are weighted by the number of vertices in that part
        # this is unlike the vert_hd_id_to_part_volume_mapping function
        batch_size = per_part_volume.shape[0]
        self.vid_in_part = self.vid_in_part.to(device)

        # count number of vertices in each part
        part_to_num_vert_dict = {}
        for part_label, part_vid_fid in self.part_vid_fid.items():
            part_to_num_vert_dict[part_label] = len(part_vid_fid['vert_id'])
        # convert dict to N x 1 matrix
        part_to_num_vert = torch.zeros([len(part_to_num_vert_dict.keys()), 1], dtype=torch.float32, device=device)
        for part_label, num_vert in part_to_num_vert_dict.items():
            part_idx = self.part_order.index(part_label)
            part_to_num_vert[part_idx, 0] = num_vert

        vid_in_part = self.vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(vid_in_part, per_part_volume[:, :, None])
        vid_to_num_verts = torch.bmm(vid_in_part, part_to_num_vert.repeat(batch_size, 1, 1))
        vid_to_vol = vid_to_vol / vid_to_num_verts
        return vid_to_vol


    def vert_hd_id_to_part_volume_mapping(self, per_part_volume, device):
        batch_size = per_part_volume.shape[0]
        self.hd_vid_in_part = self.hd_vid_in_part.to(device)
        hd_vid_in_part = self.hd_vid_in_part[None, :, :].repeat(batch_size, 1, 1)
        vid_to_vol = torch.bmm(hd_vid_in_part, per_part_volume[:, :, None])
        return vid_to_vol

    def vert_hd_id_to_part_mapping(self, hd_vert_on_fid, fid_to_part):
        vid_to_part = torch.mm(hd_vert_on_fid, fid_to_part)
        return vid_to_part

    def forward(self, vertices):
        # Note: the vertices should be aligned along y-axis and in world coordinates
        batch_size = vertices.shape[0]
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices)
        # sample 20k vertices uniformly on the smpl mesh
        vertices_hd = self.hdfy_op.hdfy_mesh(vertices)
        # get volume per vertex id in the hd mesh
        volume_per_vert_hd = self.vert_hd_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # calculate com using volume weighted mean
        com = torch.sum(vertices_hd * volume_per_vert_hd, dim=1) / torch.sum(volume_per_vert_hd, dim=1)

        # # get COM of the SMPLX mesh
        # triangles = torch.index_select(vertices, 1, self.faces.view(-1)).reshape(batch_size, -1, 3, 3)
        # triangle_centroids = torch.mean(triangles, dim=2)
        # triangle_area = self.compute_triangle_area(triangles)
        # com_naive = torch.einsum('bij,bi->bj', triangle_centroids, triangle_area) / torch.sum(triangle_area, dim=1)

        # pressure based center of support
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 1] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()
        pressure_weights = inside_mask * (1-self.cos_k*vertex_height) + outside_mask *  torch.exp(-self.cos_w * vertex_height)
        cos = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (torch.sum(pressure_weights, dim=1, keepdim=True) +eps)

        # naive center of support
        # vertex_height_robustified = GMoF_unscaled(rho=self.gmof_rho)(vertex_height)
        contact_confidence = torch.sum(pressure_weights, dim=1)
        # contact_mask = (vertex_height < self.contact_thresh).float()
        # num_contact_verts = torch.sum(contact_mask, dim=1)
        # contact_centroid_naive = torch.sum(vertices_hd * contact_mask[:, :, None], dim=1) / (torch.sum(contact_mask, dim=1) + eps)

        # project com, cos to ground plane (x-z plane)
        # weight loss by number of contact vertices to zero out if zero vertices in contact
        com_xz = torch.stack([com[:, 0], torch.zeros_like(com)[:, 0], com[:, 2]], dim=1)
        contact_centroid_xz = torch.stack([cos[:, 0], torch.zeros_like(cos)[:, 0], cos[:, 2]], dim=1)
        # stability_loss = (contact_confidence * torch.norm(com_xz - contact_centroid_xz, dim=1)).sum(dim=-1)
        stability_loss = (torch.norm(com_xz - contact_centroid_xz, dim=1))
        return stability_loss