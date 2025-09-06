import torch
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from bosRegressor.utils.mesh_utils import HDfier


from bosRegressor.core.support import StabilityLossCoP
from bosRegressor.utils.constants import PRESSURE_THRESH, CONTACT_THRESH, VELOCITY_THRESH
from bosRegressor.utils.misc_utils import compute_finite_differences
from bosRegressor.utils.pressure_utils import pressure_to_contact_mask, get_contact_mask

from bosRegressor.utils.vis_utils import plot_sequence_data

# Remove non-deterministic convs
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Solve precision issues: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#fp16-on-mi200
torch.backends.cudnn.allow_tf32 = False
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


class BiomechanicalEvaluator(StabilityLossCoP):

    def __init__(self,
                 faces,
                 fps,
                 cop_w=30,
                 cop_k=100,
                 stencil_size=3,
                 model_type='smpl',
                 exp_name='test',
                 debug=False,
                 device='cuda',
                 ):
        super().__init__(faces, cop_w, cop_k, model_type)

        self.exp_name = exp_name

        self.faces = faces
        self.cop_w = cop_w
        self.cop_k = cop_k

        self.hdfy_op = HDfier(model_type=model_type)

        self.total_mass = 72.4  # in kg
        self.gravity = torch.FloatTensor([0.0, 0.0, 9.8]).to(device)
        self.floor_normal = torch.FloatTensor([0.0, 0.0, 1.0]).to(device)
        self.fps = fps
        self.h = 1.0 / self.fps

        # set up physics information
        if stencil_size == 3:
            self.stencil_a = torch.FloatTensor([1.0, -2.0, 1.0])
        elif stencil_size == 5:
            self.stencil_a = torch.FloatTensor([-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0])
        self.stencil_a = self.stencil_a / (self.h * self.h)
        # self.stencil_a = torch.FloatTensor([1.0/90.0, -3.0/20.0, 3.0/2.0, -49.0/18.0, 3.0/2.0, -3.0/20.0, 1.0/90.0])
        self.stencil_a = self.stencil_a.to(device)
        self.stencil_size = self.stencil_a.shape[0]

        # velocity stencil
        if stencil_size == 3:
            self.stencil_v = torch.FloatTensor([-1/2, 0.0, 1/2])
        elif stencil_size == 5:
            raise NotImplementedError
        self.stencil_v = self.stencil_v / self.h
        self.stencil_v = self.stencil_v.to(device)

        self.center_frame = (self.stencil_a.shape[0] - 1) // 2

        self.debug = debug
        self.device = device

        # member attributes which are initialized later
        self.coms = None
        self.body_pose = nn.Parameter(torch.empty(0), requires_grad=True)
        self.vertices = None
        self.joints = None
        self.zmps = None
        self.init_zmps = None
        self.init_mass_per_vert = None
        self.init_per_part_multiplier = None
        self.mass_per_vert = None
        self.debug_dict = None

        # initalize the per part multiplier, which is the log of the per part density
        # zero value of per part multiplier corresponds to a density of 1.0
        per_part_multiplier = torch.zeros([1, 10]).to(device)


        # make per part multiplier a learnable parameter
        # self.register_parameter('per_part_multiplier', nn.Parameter(per_part_multiplier, requires_grad=True))
        self.register_buffer('per_part_multiplier', per_part_multiplier)
        if self.init_per_part_multiplier is None:
            self.init_per_part_multiplier = per_part_multiplier.clone().detach()

    def init_per_part_buffers(self):
        vertices = self.vertices.float()
        # calculate per part volume
        per_part_volume = self.compute_per_part_volume(vertices[:, 0]) # taking the first frame vertices
        self.register_buffer('per_part_volume', per_part_volume)

        per_part_densities = torch.exp(self.per_part_multiplier)
        self.register_buffer('per_part_densities', per_part_densities)


        # Todo: Check why the following assert fails for total volume (may be precision or some boundary vertex issue)
        # assert per_part_volume.sum() == volume_per_vert.sum(), 'per_part_volume and volume_per_vert should be equal'

        # self.register_buffer('volume_per_vert', volume_per_vert)
        # # get volume per vertex id in the hd mesh
        # volume_per_vert_hd = self.vert_hd_id_to_part_volume_mapping(per_part_volume, vertices.device)
        # # make per part volume a learnable parameter
        # self.register_parameter('volume_per_vert_hd', nn.Parameter(volume_per_vert_hd, requires_grad=True))

    def calculate_mass_per_vert(self):
        '''
        Calculate the mass per vertex using the volume per part and density per part. Note that this is proxy mass
        '''
        # clamp the per part multiplier to be between -10 and 10 to avoid numerical issues
        self.per_part_multiplier.data = torch.clamp(self.per_part_multiplier.data, min=-10, max=10)

        self.per_part_densities = torch.exp(self.per_part_multiplier)
        per_part_mass = self.per_part_volume * self.per_part_densities
        self.mass_per_vert = self.vert_id_to_part_volume_mapping(per_part_mass, self.device)


    def compute_and_plot_derivatives(self, data, name):
        data_v = compute_finite_differences(data, self.stencil_v)
        data_a = compute_finite_differences(data, self.stencil_a)
        data_pelvis_v = data_v[:, 0, :]
        data_pelvis_a = data_a[:, 0, :]
        plot_sequence_data(data_pelvis_v.squeeze(), name=f'{name}_v', exp_name=self.exp_name)
        plot_sequence_data(data_pelvis_a.squeeze(), name=f'{name}_a', exp_name=self.exp_name)
        return data_pelvis_v, data_pelvis_a

    def save_debug_logs(self):
        if self.debug:
            coms_v, coms_a = self.compute_and_plot_derivatives(self.coms, name='coms')
            pelvis_v, pelvis_a = self.compute_and_plot_derivatives(self.joints, name='pelvis')
            pose_pelvis_v, pose_pelvis_a = self.compute_and_plot_derivatives(self.body_pose, name='pose_pelvis')
            zmp_v, zmp_a = self.compute_and_plot_derivatives(self.zmps, name='zmp')
            vertex_v, vertex_a = self.compute_and_plot_derivatives(self.vertices, name='vertex')
            # pack these in a dict
            self.debug_dict = {
                'coms_v': coms_v.detach().cpu().numpy(), 'coms_a': coms_a.detach().cpu().numpy(),
                'pelvis_v': pelvis_v.detach().cpu().numpy(), 'pelvis_a': pelvis_a.detach().cpu().numpy(),
                'pose_pelvis_v': pose_pelvis_v.detach().cpu().numpy(),
                'pose_pelvis_a': pose_pelvis_a.detach().cpu().numpy(),
                'zmps_v': zmp_v.detach().cpu().numpy(), 'zmps_a': zmp_a.detach().cpu().numpy(),
                'vertex_v': vertex_v.detach().cpu().numpy(), 'vertex_a': vertex_a.detach().cpu().numpy(),
            }



    def calculate_zmps(self):
        '''
        Calculate the zero moment point (ZMP) of the current state.
        :return:
        '''

        ###### Todo: put batch dimension on 0, not frames
        # coms = torch.stack(self.coms, dim=0)
        coms = self.coms
        vertices = self.vertices

        G_a = compute_finite_differences(coms, self.stencil_a)

        # plot_first_and_second_derivatives(G_a.squeeze(), name='G_a')
        F_gi = (self.gravity - G_a) * self.total_mass

        r = vertices - coms
        r = r[:, self.center_frame:-self.center_frame, :, :]
        a = compute_finite_differences(vertices, self.stencil_a)

        # per_vertex_mass = self.total_mass / vertices_hd.shape[1]
        per_vertex_mass = (self.mass_per_vert / self.mass_per_vert.sum(dim=1, keepdim=True)) * self.total_mass
        f = a * per_vertex_mass[:, None, :, :]

        H_g = torch.cross(r, f).sum(dim=2)

        # slice the center frames for com
        coms = coms[:, self.center_frame:-self.center_frame, :, :]

        # collapse the batch and frame dimension
        bs, nf, _, _ = coms.shape
        coms = coms.reshape(bs * nf, -1, 3)
        F_gi = F_gi.reshape(bs * nf, -1, 3)
        H_g = H_g.reshape(bs * nf, -1, 3)
        G_a = G_a.reshape(bs * nf, -1, 3)
        # project com to the floor
        P = coms - torch.bmm(coms, self.floor_normal[None, :, None].repeat(coms.shape[0], 1, 1)) * self.floor_normal[
                                                                                                   None, None,
                                                                                                   :].repeat(
            coms.shape[0], 1, 1)

        PG = coms - P
        M_gi = torch.cross(PG, (self.gravity * self.total_mass)[None, None, :]) - torch.cross(PG, (
                G_a * self.total_mass)) - H_g

        eps = 1e-6
        PZ = torch.cross(self.floor_normal[None, None, :], M_gi) / (torch.bmm(F_gi,
                                                                             self.floor_normal[None, :, None].repeat(
                                                                                 F_gi.shape[0], 1, 1)) + eps)
        zmps = P - PZ

        self.zmps = zmps.view(bs, nf, 1, 3)
        if self.init_zmps is None:
            self.init_zmps = zmps.clone().detach()

    def calculate_com(self):
        """
        Note: the vertices should be aligned along y-axis and in world coordinates
        :param vertices:
        :return:
        """
        # do negative values exist in the per_part_densities?
        if torch.any(self.per_part_densities < 0):
            logger.warning('Negative values in per_part_densities. Not doing anything for now.')
        ###### Todo: put batch dimension on 0, not frames
        # save initial volume per vertex for visualization
        if self.init_mass_per_vert is None:
            self.init_mass_per_vert = self.mass_per_vert.clone().detach()

        # calculate com using volume weighted mean
        com = torch.sum(self.vertices * self.mass_per_vert[:, None, :, :], dim=2) / torch.sum(self.mass_per_vert[:, None, :, :], dim=2)
        self.coms = com[:, :, None, :]
        return com

    def calculate_cop(self):
        """
        Calculate the center of pressure (CoP) using the IPMAN heuristic
        :return:
        """
        # get uniformly sampled HD vertices
        bs, nf, nj, _ = self.vertices.shape
        vertices_hd = self.hdfy_op.hdfy_mesh(self.vertices.view([-1, nj, 3]))

        # pressure based center of pressure
        ground_plane_height = 0.0
        eps = 1e-6
        vertex_height = (vertices_hd[:, :, 2] - ground_plane_height)
        inside_mask = (vertex_height < 0.0).float()
        outside_mask = (vertex_height >= 0.0).float()


        outside_pressure = torch.exp(-self.cop_w * vertex_height)
        # Replace 'inf' values without in-place modification
        outside_pressure = torch.where(outside_pressure == float('inf'),
                                       torch.tensor(1e6, device=outside_pressure.device), outside_pressure)
        inside_pressure = (1 - self.cop_k * vertex_height)

        pressure_weights = inside_mask * inside_pressure + outside_mask * outside_pressure
        cop = torch.sum(vertices_hd * pressure_weights.unsqueeze(-1), dim=1) / (
                torch.sum(pressure_weights, dim=1, keepdim=True) + eps)
        # if all pressure_weights = 0, then set to cop to com
        com = self.coms[:, :, 0, :].view(-1, 3)
        max_pressure_wt = torch.max(pressure_weights, dim=-1).values
        # wherever max pressure weight is < 5e-4, replace cop with com
        mask = max_pressure_wt < 5e-4
        cop[mask] = com[mask]
        # project cop on the ground
        cop[:, 2] = torch.zeros_like(cop)[:, 2]
        self.cops = cop.view(bs, nf, 1, 3)

    def calculate_bos(self):
        '''
        Calculate the base of support (BOS) for every frame. Formulated as a convex hull problem.
        :return:
        '''
        # get vertices in contact
        ground_plane_height = 0.0
        # contact_mask, _ = pressure_to_contact_mask(self.vertices, cop_w=self.cop_w, cop_k=self.cop_k,
        #                                            pressure_thresh=PRESSURE_THRESH)

        with torch.no_grad():
            vertices = self.vertices.clone().detach()
            contact_mask = get_contact_mask(vertices, contact_thresh=CONTACT_THRESH, velocity_thresh=VELOCITY_THRESH)

            # combine batch and time dimension
            bs, nf, nv, _ = vertices.shape
            vertices = vertices.reshape(-1, nv, 3)
            contact_mask = contact_mask.reshape(-1, nv)

            # iterate over all frames
            self.hull_verts = []
            for i, contact_mask_el in enumerate(contact_mask):
                # filter out the vertices that are not in contact
                c_el = torch.masked_select(vertices[i], contact_mask_el[:, None]).view(-1, 3)
                # project to ground plane z=0
                c_el[:, 2] = torch.zeros_like(c_el)[:, 2]
                # find the convex hull
                try:
                    convex_hull = ConvexHull(c_el[:, :2].detach().cpu().numpy())
                    # get the vertices of the convex hull
                    hull_vert = c_el[convex_hull.vertices]
                except:
                    hull_vert = None
                # get the vertices of the convex hull
                self.hull_verts.append(hull_vert)
            # get num of frame in hull_verts with non-zero vertices
            num_non_zero = sum([1 if item is not None else 0 for item in self.hull_verts])
            # # print(f'Number of frames with non-zero vertices in hull: {num_non_zero}')
            # if num_non_zero < len(self.vertices):
            #     return False
            # else:
            #     return True

    def check_bos_interior_seq(self, no_contact_vert_tol=10):
        """
        Check if the projection of ZMP lies inside the 2D convex hull/base-of-support of the contact vertices.
        Test this by finding if the ZMP projection is a convex combination of the contact vertices.
        Ref: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl#answer-43564754
        Args:
            vertices (temporal): [batch_size, num_frames, num_verts, 3]
            faces:
            gmof_rho:

        Returns:

        """
        with torch.no_grad():
            ground_plane_height = 0.0

            # vertices_hd = torch.stack(self.vertices_hd, dim=0).squeeze()
            vertices = self.vertices
            zmps = self.zmps

            # add batch
            if vertices.ndim == 3:
                vertices = vertices[None, ...]
            if self.zmps.ndim == 3:
                zmps = self.zmps[None, ...]

            # repeat the first frame and last frame in zmp
            zmps = torch.cat([zmps[:, 0:1, :, :], zmps, zmps[:, -1:, :, :]], dim=1)

            # absorb batch and time dimension
            b, t, v, _ = vertices.shape

            contact_mask = get_contact_mask(self.vertices, contact_thresh=CONTACT_THRESH, velocity_thresh=VELOCITY_THRESH)

            vertices = vertices.reshape(b * t, v, 3)  # there was clone() here for some reason
            contact_mask = contact_mask.reshape(b * t, -1)
            zmps = zmps.reshape(b * t, -1, 3)

            num_contact_verts = torch.sum(contact_mask, dim=1, keepdim=True)
            contact_metric = torch.ones_like(num_contact_verts)
            contact_metric[num_contact_verts < no_contact_vert_tol] = 0

            # filter out the vertices that are not in contact
            # contact_vertices = vertices * contact_mask[:, :, None]
            in_hull_label = torch.zeros(b * t, dtype=torch.float)
            for i, zmps_el in enumerate(zmps):
                # filter out the vertices that are not in contact
                c_el = torch.masked_select(vertices[i], contact_mask[i, :, None]).view(-1, 3)
                if num_contact_verts[i] <= no_contact_vert_tol:
                    in_hull_label[i] = -1  # no contact
                    continue
                # project to ground plane z=0
                c_el[:, 2] = torch.zeros_like(c_el)[:, 2]
                zmps_el = zmps_el.squeeze()
                zmps_el_project = zmps_el.clone()
                zmps_el_project[2] = 0.0
                label = self.in_hull(c_el.cpu().numpy(), zmps_el_project.cpu().numpy())
                in_hull_label[i] = float(label)
            in_hull_label = torch.tensor(in_hull_label).reshape(b, t, -1)
            contact_metric = contact_metric.reshape(b, t, -1)
            contact_mask = contact_mask.reshape(b, t, -1)
        return torch.tensor(in_hull_label), contact_metric, contact_mask

    def in_hull(self, points, x):
        # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
        n_points = len(points)
        n_dim = len(x)
        c = np.zeros(n_points)
        A = np.r_[points.T, np.ones((1, n_points))]
        b = np.r_[x, np.ones(1)]
        try:
            lp = linprog(c, A_eq=A, b_eq=b, method='interior-point')
            return lp.success
        except:
            print('Linprog failed. Problem is infeasible')
            return False


def setup_biomechanical_evaluator(biomechanical_evaluator, joints, verts):
    biomechanical_evaluator.joints = joints
    biomechanical_evaluator.vertices = verts
    biomechanical_evaluator.init_per_part_buffers()
    biomechanical_evaluator.calculate_mass_per_vert()
    biomechanical_evaluator.calculate_com()
    biomechanical_evaluator.calculate_cop()
    biomechanical_evaluator.calculate_zmps()
