import torch.nn as nn
import numpy as np
import torch

from bosRegressor.losses.l2 import L2Loss
from bosRegressor.losses.gmm import MaxMixturePrior
from bosRegressor.utils.constants import SMPLH_FOOT_JOINTS


def build_loss(loss_cfg, body_model_type='smplx'):
    loss_type = loss_cfg.TYPE
    if loss_type == 'l2':
        loss = L2Loss(squared=loss_cfg.SQUARED,
                      d1_aggregation=loss_cfg.D1_AGGREGATION)
    elif loss_type == 'signed_distance':
        loss = min_signed_distance_to_convex_hull
    # elif loss_type == 'gmm':
    #     loss = MaxMixturePrior(model_type=body_model_type, **loss_cfg)
    # elif loss_type == '':
    #     loss = Placeholder()
    else:
        raise ValueError(f'Loss {loss_type} not implemented')
    return loss

def min_signed_distance_to_convex_hull(points, c_hull):
    """
    This function first calculates the signed distance to each edge of the convex hull and returns the minimum of these
    distances as the signed distance to the convex hull.
    The sign indicates whether the point is inside (negative) or outside (positive) the convex hull.
    """
    def normalize(v):
        norm = torch.norm(v, dim=-1)
        return v / norm.unsqueeze(-1).clamp(min=1e-12)

    def point_on_line(line_start, line_end, point):
        line_vector = line_end - line_start
        point_vector = point - line_start
        # no sqrt in denomintor since check the next two lines
        projection = torch.sum(point_vector * line_vector, dim=-1) / torch.sum(line_vector * line_vector, dim=-1)
        projection = torch.clamp(projection, 0, 1)
        # note here we don't use unit vector since we didn't sqrt two lines back
        closest_point = line_start + projection.unsqueeze(-1) * line_vector
        return closest_point

    def signed_distance_to_edges(edge_starts, edge_ends, point):
        closest_points = point_on_line(edge_starts, edge_ends, point.unsqueeze(1))
        edge_vector = edge_ends - edge_starts
        # rotate by 90 degrees since convex hull ordering in counter-clockwise for outward normal
        outward_normals = normalize(torch.stack((edge_vector[:, 1], -edge_vector[:, 0], torch.zeros_like(edge_vector[:, 0])), dim=-1))
        distance_vector = point.unsqueeze(1) - closest_points
        distances = torch.norm(distance_vector, dim=-1)
        # get sign
        # alpha = 100 # controls the steepness of the tanh around zero
        # sign = torch.tanh(alpha * torch.sum(distance_vector * outward_normals, dim=-1))
        # since you don't need backprop
        sign = torch.sign(torch.sum(distance_vector * outward_normals, dim=-1))
        return distances, sign

    batch_size = points.shape[0]
    min_positive_distances = torch.empty(batch_size, dtype=points.dtype, device=points.device)
    min_signed_distances = torch.empty(batch_size, dtype=points.dtype, device=points.device)
    for i in range(batch_size):
        point = points[i]
        hull_verts = c_hull[i]
        if hull_verts is not None:
            # Create tensors for edge start and end points
            edge_starts = hull_verts
            edge_ends = torch.roll(hull_verts, shifts=-1, dims=0)

            distances, sign = signed_distance_to_edges(edge_starts, edge_ends, point)
            # find minimum positive signed distance
            min_positive_distances[i] = torch.min(distances[sign > 0]) if torch.any(sign > 0) else torch.tensor(
                0.0).requires_grad_(True)
            signed_distances = distances * sign
            # find minimum distance if positive and maximum distance if negative
            min_signed_distances[i] = torch.min(signed_distances[sign > 0]) if torch.any(sign > 0) else torch.max(
                signed_distances[sign <= 0])
        else:
            min_positive_distances[i] = torch.tensor(0.0).requires_grad_(True)
            min_signed_distances[i] = torch.tensor(0.0).requires_grad_(True)
    return min_positive_distances, min_signed_distances

def all_physics_metrics(verts, joints, tol=0.01, device="cuda"):
    # calculate the physics metrics
    lowest_vertex_z = verts.min(dim=1).values[:, 2]

    # Note: for both penetration and float, we have a tolerance of 5 mm to account for geometry approximations

    # calculate the distance of the lowest mesh vertex below the ground (z=0) for getting penetration
    penetration_dist = torch.abs(lowest_vertex_z[lowest_vertex_z < -tol])
    penetration_dist = penetration_dist.mean() if penetration_dist.numel() > 0 else torch.tensor(0.0).to(device)

    # calculate the distance of the lowest mesh vertex above the ground (z=0) for getting float
    float_dist = torch.abs(lowest_vertex_z[lowest_vertex_z >= tol])
    float_dist = float_dist.mean() if float_dist.numel() > 0 else torch.tensor(0.0).to(device)

    # calculate foot sliding by finding foot joints that contact the ground in two adjacent frame and computing their
    # horizontal displacement within the frames (in x and y)
    # get the foot joints
    foot_joints = joints[:, SMPLH_FOOT_JOINTS, :]
    # calculate foot joint displacement in two adjacent frames (only in x and y directions)
    foot_joint_displacements_xy = foot_joints.diff(dim=0)[:, :, :2]
    # get the ground contact
    foot_joints_in_contact = foot_joints[:, :, [2]] < tol
    # get the foot joints in contact in two adjacent frames
    foot_ground_contact = torch.logical_and(foot_joints_in_contact[:-1, :, :], foot_joints_in_contact[1:, :, :])
    # get the horizontal displacement of the foot joints in contact
    foot_ground_disp = torch.abs(foot_joint_displacements_xy * foot_ground_contact).sum(dim=-1)
    # get the mean displacement of the foot joints in contact
    foot_ground_disp = foot_ground_disp.mean() if foot_ground_disp.numel() > 0 else torch.tensor(0.0).to(device)
    return penetration_dist, float_dist, foot_ground_disp

class OptiLoss(nn.Module):
    def __init__(
            self,
            losses_cfgs,
            biomechanical_evaluator,
            body_model_type='smplx',
    ):
        super(OptiLoss, self).__init__()

        self.cfg = losses_cfgs
        self.biomechanical_evaluator = biomechanical_evaluator

        # when loss weights are != 0, add loss as member variable
        for name, cfg in losses_cfgs.items():
            if name == 'debug':
                continue

            # add loss weight as member variable
            weight = cfg.WEIGHT
            setattr(self, name.lower() + '_weights', weight)

            # add criterion as member variable when weight != 0 exists
            if sum([x != 0 for x in cfg.WEIGHT]) > 0:
                function = build_loss(cfg, body_model_type)
                setattr(self, name.lower() + '_crit', function)

                # # check if the criterion / loss is used in forward pass
                # method = 'get_' + name.lower() + '_loss'
                # assert callable(getattr(self, method)), \
                #     f'Method {method} not implemented in OptiLoss'

        self.set_weights(stage=0)  # init weights with first stage

        self.debug = []

    def set_weights(self, stage, default_stage=-1):

        for name, cfg in self.cfg.items():
            if name == 'debug':
                continue

            weight = getattr(self, name.lower() + '_weights')

            # use default stage value if weight for stage not specified
            weight_stage = default_stage if len(weight) <= stage else stage

            setattr(self, name.lower() + '_weight', weight[weight_stage])

    def get_shape_prior_loss(self, betas):
        shape_prior_loss = self.shape_prior_crit(
            betas, y=None) * self.shape_prior_weight
        return shape_prior_loss

    def get_smooth_zmp_loss(self, zmp):
        # calculate zmp velocities
        zmp_vel = zmp[1:, :] - zmp[:-1, :]
        smooth_zmp_loss = self.smooth_zmp_crit(
            zmp_vel) * self.smooth_zmp_weight
        return smooth_zmp_loss

    def get_smooth_vertices_loss(self, vertices):
        # calculate vertices velocities
        vertices_vel = vertices[1:, :] - vertices[:-1, :]
        smooth_vertices_loss = self.smooth_vertices_crit(
            vertices_vel) * self.smooth_vertices_weight
        return smooth_vertices_loss

    def get_pose_prior_loss(self, pose):
        pose_prior_loss = torch.sum(self.pose_prior_crit(
            pose)) * self.pose_prior_weight
        return pose_prior_loss

    def get_init_per_part_multiplier_loss(self, init_per_part_multiplier, est_per_part_multiplier, device):
        init_per_part_multiplier = init_per_part_multiplier.to(device)

        init_per_part_multiplier_loss = self.init_per_part_multiplier_crit(
            init_per_part_multiplier, est_per_part_multiplier
        ) * self.init_per_part_multiplier_weight

        return init_per_part_multiplier_loss

    def get_init_pose_loss(self, init, est_body_pose, device):
        init_pose = init['body_pose'].to(device)

        if len(init_pose.shape) == 1:
            init_pose = init_pose.unsqueeze(0)

        init_pose_prior_loss = self.init_pose_crit(
            init_pose, est_body_pose
        ) * self.init_pose_weight

        return init_pose_prior_loss

    def get_init_transl_loss(self, init, est_body_transl, device):
        init_transl = init['transl'].to(device)

        if len(init_transl.shape) == 1:
            init_transl = init_transl.unsqueeze(0)

        init_transl_prior_loss = self.init_transl_crit(
            init_transl, est_body_transl
        ) * self.init_transl_weight

        return init_transl_prior_loss

    def get_init_global_orient_loss(self, init, est_body_global_orient, device):
        init_global_orient = init['global_orient'].to(device)

        if len(init_global_orient.shape) == 1:
            init_global_orient = init_global_orient.unsqueeze(0)

        init_global_orient_prior_loss = self.init_global_orient_crit(
            init_global_orient, est_body_global_orient
        ) * self.init_global_orient_weight

        return init_global_orient_prior_loss

    def get_init_shape_loss(self, init, est_shape, device):
        init_shape = init['betas'].unsqueeze(0).to(device)
        init_shape_loss = self.init_pose_crit(
            init_shape, est_shape
        ) * self.init_shape_weight
        return init_shape_loss

    def get_zmp_cop_loss(self, zmps, gt_cop):
        zmp_cop_loss = self.zmp_cop_prior_crit(zmps, gt_cop) * self.zmp_cop_prior_weight
        return zmp_cop_loss

    def get_com_cop_loss(self, coms, gt_cop):
        com_cop_loss = self.com_cop_prior_crit(coms, gt_cop) * self.com_cop_prior_weight
        return com_cop_loss

    def get_dynamic_stability_loss(self, zmps, hull_verts):
        min_positive_distances, _ = self.dynamic_stability_crit(zmps, hull_verts)
        dynamic_stability_loss = min_positive_distances * self.dynamic_stability_weight
        return dynamic_stability_loss

    def get_physics_float_loss(self, verts, joints):
        _, float_dist, _ = all_physics_metrics(verts, joints)
        physics_float_loss = float_dist * self.physics_float_weight
        return physics_float_loss

    def get_physics_penetration_loss(self, verts, joints):
        penetration_dist, _, _ = all_physics_metrics(verts, joints)
        physics_penetrate_loss = penetration_dist * self.physics_penetrate_weight
        return physics_penetrate_loss

    def get_physics_sliding_loss(self, verts, joints):
        _, _, foot_ground_disp = all_physics_metrics(verts, joints)
        physics_slide_loss = foot_ground_disp * self.physics_slide_weight
        return physics_slide_loss

    def forward_fitting(
            self,
            smpl_output,  # the current estimate of person a
            init_h1,  # the initail estimate of person a (from BEV)
    ):
        bs, num_joints, _ = smpl_output.joints.shape
        device = smpl_output.joints.device

        ld = {}  # store losses in dict for printing
        bio_out = {}  # store biomechanical attributes in dict for printing

        # init_betas = init_h1['betas'].unsqueeze(0).to(device)

        # # pose prior loss
        # ld['pose_prior_loss'] = 0.0
        # if self.pose_prior_weight > 0:
        #     ld['pose_prior_loss'] += self.get_pose_prior_loss(
        #         smpl_output.body_pose)

        # per part volume losses for each human
        ld['init_per_part_multiplier_losses'] = 0.0
        if self.init_per_part_multiplier_weight > 0:
            ld['init_per_part_multiplier_losses'] += self.get_init_per_part_multiplier_loss(
                self.biomechanical_evaluator.init_per_part_multiplier, self.biomechanical_evaluator.per_part_multiplier, device)

        # pose prior losses for each human
        ld['init_pose_losses'] = 0.0
        if self.init_pose_weight > 0:
            ld['init_pose_losses'] += self.get_init_pose_loss(
                init_h1, smpl_output.body_pose, device)

        # transl prior losses for each human
        ld['init_transl_losses'] = 0.0
        if self.init_transl_weight > 0:
            ld['init_transl_losses'] += self.get_init_transl_loss(
                init_h1, smpl_output.transl, device)

        # pose prior losses for each human
        ld['init_global_orient_losses'] = 0.0
        if self.init_global_orient_weight > 0:
            ld['init_global_orient_losses'] += self.get_init_global_orient_loss(
                init_h1, smpl_output.global_orient, device)

        # # shape prior losses for each human
        # ld['init_shape_losses'] = 0.0
        # if self.init_shape_weight > 0:
        #     ld['init_shape_losses'] += self.get_init_shape_loss(
        #         init_h1, smpl_output.betas, device)

        self.biomechanical_evaluator.calculate_com()
        self.biomechanical_evaluator.calculate_zmps()
        self.biomechanical_evaluator.calculate_bos()
        bio_out['init_zmps'] = self.biomechanical_evaluator.init_zmps.detach().clone()
        bio_out['zmps'] = self.biomechanical_evaluator.zmps.detach().clone()
        bio_out['coms'] = self.biomechanical_evaluator.coms.detach().clone()
        bio_out['hull_verts'] = [item.detach().clone() if item is not None else item for item in self.biomechanical_evaluator.hull_verts]

        # smooth zmp losses for each human
        ld['smooth_zmp_losses'] = 0.0
        if self.smooth_zmp_weight > 0:
            zmps = self.biomechanical_evaluator.zmps
            ld['smooth_zmp_losses'] += self.get_smooth_zmp_loss(zmps)

        # smooth zmp losses for each human
        ld['smooth_vertices_losses'] = 0.0
        if self.smooth_vertices_weight > 0:
            vertices = smpl_output.vertices
            ld['smooth_vertices_losses'] += self.get_smooth_vertices_loss(vertices)

        # shape prior losses for each human
        ld['zmp_cop_losses'] = 0.0
        if self.zmp_cop_prior_weight > 0:
            zmps = self.biomechanical_evaluator.zmps
            # project zmps to ground
            zmps[:, :, 2] = 0.0
            # align gt_cops with zmps
            gt_cops = init_h1["gt_cops"].permute(1, 0, 2)[
                      self.biomechanical_evaluator.center_frame:-self.biomechanical_evaluator.center_frame, :, :]
            ld['zmp_cop_losses'] += self.get_zmp_cop_loss(
                zmps, gt_cops)

        # shape prior losses for each human
        ld['com_cop_losses'] = 0.0
        if self.com_cop_prior_weight > 0:
            coms = self.biomechanical_evaluator.coms
            # project coms to ground
            coms[:, :, 2] = 0.0
            # project
            gt_cops = init_h1["gt_cops"].permute(1, 0, 2)
            ld['com_cop_losses'] += self.get_com_cop_loss(
                coms, gt_cops)

        # dynamic stability losses for each human
        ld['dynamic_stability_losses'] = 0.0
        if self.dynamic_stability_weight > 0:
            zmps = self.biomechanical_evaluator.zmps
            # project zmps to ground
            zmps[:, :, 2] = 0.0
            hull_verts = self.biomechanical_evaluator.hull_verts
            # align hull_verts with zmps
            hull_verts = hull_verts[self.biomechanical_evaluator.center_frame:-self.biomechanical_evaluator.center_frame]
            ld['dynamic_stability_losses'] += self.get_dynamic_stability_loss(zmps, hull_verts)

        # physics float losses for each human
        ld['physics_float_losses'] = 0.0
        if self.physics_float_weight > 0:
            verts = smpl_output.vertices
            joints = smpl_output.joints
            ld['physics_float_losses'] += self.get_physics_float_loss(verts, joints)

        # physics penetration losses for each human
        ld['physics_penetrate_losses'] = 0.0
        if self.physics_penetrate_weight > 0:
            verts = smpl_output.vertices
            joints = smpl_output.joints
            ld['physics_penetrate_losses'] += self.get_physics_penetration_loss(verts, joints)

        # physics foot sliding losses for each human
        ld['physics_slide_losses'] = 0.0
        if self.physics_slide_weight > 0:
            verts = smpl_output.vertices
            joints = smpl_output.joints
            ld['physics_slide_losses'] += self.get_physics_sliding_loss(verts, joints)

        # average the losses over batch
        ld_out = {}
        for k, v in ld.items():
            if type(v) == torch.Tensor:
                ld_out[k] = v.mean()

        # final loss value
        fitting_loss = sum(ld_out.values())
        ld_out['total_fitting_loss'] = fitting_loss

        return fitting_loss, ld_out, bio_out

    def forward(
            self,
            smpl_output,
            init_h1,
    ):
        """
        Compute all losses in the current optimization iteration.
        The current estimate is smpl_output/smpl_output_h2, which
        we pass to the L_fitting and L_diffusion modules. The final
        loss is the sum of both losses.
        """

        # fitting losses (keypoints, pose / shape prior etc.)
        fitting_loss, fitting_ld_out, bio_out = self.forward_fitting(
            smpl_output,
            init_h1,
        )

        # update loss dict and sum up losses
        total_loss = fitting_loss
        ld_out = fitting_ld_out

        ld_out['total_loss'] = total_loss
        return total_loss, ld_out, bio_out
