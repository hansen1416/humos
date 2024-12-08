import numpy as np
import torch
from humos.utils.constants import SMPLH_FOOT_JOINTS
from humos.utils.mesh_utils import smplh_breakdown
from humos.utils.metrics_utils import (calculate_activation_statistics_np, calculate_diversity_np,
                                            calculate_frechet_distance_np)
from bosRegressor.core.biomechanics import setup_biomechanical_evaluator
from bosRegressor.losses.build_losses import min_signed_distance_to_convex_hull


def print_latex_metrics(metrics):
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    t2m_keys = [f"t2m/R{i}" for i in vals] + ["t2m/MedR"]
    m2t_keys = [f"m2t/R{i}" for i in vals] + ["m2t/MedR"]

    keys = t2m_keys + m2t_keys

    def ff(val_):
        val = str(val_).ljust(5, "0")
        # make decimal fine when only one digit
        if val[1] == ".":
            val = str(val_).ljust(4, "0")
        return val

    str_ = "& " + " & ".join([ff(metrics[key]) for key in keys]) + r" \\"
    dico = {key: ff(metrics[key]) for key in keys}
    print(dico)
    print("Number of samples: {}".format(int(metrics["t2m/len"])))
    print(str_)


def all_physics_metrics(verts, joints, tol=0.005, device="cuda"):
    # calculate the physics metrics
    lowest_vertex_z = verts.min(dim=2).values[:, :, 2]

    # Note: for both penetration and float, we have a tolerance of 5 mm to account for geometry approximations

    # calculate the distance of the lowest mesh vertex below the ground (z=0) for getting penetration
    penetration_dist = torch.abs(lowest_vertex_z[lowest_vertex_z < -tol])
    penetration_dist = penetration_dist.mean() if penetration_dist.numel() > 0 else torch.tensor(0.0).to(device)

    # calculate the distance of the lowest mesh vertex above the ground (z=0) for getting float
    float_dist = torch.abs(lowest_vertex_z[lowest_vertex_z >= tol])
    float_dist = float_dist.mean() if float_dist.numel() > 0 else torch.tensor(0.0).to(device)

    # calculate foot sliding by finding foot joints that contact the ground in two adjacent frame and computing their
    # horizontal displacement within the frames (in x and y)
    skate_tol = tol + 0.01 # to avoid the issue that skate loss minimization results in floating meshes.
    # get the foot joints
    foot_joints = joints[:, :, SMPLH_FOOT_JOINTS, :]
    # calculate foot joint displacement in two adjacent frames (only in x and y directions)
    foot_joint_displacements_xy = torch.norm(foot_joints[:, 1:, :, :2] - foot_joints[:, :-1, :, :2], dim=-1)
    # get the ground contact
    foot_joints_in_contact = foot_joints[:, :, :, 2] < skate_tol
    # get the foot joints in contact in two adjacent frames
    foot_ground_contact = torch.logical_and(foot_joints_in_contact[:, :-1, :], foot_joints_in_contact[:, 1:, :])
    # get the horizontal displacement of the foot joints in contact
    foot_ground_disp = torch.abs(foot_joint_displacements_xy * foot_ground_contact)
    # get the mean displacement of the foot joints in contact
    foot_ground_disp_mean = foot_ground_disp.mean() if foot_ground_disp.numel() > 0 else torch.tensor(0.0).to(device)

    # extra skate metrics
    # compute percentage of frames that have foot_ground_displacement above 0.005m (5cm)
    skate_th = 0.0
    foot_ground_disp_perc = (foot_ground_disp > skate_th).float().mean(dim=(1,2))
    # Get total foot joint skate
    foot_ground_disp_sum = foot_ground_disp.sum(dim=(1,2)) if foot_ground_disp.numel() > 0 else torch.tensor(0.0).to(device)
    # get stddev of foot joint skate
    foot_ground_disp_std = foot_ground_disp.std(dim=(1,2)) if foot_ground_disp.numel() > 0 else torch.tensor(0.0).to(device)
    return penetration_dist, float_dist, foot_ground_disp_mean, foot_ground_disp_std, foot_ground_disp_sum, foot_ground_disp_perc


def all_contrastive_metrics(sims, emb=None, threshold=None, rounding=2, return_cols=False):
    text_selfsim = None
    if emb is not None:
        text_selfsim = emb @ emb.T

    t2m_m, t2m_cols = contrastive_metrics(
        sims, text_selfsim, threshold, return_cols=True, rounding=rounding
    )
    m2t_m, m2t_cols = contrastive_metrics(
        sims.T, text_selfsim, threshold, return_cols=True, rounding=rounding
    )

    all_m = {}
    for key in t2m_m:
        all_m[f"t2m/{key}"] = t2m_m[key]
        all_m[f"m2t/{key}"] = m2t_m[key]

    all_m["t2m/len"] = float(len(sims))
    all_m["m2t/len"] = float(len(sims[0]))
    if return_cols:
        return all_m, t2m_cols, m2t_cols
    return all_m


def contrastive_metrics(
        sims,
        text_selfsim=None,
        threshold=None,
        return_cols=False,
        rounding=2,
        break_ties="averaging",
):
    n, m = sims.shape
    assert n == m
    num_queries = n

    dists = -sims
    sorted_dists = np.sort(dists, axis=1)
    # GT is in the diagonal
    gt_dists = np.diag(dists)[:, None]

    if text_selfsim is not None and threshold is not None:
        real_threshold = 2 * threshold - 1
        idx = np.argwhere(text_selfsim > real_threshold)
        partition = np.unique(idx[:, 0], return_index=True)[1]
        # take as GT the minimum score of similar values
        gt_dists = np.minimum.reduceat(dists[tuple(idx.T)], partition)
        gt_dists = gt_dists[:, None]

    rows, cols = np.where((sorted_dists - gt_dists) == 0)  # find column position of GT

    # if there are ties
    if rows.size > num_queries:
        assert np.unique(rows).size == num_queries, "issue in metric evaluation"
        if break_ties == "optimistically":
            opti_cols = break_ties_optimistically(sorted_dists, gt_dists)
            cols = opti_cols
        elif break_ties == "averaging":
            avg_cols = break_ties_average(sorted_dists, gt_dists)
            cols = avg_cols

    msg = "expected ranks to match queries ({} vs {}) "
    assert cols.size == num_queries, msg

    if return_cols:
        return cols2metrics(cols, num_queries, rounding=rounding), cols
    return cols2metrics(cols, num_queries, rounding=rounding)


def break_ties_average(sorted_dists, gt_dists):
    # fast implementation, based on this code:
    # https://stackoverflow.com/a/49239335
    locs = np.argwhere((sorted_dists - gt_dists) == 0)

    # Find the split indices
    steps = np.diff(locs[:, 0])
    splits = np.nonzero(steps)[0] + 1
    splits = np.insert(splits, 0, 0)

    # Compute the result columns
    summed_cols = np.add.reduceat(locs[:, 1], splits)
    counts = np.diff(np.append(splits, locs.shape[0]))
    avg_cols = summed_cols / counts
    return avg_cols


def break_ties_optimistically(sorted_dists, gt_dists):
    rows, cols = np.where((sorted_dists - gt_dists) == 0)
    _, idx = np.unique(rows, return_index=True)
    cols = cols[idx]
    return cols


def cols2metrics(cols, num_queries, rounding=2):
    metrics = {}
    vals = [str(x).zfill(2) for x in [1, 2, 3, 5, 10]]
    for val in vals:
        metrics[f"R{val}"] = 100 * float(np.sum(cols < int(val))) / num_queries

    metrics["MedR"] = float(np.median(cols) + 1)

    if rounding is not None:
        for key in metrics:
            metrics[key] = round(metrics[key], rounding)
    return metrics


def calculate_recons_metrics(fk, pred_data, gt_data):
    smpl_params_pred = smplh_breakdown(pred_data, fk=fk)
    smpl_params_gt = smplh_breakdown(gt_data, fk=fk)

    recons_keys = ["pose_body", "root_orient", "trans"]

    # calculate the recons metrics
    recons_metrics = {}

    for key in recons_keys:
        bs, nf, nfeats = smpl_params_pred[key].shape
        squared_diff = (smpl_params_pred[key] - smpl_params_gt[key]) ** 2
        # get the mean squared error across all dimensions except the batch dimension
        sum_squared_diff = squared_diff.sum(dim=(1,2))
        mean_squared_diff = sum_squared_diff / (nf * nfeats)
        recons_metrics[key] = mean_squared_diff.detach().cpu().numpy()
    recons_metrics["net_recons"] = sum(recons_metrics.values())
    return recons_metrics


def calculate_dyn_stability_metric(biomechanical_evaluator, joints, verts):
    bs, nf, nv, _ = verts.shape
    zmps = biomechanical_evaluator.zmps
    # repeat first and last frame in zmps to match the vertices frames. zmps is of shape (bs, nf, 1, 3)
    zmps = torch.cat([zmps[:, 0:1], zmps, zmps[:, -1:]], dim=1)
    # collapse zmps batch and frame dimensions
    zmps = zmps.view(-1, 1, 3)

    biomechanical_evaluator.calculate_bos()
    hull_verts = biomechanical_evaluator.hull_verts
    min_positive_distances, min_signed_distances = min_signed_distance_to_convex_hull(zmps, hull_verts)
    # count percentage of frames with negative signed distance
    perc_neg_signed_distance = (min_signed_distances < 0).float().mean()
    # get the mean of the minimum positive distance
    mean_min_positive_distance = min_positive_distances[min_positive_distances>0].mean()

    min_pos_distance = min_positive_distances.view(bs, nf)
    min_signed_distances = min_signed_distances.view(bs, nf)
    return perc_neg_signed_distance, mean_min_positive_distance, min_pos_distance, min_signed_distances


class MotionPriorMetric:
    def __init__(self, pretrained_motion_encoder,
                 fact = None,
                 sample_mean = False):
        super(MotionPriorMetric, self).__init__()

        self.pretrained_motion_encoder = pretrained_motion_encoder
        # Set pretrained motion encoder to frozen and eval mode
        for param in self.pretrained_motion_encoder.parameters():
            param.requires_grad = False
        self.pretrained_motion_encoder.eval()

        self.diversity_times = 300  # should be these many seq at min
        self.vae = True
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

    def encode(
        self,
        inputs,
        sample_mean = None,
        fact = None,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoded = self.pretrained_motion_encoder(inputs)

        # Sampling
        if self.vae:
            dists = encoded.unbind(1)
            mu, logvar = dists
            if sample_mean:
                latent_vectors = mu
            else:
                # Reparameterization trick
                std = logvar.exp().pow(0.5)
                eps = std.data.new(std.size()).normal_()
                latent_vectors = mu + fact * eps * std
        else:
            dists = None
            (latent_vectors,) = encoded.unbind(1)

        return latent_vectors

    def compute_fid_diversity(self, pred, gt):
        # inputs are embeddings
        all_pred = torch.cat(pred, dim=0).cpu().numpy()
        all_gt = torch.cat(gt, dim=0).cpu().numpy()

        # # shuffle all embeddings since MLD did it
        # shuffle_idx = torch.randperm(all_pred.size(0))
        # all_pred = all_pred[shuffle_idx].numpy()
        # all_gt = all_gt[shuffle_idx].numpy()

        # Compute fid
        mu, cov = calculate_activation_statistics_np(all_pred)
        # gt_mu, gt_cov = calculate_activation_statistics_np(all_gtmotions)
        gt_mu, gt_cov = calculate_activation_statistics_np(all_gt)
        fid = calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov)

        # Compute diversity
        pred_diversity = calculate_diversity_np(all_pred,
                                                self.diversity_times)
        gt_diversity = calculate_diversity_np(
            all_gt, self.diversity_times)

        return fid, pred_diversity, gt_diversity
