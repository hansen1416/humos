import torch
import torch.nn.functional as F6
from torch import nn
from humos.utils.rotation_conversions import rotation_6d_to_matrix


# For reference
# https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence
class KLLoss:
    def __call__(self, q, p):
        mu_q, logvar_q = q
        mu_p, logvar_p = p

        log_var_ratio = logvar_q - logvar_p
        t1 = (mu_p - mu_q).pow(2) / logvar_p.exp()
        div = 0.5 * (log_var_ratio.exp() + t1 - 1 - log_var_ratio)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"

class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2, epsilon=1e-7):
        """ Compute the geodesic distance between two rotation matrices.
        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.permute(0, 2, 1))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        # cos = (m.diagonal(dim1=-2, dim2=-1).sum(-1) -1) /2
        # cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        # cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
        cos = torch.clamp(cos, -1 + epsilon, 1 - epsilon)
        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')


class ReconLoss(nn.Module):
    def __init__(self):
        super(ReconLoss, self).__init__()
        self.SmoothL1Recon = torch.nn.SmoothL1Loss(reduction="mean")
        self.GeoLoss = GeodesicLoss()

    def __call__(self, input_data, recon_data):
        loss = 0
        for k, v in recon_data.items():
            if 'pose_6d' in k:
                bs, T, _, _ = recon_data[k].shape
                rotmat = rotation_6d_to_matrix(recon_data[k].view(bs, T, -1, 6))
                rotmat_gt = rotation_6d_to_matrix(input_data[k].view(bs, T, -1, 6))
                recon_loss = self.GeoLoss(rotmat.view(-1, 3, 3), rotmat_gt.view(-1, 3, 3))
                loss += recon_loss
            else:
                recon_loss = self.SmoothL1Recon(v, input_data[k])
                loss += recon_loss
        return loss

    def __repr__(self):
        return f"ReconLoss()"


class MotionPriorLoss(nn.Module):
    def __init__(self, pretrained_motion_encoder,
                 fact = None,
                 sample_mean = False):
        super(MotionPriorLoss, self).__init__()

        self.SmoothL1Recon = torch.nn.SmoothL1Loss(reduction="mean")

        self.pretrained_motion_encoder = pretrained_motion_encoder
        # Set pretrained motion encoder to frozen and eval mode
        for param in self.pretrained_motion_encoder.parameters():
            param.requires_grad = False
        self.pretrained_motion_encoder.eval()

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

    def __call__(self, inputs_A, inputs_B):
        # Encoding the inputs and sampling if needed
        latent_vectors_A = self.encode(inputs_A)
        latent_vectors_B = self.encode(inputs_B)

        # Computing the loss
        loss = self.SmoothL1Recon(latent_vectors_A, latent_vectors_B)
        return loss

    def __repr__(self):
        return f"MotionPriorLoss()"

class DynStabilityLoss(nn.Module):
    def __init__(self, gmof_rho=30):
        """
        Initializes the dynamic stability loss module.
        """
        super(DynStabilityLoss, self).__init__()

        self.gmof_rho = gmof_rho
        self.epsilon = 1e-8

    def gmof(self, residual):
        """
        Applies the GMOF transformation to the residuals.
        The transformation is designed to be less sensitive to outliers than the traditional squared error.
        """
        squared_res = residual ** 2
        # Adding epsilon for numerical stability
        dist = torch.div(squared_res, squared_res + self.gmof_rho ** 2 + self.epsilon)
        # Note: The loss is scaled by gmof_rho ** 2, which affects the magnitude of the loss and optimization dynamics.
        return self.gmof_rho ** 2 * dist

    def forward(self, inputs_A, inputs_B):
        """
        Computes the loss between two inputs using the GMOF transformation on their residuals.

        Parameters:
        - inputs_A (Tensor): The first input tensor.
        - inputs_B (Tensor): The second input tensor, to be compared against inputs_A.
        """
        residual = inputs_A - inputs_B
        loss = self.gmof(residual)
        # zero out nans in loss.. these happens due to singularity instability in zmps when upward acceleration is 1G
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = loss.mean()
        return loss

    def __repr__(self):
        return f"DynStabilityLoss(gmof_rho={self.gmof_rho})"



class InfoNCE_with_filtering(nn.Module):
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        super(InfoNCE_with_filtering, self).__init__()
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def get_sim_matrix(self, x, y):
        x_logits = torch.nn.functional.normalize(x, dim=-1)
        y_logits = torch.nn.functional.normalize(y, dim=-1)
        sim_matrix = x_logits @ y_logits.T
        return sim_matrix

    def __call__(self, x, y, sent_emb=None):
        bs, device = len(x), x.device
        sim_matrix = self.get_sim_matrix(x, y) / self.temperature

        if sent_emb is not None and self.threshold_selfsim:
            # put the threshold value between -1 and 1
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            # Filtering too close values
            # mask them by putting -inf in the sim_matrix
            selfsim = sent_emb @ sent_emb.T
            selfsim_nodiag = selfsim - selfsim.diag().diag()
            idx = torch.where(selfsim_nodiag > real_threshold_selfsim)
            sim_matrix[idx] = -torch.inf

        labels = torch.arange(bs, device=device)

        total_loss = (
            F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
        ) / 2

        return total_loss

    def __repr__(self):
        return f"Constrastive(temp={self.temp})"
