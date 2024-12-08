from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule

from humos.src.model.losses import KLLoss, ReconLoss
from humos.utils.misc_utils import length_to_mask


class TEMOS(LightningModule):
    r"""TEMOS: Generating diverse human motions
    from textual descriptions
    Find more information about the model on the following website:
    https://mathis.petrovich.fr/temos

    Args:
        motion_encoder: a module to encode the input motion features in the latent space (required).
        motion_decoder: a module to decode the latent vector into motion features (required).
        vae: a boolean to make the model probabilistic (required).
        fact: a scaling factor for sampling the VAE (optional).
        sample_mean: sample the mean vector instead of random sampling (optional).
        lmd: dictionary of losses weights (optional).
        lr: learninig rate for the optimizer (optional).
    """

    def __init__(
        self,
        motion_encoder: nn.Module,
        motion_decoder: nn.Module,
        normalizer: nn.Module,
        vae: bool,
        fact: Optional[float] = None,
        sample_mean: Optional[bool] = False,
        lmd: Dict = {"recons": 1.0, "latent": 1.0e-5, "kl": 1.0e-5},
        compute_metrics: Dict = {"dyn_stability": False, "recons": True, "physics": True, "motion_prior": True},
        lr: float = 1e-4,
        fps: float =20.0,
    ) -> None:
        super().__init__()

        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.normalizer = normalizer

        # sampling parameters
        self.vae = vae
        self.fact = fact if fact is not None else 1.0
        self.sample_mean = sample_mean

        # losses
        self.reconstruction_loss_fn = ReconLoss()
        self.joint_reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.latent_loss_fn = torch.nn.SmoothL1Loss(reduction="mean")
        self.kl_loss_fn = KLLoss()

        # lambda weighting for the losses
        self.lmd = lmd
        self.compute_metrics = compute_metrics
        self.lr = lr

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def _find_encoder(self, inputs, modality):
        assert modality in ["text", "motion", "auto"]

        if modality == "text":
            return self.text_encoder
        elif modality == "motion":
            return self.motion_encoder

        m_nfeats = self.motion_encoder.nfeats

        # if m_nfeats == t_nfeats:
        #     raise ValueError(
        #         "Cannot automatically find the encoder, as they share the same input space."
        #     )

        nfeats = inputs.shape[-1]
        if nfeats == m_nfeats:
            return self.motion_encoder
        else:
            raise ValueError("The inputs is not recognized.")

    def encode(
        self,
        inputs,
        identity: Optional[Tensor] = None,
        modality: str = "motion",
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_distribution: bool = False,
    ):
        sample_mean = self.sample_mean if sample_mean is None else sample_mean
        fact = self.fact if fact is None else fact

        # Encode the inputs
        encoder = self._find_encoder(inputs, modality)
        if identity is not None:
            inputs["identity"] = identity
        encoded = encoder(inputs)

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

        if return_distribution:
            return latent_vectors, dists

        return latent_vectors

    def decode(
        self,
        latent_vectors: Tensor,
        identity: Tensor,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
    ):
        mask = mask if mask is not None else length_to_mask(lengths, device=self.device)
        z_dict = {"z": latent_vectors, "mask": mask, "identity": identity}
        motions = self.motion_decoder(z_dict)
        return motions

    # Forward: X => motions
    def forward(
        self,
        inputs,
        identity,
        lengths: Optional[List[int]] = None,
        mask: Optional[Tensor] = None,
        sample_mean: Optional[bool] = None,
        fact: Optional[float] = None,
        return_all: bool = False,
    ) -> List[Tensor]:
        # Encoding the inputs and sampling if needed
        latent_vectors, distributions = self.encode(
            inputs, identity, sample_mean=sample_mean, fact=fact, return_distribution=True
        )
        # Decoding the latent vector: generating motions
        motions = self.decode(latent_vectors, identity, lengths, mask)

        if return_all:
            return motions, latent_vectors, distributions

        return motions

    def compute_loss(self, batch: Dict) -> Dict:
        text_x_dict = batch["text_x_dict"]
        motion_x_dict = batch["motion_x_dict"]

        mask = motion_x_dict["mask"]
        ref_motions = motion_x_dict["x"]

        # Get shape parameters
        betas = ref_motions[:, 0, -11:-1]

        # motion -> motion
        m_motions, m_latents, m_dists = self(motion_x_dict, betas, mask=mask, return_all=True)

        # Store all losses
        losses = {}

        # Reconstructions losses
        # fmt: off

        losses["recons"] = (
            + self.reconstruction_loss_fn(m_motions[:, :, :-11],
                                          ref_motions[:, :, :-11])  # motion -> motion
        )

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

        # Weighted average of the losses
        losses["loss"] = sum(
            self.lmd[x] * val for x, val in losses.items() if x in self.lmd
        )

        # Weight all loss values with config weight
        losses = {x: self.lmd[x] * val if x in self.lmd else val for x, val in losses.items()}

        return losses

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["trans"])
        losses = self.compute_loss(batch,
                                   skinning=False)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"train_step/{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        bs = len(batch["motion_x_dict"]["x"])
        losses = self.compute_loss(batch)

        for loss_name in sorted(losses):
            loss_val = losses[loss_name]
            self.log(
                f"val_step/val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=True,
                batch_size=bs,
            )
        return losses["loss"]
