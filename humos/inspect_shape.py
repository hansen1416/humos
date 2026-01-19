import pickle
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# identity_pkl = "./datasets/splits/identity_dict_test_split_smpl.pkl"
# with open(identity_pkl, "rb") as f:
#     identity_dict_smpl = pickle.load(f)

"""
keyids_A: dict_keys(['keyid_B', 'betas_B', 'gender_B', 'identity_B', 'betas_B_norm', 'gender_B_norm', 'identity_B_norm'])

'identity_B_norm' are just -1 and 1
"""


def _stack_vectors(identity_dict: Dict[str, Dict[str, Any]], key: str) -> np.ndarray:
    """
    Extracts `entry[key]` for all entries in the identity_dict and stacks to (N, D).
    Handles list/tuple/np arrays and squeezes singleton dims.
    """
    betas = []
    for _, entry in identity_dict.items():

        beta = np.asarray(entry[key][0], dtype=np.float64).squeeze()

        betas.append(beta)

    # betas = np.array(betas)
    betas = np.stack(betas, axis=0)

    if betas.ndim != 2:
        raise ValueError(
            f"Expected stacked array to be 2D (N,D) for key='{key}', got shape {betas.shape}"
        )
    return betas


def estimate_mu_sigma_from_affine_norm(
    pkl_path: str = "./datasets/splits/identity_dict_test_split_smpl.pkl",
    raw_key: str = "betas_B",
    norm_key: str = "betas_B_norm",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Estimate per-dimension (mu, sigma) assuming the normalization used:
        norm = (raw - mu) / sigma
    equivalently:
        raw = sigma * norm + mu

    We estimate sigma and mu for each dimension via closed-form least squares:
        sigma = Cov(norm, raw) / Var(norm)
        mu    = E[raw] - sigma * E[norm]

    Returns:
        {
          "mu": (D,) np.ndarray,
          "sigma": (D,) np.ndarray,
          "rmse": (D,) np.ndarray,   # fit error per dim
          "r2": (D,) np.ndarray,     # fit quality per dim
          "raw_mean": (D,), "raw_std": (D,),
          "norm_mean": (D,), "norm_std": (D,),
          "n": int, "d": int,
        }
    """
    with open(pkl_path, "rb") as f:
        identity_dict = pickle.load(f)

    raw = _stack_vectors(identity_dict, raw_key)  # (N, D)

    print("min raw", np.min(raw))
    print("max raw", np.max(raw))

    norm = _stack_vectors(identity_dict, norm_key)  # (N, D)
    if raw.shape != norm.shape:
        raise ValueError(f"Shape mismatch: raw {raw.shape} vs norm {norm.shape}")

    # Means
    raw_mean = raw.mean(axis=0)
    norm_mean = norm.mean(axis=0)

    # Centered
    raw_c = raw - raw_mean
    norm_c = norm - norm_mean

    # Var(norm) and Cov(norm, raw) per dimension
    var_norm = (norm_c * norm_c).mean(axis=0)
    cov = (norm_c * raw_c).mean(axis=0)

    # sigma, guarding against degenerate dimensions
    sigma = cov / np.maximum(var_norm, eps)
    mu = raw_mean - sigma * norm_mean

    # Diagnostics: predict raw from norm and evaluate
    raw_hat = sigma[None, :] * norm + mu[None, :]
    resid = raw_hat - raw
    rmse = np.sqrt((resid * resid).mean(axis=0))

    # R^2 per dim
    ss_res = (resid * resid).sum(axis=0)
    ss_tot = ((raw - raw_mean) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, eps)

    return {
        "mu": mu,
        "sigma": sigma,
        "rmse": rmse,
        "r2": r2,
        "raw_mean": raw_mean,
        "raw_std": raw.std(axis=0),
        "norm_mean": norm_mean,
        "norm_std": norm.std(axis=0),
        "n": raw.shape[0],
        "d": raw.shape[1],
        "raw_key": raw_key,
        "norm_key": norm_key,
    }


# Example usage:
stats = estimate_mu_sigma_from_affine_norm(
    "./datasets/splits/identity_dict_test_split_smpl.pkl",
    raw_key="betas_B",
    norm_key="betas_B_norm",
)
print("mu:", stats["mu"])
print("sigma:", stats["sigma"])
print("mean RMSE:", stats["rmse"].mean(), "min R2:", stats["r2"].min())


def normalize_betas_np(
    betas_raw_10: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    betas_raw_10 = np.asarray(betas_raw_10, dtype=np.float32).reshape(-1)
    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(-1)
    assert betas_raw_10.shape[0] == mu.shape[0] == sigma.shape[0] == 10
    return (betas_raw_10 - mu) / (sigma + eps)


def normalize_betas_torch(
    betas_raw_10: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    betas_raw_10 = betas_raw_10.reshape(-1).to(dtype=torch.float32)
    mu = mu.reshape(-1).to(dtype=torch.float32, device=betas_raw_10.device)
    sigma = sigma.reshape(-1).to(dtype=torch.float32, device=betas_raw_10.device)
    assert betas_raw_10.numel() == mu.numel() == sigma.numel() == 10
    return (betas_raw_10 - mu) / (sigma + eps)


beta1 = torch.load("/home/hlz/repos/ASE/ase/data/assets/mjcf/smpl/0a1ece18_betas.pt")

beta1 = beta1[0].cpu().detach().numpy()
print(beta1)

beta1_norm = normalize_betas_np(beta1, stats["mu"], stats["sigma"])

print(beta1_norm)


# If you also want to estimate for the full identity vector (betas+gender), use:
# stats_id = estimate_mu_sigma_from_affine_norm(
#     pkl_path, raw_key="identity_B", norm_key="identity_B_norm"
# )

# # And for gender alone:
# stats_g = estimate_mu_sigma_from_affine_norm(
#     pkl_path, raw_key="gender_B", norm_key="gender_B_norm"
# )
