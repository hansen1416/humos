import numpy as np
import torch

colors = {
    'pink': [.6, .0, .4],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .2, .1],
    'brown-light': [0.654, 0.396, 0.164],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [1., .2, 0],

    'grey': [.7, .7, .7],
    'grey-blue': [0.345, 0.580, 0.713],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}
def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def compute_accel_naive(points):
    """
        Computes acceleration of 3D points.
        Args:
            joints (N).
        Returns:
            Accelerations (N-2).
    """
    velocities = points[1:] - points[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = torch.linalg.norm(acceleration, dim=2)
    return torch.mean(acceleration_normed, dim=1)

def compute_finite_differences(points, stencil):
    """
        Computes acceleration of 3D points usign one shot finite differences. Is more stable.
        https://en.wikipedia.org/wiki/Finite_difference_coefficient
        Instead of using conv1d, we use conv2d and treat the tensor points with shape [nframes, npoints, 3] as an image of size [nframes, npoints * 3]
        Args:
            points (N).
        Returns:
            Accelerations (N-2).
    """
    bs, nf, np, _ = points.shape
    gv = torch.nn.functional.conv2d(points.reshape(bs, nf, -1)[:, None, ...],
                                    stencil[None, None, ..., None])
    gv = gv[:, 0].reshape(bs, gv.shape[2], -1, 3)
    return gv

def calculate_zmps_sliding_window(points, stencil):
    stencil_size = stencil.shape[0]
    g = []
    for i in range(points.shape[0] - stencil_size + 1):
        g.append((points[i:i + stencil_size] * stencil[..., None, None]).sum(dim=-3))
    g = torch.stack(g)
    return g

# Apply Gaussian filter
def gaussian_filter(points, sigma=5):
    """
        Instead of using conv1d, we use conv2d and treat the tensor points with shape [nframes, npoints, 3] as an image of size [nframes, npoints * 3]
        Args:
            points (N).
        Returns:
            smoothed_points (N).
    """

    # Calculate the Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    x = torch.linspace(-3 * sigma, 3 * sigma, kernel_size)
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    # kernel_size = 2 * sigma + 1
    # gaussian_kernel = torch.tensor(
    #     [torch.exp(-(x - sigma) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
    gaussian_kernel = gaussian_kernel.to(points.device).to(points.dtype)

    # Normalize the kernel
    gaussian_kernel /= gaussian_kernel.sum()

    smoothed_data = torch.nn.functional.conv2d(points.reshape(points.shape[0], -1)[None, None, ...],
                                    gaussian_kernel[None, None, ..., None], padding='same')
    smoothed_data = smoothed_data[0, 0].reshape(smoothed_data.shape[2], -1, 3)
    return smoothed_data

def gaussian_filter_params(params, sigma=5):
    """
        Instead of using conv1d, we use conv2d and treat the tensor params with shape [nframes, nparams] as an image of size [nframes, nparams]
        Args:
            params (N).
        Returns:
            smoothed_params (N).
    """

    # Calculate the Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    x = torch.linspace(-3 * sigma, 3 * sigma, kernel_size)
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    # kernel_size = 2 * sigma + 1
    # gaussian_kernel = torch.tensor(
    #     [torch.exp(-(x - sigma) ** 2 / float(2 * sigma ** 2)) for x in range(kernel_size)])
    gaussian_kernel = gaussian_kernel.to(params.device).to(params.dtype)

    # Normalize the kernel
    gaussian_kernel /= gaussian_kernel.sum()

    smoothed_data = torch.nn.functional.conv2d(params.reshape(params.shape[0], -1)[None, None, ...],
                                    gaussian_kernel[None, None, ..., None], padding='same')
    smoothed_data = smoothed_data[0, 0]
    return smoothed_data