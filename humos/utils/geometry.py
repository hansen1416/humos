import torch
from humos.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d


def estimate_root_diff(data_seq):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for each timestep by taking frame differences.
    - h : step size
    '''
    # Compute consecutive frame differences using PyTorch's diff function
    diff_seq = torch.diff(data_seq, dim=1)

    # Append a zero velocity at the beginning to match the input size
    zero_vel = torch.zeros_like(diff_seq[:, -1:])
    vel_seq = torch.cat([zero_vel, diff_seq], dim=1)

    return vel_seq

def integrate_root_diff_to_joints(vel_seq):
    '''
    Given a sequence of velocities (vel_seq), where the first frame are the inital joints
    integrates the velocities to reconstruct the joint positions over time.
    '''
    # Calculate cumulative sum of velocities to get displacements
    displacements = torch.cumsum(vel_seq[:, 1:, :], dim=1)

    # add zero displacement at beginning
    zero_disp = torch.zeros_like(displacements[:, :1, :])
    displacements = torch.cat([zero_disp, displacements], dim=1)

    initial_joints = vel_seq[:, [0]]
    # Reconstruct the joint positions by adding displacements to the initial positions
    joint_positions = initial_joints + displacements

    return joint_positions

def get_root_orient_diff(root_orient_6d):
    """
    dim 0 is considered to be the time dimension, this is where the shift will happen
    the input root_orient_6d needs to be in 6d format
    root_orient_6d: (B, T, 3) in axis-angle format
    """
    # check if 6d
    if root_orient_6d.shape[-1] != 6:
        raise ValueError(f"specified conversion format is unsupported: {in_format}")

    rots1 = rotation_6d_to_matrix(root_orient_6d)
    rots2 = rots1
    rots1 = rots2.roll(1, 1)

    rots_diff = torch.einsum("...ij,...ik->...jk", rots1, rots2)  # Ri.T@R_i+1

    # Note the from above, first rots_diff will be crap

    rots_diff = matrix_to_rotation_6d(rots_diff)
    return rots_diff

def apply_root_orient_diff(deltas):
    """
    deltas: (B, T, 3) in  6d format
    Note: first frame is deltas is the inital root_orient and the remaining are the root_orient_diff
    """
    bs, T, _ = deltas.shape

    first_root_orient = deltas[:, 0]

    first_root_orient = rotation_6d_to_matrix(first_root_orient)
    deltas = rotation_6d_to_matrix(deltas)

    new_rots = [first_root_orient]

    for i in range(1, T):
        new_rots.append(torch.bmm(new_rots[-1], deltas[:, i]))

    new_rots = torch.stack(new_rots, dim=1)

    new_rots = matrix_to_rotation_6d(new_rots)
    return new_rots


def estimate_angular_velocity(rot_seq, dt):
    '''
    Given a batch of sequences of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (B, T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_linear_velocity(rot_seq, dt)
    R = rot_seq
    RT = R.transpose(-1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector by averaging symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)
    return w

def compute_trajectory_complex(velocity, up, origin, dt, up_axis='z'):
    """
    Inspired by Nemf code.
    Args:
        velocity: (B, T, 3)
        up: (B, T)
        origin: (B, 3)
        up_axis: x, y, or z

    Returns:
        trajectory: (B, T, 3)
    """
    ordermap = {
        'x': 0,
        'y': 1,
        'z': 2,
    }

    if up.ndim == 3:
        up = up[:, :, 0]

    origin = origin.unsqueeze(1)  # (B, 3) => (B, 1, 3)
    # trajectory = origin.repeat(1, up.shape[1], 1)  # (B, 1, 3) => (B, T, 3)
    trajectory = []

    # Get the first frame using root height in first frame
    first_frame = torch.zeros_like(origin).repeat(up.shape[0], 1, 1)
    first_frame[:, :, ordermap[up_axis]] = up[:, 0:1]
    trajectory.append(first_frame)

    # Handle the first velocity using forward difference
    second_frame = first_frame + velocity[:, 0:1] * dt
    trajectory.append(second_frame)

    # Iterate over the middle velocities (central difference)
    for i in range(1, velocity.shape[1] - 1):
        next_frame = velocity[:, i:i + 1] * 2 * dt + trajectory[i - 1]
        trajectory.append(next_frame)

    # Concatenate all joints along the time dimension
    trajectory = torch.cat(trajectory, dim=1)

    trajectory[:, :, ordermap[up_axis]] = up

    return trajectory

def estimate_linear_velocity_complex(data_seq, dt):
    '''
    Given some batched data sequences of T timesteps in the shape (B, T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    The first and last frames are with forward and backward first-order
    differences, respectively
    - h : step size
    '''
    # first steps is forward diff (t+1 - t) / dt
    init_vel = (data_seq[:, 1:2] - data_seq[:, :1]) / dt
    # middle steps are second order (t+1 - t-1) / 2dt
    middle_vel = (data_seq[:, 2:] - data_seq[:, 0:-2]) / (2 * dt)
    # last step is backward diff (t - t-1) / dt
    final_vel = (data_seq[:, -1:] - data_seq[:, -2:-1]) / dt

    vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    return vel_seq

def integrate_velocity_to_joints_complex(vel_seq, dt, initial_joints=None):
    '''
    Recover the original positions from the velocity sequence. Assumes velocities calculated using central difference for middle frames and forward/backward difference for first/last frames.
    - vel_seq: Sequence of velocities (B, T, ...)
    - dt: timestep
    - initial_position: The starting position (B, ...)
    '''
    B, T, _, _ = vel_seq.shape

    # Initialize the positions array with the initial joints
    while initial_joints.ndim != 4:
        initial_joints = initial_joints[None, ...]
    joints = [initial_joints]

    # Handle the first velocity using forward difference
    first_joint = initial_joints + vel_seq[:, 0:1] * dt
    joints.append(first_joint)

    # Iterate over the middle velocities (central difference)
    for i in range(1, vel_seq.shape[1] - 1):
        next_joint = vel_seq[:, i:i + 1] * 2 * dt + joints[i - 1]
        joints.append(next_joint)

    # Concatenate all joints along the time dimension
    joint_seq = torch.cat(joints, dim=1)

    return joint_seq
