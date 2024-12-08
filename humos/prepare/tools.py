import os
from glob import glob
import logging

import numpy as np
import torch
from humos.utils.rotation_conversions import (axis_angle_to_matrix, matrix_to_axis_angle,
                                                   matrix_to_euler_angles, euler_angles_to_matrix,
                                                   rotation_6d_to_matrix, matrix_to_rotation_6d)
from tqdm import tqdm

from humos.utils.misc_utils import copy2cpu as c2c
from humos.utils import constants

logger = logging.getLogger(__name__)


def loop_amass(
        base_folder,
        new_base_folder,
        ext=".npz",
        newext=".npz",
        force_redo=False,
        exclude=None,
):
    match_str = f"**/*{ext}"

    # for motion_file in tqdm(glob(match_str, root_dir=base_folder, recursive=True)):
    for motion_path in tqdm(glob(f"{base_folder}/**/{match_str}")):
        if exclude and exclude in motion_path:
            continue

        motion_file = os.path.relpath(motion_path, base_folder)
        # motion_path = os.path.join(base_folder, motion_file)

        if motion_path.endswith("shape.npz"):
            continue

        new_motion_path = os.path.join(
            new_base_folder, motion_file.replace(ext, newext)
        )
        if not force_redo and os.path.exists(new_motion_path):
            continue

        new_folder = os.path.split(new_motion_path)[0]
        os.makedirs(new_folder, exist_ok=True)

        yield motion_path, new_motion_path


def flip_pose(pose):
    """Flip pose.
    The flipping is based on SMPL parameters.
    """
    flipped_parts = constants.SMPLH_POSE_FLIP_PERM
    pose = pose[:, flipped_parts]
    # we also negate the second and the third dimension of the axis-angle
    pose[:, 1::3] = -pose[:, 1::3]
    pose[:, 2::3] = -pose[:, 2::3]
    return pose


def swap_left_right(data: dict):
    data_m = data.copy()

    # Flip the pose parameters
    # join root orient and pose
    pose = np.concatenate((data_m["root_orient"], data_m["pose_body"]), axis=-1)
    pose = flip_pose(pose)
    data_m["root_orient"] = pose[:, :3]
    data_m["pose_body"] = pose[:, 3:]

    # flip translation
    trans = data_m["trans"]
    x, y, z = np.split(trans, 3, axis=-1)
    mirrored_transl = np.stack((-x, y, z), axis=-1)
    data_m["trans"] = mirrored_transl[:, 0, :]
    return data_m


def take_out_z_rotation(root_orient, trans, in_format="aa"):
    # Look at normalize_root in AITViewer models/smpl.py

    # get root orientation along z-axis for first frame
    if in_format == "aa":
        first_frame_root_orient_z = root_orient[[0], :]
        first_frame_root_orient_z[:, :2] = 0
        # convert to rotmat
        first_frame_root_orient_z_rotmat = axis_angle_to_matrix(first_frame_root_orient_z)
    elif in_format == "6d":
        first_frame_root_orient_z = rotation_6d_to_matrix(root_orient[[0], :])
        first_frame_root_orient_z = matrix_to_euler_angles(first_frame_root_orient_z, "XYZ")
        first_frame_root_orient_z[:, :2] = 0
        first_frame_root_orient_z_rotmat = euler_angles_to_matrix(first_frame_root_orient_z, "XYZ")
    else:
        raise ValueError(f"Unknown input format: {in_format}")

    # invert
    first_frame_root_orient_z_rotmat_inv = torch.transpose(first_frame_root_orient_z_rotmat, -2, -1)
    first_frame_root_orient_z_rotmat_inv = first_frame_root_orient_z_rotmat_inv[:, :3, :3]
    # multiply all root orient by the inverse
    if in_format == "aa":
        root_orient_rotmat = axis_angle_to_matrix(root_orient)
    elif in_format == "6d":
        root_orient_rotmat = rotation_6d_to_matrix(root_orient)
    root_orient_rotmat = torch.einsum('bij,bjk->bik', first_frame_root_orient_z_rotmat_inv, root_orient_rotmat)
    # convert back to axis-angle
    if in_format == "aa":
        root_orient = matrix_to_axis_angle(root_orient_rotmat)
    elif in_format == "6d":
        root_orient = matrix_to_rotation_6d(root_orient_rotmat)

    # apply the same rotations to translations
    # trans has dimensions N x 3
    trans = trans[:, :, None]
    trans = torch.einsum('bij,bjk->bik', first_frame_root_orient_z_rotmat_inv, trans)[..., 0]
    return root_orient, trans


def canonicalize_smplh(data: dict, ground_the_human: bool = True, in_format: str = "aa"):
    # check if the data values are tensor and not of str type, if not, convert
    data = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) and k in constants.SMPL_KEYS else v for k, v in
            data.items()}

    # Note: input mesh is Z up
    # canonicalize the pose by taking out Z root rotation, apply the same
    # rotation correct to translations
    root_orient_key = 'root_orient' if in_format == "aa" else 'root_orient_6d'
    data[root_orient_key], data['trans'] = take_out_z_rotation(data[root_orient_key], data['trans'], in_format)

    # Subtract first frame x and y from all frames
    first_frame_xy = data['trans'][0, :2].clone()
    data['trans'][:, :2] -= first_frame_xy

    if ground_the_human:

        device = data['root_orient'].device

        gender = check_byte_string(data['gender'][0])
        if gender == 'male' or gender == 1:
            bm = constants.male_bm.to(device)
        elif gender == 'female' or gender == -1:
            bm = constants.female_bm.to(device)
        else:
            logger.error("Gender not recognized. Exiting.")
            import ipdb;
            ipdb.set_trace()

        verts, joints = bm(poses_body=data['pose_body'],
                           betas=data['betas'],
                           poses_root=data['root_orient'],
                           trans=data['trans'])

        # discard the sequence if the the lowest vertex is above 0.25 m from the floor
        # for more than 5 frames
        if (verts.min(dim=1).values[:, 2] > 0.25).sum() > 5:
            return None

        # Get the lowest vertex across all frame and put it beneath the floor by 2 cm
        # Adjust all frames by this amount
        floor_height = verts.min(dim=0).values.min(dim=0).values[2] + 0.02
        data['trans'][:, 2] -= floor_height

    return data


def check_byte_string(input_string):
    # Check if the input is a byte string
    if isinstance(input_string, bytes):
        # Attempt to decode using utf-8 or any other preferred encoding
        try:
            # Decode the byte string to a "normal" string (Unicode in Python 3)
            return input_string.decode('utf-8')
        except UnicodeDecodeError as e:
            # Handle decoding error (e.g., invalid utf-8 sequence)
            print(f"Error decoding byte string: {e}")
            return None
    elif isinstance(input_string, str):
        # Input is already a normal string (Unicode)
        return input_string
    else:
        # Input is neither bytes nor string (Unicode)
        print(f"Input is neither a byte string nor a normal string: {type(input_string)}")
        return None


def save_meshes(data, out_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gender = check_byte_string(data['gender'][0])
    if gender == 'male' or gender == 1:
        bm = constants.male_bm.to(device)
    elif gender == 'female' or gender == -1:
        bm = constants.female_bm.to(device)
    else:
        logger.error("Gender not recognized. Exiting.")
        import ipdb;
        ipdb.set_trace()

    smpl_output = bm(pose_body=torch.tensor(data['pose_body']).to(device),
                     # pose_hand=torch.tensor(data['pose_hand']).to(device),
                     betas=torch.tensor(data['betas']).to(device),
                     root_orient=torch.tensor(data['root_orient']).to(device))

    verts = smpl_output.v + torch.tensor(data['trans'][:, None, :]).to(device)
    joints = smpl_output.Jtr + torch.tensor(data['trans'][:, None, :]).to(device)

    # Visualize the meshes
    # save the mesh as an obj file
    for i, vert in enumerate(verts):
        import trimesh
        body_mesh = trimesh.Trimesh(vertices=c2c(vert), faces=c2c(bm.f),
                                    vertex_colors=np.tile([255, 200, 200, 255], (6890, 1)))
        body_mesh.export(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
        print(os.path.join(out_folder, f'body_mesh_{i:04d}.obj'))
