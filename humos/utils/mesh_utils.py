import torch
from humos.utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, \
    matrix_to_rotation_6d
from humos.utils.geometry import (integrate_root_diff_to_joints, apply_root_orient_diff)

def smplh_breakdown(data, fk=None):
    smpl_params = {}
    bs, fn, _ = data['betas'].shape

    smpl_params["betas"] = data['betas']
    smpl_params["gender"] = data['gender']

    # Get root_orient and pose_body
    if "root_orient_6d_root_rel" in data:
        root_orient_6d_root_rel = data["root_orient_6d_root_rel"]
    elif "root_orient_diff_6d_root_rel" in data:
        root_orient_diff_6d_root_rel = data["root_orient_diff_6d_root_rel"]
        root_orient_6d_root_rel = apply_root_orient_diff(root_orient_diff_6d_root_rel)
    else:
        raise ValueError("No root_orient_6d_root_rel or root_orient_diff_6d_root_rel found in data")
    root_orient_6d_root_rel = root_orient_6d_root_rel.view(bs, fn, -1, 6)
    root_orient_root_rel_mat = rotation_6d_to_matrix(root_orient_6d_root_rel)

    if "pose_6d_root_rel" in data:
        pose6d_root_rel_mat = data["pose_6d_root_rel"].view(bs, fn, -1, 6)
        pose6d_root_rel_mat = rotation_6d_to_matrix(pose6d_root_rel_mat)
    else:
        raise ValueError("No pose_6d_root_rel found in data")

    rot6d_root_rel_mat = torch.cat((root_orient_root_rel_mat, pose6d_root_rel_mat), dim=2)
    rot6d_mat = fk.global_to_local(rot6d_root_rel_mat.view(bs * fn, -1, 3, 3))
    pose = matrix_to_axis_angle(rot6d_mat)
    pose = pose.view(bs, fn, -1)
    smpl_params["root_orient"] = pose[:, :, :3]
    smpl_params["pose_body"] = pose[:, :, 3:66]

    # Get trans
    if "trans" in data:
        smpl_params["trans"] = data["trans"]
    elif "root_trans_diff" in data:
        root_trans_diff = data["root_trans_diff"]
        smpl_params["trans"] = integrate_root_diff_to_joints(root_trans_diff)
    else:
        raise ValueError("No trans or root_trans_diff found in data")
    return smpl_params

def smplh_consolidate(data):
    # This expects a single seq dict, not batched
    # concatenate body pose betas and root_orient. Keep betas at the end of the feature vector
    features = data['root_orient']
    features = torch.concatenate((features, data['pose_body']), dim=-1)
    # Convert axis-angle to 6d rotation representation
    # Unflatten the features to have last dimension of 3
    features = features.reshape(features.shape[0], -1, 3)
    features = axis_angle_to_matrix(torch.tensor(features) if not isinstance(features, torch.Tensor) else features)
    features = matrix_to_rotation_6d(features)
    # Flatten the features again
    features = features.reshape(features.shape[0], -1)
    features = torch.concatenate((features, data['transl']), dim=-1)
    features = torch.concatenate((features, data['betas']), dim=-1)
    features = torch.concatenate((features, data['gender']), dim=-1)
    return features
