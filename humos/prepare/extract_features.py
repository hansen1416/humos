import logging
import numpy as np
import torch

from humos.utils import constants
from humos.prepare.tools import check_byte_string

from humos.utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
from humos.utils.geometry import (estimate_angular_velocity, estimate_root_diff, integrate_root_diff_to_joints,
                                       get_root_orient_diff, apply_root_orient_diff)
from humos.utils.misc_utils import visualize_joints


logger = logging.getLogger(__name__)



def get_input_features(data, fps=20, z_up=True, training=False):

    # use training = True if you want to use the training features

    features = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if training:
        gender = data['gender'][0]
        gender_feats = data['gender']
    else:
        gender = check_byte_string(data['gender'][0])

        if gender == 'male' or gender == 1:
            gender_feats = np.ones((data['root_orient'].shape[0], 1))
        elif gender == 'female' or gender == -1:
            gender_feats = -1 * np.ones((data['root_orient'].shape[0], 1))
        else:
            logger.error("Gender not recognized. Exiting.")
            import ipdb;
            ipdb.set_trace()

    if gender == 'male' or gender == 1:
        # bm = constants.male_bm.to(device)
        fk = constants.male_fk
    elif gender == 'female' or gender == -1:
        # bm = constants.female_bm.to(device)
        fk = constants.female_fk
    else:
        logger.error("Gender not recognized. Exiting.")
        import ipdb;
        ipdb.set_trace()

        # verts, joints = bm(poses_body=torch.tensor(data['pose_body']).to(device),
        #                  # pose_hand=torch.tensor(data['pose_hand']).to(device),
        #                  betas=torch.tensor(data['betas']).to(device),
        #                  poses_root=torch.tensor(data['root_orient']).to(device),
        #                  trans=torch.tensor(data['trans']).to(device),
        #                  skinning=True)

        # Get gender features. If gender is male, create a feature vector of 1s, else -1s


    # add smpl features
    features['root_orient'] = data['root_orient']
    features['pose_body'] = data['pose_body']
    features['betas'] = data['betas']
    features['trans'] = data['trans']
    features['gender'] = gender_feats

    if training:
        # concatenate body pose betas and root_orient. Keep betas at the end of the feature vector
        pose = data['root_orient']
        pose = torch.concatenate((pose, data['pose_body']), dim=-1)
        # Unflatten the features to have last dimension of 3 for axis-angle representation
        pose = pose.reshape(pose.shape[0], -1, 3)
        # Convert axis-angle to 6d rotation representation
        rotmat = axis_angle_to_matrix(pose) # rotation matrix (T, J, 3, 3)
        # features['pose_rotmat'] = rotmat.cpu()
        rot6d = matrix_to_rotation_6d(rotmat) # rotation 6d (T, J, 6)velocity
        # Flatten the features again
        # rot6d = rot6d.reshape(rot6d.shape[0], -1)
        # features['pose_6d'] = rot6d

        # # Get angular velocity from local pose
        # rot_seq = rotmat[None, ...].clone() # adding batch dim
        # angular = estimate_angular_velocity(rot_seq, dt=1.0 / fps)
        # angular = angular[0, ...] # angular velocity of all the joints (T, J, 3)
        # features['angular_parent_rel'] = angular.cpu()
        joints_root_rel, rotmat_root_rel = fk(rot6d.view(-1, constants.SMPLH_BODY_JOINTS, 6))

        # global poses (wrt root joint)
        rotmat_root_rel = rotmat_root_rel[..., :3, :3] # (T, J, 3, 3)
        # features['rotmat_root_rel'] = rotmat_root_rel.cpu()
        rot6d_root_rel = matrix_to_rotation_6d(rotmat_root_rel) # rotation 6d (T, J, 6)
        features['root_orient_6d_root_rel'] = rot6d_root_rel[:, 0, ...]
        features['pose_6d_root_rel'] = rot6d_root_rel[:, 1:, ...]

        # # Get joint positions
        # features['joints_root_rel'] = joints_root_rel.cpu()

        # # Get joint velocities
        # joint_seq = joints_root_rel[None, ...].clone() # adding batch dim
        # velocity = estimate_linear_velocity(joint_seq, dt=1.0 / fps)
        # velocity = velocity[0, ...] # linear velocity of all the joints (T, J, 3)
        # features['velocity_root_rel'] = velocity.cpu()

        # get joints from velocities using euler integration
        # joints_from_vel = integrate_velocity_to_joints(velocity[None, ...], dt=1.0 / fps, initial_joints=joint_seq[0, 0, ...])

        # Get all joint height (nemf style)
        # root_rotation = rotmat[:, [0], :, :]
        # root_rotation = root_rotation.repeat(1, constants.SMPLH_BODY_JOINTS, 1, 1)  # (T, J, 3, 3)
        # global_pos = torch.matmul(root_rotation, joints_root_rel.cpu().unsqueeze(-1)).squeeze(-1)
        # height = global_pos + trans[:, None, :]
        # height = height[:, :, 2] # (T, J)

        # Get root height
        root_height = data['trans'][:, 2] # (T, 1)
        # features['root_height'] = root_height

        # get root joint velocity
        trans_seq = data['trans'][None, ...].clone() # adding batch dim
        root_trans_diff = estimate_root_diff(trans_seq)
        root_trans_diff = root_trans_diff[0, ...] # linear velocity of root joints (T, 3)

        # replace first root_trans_diff with origin
        first_root_height = root_height[0, ...]
        first_root_height = first_root_height[None, ...]
        origin = torch.zeros(1, 3)
        # add the first frame root height to the origin
        origin[:, 2] = first_root_height
        root_trans_diff[0] = origin
        features['root_trans_diff'] = root_trans_diff

        # # # recover the trans from root_height and root_trans_diff
        # root_trans_diff = root_trans_diff[None, ...]
        # trans = integrate_root_diff_to_joints(root_trans_diff)
        # diff = torch.sum(trans - torch.tensor(data['trans']))
        # print(diff)

        # # Get RotMat differences for z root orient
        # import ipdb; ipdb.set_trace()
        # root_orient_6d = rot6d[:, [0], ...]
        # root_orient_6d = root_orient_6d[None, ...]
        # rot_orient_diff_6d = get_root_orient_diff(root_orient_6d)
        # rot_orient_diff_6d = rot_orient_diff_6d[0, ...] # (T, 3)
        # features['rot_orient_diff_6d'] = rot_orient_diff_6d
        #
        #
        # # recover the root_orient from root_orient_diff
        # # Get original root_orient from first_root_orient
        # first_root_orient = root_orient_6d[:, 0, ...]
        # rot_orient_diff_6d = rot_orient_diff_6d[None, ...]
        # new_root_orient_6d = apply_root_orient_diff(first_root_orient, rot_orient_diff_6d)
        # diff = torch.sum(new_root_orient_6d - root_orient_6d)
        # print(diff)

        # Get RotMat differences for z root orient (Root Relative)
        root_orient_6d_root_rel = rot6d_root_rel[:, 0, ...]
        root_orient_6d_root_rel = root_orient_6d_root_rel[None, ...]
        root_orient_diff_6d_root_rel = get_root_orient_diff(root_orient_6d_root_rel)
        root_orient_diff_6d_root_rel = root_orient_diff_6d_root_rel[0, ...]  # (T, 3)
        # replace first frame root_orient_diff with first_frame root_orient
        root_orient_diff_6d_root_rel[0] = root_orient_6d_root_rel[:, 0, ...]
        features['root_orient_diff_6d_root_rel'] = root_orient_diff_6d_root_rel


        # # recover the root_orient from root_orient_diff
        # # Get original root_orient from first_root_orient
        # root_orient_diff_6d_root_rel = root_orient_diff_6d_root_rel[None, ...]
        # new_root_orient_6d_root_rel = apply_root_orient_diff(root_orient_diff_6d_root_rel)
        # diff = torch.sum(new_root_orient_6d_root_rel - root_orient_6d_root_rel)
        # print(diff)
        # import ipdb; ipdb.set_trace()


        # # flattened all features to T x D
        # for k, v in features.items():
        #     features[k] = v.reshape(v.shape[0], -1)

        # remove some keys
        features.pop('root_orient')
        features.pop('pose_body')

    return features
