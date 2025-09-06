import argparse
import logging
import os

import numpy as np

from humos.prepare.extract_features import get_input_features
from humos.prepare.tools import (loop_amass, swap_left_right, save_meshes, canonicalize_smplh)
from humos.utils import constants
import torch

logger = logging.getLogger(__name__)


def extract_h3d(feats):
    from einops import unpack

    root_data, ric_data, rot_data, local_vel, feet_l, feet_r = unpack(
        feats, [[4], [63], [126], [66], [2], [2]], "i *"
    )
    return root_data, ric_data, rot_data, local_vel, feet_l, feet_r


def compute_3dfeats(args):
    base_folder = args.base_folder  # datasets/pose_data
    output_folder = args.output_folder  # datasets/humos3dfeats
    force_redo = args.force_redo

    output_folder_M = os.path.join(output_folder, "M")

    print("Get h3d features")
    print("The processed motions will be stored in this folder:")
    print(output_folder)

    iterator = loop_amass(
        base_folder, output_folder, ext=".npz", newext=".tensor", force_redo=force_redo
    )

    # count = 0
    for motion_path, new_motion_path in iterator:
        # count += 1
        # if count < 4265:
        #     continue

        # replace / in path with _
        motion_name = '_'.join(motion_path.split('/')[2:]).replace('.npz', '')

        data = dict(np.load(motion_path, allow_pickle=True))

        # rename 'transl' key to 'trans' if exists
        if 'transl' in data:
            data['trans'] = data.pop('transl')

        # if only 1 frame, then skip
        if len(data["trans"]) == 1:
            continue

        # out_dir = f'./meshes/{motion_name}'
        # os.makedirs(out_dir, exist_ok=True)
        # save_meshes(data, out_dir)

        data_m = swap_left_right(data)
        # out_dir = './meshes_m'
        # os.makedirs(out_dir, exist_ok=True)
        # save_meshes(data_m, out_dir)

        # canonicalize all meshes
        data = canonicalize_smplh(data, in_format='aa')
        if data is None:
            continue
        # convert data values to numpy
        data = {k: v.cpu().numpy() if k in constants.SMPL_KEYS else v for k, v in data.items()}
        # out_dir = f'./meshes_can/{motion_name}'
        # os.makedirs(out_dir, exist_ok=True)
        # save_meshes(data, out_dir)

        data_m = canonicalize_smplh(data_m, in_format='aa')
        if data_m is None:
            continue
        # convert data values to numpy
        data_m = {k: v.cpu().numpy() if k in constants.SMPL_KEYS else v for k, v in data_m.items()}

        # out_dir = f'./meshes_can_m/{motion_name}'
        # os.makedirs(out_dir, exist_ok=True)
        # save_meshes(data_m, out_dir)
        # try:
        features = get_input_features(data, fps=args.fps, training=False)
        features_m = get_input_features(data_m, fps=args.fps, training=False)
        # except (IndexError, ValueError):
        #     import ipdb;
        #     ipdb.set_trace()
        # The sequence should be only 1 frame long
        # so we cannot compute features (which involve velocities etc)
        # assert len(data["poses"]) == 1
        # continue

        # Todo: discard jumping and stair climbing using the text prompts

        # save the features
        torch.save(features, new_motion_path)

        # save the mirrored features as well
        new_motion_path_M = new_motion_path.replace(output_folder, output_folder_M)
        os.makedirs(os.path.split(new_motion_path_M)[0], exist_ok=True)
        torch.save(features_m, new_motion_path_M)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, default="./datasets/pose_data")
    parser.add_argument("--output_folder", type=str, default="./datasets/humos3dfeats")
    parser.add_argument("--fps", type=int, required=True, default=20, help="fps of the input motion")
    parser.add_argument("--force_redo", action="store_true")
    args = parser.parse_args()

    compute_3dfeats(args)
