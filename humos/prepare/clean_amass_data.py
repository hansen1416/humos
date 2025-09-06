import glob
import os
import argparse
import time
import random
import shutil
from humos.prepare.tools import loop_amass

# which sub-datasets to clean up
BML_NTroje = True
MPI_HDM05 = True

def main(args):
    base_folder = args.data
    backup_folder = args.backup
    force_redo = args.force_redo

    iterator = loop_amass(
        base_folder, backup_folder, ext=".npz", newext=".npz", force_redo=force_redo
    )

    for motion_path, new_motion_path in iterator:
        if BML_NTroje and 'BioMotionLab_NTroje' in motion_path:
            # remove treadmill clips from BML_NTroje
            # contains treadmill_
            # contains normal_

            motion_name = motion_path.split('/')[-1]
            motion_type = motion_name.split('_')[1]
            # print(motion_type)
            if motion_type == 'treadmill' or motion_type == 'normal':
                shutil.move(motion_path, new_motion_path)
                print(f'moved {motion_path} to {new_motion_path}')


        if MPI_HDM05 and 'MPI_HDM05' in motion_path:
            # remove ice skating clips from MPI_HDM05
            # dg/HDM_dg_07-01* is inline skating
            if 'HDM_dg_07-01' in motion_path:
                shutil.move(motion_path, new_motion_path)
                print(f'moved {motion_path} to {new_motion_path}')

    # In backup folder, delete all subfolders that contain no files recursively
    for root, dirs, files in os.walk(backup_folder, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                print(f'removed empty folder {dir_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./datasets/pose_data',
                        help='Root dir of processed AMASS data')
    parser.add_argument('--backup', type=str, default='./datasets/pose_data_backup',
                        help='Root directory to save removed data to.')
    parser.add_argument("--force_redo", action="store_true")

    config = parser.parse_known_args()
    config = config[0]

    main(config)