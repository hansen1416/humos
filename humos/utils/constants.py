from aitviewer.models.smpl import SMPLLayer
from humos.utils.fk import ForwardKinematicsLayer

SMPL_KEYS = ["root_orient", "pose_body", "pose_hand", "betas", "trans"]

male_bm = SMPLLayer(model_type="smplh", gender="male")
female_bm = SMPLLayer(model_type="smplh", gender="female")

SMPLH_BODY_JOINTS = 22

SMPL_PATH = "./body_models/smpl"

male_fk = ForwardKinematicsLayer(SMPL_PATH, gender="male", num_joints=SMPLH_BODY_JOINTS, device='cpu')
female_fk = ForwardKinematicsLayer(SMPL_PATH, gender="female", num_joints=SMPLH_BODY_JOINTS, device='cpu')

SMPLH_FOOT_JOINTS = [7, 10, 8, 11]# left foot, left-toe base, right foot, right-toe base

# Permutation of SMPL pose parameters when flipping the shape
SMPLH_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
SMPLH_POSE_FLIP_PERM = []
for i in SMPLH_JOINTS_FLIP_PERM:
    SMPLH_POSE_FLIP_PERM.append(3 * i)
    SMPLH_POSE_FLIP_PERM.append(3 * i + 1)
    SMPLH_POSE_FLIP_PERM.append(3 * i + 2)
