from aitviewer.models.smpl import SMPLLayer

ESSENTIALS_DIR = './bosRegressor/data/essentials'

PRESSURE_THRESH = 0.9
CONTACT_THRESH = 0.05
VELOCITY_THRESH = 0.02 # obtained by visualizing foot joint velocities
SMPLH_FOOT_JOINTS = [7, 10, 8, 11]# left foot, left-toe base, right foot, right-toe base


SMPL_KEYS = ["root_orient", "pose_body", "pose_hand", "betas", "trans"]

male_bm = SMPLLayer(model_type="smplh", gender="male")

SMPLH_BODY_JOINTS = 22

SMPL_PATH = "./body_models/smpl"

NUM_BETAS = 10

# Permutation of SMPL pose parameters when flipping the shape
SMPLH_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
SMPLH_POSE_FLIP_PERM = []
for i in SMPLH_JOINTS_FLIP_PERM:
    SMPLH_POSE_FLIP_PERM.append(3 * i)
    SMPLH_POSE_FLIP_PERM.append(3 * i + 1)
    SMPLH_POSE_FLIP_PERM.append(3 * i + 2)
