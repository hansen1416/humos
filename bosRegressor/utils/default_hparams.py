from yacs.config import CfgNode as CN

# Set default hparams to construct new default config
# Make sure the defaults are same as in parser
hparams = CN()

# General settings
hparams.EXP_NAME = 'default'
hparams.OUTPUT_DIR = 'logs/'
hparams.CONDOR_DIR = 'condor/'
hparams.LOG_DIR = 'logs/'
hparams.SEED = 1234
hparams.DEVICE = 'cuda'
hparams.VISUALIZE = False
hparams.DEBUG = False

# Input configurations
hparams.INPUT = CN()
hparams.INPUT.DATA = 'humanml3d'  # 'humanml3d' or 'kitml' or 'babel'
hparams.INPUT.PRESSURE_XML_FOLDER = 'data/MOYO/20220923_20220926_with_hands/pressure/train/xml'
hparams.INPUT.PRESSURE_CSV_FOLDER = 'data/MOYO/20220923_20220926_with_hands/pressure/train/single_csv'

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.BATCH_SIZE = 64
hparams.DATASET.NUM_WORKERS = 4

# Biomechanics params
hparams.BIOMECHANICS = CN()
hparams.BIOMECHANICS.STENCIL_SIZE = 3

# Training hparams
hparams.TRAINING = CN()
hparams.TRAINING.MODEL_TYPE = 'mlp'
hparams.TRAINING.NUM_INPUT_FEATS = 147
hparams.TRAINING.TRAIN_FEATS = ["pose_6d", "trans", "betas", "gender"]
hparams.TRAINING.NUM_EPOCHS = 50
hparams.TRAINING.SUMMARY_STEPS = 100
hparams.TRAINING.CHECKPOINT_EPOCHS = 5
hparams.TRAINING.NUM_EARLY_STOP = 10
hparams.TRAINING.BEST_MODEL_PATH = './best_model.pth'
hparams.TRAINING.LOSS_WEIGHTS = 1.

# Optimizer hparams
hparams.OPTIMIZER = CN()
hparams.OPTIMIZER.TYPE = 'adam'
hparams.OPTIMIZER.LR = 1e-4
hparams.OPTIMIZER.NUM_UPDATE_LR = 300
hparams.OPTIMIZER.LR_GAMMA = 0.9
hparams.OPTIMIZER.NUM_PREV_STEPS = 50
hparams.OPTIMIZER.MAX_ITERS = 100
hparams.OPTIMIZER.MAX_EPOCHS = 100
hparams.OPTIMIZER.WEIGHT_DECAY = 0.0001
hparams.OPTIMIZER.SLOPE_TOL = -1e-4

# Text Motion Loader hparams
hparams.TM_LOADER = CN()
hparams.TM_LOADER.PATH = f'../humos/annotations/{hparams.INPUT.DATA}'
hparams.TM_LOADER.PRELOAD = True

# Motion Loader hparams
hparams.TM_LOADER.MOTION_LOADER = CN()
hparams.TM_LOADER.MOTION_LOADER.BASE_DIR = '../datasets/guoh3dfeats'
hparams.TM_LOADER.MOTION_LOADER.FPS =20
hparams.TM_LOADER.MOTION_LOADER.NFEATS = 146
hparams.TM_LOADER.MOTION_LOADER.CANONICALIZE_CROPS = False

# Normalizer hparams
hparams.TM_LOADER.NORMALIZER = CN()
hparams.TM_LOADER.NORMALIZER.BASE_DIR = f'../humos/stats/{hparams.INPUT.DATA}/guo3dfeats'
hparams.TM_LOADER.NORMALIZER.EPS = 1e-12
hparams.TM_LOADER.NORMALIZER.DISABLE = False

