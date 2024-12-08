from yacs.config import CfgNode as CN

# Set default hparams to construct new default config
# Make sure the defaults are same as in parser
hparams = CN()

# General settings
hparams.EXP_NAME = 'default'
hparams.OUTPUT_DIR = 'logs/'
hparams.CONDOR_DIR = 'condor/'
hparams.LOG_DIR = 'logs/'
hparams.RESUME_CKPT = None
hparams.SEED = 1234
hparams.DEVICE = 'cuda'
hparams.VISUALIZE = False
hparams.DEBUG = False

# Input configurations
hparams.INPUT = CN()
hparams.INPUT.DATA = 'humanml3d'  # 'humanml3d' or 'kitml' or 'babel'

# Dataset hparams
hparams.DATASET = CN()
hparams.DATASET.BATCH_SIZE = 32
hparams.DATASET.NUM_WORKERS = 8

# Checkpointing hparams
hparams.CHECKPOINT = CN()
hparams.CHECKPOINT.EVERY_N_EPOCHS = 100
hparams.CHECKPOINT.SAVE_TOP_K = -1
hparams.CHECKPOINT.PRECISION = 3

# Training hparams
hparams.TRAINING = CN()
hparams.TRAINING.MAX_EPOCHS = 1000
hparams.TRAINING.LOG_EVERY_N_STEPS = 50
hparams.TRAINING.NUM_SANITY_VAL_STEPS = 0
hparams.TRAINING.CHECK_VAL_EVERY_N_EPOCH = 1
hparams.TRAINING.DEVICES = 1
hparams.TRAINING.DROP_TEXT_ENCODER = False
hparams.TRAINING.TMR_CYCLE = False
hparams.TRAINING.TRAIN_FEATS = ["pose_6d", "trans", "betas", "gender"]

# Text Motion Loader hparams
hparams.TM_LOADER = CN()
hparams.TM_LOADER.PATH = f'annotations/{hparams.INPUT.DATA}'
hparams.TM_LOADER.PRELOAD = True

# Motion Loader hparams
hparams.TM_LOADER.MOTION_LOADER = CN()
hparams.TM_LOADER.MOTION_LOADER.BASE_DIR = '../datasets/guoh3dfeats'
hparams.TM_LOADER.MOTION_LOADER.FPS =20
hparams.TM_LOADER.MOTION_LOADER.NFEATS = 146
hparams.TM_LOADER.MOTION_LOADER.CANONICALIZE_CROPS = False

# Normalizer hparams
hparams.TM_LOADER.NORMALIZER = CN()
hparams.TM_LOADER.NORMALIZER.BASE_DIR = f'stats/{hparams.INPUT.DATA}/guo3dfeats'
hparams.TM_LOADER.NORMALIZER.EPS = 1e-12
hparams.TM_LOADER.NORMALIZER.DISABLE = False

# Sentence Embeddings hparams
hparams.TM_LOADER.SENTENCE_EMBEDDINGS = CN()
hparams.TM_LOADER.SENTENCE_EMBEDDINGS.MODELNAME = 'sentence-transformers/all-mpnet-base-v2'
hparams.TM_LOADER.SENTENCE_EMBEDDINGS.PATH = f'annotations/{hparams.INPUT.DATA}'
hparams.TM_LOADER.SENTENCE_EMBEDDINGS.PRELOAD = True
hparams.TM_LOADER.SENTENCE_EMBEDDINGS.DISABLE = False

# Token Embeddings hparams
hparams.TM_LOADER.TOKEN_EMBEDDINGS = CN()
hparams.TM_LOADER.TOKEN_EMBEDDINGS.MODELNAME = 'distilbert-base-uncased'
hparams.TM_LOADER.TOKEN_EMBEDDINGS.PATH = f'annotations/{hparams.INPUT.DATA}'
hparams.TM_LOADER.TOKEN_EMBEDDINGS.PRELOAD = True
hparams.TM_LOADER.TOKEN_EMBEDDINGS.DISABLE = False

# TMR hparams
hparams.TMR = CN()
hparams.TMR.RUN_CYCLE = True
hparams.TMR.VAE = True
hparams.TMR.LR = 1e-4
hparams.TMR.TEMPERATURE = 0.1
hparams.TMR.THRESHOLD_SELFSIM = 0.80
hparams.TMR.THRESHOLD_SELFSIM_METRICS = 0.95

# Motion Encoder hparams
hparams.TMR.MOTION_ENCODER = CN()
hparams.TMR.MOTION_ENCODER.NFEATS = 146
hparams.TMR.MOTION_ENCODER.VAE = True
hparams.TMR.MOTION_ENCODER.LATENT_DIM = 256
hparams.TMR.MOTION_ENCODER.FF_SIZE = 1024
hparams.TMR.MOTION_ENCODER.NUM_LAYERS = 6
hparams.TMR.MOTION_ENCODER.NUM_HEADS = 4
hparams.TMR.MOTION_ENCODER.DROPOUT = 0.1
hparams.TMR.MOTION_ENCODER.ACTIVATION = "gelu"

# Text Encoder hparams
hparams.TMR.TEXT_ENCODER = CN()
hparams.TMR.TEXT_ENCODER.NFEATS = 768
hparams.TMR.TEXT_ENCODER.VAE = True
hparams.TMR.TEXT_ENCODER.LATENT_DIM = 256
hparams.TMR.TEXT_ENCODER.FF_SIZE = 1024
hparams.TMR.TEXT_ENCODER.NUM_LAYERS = 6
hparams.TMR.TEXT_ENCODER.NUM_HEADS = 4
hparams.TMR.TEXT_ENCODER.DROPOUT = 0.1
hparams.TMR.TEXT_ENCODER.ACTIVATION = "gelu"

# Motion Decoder hparams
hparams.TMR.MOTION_DECODER = CN()
hparams.TMR.MOTION_DECODER.NFEATS = 146
hparams.TMR.MOTION_DECODER.LATENT_DIM = 256
hparams.TMR.MOTION_DECODER.FF_SIZE = 1024
hparams.TMR.MOTION_DECODER.NUM_LAYERS = 6
hparams.TMR.MOTION_DECODER.NUM_HEADS = 4
hparams.TMR.MOTION_DECODER.DROPOUT = 0.1
hparams.TMR.MOTION_DECODER.ACTIVATION = "gelu"

# Loss weights
hparams.TMR.LOSSES = CN()
hparams.TMR.LOSSES.RECONS = 1.0
hparams.TMR.LOSSES.JOINT_RECONS = 1.0
hparams.TMR.LOSSES.RECONS_TYPE = 'rotations'
hparams.TMR.LOSSES.LATENT = 1.0e-5
hparams.TMR.LOSSES.CYCLE_LATENT = 1.0e-5
hparams.TMR.LOSSES.KL = 1.0e-5
hparams.TMR.LOSSES.CONTRASTIVE = 0.1
hparams.TMR.LOSSES.MOTION_PRIOR = 1.0
hparams.TMR.LOSSES.PHYSICS_FLOAT = 1.0
hparams.TMR.LOSSES.PHYSICS_PENETRATE = 1.0
hparams.TMR.LOSSES.PHYSICS_SKATE = 1.0
hparams.TMR.LOSSES.DYN_STABILITY = 1.0

# Biomechanics hparams
hparams.BIOMECHANICS = CN()
hparams.BIOMECHANICS.COP_W = 30
hparams.BIOMECHANICS.COP_K = 100

# Metrics
hparams.METRICS = CN()
hparams.METRICS.DYN_STABILITY = False
hparams.METRICS.RECONS = True
hparams.METRICS.PHYSICS = True
hparams.METRICS.MOTION_PRIOR = True
