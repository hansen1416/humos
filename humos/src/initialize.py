from loguru import logger
import torch

from humos.src.data.motion import AMASSMotionLoader, Normalizer
from humos.src.data.text import TokenEmbeddings, SentenceEmbeddings
from humos.src.model import TMR, CYCLIC_TMR, ACTORStyleEncoder, ACTORStyleDecoder


def initialize_dataloaders(hparams):
    motion_loader = AMASSMotionLoader(
        base_dir=hparams.TM_LOADER.MOTION_LOADER.BASE_DIR,
        fps=hparams.TM_LOADER.MOTION_LOADER.FPS,
        normalizer=Normalizer(
            base_dir=hparams.TM_LOADER.NORMALIZER.BASE_DIR,
            eps=hparams.TM_LOADER.NORMALIZER.EPS,
            disable=hparams.TM_LOADER.NORMALIZER.DISABLE,
        ),
        canonicalize_crops=hparams.TM_LOADER.MOTION_LOADER.CANONICALIZE_CROPS,
    )

    # text_to_send_emb = SentenceEmbeddings(
    #     modelname=hparams.TM_LOADER.SENTENCE_EMBEDDINGS.MODELNAME,
    #     path=hparams.TM_LOADER.SENTENCE_EMBEDDINGS.PATH,
    #     device=hparams.DEVICE,
    #     preload=hparams.TM_LOADER.SENTENCE_EMBEDDINGS.PRELOAD,
    #     disable=hparams.TM_LOADER.SENTENCE_EMBEDDINGS.DISABLE,
    # )
    text_to_send_emb = None

    # text_to_token_emb = TokenEmbeddings(
    #     modelname=hparams.TM_LOADER.TOKEN_EMBEDDINGS.MODELNAME,
    #     path=hparams.TM_LOADER.TOKEN_EMBEDDINGS.PATH,
    #     device=hparams.DEVICE,
    #     preload=hparams.TM_LOADER.TOKEN_EMBEDDINGS.PRELOAD,
    #     disable=hparams.TM_LOADER.TOKEN_EMBEDDINGS.DISABLE,
    # )

    text_to_token_emb = None
    return motion_loader, text_to_send_emb, text_to_token_emb


def initialize_model(hparams, ckpt_path=None, renderer=None):
    motion_encoder = ACTORStyleEncoder(
        nfeats=hparams.TMR.MOTION_ENCODER.NFEATS,
        vae=hparams.TMR.MOTION_ENCODER.VAE,
        latent_dim=hparams.TMR.MOTION_ENCODER.LATENT_DIM,
        ff_size=hparams.TMR.MOTION_ENCODER.FF_SIZE,
        num_layers=hparams.TMR.MOTION_ENCODER.NUM_LAYERS,
        num_heads=hparams.TMR.MOTION_ENCODER.NUM_HEADS,
        dropout=hparams.TMR.MOTION_ENCODER.DROPOUT,
        activation=hparams.TMR.MOTION_ENCODER.ACTIVATION,
    )

    pretrained_motion_encoder = ACTORStyleEncoder(
        nfeats=hparams.TMR.MOTION_ENCODER.NFEATS,
        vae=hparams.TMR.MOTION_ENCODER.VAE,
        latent_dim=hparams.TMR.MOTION_ENCODER.LATENT_DIM,
        ff_size=hparams.TMR.MOTION_ENCODER.FF_SIZE,
        num_layers=hparams.TMR.MOTION_ENCODER.NUM_LAYERS,
        num_heads=hparams.TMR.MOTION_ENCODER.NUM_HEADS,
        dropout=hparams.TMR.MOTION_ENCODER.DROPOUT,
        activation=hparams.TMR.MOTION_ENCODER.ACTIVATION,
    )

    if hparams.TRAINING.DROP_TEXT_ENCODER:
        text_encoder = None
    else:
        text_encoder = ACTORStyleEncoder(
            nfeats=hparams.TMR.TEXT_ENCODER.NFEATS,
            vae=hparams.TMR.TEXT_ENCODER.VAE,
            latent_dim=hparams.TMR.TEXT_ENCODER.LATENT_DIM,
            ff_size=hparams.TMR.TEXT_ENCODER.FF_SIZE,
            num_layers=hparams.TMR.TEXT_ENCODER.NUM_LAYERS,
            num_heads=hparams.TMR.TEXT_ENCODER.NUM_HEADS,
            dropout=hparams.TMR.TEXT_ENCODER.DROPOUT,
            activation=hparams.TMR.TEXT_ENCODER.ACTIVATION,
        )

    motion_decoder = ACTORStyleDecoder(
        nfeats=hparams.TMR.MOTION_DECODER.NFEATS,
        latent_dim=hparams.TMR.MOTION_DECODER.LATENT_DIM,
        ff_size=hparams.TMR.MOTION_DECODER.FF_SIZE,
        num_layers=hparams.TMR.MOTION_DECODER.NUM_LAYERS,
        num_heads=hparams.TMR.MOTION_DECODER.NUM_HEADS,
        dropout=hparams.TMR.MOTION_DECODER.DROPOUT,
        activation=hparams.TMR.MOTION_DECODER.ACTIVATION,
        add_gender=True if hparams.TRAINING.TMR_CYCLE else False,
    )

    normalizer = Normalizer(
        base_dir=hparams.TM_LOADER.NORMALIZER.BASE_DIR,
        eps=hparams.TM_LOADER.NORMALIZER.EPS,
        disable=hparams.TM_LOADER.NORMALIZER.DISABLE,
    )

    logger.info("Loading the model")
    if hparams.TRAINING.TMR_CYCLE:
        model = CYCLIC_TMR(
            motion_encoder=motion_encoder,
            motion_decoder=motion_decoder,
            pretrained_motion_encoder=pretrained_motion_encoder,
            normalizer=normalizer,
            vae=hparams.TMR.VAE,
            train_feats=hparams.TRAINING.TRAIN_FEATS,
            lmd={
                "recons": hparams.TMR.LOSSES.RECONS,
                "joint_recons": hparams.TMR.LOSSES.JOINT_RECONS,
                "recons_type": hparams.TMR.LOSSES.RECONS_TYPE,
                "latent": hparams.TMR.LOSSES.LATENT,
                "cycle_latent": hparams.TMR.LOSSES.CYCLE_LATENT,
                "kl": hparams.TMR.LOSSES.KL,
                "contrastive": hparams.TMR.LOSSES.CONTRASTIVE,
                "motion_prior": hparams.TMR.LOSSES.MOTION_PRIOR,
                "physics_float": hparams.TMR.LOSSES.PHYSICS_FLOAT,
                "physics_penetrate": hparams.TMR.LOSSES.PHYSICS_PENETRATE,
                "physics_skate": hparams.TMR.LOSSES.PHYSICS_SKATE,
                "dyn_stability": hparams.TMR.LOSSES.DYN_STABILITY,
            },
            compute_metrics={"recons": hparams.METRICS.RECONS,
                             "dyn_stability": hparams.METRICS.DYN_STABILITY,
                             "physics": hparams.METRICS.PHYSICS,
                             "motion_prior": hparams.METRICS.MOTION_PRIOR},
            bs=hparams.DATASET.BATCH_SIZE,
            lr=hparams.TMR.LR,
            fps=hparams.TM_LOADER.MOTION_LOADER.FPS,
            cop_w=hparams.BIOMECHANICS.COP_W,
            cop_k=hparams.BIOMECHANICS.COP_K,
            temperature=hparams.TMR.TEMPERATURE,
            threshold_selfsim=hparams.TMR.THRESHOLD_SELFSIM,
            threshold_selfsim_metrics=hparams.TMR.THRESHOLD_SELFSIM_METRICS,
            run_cycle=hparams.TMR.RUN_CYCLE,
            demo=hparams.DEMO,
            renderer=renderer,
        )
    else:
        model = TMR(
            motion_encoder=motion_encoder,
            motion_decoder=motion_decoder,
            normalizer=normalizer,
            vae=hparams.TMR.VAE,
            lmd={
                "recons": hparams.TMR.LOSSES.RECONS,
                "joint_recons": hparams.TMR.LOSSES.JOINT_RECONS,
                "recons_type": hparams.TMR.LOSSES.RECONS_TYPE,
                "latent": hparams.TMR.LOSSES.LATENT,
                "kl": hparams.TMR.LOSSES.KL,
                "contrastive": hparams.TMR.LOSSES.CONTRASTIVE,
            },
            lr=hparams.TMR.LR,
            fps=hparams.TM_LOADER.MOTION_LOADER.FPS,
            temperature=hparams.TMR.TEMPERATURE,
            threshold_selfsim=hparams.TMR.THRESHOLD_SELFSIM,
            threshold_selfsim_metrics=hparams.TMR.THRESHOLD_SELFSIM_METRICS,
        )

    # Pretrain the models if checkpoint is provided
    if ckpt_path is not None:
        logger.info("Loading the checkpoint for TMR")
        checkpoint = torch.load(ckpt_path)
        cktp_state_dict = checkpoint["state_dict"]
        for k, v in model.state_dict().items():
            if k in cktp_state_dict.keys():
                model.state_dict()[k].copy_(cktp_state_dict[k])
            else:
                ckpt_k = k.replace("motion_prior_loss_fn.pretrained_motion_encoder.", "motion_encoder.")
                if ckpt_k in cktp_state_dict.keys():
                    model.state_dict()[k].copy_(cktp_state_dict[ckpt_k])
                else:
                    logger.error(f"Key {k} not found in checkpoint")
    return model
