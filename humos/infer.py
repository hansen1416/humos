import os

import wandb
from aitviewer.headless import HeadlessRenderer
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from humos.src.callback import ProgressLogger, TQDMProgressBar
from humos.src.data.text_motion import TextMotionDataset
from humos.src.initialize import initialize_dataloaders, initialize_model
from humos.utils.config import parse_args, run_grid_search_experiments

os.system("Xvfb :13 -screen 1 640x480x24 &")
os.environ["DISPLAY"] = ":13"

import pytorch_lightning as pl


def get_text_motion_dataset(hparams, split):
    motion_loader, text_to_send_emb, text_to_token_emb = initialize_dataloaders(hparams)

    dataset = TextMotionDataset(
        path=hparams.TM_LOADER.PATH,
        motion_loader=motion_loader,
        text_to_sent_emb=text_to_send_emb,
        text_to_token_emb=text_to_token_emb,
        split=split,
        preload=hparams.TM_LOADER.PRELOAD,
        demo=hparams.DEMO,
    )

    return dataset


def test(hparams):
    # TODO: remove hydra dependencies from here
    # # Resuming if needed
    ckpt = hparams.RESUME_CKPT
    if ckpt is not None:
        ckpt = os.path.join(hparams.OUTPUT_DIR, ckpt)
        logger.info("Resuming training")
        logger.info(f"The checkpoint is loaded from: \n{ckpt}")
    else:
        raise ValueError("No checkpoint provided")

    pl.seed_everything(hparams.SEED)

    logger.info("Loading the dataloaders")

    val_dataset = get_text_motion_dataset(hparams, split="test")

    print(len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.DATASET.BATCH_SIZE,
        num_workers=hparams.DATASET.NUM_WORKERS,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
        drop_last=True,
    )

    # ait_renderer = HeadlessRenderer(size=(320, 240))
    ait_renderer = None
    # Get the TMR model
    model = initialize_model(hparams, ckpt, renderer=ait_renderer)

    logger.info("Training")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=hparams.TRAINING.DEVICES,  # e.g., 1 or [0]
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[],  # or keep only what you truly need
        num_sanity_val_steps=0,  # irrelevant for predict, but keep safe
        limit_train_batches=0,  # optional; makes intent explicit
        limit_val_batches=0,  # optional
    )

    preds = trainer.predict(model, dataloaders=val_dataloader, return_predictions=False)

    print(preds)


if __name__ == "__main__":
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script="train.py",
    )
    wandb.init(project="humos", config=hparams, name=hparams.EXP_NAME)
    logger.info(f"Running experiment: {hparams.EXP_NAME}")
    test(hparams)
