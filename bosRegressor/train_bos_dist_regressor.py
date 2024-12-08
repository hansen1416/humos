import torch
import torch.nn as nn
import wandb
from loguru import logger
from bosRegressor.models.tipman import BoSDistMLP, BoSDistTransformer
from torch.utils.data import DataLoader

from bosRegressor.core.biomechanics import BiomechanicalEvaluator
from bosRegressor.core.trainer import BoSTrainer
from bosRegressor.datautils.build_datasets import BosDistDataset
from bosRegressor.utils.config import parse_args, run_grid_search_experiments
from bosRegressor.utils.constants import male_bm
from humos.src.data.motion import Normalizer
from humos.src.initialize import initialize_dataloaders


def get_bos_dataset(hparams, split):
    motion_loader, text_to_send_emb, text_to_token_emb = initialize_dataloaders(hparams)

    dataset = BosDistDataset(
        path=hparams.TM_LOADER.PATH,
        motion_loader=motion_loader,
        split=split,
        preload=hparams.TM_LOADER.PRELOAD,
    )

    return dataset


def main(hparams):
    # faces = male_bm.faces.astype(np.int64)
    faces = torch.tensor(male_bm.faces, dtype=torch.int64).to(hparams.DEVICE)
    biomechanical_evaluator = BiomechanicalEvaluator(faces=faces, fps=hparams.TM_LOADER.MOTION_LOADER.FPS,
                                                     stencil_size=hparams.BIOMECHANICS.STENCIL_SIZE,
                                                     exp_name=hparams.EXP_NAME,
                                                     debug=hparams.DEBUG, device=hparams.DEVICE)
    train_dataset = get_bos_dataset(hparams, split="all")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.DATASET.BATCH_SIZE,
        num_workers=hparams.DATASET.NUM_WORKERS,
        shuffle=True,
        drop_last=True,
    )

    val_dataset = get_bos_dataset(hparams, split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.DATASET.BATCH_SIZE,
        num_workers=hparams.DATASET.NUM_WORKERS,
        shuffle=False,
        drop_last=True,
    )

    # create model
    if hparams.TRAINING.MODEL_TYPE == 'mlp':
        bos_dist_model = BoSDistMLP(
            # input_size=31428,
            input_size=hparams.TRAINING.NUM_INPUT_FEATS,
            device=hparams.DEVICE,
        ).to(hparams.DEVICE)
    elif hparams.TRAINING.MODEL_TYPE == 'transformer':
        bod_dist_model = BoSDistTransformer(
            nhead=8,
            num_encoder_layers=3,
            d_model=256,
            num_intermediate_nodes=512,
            dropout=0.1,
            device=hparams.DEVICE,
        ).to(hparams.DEVICE)

    # create losses
    criterion = nn.MSELoss(reduction='none').to(hparams.DEVICE)

    # create Adam optimizer module
    optimizer = torch.optim.Adam(
        bos_dist_model.parameters(),
        lr=hparams.OPTIMIZER.LR,
        weight_decay=hparams.OPTIMIZER.WEIGHT_DECAY,
    )

    # to normalize and unnormalize data
    normalizer = Normalizer(
        base_dir=hparams.TM_LOADER.NORMALIZER.BASE_DIR,
        eps=hparams.TM_LOADER.NORMALIZER.EPS,
        disable=hparams.TM_LOADER.NORMALIZER.DISABLE,
    )

    trainer = BoSTrainer(hparams, bos_dist_model, criterion, optimizer, normalizer, biomechanical_evaluator)

    for epoch in range(hparams.TRAINING.NUM_EPOCHS):
        trainer.train(train_dataloader, epoch)
        trainer.validate(val_dataloader, epoch)


if __name__ == "__main__":
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script='train_bos_dist_regressor.py',
    )
    wandb.init(project="bosRegressor", config=hparams, name=hparams.EXP_NAME)
    logger.info(f'Running experiment: {hparams.EXP_NAME}')
    main(hparams)
