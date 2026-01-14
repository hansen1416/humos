from loguru import logger

import torch

from humos.src.data.text_motion import TextMotionDataset
from humos.src.initialize import initialize_dataloaders
from humos.utils.config import parse_args, run_grid_search_experiments


def inspect_one_motion(hparams, split="train", index=0):
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

    sample = dataset[index]

    logger.info("Loaded sample index={} keyid={}", index, sample.get("keyid"))
    logger.info("Text: {}", sample.get("text"))

    motion_x_dict = sample.get("motion_x_dict")
    if isinstance(motion_x_dict, dict):
        for key, value in motion_x_dict.items():

            print(f"{key}: ")

            if torch.is_tensor(value):
                print(value.shape)
            else:
                print(value)

    else:
        logger.error("motion_x_dict is not a dict")


if __name__ == "__main__":
    import sys

    if "--cfg" not in sys.argv:
        sys.argv.extend(["--cfg", "humos/configs/cfg_template_test.yml"])
    args = parse_args()
    hparams = run_grid_search_experiments(
        args,
        script="inspect_motion.py",
    )
    inspect_one_motion(hparams)
