import logging
from tqdm import tqdm
import argparse
import torch
from humos.src.data.text_motion import TextMotionDataset
from humos.src.data.motion import AMASSMotionLoader, Normalizer
from humos.src.data.text import TokenEmbeddings, SentenceEmbeddings
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def motion_stats(args):
    logger.info("Computing motion stats")
    import humos.src.prepare  # noqa

    motion_loader = AMASSMotionLoader(
            base_dir='./datasets/humos3dfeats',
            fps =20.0,
            normalizer=Normalizer(
                base_dir=f'humos/stats/{args.data}/humos3dfeats',
                eps=1e-12,
                disable=True,
            ),
            canonicalize_crops=True,
    )

    text_to_send_emb = SentenceEmbeddings(
            modelname='sentence-transformers/all-mpnet-base-v2',
            path=f'humos/annotations/{args.data}',
            device=args.device,
            preload=True,
            disable=True,
        )

    text_to_token_emb = TokenEmbeddings(
            modelname='distilbert-base-uncased',
            path=f'humos/annotations/{args.data}',
            device=args.device,
            preload=True,
            disable=True,
        )

    train_dataset = TextMotionDataset(
        path=f'humos/annotations/{args.data}',
        motion_loader=motion_loader,
        text_to_sent_emb=text_to_send_emb,
        text_to_token_emb=text_to_token_emb,
        split="train",
        preload=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=0,
        # collate_fn=train_dataset.collate_fn,
        shuffle=False,
    )

    accumulator = {}
    for x in tqdm(train_dataloader):
        for k, v in x["motion_x_dict"].items():
            if k in 'length':
                continue
            if k not in accumulator:
                accumulator[k] = []
            accumulator[k].append(v)

    mean_feat_dict = {k: torch.stack(v).mean(0)[0] for k, v in accumulator.items()}
    std_feat_dict = {k: torch.stack(v).std(0)[0] for k, v in accumulator.items()}

    logging.info(f"Making the gender mean to be 0. Make sure gender is the last feature.")
    mean_feat_dict['gender'] *= 0.
    std_feat_dict['gender'] /= std_feat_dict['gender']

    normalizer = train_dataset.motion_loader.normalizer
    logger.info(f"Saving them in {normalizer.base_dir}")
    normalizer.save(mean_feat_dict, std_feat_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="humanml3d", choices=["humanml3d", "kitml"])
    args = parser.parse_args()
    motion_stats(args)
