import logging
import argparse

logger = logging.getLogger(__name__)


def text_embeddings(args):
    device = args.device

    import humos.src.prepare  # noqa
    from humos.src.data.text import save_token_embeddings, save_sent_embeddings

    # Compute token embeddings
    modelname = args.text_to_token_model_name
    logger.info(f"Compute token embeddings for {modelname}")
    path = f"annotations/{args.data}"
    save_token_embeddings(path, modelname=modelname, device=device)

    # Compute sent embeddings
    modelname = args.text_to_sent_model_name
    logger.info(f"Compute sentence embeddings for {modelname}")
    path =  f"annotations/{args.data}"
    save_sent_embeddings(path, modelname=modelname, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=str, default="humanml3d", choices=["humanml3d", "kitml"])
    parser.add_argument("--text_to_token_model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--text_to_sent_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    args = parser.parse_args()

    text_embeddings(args)
