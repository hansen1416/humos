import os
import pickle
import numpy as np


def pkl_to_npz(pkl_path: str, npz_path: str):
    # SMPL v1.1.0 pickles are Python2-era; latin1 avoids UnicodeDecodeError
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Many SMPL pkls store J_regressor as a scipy sparse matrix.
    # To keep this dependency-free and robust, densify when possible.
    out = {}
    for k, v in data.items():
        if hasattr(v, "toarray"):  # e.g., scipy.sparse
            out[k] = v.toarray()
        else:
            out[k] = v
    np.savez(npz_path, **out)


if __name__ == "__main__":
    roots = [
        ("body_models/smpl/female/model.pkl", "body_models/smpl/female/model.npz"),
        ("body_models/smpl/male/model.pkl", "body_models/smpl/male/model.npz"),
        ("body_models/smpl/neutral/model.pkl", "body_models/smpl/neutral/model.npz"),
    ]
    for src, dst in roots:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"{src} -> {dst}")
        pkl_to_npz(src, dst)
