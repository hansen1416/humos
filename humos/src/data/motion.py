import os

import numpy as np
import torch

from humos.prepare.tools import canonicalize_smplh, save_meshes
from humos.prepare.extract_features import get_input_features
from humos.utils.rotation_conversions import matrix_to_rotation_6d


class AMASSMotionLoader:
    def __init__(
        self,
        base_dir,
        fps,
        normalizer=None,
        disable: bool = False,
        canonicalize_crops: bool = False,
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.normalizer = normalizer
        self.disable = disable
        self.canonicalize_crops = canonicalize_crops
        self.num_frames = 200  # have at num_frames many frames in the motion crops

    def __call__(self, path, start, end):
        if self.disable:
            return {"x": path, "length": int(self.fps * (end - start))}

        begin = int(start * self.fps)
        end = int(end * self.fps)
        if path not in self.motions:
            motion_path = os.path.join(self.base_dir, path + ".tensor")
            motion = torch.load(motion_path)
            # convert all features to torch tensor
            motion = {k: torch.tensor(v) for k, v in motion.items()}
            self.motions[path] = motion

        # if number of frames between begin and end are less than num_frames, then extend begin and end to num_frames
        # if not enough frames in motion, then repeat the last frame
        if end - begin < self.num_frames:
            prev_begin = begin - (self.num_frames - (end - begin)) // 2
            next_end = end + (self.num_frames - (end - begin)) // 2
            # if not enough frames before begin, then extend end
            if prev_begin < 0:
                prev_begin = 0
                next_end = next_end + (self.num_frames - (next_end - prev_begin))
            # if not enough frames after end
            if next_end > len(self.motions[path]["trans"]):
                next_end = len(self.motions[path]["trans"])
        else:
            prev_begin = begin
            next_end = end

        motion = {}
        for k, v in self.motions[path].items():
            motion[k] = v[prev_begin:next_end]

        # if number of frames between begin and end are less than num_frame, repeat the last frame
        if len(motion["trans"]) < self.num_frames:
            for (
                k,
                v,
            ) in motion.items():
                # get ndims
                ndims = len(v.shape)
                # handle root_diff and root_orient_diffs
                if "root_trans_diff" in k:  # if velocity, then append zeros at the end
                    motion[k] = torch.cat(
                        (v, torch.zeros(self.num_frames - len(v), v.shape[1])), dim=0
                    )
                elif (
                    "rot_orient_diff" in k
                ):  # if rot_orint_diff, append identity 6d pose at the end
                    identity = torch.eye(3).repeat(self.num_frames - len(v), 1, 1)
                    # convert to 6d
                    identity = matrix_to_rotation_6d(identity)
                    motion[k] = torch.cat((v, identity), dim=0)
                elif "vel" in k or "angular" in k:
                    raise NotImplementedError
                else:
                    motion[k] = torch.cat(
                        (
                            v,
                            v[-1].repeat(
                                self.num_frames - len(v), *([1] * (ndims - 1))
                            ),
                        ),
                        dim=0,
                    )

        # canonicalize crops to always start from origin and facing the same direction
        if self.canonicalize_crops:
            motion = canonicalize_smplh(motion, ground_the_human=False, in_format="aa")

        motion = get_input_features(motion, training=True)

        print(motion)
        exit()

        if self.normalizer is not None:
            motion = self.normalizer(motion)

        # add motion dict ot x_dict
        x_dict = {"length": len(motion["trans"])}
        x_dict.update(motion)
        return x_dict


class Normalizer:
    def __init__(self, base_dir: str, eps: float = 1e-12, disable: bool = False):
        self.base_dir = base_dir
        self.eps = eps

        self.disable = disable
        if not disable:
            self.load()

    def load(self):
        mean = {}
        std = {}
        for f in os.listdir(self.base_dir):
            if f.endswith("_mean.pt"):
                mean[f.replace("_mean.pt", "")] = torch.load(
                    os.path.join(self.base_dir, f)
                )
            if f.endswith("_std.pt"):
                std[f.replace("_std.pt", "")] = torch.load(
                    os.path.join(self.base_dir, f)
                )
        self.mean = mean
        self.std = std

    def save(self, mean, std):
        os.makedirs(self.base_dir, exist_ok=True)
        for k, v in mean.items():
            torch.save(v, os.path.join(self.base_dir, f"{k}_mean.pt"))
        for k, v in std.items():
            torch.save(v, os.path.join(self.base_dir, f"{k}_std.pt"))

    def __call__(self, x):
        if self.disable:
            return x
        for k, v in x.items():
            x[k] = (x[k] - self.mean[k]) / (self.std[k] + self.eps)
        return x

    def inverse(self, x):
        if self.disable:
            return x
        for k, v in x.items():
            mean = self.mean[k].to(v.device)
            std = self.std[k].to(v.device)
            x[k] = x[k] * (std + self.eps) + mean
        return x
