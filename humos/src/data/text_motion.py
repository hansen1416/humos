import os
import codecs as cs
import orjson  # loading faster than json
import json

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from .collate import collate_text_motion


def read_split(path, split):
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


def load_annotations(path, name="annotations_processed.json"):
    json_path = os.path.join(path, name)
    with open(json_path, "rb") as ff:
        return orjson.loads(ff.read())


class TextMotionDataset(Dataset):
    def __init__(
        self,
        path: str,
        motion_loader,
        text_to_sent_emb,
        text_to_token_emb,
        split: str = "train",
        min_seconds: float = 2.0,
        max_seconds: float = 10.0,
        preload: bool = True,
        demo: bool = False,
        tiny: bool = False,
    ):
        if tiny:
            split = split + "_tiny"

        self.collate_fn = collate_text_motion
        self.split = split
        self.keyids = read_split(path, split)

        self.text_to_sent_emb = text_to_sent_emb
        self.text_to_token_emb = text_to_token_emb
        self.motion_loader = motion_loader

        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

        # remove too short or too long annotations
        self.annotations = load_annotations(path)

        # filter annotations (min/max)
        # but not for the test set
        # otherwise it is not fair for everyone
        if "test" not in split:
            self.annotations = self.filter_annotations(self.annotations)

        self.is_training = split == "train"
        self.keyids = [keyid for keyid in self.keyids if keyid in self.annotations]
        if "test" in split:
            # open mld files
            mdl_test_keyids = "./datasets/splits/mld_test_split_keyids.txt"
            with open(mdl_test_keyids, "r") as f:
                mdl_test_keyids = f.readlines()
                mdl_test_keyids = [x.strip() for x in mdl_test_keyids]
            # take only the keyids that are in the mdl_test_keyids
            self.keyids = [keyid for keyid in self.keyids if keyid in mdl_test_keyids]
            # remove the keyids with _M
            self.keyids = [keyid for keyid in self.keyids if "M" not in keyid]
        if demo:
            self.keyids = ["000073", "000419", "001262", "004117"]
            # Repeat this same list 25 times to have a larger dataset
            self.keyids = [keyid for _ in range(25) for keyid in self.keyids]
        if preload:
            for _ in tqdm(self, desc="Preloading the dataset"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        output = self.load_keyid(keyid)
        # if output is None:
        #     TODO: this might cause issues in test time stocasticity in results
        # if self.is_training:
        #     index = np.random.randint(len(self.keyids))
        # else:
        #     index = index - 1
        # return self.__getitem__(index)
        return output

    def load_keyid(self, keyid):
        annotations = self.annotations[keyid]

        # Take the first one for testing/validation
        # Otherwise take a random one
        index = 0
        if self.is_training:
            index = np.random.randint(len(annotations["annotations"]))
        annotation = annotations["annotations"][index]

        text = annotation["text"]

        # # Filter out 'chair' and 'stairs' annotations
        # if 'chair' in text or 'stairs' in text:
        #     return None

        # text_x_dict = self.text_to_token_emb(text)
        # try:
        motion_x_dict = self.motion_loader(
            path=annotations["path"],
            start=annotation["start"],
            end=annotation["end"],
        )
        # except:
        #     # to handle motions filteres by min_vertex > 0.25m
        #     return None
        # sent_emb = self.text_to_sent_emb(text)

        output = {
            "motion_x_dict": motion_x_dict,
            # "text_x_dict": text_x_dict,
            "text": text,
            "keyid": keyid,
            # "sent_emb": sent_emb,
        }
        return output

    def check_problem_paths(self, data):
        # Remove humanact12
        # Remove treadmill from BMLrub (treadmill_ and normal_)
        # Remove ice-skating (HDM_dg_07-01) in MPI_HDM05
        if "humanact12" in data["path"]:
            return False
        if "treadmill" in data["path"] or "normal" in data["path"]:
            return False
        if "HDM_dg_07-01" in data["path"]:
            return False
        return True

    def filter_annotations(self, annotations):
        filtered_annotations = {}
        for key, val in annotations.items():

            # Todo: Add back humanact12 later
            check = self.check_problem_paths(val)
            if not check:
                continue

            annots = val.pop("annotations")
            filtered_annots = []
            for annot in annots:
                duration = annot["end"] - annot["start"]
                if self.max_seconds >= duration >= self.min_seconds:
                    filtered_annots.append(annot)

            if filtered_annots:
                val["annotations"] = filtered_annots
                filtered_annotations[key] = val

        return filtered_annotations


def write_json(data, path):
    with open(path, "w") as ff:
        ff.write(json.dumps(data, indent=4))
