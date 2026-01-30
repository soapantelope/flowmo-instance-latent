import io
import json
import os
from collections import defaultdict
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, data_root, size=256, random_crop=False):
        self.data_root = data_root
        self.size = size
        
        self.rescaler = T.Resize(size)
        self.cropper = T.RandomCrop((size, size)) if random_crop else T.CenterCrop((size, size))
        self.preprocessor = T.Compose([self.rescaler, self.cropper])
        
        self.instances = []
        self.instance_to_frames = defaultdict(list) # instance : list of its frames

        self.num_frames = 0
        self._load_all_images(data_root)
    
    def _load_all_images(self, data_root):
        print("loading images")
        
        for root, dirs, files in os.walk(data_root):
            for i, file in enumerate(files):
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if i % 500 == 0:
                    print(f"{i} images loaded into memory so far...")

                instance, _ = file.rsplit('_', 1)

                path = os.path.join(root, file)
                image = Image.open(path).convert("RGB")
                image = self.preprocessor(image)
                image = np.array(image)
                image = (image / 127.5 - 1.0).astype(np.float32)
                
                if instance not in self.instance_to_frames:
                    self.instances.append(instance)

                self.instance_to_frames[instance].append(image)
                self.num_frames += 1
                
        print(f"all images loaded into memory!")
    
    def __len__(self):
        return len(self.instances) * 100 # each epoch will just be num_instances pairs * 100 random pairs
    
    def __getitem__(self, idx):
        idx %= len(self.instances)
        p1, p2 = random.sample(self.instance_to_frames[self.instances[idx]], 2)
        
        return {
            "images": np.stack([p1, p2], axis=0),
        }


class IndexedTarDataset(Dataset):
    def __init__(
        self,
        imagenet_tar,
        imagenet_index,
        size=None,
        random_crop=False,
        aug_mode="default",
    ):
        self.size = size
        self.random_crop = random_crop

        self.aug_mode = aug_mode

        if aug_mode == "default":
            assert self.size is not None and self.size > 0
            self.rescaler = T.Resize(self.size)
            if not self.random_crop:
                self.cropper = T.CenterCrop((self.size, self.size))
            else:
                self.cropper = T.RandomCrop((self.size, self.size))
            self.preprocessor = T.Compose([self.rescaler, self.cropper])
        else:
            raise NotImplementedError

        # Tar setup
        self.imagenet_tar = imagenet_tar
        self.imagenet_index = imagenet_index
        with open(self.imagenet_index, "r") as fp:
            self.index = json.load(fp)
        self.index = sorted(self.index, key=lambda d: d["name"].split("/")[-1])
        self.id_to_handle = {}

    def __len__(self):
        return len(self.index)

    def get_image(self, image_info):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        if worker_id not in self.id_to_handle:
            self.id_to_handle[worker_id] = open(self.imagenet_tar, "rb")
        handle = self.id_to_handle[worker_id]

        handle.seek(image_info["offset"])
        img_bytes = handle.read(image_info["size"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image.load()
        return image

    def preprocess_image(self, image_info):
        image = self.get_image(image_info)
        image = self.preprocessor(image)
        image = np.array(image)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.index[i])
        return example
