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

class QuadrupletDataset(Dataset):
    def __init__(self, data_root, size=256, random_crop=False):
        self.data_root = data_root
        self.size = size
        
        self.rescaler = T.Resize(size)
        self.cropper = T.RandomCrop((size, size)) if random_crop else T.CenterCrop((size, size))
        self.preprocessor = T.Compose([self.rescaler, self.cropper])
        
        self.instance_pose_to_images = defaultdict(list)
        self.instances_list = []
        self.poses_list = []
        self.quadruplets = []
        
        self._load_all_images(data_root)
        self._generate_quadruplets()
    
    def _load_all_images(self, data_root):
        print("loading images")
        
        image_paths = []
        image_metadata = []
        instances = set()
        poses = set()
        
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                instance, pose_with_ext = file.rsplit('_', 1)
                pose = pose_with_ext.rsplit('.', 1)[0]
                
                instances.add(instance)
                poses.add(pose)
                
                path = os.path.join(root, file)
                image_paths.append(path)
                image_metadata.append((instance, pose))
        
        self.instances_list = list(instances)
        self.poses_list = list(poses)
        
        print(f"{len(image_paths)} images with {len(self.instances_list)} instances and {len(self.poses_list)} poses")
        print("loading all images into memory...")
        
        for i, (path, (instance, pose)) in enumerate(zip(image_paths, image_metadata)):
            if i % 500 == 0:
                print(f"  Loaded {i}/{len(image_paths)} images...")
            
            image = Image.open(path).convert("RGB")
            image = self.preprocessor(image)
            image = np.array(image)
            image = (image / 127.5 - 1.0).astype(np.float32)
            
            self.instance_pose_to_images[(instance, pose)].append(image)
        
        print(f"all {len(image_paths)} images loaded into memory!")
        print(f"{len(self.instance_pose_to_images)} (instance, pose) combos")
    
    def _generate_quadruplets(self):
        instances_shuffled = self.instances_list.copy()
        poses_shuffled = self.poses_list.copy()
        random.shuffle(instances_shuffled)
        random.shuffle(poses_shuffled)
        
        instance_pairs = [(instances_shuffled[i], instances_shuffled[i+1]) 
                          for i in range(0, len(instances_shuffled) - 1, 2)]
        pose_pairs = [(poses_shuffled[i], poses_shuffled[i+1]) 
                      for i in range(0, len(poses_shuffled) - 1, 2)]
        
        self.quadruplets = [
            (inst_i, inst_j, pose_p1, pose_p2)
            for (inst_i, inst_j) in instance_pairs
            for (pose_p1, pose_p2) in pose_pairs
        ]
        print(f"{len(self.quadruplets)} quadruplets")
    
    def shuffle(self):
        random.shuffle(self.quadruplets)
    
    def _get_image(self, instance, pose):
        images = self.instance_pose_to_images[(instance, pose)]
        return random.choice(images)
    
    def __len__(self):
        return len(self.quadruplets)
    
    def __getitem__(self, idx):
        instance_i, instance_j, pose_p1, pose_p2 = self.quadruplets[idx]

        a = self._get_image(instance_i, pose_p1)
        b = self._get_image(instance_i, pose_p2)
        c = self._get_image(instance_j, pose_p1)
        d = self._get_image(instance_j, pose_p2)

        quadruplet = np.stack([a, b, c, d], axis=0)
        
        return {
            "images": quadruplet,
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
