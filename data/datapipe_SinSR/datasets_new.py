import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from skimage.io import imread
from functools import reduce
from torchvision import transforms
import random
import cv2
from albumentations import SmallestMaxSize
from loguru import logger

# Import dataset utilities
from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .degradation_bsrgan.bsrgan_light import degradation_bsrgan_variant, degradation_bsrgan
from data.utils import util_image, util_common

def create_dataset(dataset_config):
    if dataset_config.data.train['type'] == '3dSin':
        dataset = CombinedDataset(mica_config = dataset_config.MICA.dataset, sinSR_config = dataset_config.data.train, isEval=False)
    else:
        raise NotImplementedError(dataset_config.data.train['type'])

    return dataset

class CombinedDataset(Dataset):
    def __init__(self, mica_config, sinSR_config, isEval=False):
        """
        Combines MICA-based dataset with SinSR datasets (SinSR uses the same images as MICA).
        :param mica_config: Configuration for MICA dataset.
        :param sinSR_config: Configuration for SinSR transformations.
        :param isEval: Whether to run in evaluation mode.
        """
        # Initialize MICA dataset (primary dataset)
        self.K = mica_config.K
        self.gt_size = sinSR_config.params.gt_size
        self.isEval = isEval
        self.dataset_root = mica_config.root
        self.name = mica_config.training_data[0] #!!! fix if use more dataset
        self.image_folder = 'arcface_input'
        self.flame_folder = 'FLAME_parameters'
        self.total_images = 0
        self.imagepaths = []
        self.face_dict = {}

        
        # SinSR parameters (same image order as MICA)
        self.sinSR_config = sinSR_config

        # Load image paths 
        self.initialize()


    def initialize(self):
        """Initialize MICA dataset by loading face image paths."""
        logger.info(f'[{self.name}] Initialization')
        image_list = f'/users/ps1510/scratch/Programs/Sin3dFace/datasets/image_paths/{self.name}.npy'
        self.content_image_path = Path(self.sinSR_config.params.dir_path)
        logger.info(f'[{self.name}] Load image files: {self.content_image_path}')
        self.face_dict = np.load(image_list, allow_pickle=True).item()
        self.imagepaths = list(self.face_dict.keys())
        logger.info(f'[Dataset {self.name}] Total {len(self.imagepaths)} actors loaded!')
        self.set_smallest_k()

    def set_smallest_k(self):
        """Determine the smallest K across all actors."""
        self.min_max_K = np.Inf
        max_min_k = -np.Inf
        for key in self.face_dict.keys():
            length = len(self.face_dict[key][0])
            if length < self.min_max_K:
                self.min_max_K = length
            if length > max_min_k:
                max_min_k = length

        self.total_images = reduce(lambda k, l: l + k, map(lambda e: len(self.face_dict[e][0]), self.imagepaths))
        logger.info(f'Dataset {self.name} with min K = {self.min_max_K}, max K = {max_min_k}, total images = {self.total_images}')
        return self.min_max_K

    def __len__(self):
        """Return the total number of images."""
        return len(self.imagepaths)

    def __getitem__(self, index):
        """
        Get an item from the dataset, ensuring SinSR images match MICA order.
        :param index: Index of the image.
        :return: Dictionary with image tensors and metadata.
        """
        actor = self.imagepaths[index]
        images, params_path = self.face_dict[actor]
        images = [Path(self.dataset_root, self.name, self.image_folder, path) for path in images]
        sample_list = np.random.choice(range(len(images)), size=self.K, replace=False)

        K = self.K
        if self.isEval:
            K = max(0, min(200, self.min_max_K))
            sample_list = np.array(range(len(images))[:K])

        # Load FLAME model parameters
        params = np.load(os.path.join(self.dataset_root, self.name, self.flame_folder, params_path), allow_pickle=True)
        pose = torch.tensor(params['pose']).float()
        betas = torch.tensor(params['betas']).float()

        flame = {
            'shape_params': torch.cat(K * [betas[:300][None]], dim=0),
            'expression_params': torch.cat(K * [betas[300:][None]], dim=0),
            'pose_params': torch.cat(K * [torch.cat([pose[:3], pose[6:9]])[None]], dim=0),
        }

        images_list, sinSR_list = [], []
        for i in sample_list:
            image_path = images[i]
            # Extract 'a' (parent directory name)
            a = image_path.parent.name 
            # Extract 'b' (filename without extension)
            b = image_path.stem

            matching_images = sorted(self.content_image_path.glob(f"*_{a}_{b}.png"))

            image = np.array(imread(matching_images[0])) / 255.0
            image = image.transpose(2, 0, 1)

            # degradation 
            degrade_factor = int(self.gt_size / image.shape[-1])

            #!!! test the model - will delete after can set it to train on any resolutions
            # Resize the image to self.gt_size (assuming it's square)
            image = np.array([cv2.resize(image[c], (self.gt_size, self.gt_size), interpolation=cv2.INTER_CUBIC) for c in range(image.shape[0])])

            

            # Apply SinSR transformation on the same image
            sinSR_image = self.apply_sinSR(image, degrade_factor)

            images_list.append(image)
            sinSR_list.append(sinSR_image)

        images_array = torch.from_numpy(np.array(images_list)).float()
        sinSR_array = torch.from_numpy(np.array(sinSR_list)).float()

        return {
            'lq': sinSR_array,  # Low-quality SinSR image (same order as MICA)
            'gt': images_array,
            'imagename': actor,
            'dataset': self.name,
            'flame': flame,
            'sinSR_type': self.sinSR_config['type'],  # Type of SinSR transformation
            'degrade_factor': degrade_factor
        }

    def apply_sinSR(self, image, degrade_factor = 4):
        """
        Apply SinSR degradation based on the provided configuration.
        :param image: Input image from MICA.
        :return: Transformed SinSR image.
        """
        sf = self.sinSR_config.get('scale_factor', 4) * int(degrade_factor)
        degradation_type = self.sinSR_config['type']

        if degradation_type == 'bsrgan':
            sinSR_image, _ = degradation_bsrgan_variant(image, sf=sf, use_sharp=False)
        elif degradation_type == 'bicubic':
            sinSR_image = cv2.resize(image.transpose(1, 2, 0), dsize=(image.shape[1] // sf, image.shape[2] // sf), interpolation=cv2.INTER_CUBIC)
            sinSR_image = np.clip(sinSR_image, 0.0, 1.0).transpose(2, 0, 1)  # Convert back to CxHxW
        elif degradation_type == '3dSin':
            sinSR_image = cv2.resize(image.transpose(1, 2, 0), dsize=(image.shape[1] // sf, image.shape[2] // sf), interpolation=cv2.INTER_CUBIC)
            sinSR_image = cv2.resize(sinSR_image, dsize=(sinSR_image.shape[0] * int(degrade_factor), sinSR_image.shape[1] * int(degrade_factor)))
            sinSR_image = np.clip(sinSR_image, 0.0, 1.0).transpose(2, 0, 1)  # Convert back to CxHxW
        elif degradation_type == 'realesrgan':
            sinSR_image = util_image.imresize_np(image, scale=1/sf)
        else:
            raise ValueError(f"Unsupported SinSR type: {degradation_type}")

        return sinSR_image