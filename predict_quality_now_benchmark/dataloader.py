import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import re


class AutoStructuredDataset(Dataset):
    def __init__(self, root_folders, transform=None, metrics=("median", "mean", "std"), thresholds= {
        "median": 1.5,
        "mean": 1.5,
        "std": 1.5
    }):
        """
        Args:
            root_folders (list of str): List of base folders containing sr_img and 3d_obj subfolders.
            transform (torchvision.transforms): Transformations for SR images.
            metrics (tuple of str): Error metrics to extract (e.g., "median", "mean", "std").
        """
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.metrics = metrics
        self.samples = []
        self.thresholds = thresholds

        for base_path in root_folders:
            base = Path(base_path)
            sr_img_root = base / "sr_img"
            obj_root = base / "3d_obj"
            result_txt = list(obj_root.rglob("results/RECON_computed_distances_all_output.txt"))
            if not result_txt:
                continue

            # Step 1: Parse the error metrics
            error_dict = {}
            with open(result_txt[0], 'r') as f:
                for line in f:
                    if '-' not in line:
                        continue
                    name, metric_str = line.strip().split(' - ', 1)
                    metrics_found = {}
                    for m in self.metrics:
                        match = re.search(rf"{m}:\s*([\d.]+)", metric_str)
                        if match:
                            metrics_found[m] = float(match.group(1))
                    if metrics_found:
                        error_dict[name.strip()] = metrics_found

            # Step 2: Match each 3d_obj folder with corresponding SR image and identity.npy
            # Match each mesh folder to the error entry using suffix
            for mesh_dir in obj_root.glob("*"):
                if not mesh_dir.is_dir():
                    continue
                img_name = mesh_dir.name

                # Find the full error key that ends with this mesh name
                matching_key = next((k for k in error_dict if k.endswith(img_name)), None)
                if matching_key is None:
                    continue  # skip if no match

                sr_img_path = sr_img_root / f"{img_name}.png"
                mesh_feat_path = mesh_dir / "identity.npy"

                if sr_img_path.exists() and mesh_feat_path.exists():
                    self.samples.append({
                        "img_path": sr_img_path,
                        "mesh_feat_path": mesh_feat_path,
                        "errors": error_dict[matching_key]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['img_path']).convert('RGB')
        mesh_feat = np.load(sample['mesh_feat_path'])
        errors = sample['errors']

        image = self.transform(image)

        label = [
            0.0 if errors.get("median", 0.0) < self.thresholds["median"] else 1.0,
            0.0 if errors.get("mean", 0.0) < self.thresholds["mean"] else 1.0,
            0.0 if errors.get("std", 0.0) < self.thresholds["std"] else 1.0
        ]

        return {
            'image': image,
            'mesh_feat': torch.tensor(mesh_feat, dtype=torch.float32),
            'labels': torch.tensor(label, dtype=torch.float32)
        }
