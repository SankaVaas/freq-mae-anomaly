"""
dataset.py
----------
MVTec-AD Dataset Loader for Frequency-Aware MAE Anomaly Detection.

MVTec-AD Directory Structure (expected):
    mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          ← normal images for self-supervised training
    │   └── test/
    │       ├── good/          ← normal test images
    │       ├── broken_large/  ← defect type 1
    │       ├── broken_small/  ← defect type 2
    │       └── ...
    │   └── ground_truth/
    │       ├── broken_large/  ← binary masks (.png)
    │       └── ...
    ├── cable/
    └── ...  (15 categories total)

Download:
    https://www.mvtec.com/company/research/datasets/mvtec-ad
    (~5GB, free for research)

Usage:
    # Training (self-supervised, normal images only)
    train_ds = MVTecDataset(root="data/mvtec", category="bottle", split="train")

    # Test (normal + anomaly, with ground-truth masks)
    test_ds  = MVTecDataset(root="data/mvtec", category="bottle", split="test")
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# ImageNet stats — standard for ViT pre-training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_train_transform(img_size: int = 224) -> transforms.Compose:
    """Augmentation for self-supervised training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_test_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic transform for evaluation — no augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_mask_transform(img_size: int = 224) -> transforms.Compose:
    """Transform for ground-truth anomaly masks (binary, no normalization)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MVTecDataset(Dataset):
    """
    MVTec-AD dataset for anomaly detection.

    Args:
        root        : path to mvtec/ root directory
        category    : one of MVTEC_CATEGORIES
        split       : 'train' or 'test'
        img_size    : resize all images to (img_size x img_size)
        transform   : override default image transform
    """

    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ):
        assert category in MVTEC_CATEGORIES, (
            f"Unknown category '{category}'. Choose from:\n{MVTEC_CATEGORIES}"
        )
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root = Path(root)
        self.category = category
        self.split = split
        self.img_size = img_size

        self.img_transform = transform or (
            build_train_transform(img_size) if split == "train"
            else build_test_transform(img_size)
        )
        self.mask_transform = build_mask_transform(img_size)

        self.samples: List[Dict] = []
        self._load_samples()

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_samples(self):
        split_dir = self.root / self.category / self.split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {split_dir}\n"
                f"Please download MVTec-AD to '{self.root}' and extract it."
            )

        if self.split == "train":
            # Training: only 'good' subfolder
            good_dir = split_dir / "good"
            for img_path in sorted(good_dir.glob("*.png")) + sorted(good_dir.glob("*.jpg")):
                self.samples.append({
                    "img_path"  : img_path,
                    "mask_path" : None,
                    "label"     : 0,          # 0 = normal
                    "defect"    : "good",
                })

        else:  # test
            gt_root = self.root / self.category / "ground_truth"
            for defect_dir in sorted(split_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_name = defect_dir.name
                is_anomaly = defect_name != "good"

                for img_path in sorted(defect_dir.glob("*.png")) + sorted(defect_dir.glob("*.jpg")):
                    mask_path = None
                    if is_anomaly:
                        # Ground-truth mask: same stem + '_mask.png'
                        mask_candidate = (
                            gt_root / defect_name / (img_path.stem + "_mask.png")
                        )
                        if mask_candidate.exists():
                            mask_path = mask_candidate

                    self.samples.append({
                        "img_path"  : img_path,
                        "mask_path" : mask_path,
                        "label"     : 1 if is_anomaly else 0,
                        "defect"    : defect_name,
                    })

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {split_dir}. "
                f"Check that the dataset is extracted correctly."
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image (convert grayscale to RGB for ViT)
        img = Image.open(sample["img_path"]).convert("RGB")
        img = self.img_transform(img)                  # (C, H, W) float tensor

        # Load ground-truth mask (test only, anomaly only)
        if sample["mask_path"] is not None:
            mask = Image.open(sample["mask_path"]).convert("L")
            mask = self.mask_transform(mask)           # (1, H, W) in [0, 1]
            mask = (mask > 0.5).float()                # binarize
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)

        return {
            "image"  : img,                            # (3, H, W)
            "mask"   : mask,                           # (1, H, W) pixel GT
            "label"  : torch.tensor(sample["label"], dtype=torch.long),
            "defect" : sample["defect"],
            "path"   : str(sample["img_path"]),
        }

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_normal  = sum(1 for s in self.samples if s["label"] == 0)
        n_anomaly = sum(1 for s in self.samples if s["label"] == 1)
        return (
            f"MVTecDataset(category={self.category}, split={self.split}, "
            f"total={len(self.samples)}, normal={n_normal}, anomaly={n_anomaly})"
        )

    def defect_types(self) -> List[str]:
        return sorted(set(s["defect"] for s in self.samples))


# ---------------------------------------------------------------------------
# DataLoader Factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    root: str,
    category: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders for one MVTec category.

    Args:
        root        : path to mvtec/ root
        category    : MVTec category name
        img_size    : image resolution
        batch_size  : training batch size
        num_workers : DataLoader workers (set 0 on Windows if issues arise)

    Returns:
        train_loader, test_loader
    """
    train_ds = MVTecDataset(root, category, split="train",  img_size=img_size)
    test_ds  = MVTecDataset(root, category, split="test",   img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,              # one at a time for per-image scoring
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity Check (runs without MVTec — creates dummy images)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, shutil

    print("Creating dummy MVTec-like dataset for sanity check...")

    # Build a minimal fake dataset structure
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cat = "bottle"

        # Train: good images only
        (root / cat / "train" / "good").mkdir(parents=True)
        # Test: good + one defect type
        (root / cat / "test" / "good").mkdir(parents=True)
        (root / cat / "test" / "broken_large").mkdir(parents=True)
        (root / cat / "ground_truth" / "broken_large").mkdir(parents=True)

        def make_img(path, size=(256, 256)):
            arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)

        def make_mask(path, size=(256, 256)):
            arr = np.zeros((*size,), dtype=np.uint8)
            arr[100:150, 100:150] = 255          # small anomaly region
            Image.fromarray(arr).save(path)

        # Create dummy images
        for i in range(8):
            make_img(root / cat / "train" / "good" / f"{i:03d}.png")
        for i in range(4):
            make_img(root / cat / "test" / "good" / f"{i:03d}.png")
        for i in range(4):
            make_img(root / cat / "test" / "broken_large" / f"{i:03d}.png")
            make_mask(root / cat / "ground_truth" / "broken_large" / f"{i:03d}_mask.png")

        # Test dataset loading
        train_ds = MVTecDataset(str(root), cat, split="train", img_size=224)
        test_ds  = MVTecDataset(str(root), cat, split="test",  img_size=224)

        print(train_ds)
        print(test_ds)
        print(f"Defect types: {test_ds.defect_types()}")

        # Test one batch
        batch = train_ds[0]
        print(f"\nTrain sample — image: {batch['image'].shape}, "
              f"label: {batch['label']}, defect: {batch['defect']}")

        batch = test_ds[4]          # first anomaly sample
        print(f"Test  sample — image: {batch['image'].shape}, "
              f"mask: {batch['mask'].shape}, label: {batch['label'].item()}, "
              f"defect: {batch['defect']}")

        # Test DataLoader
        train_loader, test_loader = build_dataloaders(
            str(root), cat, img_size=224, batch_size=4, num_workers=0
        )
        imgs = next(iter(train_loader))
        print(f"\nDataLoader batch — images: {imgs['image'].shape}, "
              f"labels: {imgs['label']}")

    print("\ndataset.py OK")