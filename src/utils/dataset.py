import os
from pathlib import Path
from typing import Tuple, Optional, List

from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


def _find_dir(base: Path, candidates: List[str]) -> Path:
    for c in candidates:
        p = base / c
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"None of these folders exist under {base}: {candidates}")


class LevirCDDataset(Dataset):
    """
    LEVIR-CD dataset loader.
    Expects split folder containing A/, B/, and label/ (or labels/, gt/, mask/).
    Returns: (x, y, name)
      x: float tensor [6, H, W] (A and B concatenated)
      y: float tensor [1, H, W] in {0,1}
    """
    def __init__(
        self,
        root_dir: str,
        split: str,
        img_size: Optional[Tuple[int, int]] = (256, 256),
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split

        self.dir_a = _find_dir(self.split_dir, ["A", "t1", "image1", "ImagesA"])
        self.dir_b = _find_dir(self.split_dir, ["B", "t2", "image2", "ImagesB"])
        self.dir_y = _find_dir(self.split_dir, ["label", "labels", "gt", "mask", "masks", "Label"])

        self.img_size = img_size
        self.augment = augment

        self.names = sorted([p.name for p in self.dir_a.glob("*") if p.is_file()])
        if len(self.names) == 0:
            raise FileNotFoundError(f"No images found in {self.dir_a}")

        # Keep only those present in all folders
        keep = []
        for n in self.names:
            if (self.dir_b / n).exists() and (self.dir_y / n).exists():
                keep.append(n)
        self.names = keep

        if len(self.names) == 0:
            raise FileNotFoundError("No matching filenames across A, B, label folders.")

    def __len__(self):
        return len(self.names)

    def _load_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> Image.Image:
        # binary mask; keep as L
        return Image.open(path).convert("L")

    def _resize(self, img: Image.Image) -> Image.Image:
        if self.img_size is None:
            return img
        return img.resize(self.img_size, resample=Image.BILINEAR)

    def _resize_mask(self, m: Image.Image) -> Image.Image:
        if self.img_size is None:
            return m
        return m.resize(self.img_size, resample=Image.NEAREST)

    def _to_tensor_rgb(self, img: Image.Image) -> torch.Tensor:
        # [H,W,3] -> [3,H,W], float [0,1]
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return t

    def _to_tensor_mask(self, m: Image.Image) -> torch.Tensor:
        arr = __import__("numpy").array(m)
        # convert to {0,1}
        arr = (arr > 127).astype("float32")
        t = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return t

    def __getitem__(self, idx):
        name = self.names[idx]
        a = self._load_rgb(self.dir_a / name)
        b = self._load_rgb(self.dir_b / name)
        y = self._load_mask(self.dir_y / name)

        a = self._resize(a)
        b = self._resize(b)
        y = self._resize_mask(y)

        # simple augmentation (safe + fast): random horizontal/vertical flip
        if self.augment:
            import random
            if random.random() < 0.5:
                a = a.transpose(Image.FLIP_LEFT_RIGHT)
                b = b.transpose(Image.FLIP_LEFT_RIGHT)
                y = y.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                a = a.transpose(Image.FLIP_TOP_BOTTOM)
                b = b.transpose(Image.FLIP_TOP_BOTTOM)
                y = y.transpose(Image.FLIP_TOP_BOTTOM)

        ta = self._to_tensor_rgb(a)
        tb = self._to_tensor_rgb(b)
        x = torch.cat([ta, tb], dim=0)  # [6,H,W]
        yt = self._to_tensor_mask(y)

        return x, yt, name