"""MorphoMNIST dataset loader.

Download data from https://github.com/dccastro/Morpho-MNIST and place the
six files in validation/morphomnist/data/:
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    train-morpho.csv
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz
    t10k-morpho.csv
"""

import gzip
import os
import struct

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

# Min/max for [-1, 1] normalisation (from benchmark paper)
MIN_MAX = {
    "thickness": [0.87598526, 6.255515],
    "intensity":  [66.601204, 254.90317],
    "image":      [0.0, 255.0],
}

# Causal graph and attribute dimensions (for reference by training code)
CAUSAL_GRAPH = {
    "thickness": [],
    "intensity": ["thickness"],
    "digit":     [],
    "image":     ["thickness", "intensity", "digit"],
}

ATTRIBUTE_SIZE = {"thickness": 1, "intensity": 1, "digit": 10}


# ---------------------------------------------------------------------------
# IDX binary reader (from https://github.com/dccastro/Morpho-MNIST)
# ---------------------------------------------------------------------------

def _load_uint8(f):
    idx_dtype, ndim = struct.unpack("BBBB", f.read(4))[2:]
    shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
    data = np.frombuffer(f.read(int(np.prod(shape))), dtype=np.uint8).reshape(shape)
    return data

def load_idx(path: str) -> np.ndarray:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rb") as f:
        return _load_uint8(f)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize_attr(value, name):
    lo, hi = MIN_MAX[name]
    return 2.0 * (value - lo) / (hi - lo) - 1.0

def unnormalize_attr(value, name):
    lo, hi = MIN_MAX[name]
    return (value + 1.0) / 2.0 * (hi - lo) + lo


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _get_paths(root_dir, train):
    prefix = "train" if train else "t10k"
    return (
        os.path.join(root_dir, f"{prefix}-images-idx3-ubyte.gz"),
        os.path.join(root_dir, f"{prefix}-labels-idx1-ubyte.gz"),
        os.path.join(root_dir, f"{prefix}-morpho.csv"),
    )


class MorphoMNIST(Dataset):
    """Each item is (image, attrs) where:
        image : float32 tensor (1, 32, 32) normalised to [-1, 1]
        attrs : float32 tensor (12,) = [thickness(1), intensity(1), digit_onehot(10)]
    """

    def __init__(self, data_dir, split="train", transform=None):
        assert split in ("train", "test")
        images_path, labels_path, metrics_path = _get_paths(data_dir, split == "train")

        raw_images = load_idx(images_path).astype(np.float32)   # (N, 28, 28)
        labels     = load_idx(labels_path)                       # (N,)
        metrics_df = pd.read_csv(metrics_path, index_col="index")

        # Pad 28x28 -> 32x32, normalise to [-1, 1]
        pad = transforms.Pad(padding=2)
        images_t = pad(torch.as_tensor(raw_images).unsqueeze(1))  # (N,1,32,32)
        lo, hi = MIN_MAX["image"]
        self.images = 2.0 * (images_t - lo) / (hi - lo) - 1.0

        self.thickness = torch.as_tensor(
            normalize_attr(metrics_df["thickness"].values, "thickness"), dtype=torch.float32
        )
        self.intensity = torch.as_tensor(
            normalize_attr(metrics_df["intensity"].values, "intensity"), dtype=torch.float32
        )
        self.digit = F.one_hot(
            torch.as_tensor(labels.copy(), dtype=torch.long), num_classes=10
        ).float()

        # Combined attribute vector [thickness(1), intensity(1), digit(10)] -> (N,12)
        self.attrs = torch.cat(
            [self.thickness.unsqueeze(1), self.intensity.unsqueeze(1), self.digit], dim=1
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, attrs = self.images[idx], self.attrs[idx]
        if self.transform is not None:
            img, attrs = self.transform(img, attrs)
        return img, attrs