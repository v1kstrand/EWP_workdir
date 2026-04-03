# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from torch.utils.data import Dataset
import torch
import torchvision
from PIL import Image
import numpy as np

class _LatentCacheMixin:
    def _init_common(self, dataset_root, img_file, labels_file, size_dataset, transform, experience, latent_cache_file=None):
        self.dataset_root = Path(dataset_root)
        self.samples = np.load(img_file, allow_pickle=True)
        self.labels = np.load(labels_file)
        if size_dataset > 0:
            self.samples = self.samples[:size_dataset]
            self.labels = self.labels[:size_dataset]
        assert len(self.samples) == len(self.labels)
        self.transform = transform
        self.to_tensor = torchvision.transforms.ToTensor()
        self.experience = experience
        self.latents = None
        self.quats = None
        if latent_cache_file is not None:
            cache = np.load(latent_cache_file, allow_pickle=True)
            cached_samples = cache["samples"]
            if len(cached_samples) != len(self.samples) or not np.array_equal(cached_samples, self.samples):
                raise ValueError(f"Latent cache sample list does not match manifest: {latent_cache_file}")
            self.latents = cache["latents"]
            self.quats = cache["quats"]
            mib = self.latents.nbytes / 1024 / 1024 + self.quats.nbytes / 1024 / 1024
            print(
                f"Loaded latent cache: {latent_cache_file} | latents={self.latents.shape} | "
                f"quats={self.quats.shape} | approx_ram={mib:.2f} MiB"
            )

    def _sample_root(self, i):
        return self.dataset_root / self.samples[i][1:]

    def _load_latent_pair(self, i, views):
        if self.latents is not None:
            latent_1 = self.latents[i, views[0]]
            latent_2 = self.latents[i, views[1]]
        else:
            sample_root = self._sample_root(i)
            latent_1 = np.load(sample_root / f"latent_{views[0]}.npy").astype(np.float32)
            latent_2 = np.load(sample_root / f"latent_{views[1]}.npy").astype(np.float32)
        return latent_1, latent_2


class Dataset3DIEBench(_LatentCacheMixin, Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None, latent_cache_file=None):
        self._init_common(dataset_root, img_file, labels_file, size_dataset, transform, experience, latent_cache_file=latent_cache_file)

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        sample_root = self._sample_root(i)
        img_1 = self.get_img(sample_root / f"image_{views[0]}.jpg")
        img_2 = self.get_img(sample_root / f"image_{views[1]}.jpg")

        latent_1, latent_2 = self._load_latent_pair(i, views)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]

        return img_1, img_2, torch.from_numpy(angles_1), torch.from_numpy(angles_2), label

    def __len__(self):
        return len(self.samples)

class Dataset3DIEBenchAll(_LatentCacheMixin, Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None, latent_cache_file=None):
        self._init_common(dataset_root, img_file, labels_file, size_dataset, transform, experience, latent_cache_file=latent_cache_file)

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        sample_root = self._sample_root(i)
        img_1 = self.get_img(sample_root / f"image_{views[0]}.jpg")
        img_2 = self.get_img(sample_root / f"image_{views[1]}.jpg")

        latent_1, latent_2 = self._load_latent_pair(i, views)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]

        other_params = latent_2[3:] - latent_1[3:]

        return img_1, img_2, torch.from_numpy(angles_1), torch.from_numpy(angles_2), torch.from_numpy(other_params), label

    def __len__(self):
        return len(self.samples)


class Dataset3DIEBenchRotColor(_LatentCacheMixin, Dataset):
    def __init__(self, dataset_root, img_file,labels_file,experience="quat", size_dataset=-1, transform=None, latent_cache_file=None):
        self._init_common(dataset_root, img_file, labels_file, size_dataset, transform, experience, latent_cache_file=latent_cache_file)

    def get_img(self, path):
        with open(path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img) 
        return img

    def __getitem__(self, i):
        label = self.labels[i]
        # Latent vector creation
        views = np.random.choice(50,2, replace=False)
        sample_root = self._sample_root(i)
        img_1 = self.get_img(sample_root / f"image_{views[0]}.jpg")
        img_2 = self.get_img(sample_root / f"image_{views[1]}.jpg")

        latent_1, latent_2 = self._load_latent_pair(i, views)
        angles_1 = latent_1[:3]
        angles_2 = latent_2[:3]

        other_params = latent_2[[3, 6]] - latent_1[[3, 6]]

        return img_1, img_2, torch.from_numpy(angles_1), torch.from_numpy(angles_2), torch.from_numpy(other_params), label

    def __len__(self):
        return len(self.samples)

