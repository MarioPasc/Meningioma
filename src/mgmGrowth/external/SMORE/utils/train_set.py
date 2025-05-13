import numpy as np
import torch
from torch.utils.data import Dataset
from math import ceil

import torch.nn.functional as F
from resize.pytorch import resize

from .augmentations import augment_3d_image
from .patch_ops import get_patch, get_random_centers
from .pad import target_pad
from .parse_image_file import lr_axis_to_z, normalize
from .timer import timer_context


class TrainSet(Dataset):
    def __init__(
        self,
        image,
        slice_separation,
        lr_axis,
        patch_size,
        ext_patch_crop,
        device,
        blur_kernel,
        n_patches,
        patch_sampling,
        verbose=True,
    ):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.slice_separation = slice_separation

        self.ext_patch_crop = ext_patch_crop
        self.device = device
        self.blur_kernel = blur_kernel

        image = lr_axis_to_z(image, lr_axis)

        with timer_context(
            "Gathering data augmentations (flips and transposition in-plane)...",
            verbose=verbose,
        ):
            imgs_hr = augment_3d_image(image)

        with timer_context(
            "Padding image out to extract patches correctly...", verbose=verbose
        ):
            self.imgs_hr = []
            self.pads = []

            for image in imgs_hr:
                # Pad out s.t. in-planes are at least the patch size in each direction
                target_shape = [
                    s + p for s, p in zip(image.shape[:-1], self.patch_size[:-1])
                ] + [image.shape[2]]
                
                # apply the pad
                image, pads = target_pad(image, target_shape, mode="reflect")
                self.imgs_hr.append(image)
                self.pads.append(pads)

        with timer_context(
            "Generating (weighted) random patch centers..", verbose=verbose
        ):
            if patch_sampling == "uniform":
                weighted = False
            elif patch_sampling == "gradient":
                weighted = True
            self.centers = get_random_centers(
                self.imgs_hr,
                self.patch_size,
                self.n_patches,
                weighted=weighted,
            )

    def __len__(self):
        return self.n_patches

    def __getitem__(self, i):
        # Pull the HR patch
        aug_idx, center_idx = self.centers[i]
        patch_hr = get_patch(self.imgs_hr[aug_idx], center_idx, self.patch_size)
        patch_hr = torch.from_numpy(patch_hr)

        patch_hr = patch_hr.unsqueeze(0).unsqueeze(1)
        patch_lr = F.conv2d(patch_hr, self.blur_kernel, padding="same")

        patch_hr = patch_hr[self.ext_patch_crop]
        patch_lr = patch_lr[self.ext_patch_crop]

        # Downsample the LR patches
        patch_lr = resize(patch_lr, (self.slice_separation, 1), order=3)

        patch_hr = patch_hr.squeeze(0)
        patch_lr = patch_lr.squeeze(0)
        return patch_lr, patch_hr
