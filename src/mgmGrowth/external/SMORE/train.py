import argparse
import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from math import ceil

from .models.wdsr import WDSR
from .utils.train_set import TrainSet
from .utils.timer import timer_context
from .utils.parse_image_file import parse_image
from .utils.misc_utils import parse_device, LossProgBar
from .utils.blur_kernel_ops import calc_extended_patch_size, parse_kernel


# Optimize torch
# noinspection PyUnresolvedReferences
torch.backends.cudnn.benchmark = True

import torch


def set_random_seed(seed):
    # Set the random seed for CPU
    torch.manual_seed(seed)

    # Set the random seed for all GPUs (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic algorithms for cuDNN operations
        torch.backends.cudnn.deterministic = True

        # Disable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = False


# Example usage
set_random_seed(0)


def main(args=None):
    #################### ARGUMENTS ####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--weight-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--interp-order", type=int, default=3)
    parser.add_argument(
        "--n-patches", type=int, default=832000
    )  # The sum of 4 phases from Shuo's thesis
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--slice-thickness", type=float)
    parser.add_argument("--blur-kernel", type=str, default="rf-pulse-slr")
    parser.add_argument("--blur-kernel-fpath", type=str)
    parser.add_argument("--patch-sampling", type=str, default="gradient")
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    if not Path(args.in_fpath).exists():
        raise ValueError("Input image path does not exist.")
    if args.blur_kernel_fpath is not None and not Path(args.blur_kernel_fpath).exists():
        raise ValueError("Blur kernel fpath is specified but does not exist.")

    # A nice print statement divider for the command line
    text_div = "=" * 10

    print(f"{text_div} BEGIN TRAINING {text_div}")

    lr_patch_size = [args.patch_size, args.patch_size]
    weight_dir = Path(args.weight_dir)
    n_steps = int(ceil(args.n_patches / args.batch_size))
    learning_rate = 1e-3
    device = parse_device(args.gpu_id)

    if not weight_dir.exists():
        weight_dir.mkdir(parents=True)

    with timer_context("Parsing image file...", verbose=args.verbose):
        image, slice_separation, lr_axis, blur_fwhm, *_ = parse_image(
            args.in_fpath, args.slice_thickness, normalize_image=True
        )

    # ===== MODEL SETUP =====

    model = WDSR(
        n_resblocks=args.num_blocks,
        num_channels=args.num_channels,
        scale=slice_separation,
    ).to(device)

    patch_size = model.calc_out_patch_size(lr_patch_size)

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt,
        max_lr=learning_rate,
        total_steps=n_steps + 1,
        cycle_momentum=True,
    )
    opt.step()  # necessary for the LR scheduler

    scaler = torch.cuda.amp.GradScaler()
    loss_obj = torch.nn.L1Loss().to(device)

    # ===== LOAD AND PROCESS DATA =====

    blur_kernel = parse_kernel(args.blur_kernel_fpath, args.blur_kernel, blur_fwhm)
    ext_patch_size, ext_patch_crop = calc_extended_patch_size(blur_kernel, patch_size)
    ext_patch_size = (*ext_patch_size, 1)
    ext_patch_crop = (slice(None, None), slice(None, None), *ext_patch_crop)

    dataset = TrainSet(
        image=image,
        slice_separation=slice_separation,
        lr_axis=lr_axis,
        patch_size=ext_patch_size,
        ext_patch_crop=ext_patch_crop,
        device=device,
        blur_kernel=blur_kernel,
        n_patches=args.n_patches,
        verbose=args.verbose,
        patch_sampling=args.patch_sampling,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Dataset automatically shuffles
        pin_memory=True,
        num_workers=8,
    )

    # ===== TRAIN =====
    print(f"\n{text_div} TRAINING NETWORK {text_div}\n")
    train_st = time.time()

    loss_names = ["loss"]

    with LossProgBar(args.n_patches, args.batch_size, loss_names) as pbar:
        for patches_lr, patches_hr in data_loader:
            patches_hr_device = patches_hr.to(device)
            patches_lr_device = patches_lr.to(device)

            with torch.cuda.amp.autocast():
                patches_hr_hat = model(patches_lr_device)
                loss = loss_obj(patches_hr_hat, patches_hr_device)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            # Progress bar update
            pbar.update({"loss": loss})

    # ===== SAVE MODEL CONDITIONS =====
    weight_path = weight_dir / "best_weights.pt"

    torch.save({"model": model.state_dict()}, str(weight_path))

    train_en = time.time()
    print(f"\n\tElapsed time to finish training: {train_en - train_st:.4f}s")
