import argparse
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
from resize.affine import update_affine
import torch
from tqdm import tqdm

from .models.wdsr import WDSR
from .utils.patch_ops import find_integer_p, calc_slices_to_crop

from .utils.timer import timer_context
from .utils.parse_image_file import (
    parse_image,
    inv_normalize,
    lr_axis_to_z,
    z_axis_to_lr_axis,
)
from .utils.misc_utils import parse_device
from .utils.rotate import rotate_vol_2d


def apply_to_vol(model, image, batch_size):
    result = []
    for st in tqdm(range(0, image.shape[0], batch_size)):
        en = st + batch_size
        batch = image[st:en]
        with torch.inference_mode():
            with torch.cuda.amp.autocast():
                sr = model(batch).detach().cpu()
        result.append(sr)
    result = torch.cat(result, dim=0)
    return result


def main(args=None):
    main_st = time.time()
    #################### ARGUMENTS ####################

    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=str, required=True)
    parser.add_argument("--out-fpath", type=str, required=True)
    parser.add_argument("--weight-dir", type=str, required=True)
    parser.add_argument("--num-blocks", type=int, default=16)
    parser.add_argument("--num-channels", type=int, default=32)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--n-rots", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args(args if args is not None else sys.argv[1:])

    # A nice print statement divider for the command line
    text_div = "=" * 10

    print(f"{text_div} BEGIN PREDICTION {text_div}")

    out_dir = Path(args.out_fpath).parent
    weight_dir = Path(args.weight_dir)

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    device = parse_device(args.gpu_id)

    # ===== LOAD AND PROCESS DATA =====

    image, slice_separation, lr_axis, _, header, affine, orig_min, orig_max = parse_image(
        args.in_fpath, normalize_image=True
    )
    image = lr_axis_to_z(image, lr_axis)
    # pad the number of slices out so we achieve the correct final resolution
    n_slices_pad = find_integer_p(image.shape[2], slice_separation)
    n_slices_crop = calc_slices_to_crop(n_slices_pad, slice_separation)
    image = np.pad(image, ((0, 0), (0, 0), (0, n_slices_pad)), mode="reflect")
    image = torch.from_numpy(image)

    # ===== MODEL SETUP =====
    checkpoint = torch.load(weight_dir / "best_weights.pt")

    model = WDSR(
        n_resblocks=args.num_blocks,
        num_channels=args.num_channels,
        scale=slice_separation,
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    # ===== PREDICT =====
    min_angle = 0
    max_angle = 90
    angles = range(min_angle, max_angle + 1, max_angle // (args.n_rots - 1))

    model_preds = []
    for i, angle in enumerate(angles):
        context_str = f"Super-resolving at {angle} degrees: {i+1}/{args.n_rots}"
        with timer_context(context_str, verbose=args.verbose):
            # Rotate in-plane. Image starts as (hr_axis, hr_axis, lr_axis)
            image_rot = rotate_vol_2d(image.to(device), angle)
            # Ensure the LR axis is s.t. (hr_axis, C, lr_axis, hr_axis)
            image_rot = image_rot.permute(0, 2, 1).unsqueeze(1)
            # Run model
            rot_result = apply_to_vol(model, image_rot, args.batch_size)
            # Return to (hr_axis, hr_axis, lr_axis)
            result = rot_result.squeeze(1).permute(0, 2, 1)

            model_preds.append(rotate_vol_2d(result, -angle))

    # ===== FINALIZE =====
    final_out = torch.mean(torch.stack(model_preds), dim=0)
    final_out = final_out.detach().cpu().numpy().astype(np.float32)
    final_out = inv_normalize(final_out, orig_min, orig_max, a=0, b=1)

    # Re-crop to target shape
    if n_slices_crop != 0:
        final_out = final_out[:, :, :-n_slices_crop]
    # Reorient to original orientation
    final_out = z_axis_to_lr_axis(final_out, lr_axis)

    print("Saving image...")
    # Update affine matrix
    scales = [1, 1]
    scales.insert(lr_axis, 1 / slice_separation)
    new_affine = update_affine(affine, scales)

    # Write nifti
    out_obj = nib.Nifti1Image(final_out, affine=new_affine, header=header)
    nib.save(out_obj, args.out_fpath)

    main_en = time.time()
    print("\n\nDONE\nElapsed time: {:.4f}s\n".format(main_en - main_st))
    print("\tWritten to: {}\n".format(str(Path(args.out_fpath))))
    print(f"{text_div} END PREDICTION {text_div}")
