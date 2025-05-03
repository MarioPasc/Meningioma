import os
import numpy as np
from typing import List

from mgmGrowth import Stats  # type: ignore

from tqdm import tqdm  # type:ignore


def save_pdf_arrays_for_pulse(
    base_results_folder: str,
    pulse: str,
    seeds: List[int],
    dist_for_fit: str = "rayleigh",
    num_points: int = 300,
    bandwidth_h: float = 0.5,
    output_npz_name: str = "./arrays.npz",
):
    """
    For a given pulse, loads the NPZ files (noise_volume_{pulse}_seed{seed}.npz)
    from 'base_results_folder', then for each slice in the 3D volume, computes:
      - KDE-based PDF of background
      - Rayleigh-fitted PDF of background
      - x values for background
      - KDE-based PDF of generated noise
      - Rayleigh-fitted PDF of generated noise
      - x values for generated noise

    We store these arrays with shape (n_seeds, n_slices, num_points), i.e.:
       pdf_kde_bg[seed_index, slice_index, :]       # float
       pdf_rayl_bg[seed_index, slice_index, :]      # float
       x_bg[seed_index, slice_index, :]             # float

       pdf_kde_noise[seed_index, slice_index, :]
       pdf_rayl_noise[seed_index, slice_index, :]
       x_noise[seed_index, slice_index, :]

    Then it saves them all in ONE NPZ file (by default named "pdf_arrays_{pulse}.npz").

    Parameters
    ----------
    base_results_folder : str
        Path to the folder containing the NPZ files created by your pipeline.
    pulse : str
        The pulse label (e.g. "T1").
    seeds : List[int]
        The list of seed values to process.
    dist_for_fit : str
        Distribution name for the theoretical PDF fit in compute_pdf.
        By default "rayleigh". Could be "norm", "rice", etc.
    num_points : int
        Number of x-points for the computed PDF. (passed to compute_pdf(..., num_points=...))
    bandwidth_h : float
        Bandwidth for the KDE, also passed to compute_pdf(..., h=...).
    output_npz_name : str
        Optional name for the output file. If None, we use "pdf_arrays_{pulse}.npz"
    """

    if output_npz_name is None:
        output_npz_name = f"pdf_arrays_{pulse}.npz"
    out_path = os.path.join(base_results_folder, output_npz_name)

    # --------------------------------------------------------------------------
    # 1) Use the first seed to find shape of the volume
    # --------------------------------------------------------------------------
    first_seed_path = os.path.join(
        base_results_folder, f"noise_volume_{pulse}_seed{seeds[0]}.npz"
    )
    if not os.path.exists(first_seed_path):
        raise FileNotFoundError(
            f"File not found for seed={seeds[0]}: {first_seed_path}"
        )

    first_data = np.load(first_seed_path)["data"]  # shape => (nx, ny, nz, 5)
    _, _, nz, _ = first_data.shape
    n_slices = nz
    n_seeds = len(seeds)

    # --------------------------------------------------------------------------
    # 2) Prepare arrays to store results:
    #    shape => (n_seeds, n_slices, num_points)
    # --------------------------------------------------------------------------
    pdf_kde_bg = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)
    pdf_rayl_bg = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)
    x_bg_vals = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)

    pdf_kde_noise = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)
    pdf_rayl_noise = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)
    x_noise_vals = np.zeros((n_seeds, n_slices, num_points), dtype=np.float32)

    # --------------------------------------------------------------------------
    # 3) Loop over seeds & slices
    # --------------------------------------------------------------------------
    for s_idx, seed in enumerate(seeds):
        npz_filename = f"noise_volume_{pulse}_seed{seed}.npz"
        npz_filepath = os.path.join(base_results_folder, npz_filename)
        if not os.path.exists(npz_filepath):
            print(f"Warning: file not found {npz_filepath}. Skipping seed={seed}.")
            continue

        print(f"Computing for seed: {seed}")

        data_4d = np.load(npz_filepath)["data"]  # shape => (nx, ny, nz, 5)
        vol_3d = data_4d[..., 0]  # original volume
        mask_3d = data_4d[..., 1]  # segmentation mask
        noise_3d = data_4d[..., 4]  # final noise

        # cast mask to bool in case it's not
        mask_3d = mask_3d.astype(bool)

        for slice_idx in tqdm(
            iterable=range(n_slices),
            total=n_slices,
            desc="Computing per-slice PDF ...",
            colour="green",
        ):
            vol_slice = vol_3d[:, :, slice_idx]
            mask_slice = mask_3d[:, :, slice_idx]
            noise_slice = noise_3d[:, :, slice_idx]

            # A) Background values
            bg_values = vol_slice[~mask_slice]
            # B) Generated noise
            gen_values = noise_slice.ravel()

            # C) compute_pdf for background
            #    returns -> x_common, kde_est, pdf_fit, param_str, param_series
            x_bg, kde_bg, pdf_bg_fit, _, _ = Stats.compute_pdf(
                data=bg_values, h=bandwidth_h, dist=dist_for_fit, num_points=num_points
            )
            # D) compute_pdf for noise
            x_n, kde_n, pdf_n_fit, _, _ = Stats.compute_pdf(
                data=gen_values, h=bandwidth_h, dist=dist_for_fit, num_points=num_points
            )

            # Store them in the big arrays
            # We do not unify x_bg and x_n. We store them separately, because the min/max might differ
            # from BG vs. noise distribution. If you want to unify them, you'd do interpolation.
            pdf_kde_bg[s_idx, slice_idx, :] = kde_bg
            pdf_rayl_bg[s_idx, slice_idx, :] = pdf_bg_fit
            x_bg_vals[s_idx, slice_idx, :] = x_bg

            pdf_kde_noise[s_idx, slice_idx, :] = kde_n
            pdf_rayl_noise[s_idx, slice_idx, :] = pdf_n_fit
            x_noise_vals[s_idx, slice_idx, :] = x_n

    # --------------------------------------------------------------------------
    # 4) Save everything into a single NPZ
    # --------------------------------------------------------------------------
    # We store arrays for BG and noise.
    np.savez_compressed(
        out_path,
        pdf_kde_bg=pdf_kde_bg,
        pdf_rayl_bg=pdf_rayl_bg,
        x_bg_vals=x_bg_vals,
        pdf_kde_noise=pdf_kde_noise,
        pdf_rayl_noise=pdf_rayl_noise,
        x_noise_vals=x_noise_vals,
    )

    print(f"Saved PDF arrays for pulse={pulse} to {out_path}")


def main() -> None:
    results_folder: str = (
        "/home/mariopascual/Projects/MENINGIOMA/Results/NoiseEstimation_20250210_231309"
    )
    pulse: str = "T1"
    seeds: List[int] = [123, 456, 789]
    pdf_theoretical_fit: str = "rayleigh"
    num_points: int = 1000
    bandwidth_h: float = 0.5
    output_name: str = f"{pulse}_pdf_noise_estimation_arrays.npz"

    save_pdf_arrays_for_pulse(
        base_results_folder=results_folder,
        pulse=pulse,
        seeds=seeds,
        dist_for_fit=pdf_theoretical_fit,
        num_points=num_points,
        bandwidth_h=bandwidth_h,
        output_npz_name=output_name,
    )


if __name__ == "__main__":
    main()
