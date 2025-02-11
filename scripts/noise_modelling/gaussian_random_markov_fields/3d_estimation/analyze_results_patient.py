import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # type: ignore

from Meningioma import (  # type: ignore
    Stats,
    Metrics,
)

from tqdm import tqdm  # type: ignore


def plot_js_divergence_per_slice(
    base_results_folder: str,
    pulse: str,
    seeds: list,
    h_bandwidth: float = 0.5,
    output_fig_name: str = "js_divergence_per_slice.png",
):
    """
    For a single pulse, computes and plots the per-slice JS divergence across multiple seeds.
    - X-axis: slice index (0..nz-1).
    - Y-axis: JS divergence.
    - Plots 2 lines, each with mean and std across seeds:
        (1) KDE-based JS in blue
        (2) Rayleigh-based JS in red

    The function expects NPZ files named noise_volume_{pulse}_seed{seed}.npz
    with shape: (nx, ny, nz, 5), indices:
        data[...,0] = original volume
        data[...,1] = mask
        data[...,4] = final noise

    Parameters
    ----------
    base_results_folder : str
        Where the NPZ files are located.
    pulse : str
        The pulse to visualize (e.g. "T1").
    seeds : list
        List of seed integers to combine.
    h_bandwidth : float
        KDE bandwidth parameter.
    output_fig_name : str
        Filename for the final figure (PNG, etc.).
    """

    # 1) Find shape & check slices from the first NPZ
    first_seed_path = os.path.join(
        base_results_folder, f"noise_volume_{pulse}_seed{seeds[0]}.npz"
    )
    if not os.path.exists(first_seed_path):
        print(
            f"Error: File not found for pulse={pulse}, seed={seeds[0]}: {first_seed_path}"
        )
        return

    temp_data = np.load(first_seed_path)["data"]
    nx, ny, nz, _ = temp_data.shape

    # We'll store per-slice JS for each seed in arrays of shape (len(seeds), nz)
    js_kde_values = np.zeros((len(seeds), nz), dtype=np.float32)
    js_rayl_values = np.zeros((len(seeds), nz), dtype=np.float32)

    # 2) Loop over slices
    for slice_idx in tqdm(
        range(nz), desc="Processing slices ...", total=nz, colour="green"
    ):
        # 2a) For each seed, compute the JS divergence
        for s_i, seed in enumerate(seeds):
            print(f"Seed: {seed}")
            npz_filename = f"noise_volume_{pulse}_seed{seed}.npz"
            npz_path = os.path.join(base_results_folder, npz_filename)
            if not os.path.exists(npz_path):
                print(f"Warning: File not found {npz_path} for seed={seed}, skipping.")
                continue

            data_4d = np.load(npz_path)["data"]
            vol_3d = data_4d[..., 0]
            mask_3d = data_4d[..., 1].astype(bool)
            noise_3d = data_4d[..., 4]

            vol_slice = vol_3d[:, :, slice_idx]
            mask_slice = mask_3d[:, :, slice_idx]
            noise_slice = noise_3d[:, :, slice_idx]

            # Background vs noise
            bg_values = vol_slice[~mask_slice]
            gen_values = noise_slice.ravel()

            # 2b) Get PDF data for background & noise with dist="rayleigh"
            #     This returns both the KDE (kde_est) and the distribution-based PDF (pdf_fit).
            x_bg, bg_kde, bg_rayl, _, _ = Stats.compute_pdf(
                bg_values, h=h_bandwidth, dist="rayleigh"
            )
            x_n, n_kde, n_rayl, _, _ = Stats.compute_pdf(
                gen_values, h=h_bandwidth, dist="rayleigh"
            )

            # Note: The compute_pdf might produce slightly different x-ranges for bg vs. noise.
            # We'll unify them so we can use the same x-values for JS. We can do this by:
            x_min = min(x_bg.min(), x_n.min())
            x_max = max(x_bg.max(), x_n.max())
            n_points = max(len(x_bg), len(x_n))  # approximate
            x_common = np.linspace(x_min, x_max, n_points)

            # Interpolate the two BG PDFs onto x_common
            f_bg_kde = interp1d(x_bg, bg_kde, bounds_error=False, fill_value=0)
            f_bg_rayl = interp1d(x_bg, bg_rayl, bounds_error=False, fill_value=0)

            bg_kde_res = f_bg_kde(x_common)
            bg_rayl_res = f_bg_rayl(x_common)

            # Interpolate the two Noise PDFs onto x_common
            f_n_kde = interp1d(x_n, n_kde, bounds_error=False, fill_value=0)
            f_n_rayl = interp1d(x_n, n_rayl, bounds_error=False, fill_value=0)

            n_kde_res = f_n_kde(x_common)
            n_rayl_res = f_n_rayl(x_common)

            # 2c) Compute JS Divergence (KDE-based vs. Rayleigh-based)
            js_kde = Metrics.compute_jensen_shannon_divergence_pdfs(
                bg_kde_res, n_kde_res, x_common
            )
            js_rayl = Metrics.compute_jensen_shannon_divergence_pdfs(
                bg_rayl_res, n_rayl_res, x_common
            )

            js_kde_values[s_i, slice_idx] = js_kde
            js_rayl_values[s_i, slice_idx] = js_rayl

    # 3) Compute mean & std for each slice across seeds
    mean_js_kde = np.mean(js_kde_values, axis=0)
    std_js_kde = np.std(js_kde_values, axis=0)

    mean_js_rayl = np.mean(js_rayl_values, axis=0)
    std_js_rayl = np.std(js_rayl_values, axis=0)

    # 4) Plot
    slices = np.arange(nz)
    plt.figure(figsize=(10, 6))
    plt.title(f"JS Divergence per Slice\nPulse={pulse}, seeds={seeds}")

    # Plot KDE-based in blue
    plt.errorbar(
        slices,
        mean_js_kde,
        yerr=std_js_kde,
        fmt="-o",
        color="blue",
        ecolor="lightblue",
        capsize=3,
        label="KDE-based JS",
    )

    # Plot Rayleigh-based in red
    plt.errorbar(
        slices,
        mean_js_rayl,
        yerr=std_js_rayl,
        fmt="-o",
        color="red",
        ecolor="salmon",
        capsize=3,
        label="Rayleigh-based JS",
    )

    plt.xlabel("Slice Index")
    plt.ylabel("JS Divergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = os.path.join(base_results_folder, output_fig_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved JS divergence figure to: {out_path}")


def visualize_middle_slices_across_pulses(
    base_results_folder: str,
    pulses: list,
    seed: int,
    h_bandwidth: float = 0.5,
    dist_for_fit: str = "rayleigh",
    output_fig_name: str = "compare_middle_slices.png",
    fig_title: str = "Comparison of Middle Slice & Noise PDF",
):
    """
    Shows, for a single seed, a 2-row x len(pulses)-column figure:
    - Row 0: The middle slice of the original volume with the mask in red overlay
    - Row 1: Histograms + PDF comparisons (KDE & dist_for_fit) for BG vs. noise.

    Parameters
    ----------
    base_results_folder : str
        Folder where NPZ files (noise_volume_{pulse}_seed{seed}.npz) are located.
    pulses : list
        E.g. ["T1", "T1SIN", "T2", "SUSC"].
    seed : int
        The seed number whose NPZ files we want to compare across pulses.
    h_bandwidth : float
        Bandwidth (h) for the KDE.
    dist_for_fit : str
        Distribution name to pass to Stats.compute_pdf. Typically "rayleigh" or "norm".
    output_fig_name : str
        Filename for the output figure (PNG, SVG, etc.).
    fig_title : str
        Overall title for the figure.
    """

    # Create figure
    n_pulses = len(pulses)
    fig, axs = plt.subplots(nrows=2, ncols=n_pulses, figsize=(6 * n_pulses, 12))
    plt.suptitle(f"{fig_title} (seed={seed})", fontsize=16, y=0.97)

    for col_idx, pulse in enumerate(pulses):
        # ---------------------------------------------------------------------
        # 1) Load NPZ data
        # ---------------------------------------------------------------------
        npz_filename = f"noise_volume_{pulse}_seed{seed}.npz"
        npz_path = os.path.join(base_results_folder, npz_filename)
        if not os.path.exists(npz_path):
            print(f"Warning: file not found {npz_path}, skipping {pulse}.")
            continue

        data_obj = np.load(npz_path)
        data_4d = data_obj["data"]  # shape: (nx, ny, nz, 5)

        volume_3d = data_4d[..., 0]
        mask_3d = data_4d[..., 1].astype(bool)  # 0 or 1 → bool
        noise_3d = data_4d[..., 4]  # final combined noise

        nx, ny, nz, _ = data_4d.shape
        mid_slice_index = nz // 2

        # ---------------------------------------------------------------------
        # 2) Extract the middle slice
        # ---------------------------------------------------------------------
        volume_slice = volume_3d[:, :, mid_slice_index]
        mask_slice = mask_3d[:, :, mid_slice_index]
        noise_slice = noise_3d[:, :, mid_slice_index]

        # ---------------------------------------------------------------------
        # Row 0: volume slice + mask overlay
        # ---------------------------------------------------------------------
        ax0 = axs[0, col_idx]
        ax0.imshow(volume_slice, cmap="gray", origin="lower")
        mask_overlay = np.ma.masked_where(~mask_slice, mask_slice)
        ax0.imshow(mask_overlay, cmap="Reds_r", alpha=0.5, origin="lower")

        ax0.set_title(f"{pulse}\nMiddle Slice z={mid_slice_index}", fontsize=12)
        ax0.set_axis_off()

        # ---------------------------------------------------------------------
        # Row 1: Histograms + PDF (BG vs. noise)
        # ---------------------------------------------------------------------
        ax1 = axs[1, col_idx]

        # A) Background values from that slice
        bg_values = volume_slice[~mask_slice]
        # B) Generated noise from that slice
        gen_values = noise_slice.ravel()

        # Histograms as PMFs (density=True)
        num_bins = 50
        min_val = min(bg_values.min(), gen_values.min())
        max_val = max(bg_values.max(), gen_values.max())

        # BG histogram
        hist_bg, bins_bg = np.histogram(
            bg_values, bins=num_bins, range=(min_val, max_val), density=True
        )
        bin_centers_bg = 0.5 * (bins_bg[:-1] + bins_bg[1:])
        ax1.bar(
            bin_centers_bg,
            hist_bg,
            width=(bins_bg[1] - bins_bg[0]),
            alpha=0.3,
            color="red",
            label="BG Hist" if col_idx == 0 else "",
        )

        # Noise histogram
        hist_noise, bins_noise = np.histogram(
            gen_values, bins=num_bins, range=(min_val, max_val), density=True
        )
        bin_centers_noise = 0.5 * (bins_noise[:-1] + bins_noise[1:])
        ax1.bar(
            bin_centers_noise,
            hist_noise,
            width=(bins_noise[1] - bins_noise[0]),
            alpha=0.3,
            color="blue",
            label="Noise Hist" if col_idx == 0 else "",
        )

        # Fit & plot PDF (KDE & Theoretical) for BG
        x_bg, kde_bg, pdf_bg_fit, str_bg, _ = Stats.compute_pdf(
            bg_values, h=h_bandwidth, dist=dist_for_fit
        )
        ax1.plot(
            x_bg,
            kde_bg,
            color="red",
            linewidth=2,
            label="BG KDE" if col_idx == 0 else "",
        )
        ax1.plot(
            x_bg,
            pdf_bg_fit,
            color="red",
            ls="--",
            label=f"BG {dist_for_fit} Fit" if col_idx == 0 else "",
        )

        # Fit & plot PDF (KDE & Theoretical) for Noise
        x_n, kde_n, pdf_n_fit, str_n, _ = Stats.compute_pdf(
            gen_values, h=h_bandwidth, dist=dist_for_fit
        )
        ax1.plot(
            x_n,
            kde_n,
            color="blue",
            linewidth=2,
            label="Noise KDE" if col_idx == 0 else "",
        )
        ax1.plot(
            x_n,
            pdf_n_fit,
            color="blue",
            ls="--",
            label=f"Noise {dist_for_fit} Fit" if col_idx == 0 else "",
        )

        ax1.set_title(f"{pulse} - BG vs Noise PDFs", fontsize=11)
        ax1.set_xlabel("Intensity")
        ax1.set_ylabel("Density")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

    # Common legend
    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(base_results_folder, output_fig_name)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")


def main() -> None:

    results_folder: str = (
        "/home/mariopasc/Python/Results/Meningioma/noise_modelling/NoiseEstimation_20250210_231309"
    )

    visualize_middle_slices_across_pulses(
        base_results_folder=results_folder,
        pulses=["T1", "T1SIN", "T2", "SUSC"],
        seed=123,
        h_bandwidth=0.5,
        dist_for_fit="rayleigh",
        output_fig_name="middle_slices_seed123.svg",
        fig_title="Comparison of Middle Slice & Noise PDF",
    )

    print("Plotting JS divergence results")

    plot_js_divergence_per_slice(
        base_results_folder=results_folder,
        pulse="T1",
        seeds=[123, 456, 789],  # however many seeds you used
        h_bandwidth=0.5,
        output_fig_name="js_divergence_T1.svg",
    )


if __name__ == "__main__":
    main()
