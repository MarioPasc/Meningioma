import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # type: ignore
from typing import List

# Meningioma imports
from Meningioma import Metrics  # type: ignore

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift  # type: ignore
from typing import List

# Assuming your Meningioma imports:
from Meningioma import Stats
from tqdm import tqdm  # type: ignore

import scienceplots  # type: ignore

# Set up plotting style.
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


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
    plt.show()
    print(f"Saved figure: {out_path}")


def generate_abstract_figures(
    base_results_folder: str,
    pulse: str,
    seed: int,
    slices_of_interest: List[int],
    output_folder_name: str = "abstract_figures",
    bandwidth_h: float = 0.5,
    num_bins: int = 50,
):
    """
    Creates and saves multiple figure files for the chosen pulse, seed, and slices:
      1) Original slice in grayscale
      2) Original slice + overlayed mask (alpha=0.5, red)
      3) For each of real, imaginary, and final noise:
         - Gray image of that noise slice
         - PDF/PMF comparison: histogram + KDE + Theoretical distribution
            * Real/Imag => 'norm' distribution
            * Final => 'rayleigh'
            * Includes param_str in the legend
         - 2D FFT power spectrum (log scale)

    All in .svg format, with no titles, no colorbars, and minimal/no borders.

    Subfolders in abstract_figures/:
        slices/
        slices_overlay/
        real_noise/
        imag_noise/
        final_noise/

    Parameters
    ----------
    base_results_folder : str
        Folder where your NPZ file (noise_volume_{pulse}_seed{seed}.npz) is located.
    pulse : str
        Pulse type (e.g. 'T1').
    seed : int
        Specific seed to load from the NPZ.
    slices_of_interest : List[int]
        The slice indices to plot.
    output_folder_name : str
        Name of the top-level folder to create for storing the figures.
    bandwidth_h : float
        Bandwidth for KDE (Stats.compute_pdf).
    num_bins : int
        Number of bins for the histogram (PMF).
    """

    # 1) Prepare output subfolders
    top_output = os.path.join(base_results_folder, output_folder_name)
    os.makedirs(top_output, exist_ok=True)

    folder_slices = os.path.join(top_output, "slices")
    folder_slices_overlay = os.path.join(top_output, "slices_overlay")
    folder_real = os.path.join(top_output, "real_noise")
    folder_imag = os.path.join(top_output, "imag_noise")
    folder_final = os.path.join(top_output, "final_noise")

    for fdir in [
        folder_slices,
        folder_slices_overlay,
        folder_real,
        folder_imag,
        folder_final,
    ]:
        os.makedirs(fdir, exist_ok=True)

    # 2) Load data from NPZ
    npz_filename = f"noise_volume_{pulse}_seed{seed}.npz"
    npz_path = os.path.join(base_results_folder, npz_filename)
    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found: {npz_path}")
        return

    data_4d = np.load(npz_path)["data"]  # shape => (nx, ny, nz, 5)
    volume_3d = data_4d[..., 0]
    mask_3d = data_4d[..., 1].astype(bool)
    real_3d = data_4d[..., 2]
    imag_3d = data_4d[..., 3]
    final_3d = data_4d[..., 4]

    nx, ny, nz, _ = data_4d.shape

    # 3) Helper function: generate a 2D FFT power spectrum for a slice
    def compute_2d_fft_power_spectrum(image_2d: np.ndarray) -> np.ndarray:
        """
        Returns log(1 + power_spectrum) of the 2D FFT of image_2d.
        """
        fft_res = fft2(image_2d)
        fft_shifted = fftshift(fft_res)
        mag_spectrum = np.abs(fft_shifted)
        power_spectrum = mag_spectrum**2
        return np.log1p(power_spectrum)

    # 4) Iterate over the requested slices
    for slice_idx in slices_of_interest:
        if slice_idx < 0 or slice_idx >= nz:
            print(
                f"Warning: slice index {slice_idx} out of range [0, {nz-1}]. Skipping."
            )
            continue

        slice_2d = volume_3d[:, :, slice_idx]
        mask_2d = mask_3d[:, :, slice_idx]

        # ---------------------------------------------------------------------
        # (A) Save the slice in grayscale (no titles, no borders)
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_2d, cmap="gray", origin="lower")
        ax.set_axis_off()
        out_path_slice = os.path.join(
            folder_slices, f"{pulse}_seed{seed}_slice{slice_idx}.svg"
        )
        plt.savefig(out_path_slice, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # ---------------------------------------------------------------------
        # (B) Save the slice + overlay in red
        # ---------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_2d, cmap="gray", origin="lower")
        mask_overlay = np.ma.masked_where(~mask_2d, mask_2d)
        ax.imshow(mask_overlay, cmap="Reds_r", alpha=0.5, origin="lower")
        ax.set_axis_off()
        out_path_slice_overlay = os.path.join(
            folder_slices_overlay, f"{pulse}_seed{seed}_slice{slice_idx}_overlay.svg"
        )
        plt.savefig(out_path_slice_overlay, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # ---------------------------------------------------------------------
        # (C) For each noise field: real, imaginary, final
        # ---------------------------------------------------------------------
        real_slice_2d = real_3d[:, :, slice_idx]
        imag_slice_2d = imag_3d[:, :, slice_idx]
        final_slice_2d = final_3d[:, :, slice_idx]

        # Helper function to handle plotting
        def plot_noise_slice_and_save(noise_slice_2d: np.ndarray, channel_name: str):
            """
            channel_name in {"real", "imag", "final"} used to pick subfolder & distribution fit.
            """
            if channel_name == "real":
                subfolder = folder_real
                dist_name = "norm"
            elif channel_name == "imag":
                subfolder = folder_imag
                dist_name = "norm"
            else:
                subfolder = folder_final
                dist_name = "rayleigh"

            # ------------------------------------------------------------
            # (1) Noise slice in grayscale
            # ------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(noise_slice_2d, cmap="gray", origin="lower")
            ax.set_axis_off()
            out_img_path = os.path.join(
                subfolder, f"{channel_name}_slice{slice_idx}_field.svg"
            )
            plt.savefig(out_img_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # ------------------------------------------------------------
            # (2) PDF/PMF figure
            # ------------------------------------------------------------
            data_flat = noise_slice_2d.ravel()
            min_val, max_val = data_flat.min(), data_flat.max()

            fig, ax = plt.subplots(figsize=(7, 5))

            # Turn off everything except the legend
            ax.set_axis_off()

            # We'll still plot our histogram and lines, but no axis or spines
            hist_vals, bin_edges = np.histogram(
                data_flat, bins=num_bins, range=(min_val, max_val), density=True
            )
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(
                bin_centers,
                hist_vals,
                width=(bin_edges[1] - bin_edges[0]),
                alpha=0.3,
                color="gray",
                label="Histogram (PMF)",
            )

            # compute_pdf => (x_common, kde_est, pdf_fit, param_str, param_series)
            x_c, kde_est, pdf_fit, param_str, _ = Stats.compute_pdf(
                data=data_flat, h=bandwidth_h, dist=dist_name
            )

            ax.plot(x_c, kde_est, color="blue", lw=2, label="KDE")
            # Put the param_str in the legend as requested
            ax.plot(
                x_c,
                pdf_fit,
                color="red",
                ls="--",
                label=f"{dist_name} Fit\n{param_str}",
            )

            # Because the axis is off, the legend can float inside the figure
            ax.legend()

            out_pdf_path = os.path.join(
                subfolder, f"{channel_name}_slice{slice_idx}_pdf.svg"
            )
            plt.savefig(out_pdf_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            # ------------------------------------------------------------
            # (3) FFT power spectrum (log(1 + power)), no colorbar
            # ------------------------------------------------------------
            power_2d = compute_2d_fft_power_spectrum(noise_slice_2d)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(power_2d, cmap="gray", origin="lower")
            ax.set_axis_off()  # no ticks, spines, or axis labels

            out_fft_path = os.path.join(
                subfolder, f"{channel_name}_slice{slice_idx}_fft.svg"
            )
            # No colorbar, no title, remove borders
            plt.savefig(out_fft_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        # Now call that helper for real, imag, final
        plot_noise_slice_and_save(real_slice_2d, "real")
        plot_noise_slice_and_save(imag_slice_2d, "imag")
        plot_noise_slice_and_save(final_slice_2d, "final")

    print(f"All figures saved to: {top_output}")


def visualize_noise_components_2x3(
    npz_path: str,
    slice_index: int = 0,
    h_bandwidth: float = 0.5,
    output_filename: str = "noise_2x3_visualization.png",
):
    """
    Loads a single NPZ file containing [volume, mask, real_noise, imaginary_noise, final_noise],
    extracts one slice (default: middle), and creates a 2x3 figure:

    Row 0: 3 subplots showing real noise slice, imaginary noise slice, and final noise slice.
    Row 1: 3 subplots showing hist + PDF fits (KDE & theoretical):
        - real noise (normal fit),
        - imaginary noise (normal fit),
        - final noise (rayleigh fit).

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file with 'data' shaped (nx, ny, nz, 5).
    slice_index : int, optional
        Index of the slice to visualize. If None, uses the middle slice.
    h_bandwidth : float
        KDE bandwidth parameter for Stats.compute_pdf.
    output_filename : str
        Name of the output image file.
    """

    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found: {npz_path}")
        return

    # Load data
    data_obj = np.load(npz_path)
    data_4d = data_obj["data"]  # shape = (nx, ny, nz, 5)

    # Indices: 0=volume, 1=mask, 2=real, 3=imag, 4=final
    real_3d = data_4d[..., 2]
    imag_3d = data_4d[..., 3]
    final_3d = data_4d[..., 4]

    nx, ny, nz, _ = data_4d.shape

    if slice_index is None:
        slice_index = nz // 2  # middle slice by default

    if slice_index < 0 or slice_index >= nz:
        print(
            f"Warning: slice_index={slice_index} is out of range [0..{nz-1}]. Using the middle slice."
        )
        slice_index = nz // 2

    # Extract slices
    real_slice = real_3d[:, :, slice_index]
    imag_slice = imag_3d[:, :, slice_index]
    final_slice = final_3d[:, :, slice_index]

    # Create figure: 2 rows x 3 columns
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    # ---------------------------------------------------
    # Row 0: Show real, imaginary, final as images
    # ---------------------------------------------------
    # Real
    axs[0, 0].imshow(real_slice, cmap="gray", origin="lower")
    axs[0, 0].set_title("Real Noise (Slice)", fontsize=12)
    axs[0, 0].axis("off")

    # Imag
    axs[0, 1].imshow(imag_slice, cmap="gray", origin="lower")
    axs[0, 1].set_title("Imag Noise (Slice)", fontsize=12)
    axs[0, 1].axis("off")

    # Final
    axs[0, 2].imshow(final_slice, cmap="gray", origin="lower")
    axs[0, 2].set_title("Final Noise (Slice)", fontsize=12)
    axs[0, 2].axis("off")

    # ---------------------------------------------------
    # Row 1: Hist + PDF fits
    # ---------------------------------------------------
    # We assume:
    # - Real noise => normal
    # - Imag noise => normal
    # - Final noise => rayleigh

    # Helper function to plot histogram + PDF in a single axis
    def plot_noise_pdf(ax, data_slice, dist_name, color_hist, label_prefix):
        data_flat = data_slice.ravel()
        min_val, max_val = data_flat.min(), data_flat.max()

        num_bins = 40
        # histogram as PMF
        hist_vals, bin_edges = np.histogram(
            data_flat, bins=num_bins, range=(min_val, max_val), density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.bar(
            bin_centers,
            hist_vals,
            width=(bin_edges[1] - bin_edges[0]),
            alpha=0.3,
            color="gray",
            label=f"{label_prefix} Hist",
        )

        # compute_pdf => (x_common, kde_est, pdf_fit, param_str, param_series)
        x_c, kde_est, pdf_fit, param_str, _ = Stats.compute_pdf(
            data=data_flat, h=h_bandwidth, dist=dist_name
        )

        # Plot KDE
        ax.plot(x_c, kde_est, color=color_hist, lw=2, label=f"{label_prefix} KDE")
        # Plot theoretical PDF
        ax.plot(
            x_c,
            pdf_fit,
            color="black",
            ls="--",
            label=f"{dist_name.capitalize()} Fit\n{param_str}",
        )

        ax.set_xlabel("Intensity")
        ax.set_ylabel("PDF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    # --- Real => normal
    plot_noise_pdf(
        ax=axs[1, 0],
        data_slice=real_slice,
        dist_name="norm",
        color_hist="red",
        label_prefix="Real",
    )

    # --- Imag => normal
    plot_noise_pdf(
        ax=axs[1, 1],
        data_slice=imag_slice,
        dist_name="norm",
        color_hist="red",
        label_prefix="Imag",
    )

    # --- Final => rayleigh
    plot_noise_pdf(
        ax=axs[1, 2],
        data_slice=final_slice,
        dist_name="rayleigh",
        color_hist="red",
        label_prefix="Final",
    )

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(npz_path), output_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved figure to: {out_path}")


def plot_js_divergence_per_slice_from_pdf(
    base_results_folder: str,
    pulse: str,
    seeds: List[int],
    output_fig_name: str = "js_divergence_per_slice.png",
):
    """
    For a single pulse, computes and plots the per-slice JS divergence across multiple seeds,
    using the *already-computed* PDF arrays (KDE & Rayleigh) from 'pdf_arrays_{pulse}.npz'.

    - X-axis: slice index (0..n_slices-1).
    - Y-axis: JS divergence.
    - Plots 2 lines, each with mean and std across seeds:
        (1) KDE-based JS in blue
        (2) Rayleigh-based JS in red

    The NPZ from save_pdf_arrays_for_pulse(...) is expected to have the following arrays, all
    shape: (n_seeds, n_slices, num_points):

        pdf_kde_bg
        pdf_rayl_bg
        x_bg_vals

        pdf_kde_noise
        pdf_rayl_noise
        x_noise_vals

    We assume that the *first* dimension (axis=0) corresponds to seed index in the same order
    as the given 'seeds' list.

    Parameters
    ----------
    base_results_folder : str
        Where the NPZ file 'pdf_arrays_{pulse}.npz' is located.
    pulse : str
        The pulse to visualize (e.g. "T1").
    seeds : List[int]
        The list of seed integers (in the order you saved them).
    output_fig_name : str
        Filename for the final figure (PNG, etc.).
    """

    # 1) Load the PDF npz file
    pdf_npz_name = f"{pulse}_pdf_noise_estimation_arrays.npz"
    pdf_npz_path = os.path.join(base_results_folder, pdf_npz_name)
    if not os.path.exists(pdf_npz_path):
        print(f"Error: PDF NPZ file not found for pulse={pulse} at: {pdf_npz_path}")
        return

    data = np.load(pdf_npz_path)

    # Each has shape => (n_seeds, n_slices, num_points)
    pdf_kde_bg = data["pdf_kde_bg"]  # background KDE
    pdf_rayl_bg = data["pdf_rayl_bg"]  # background Rayleigh (or chosen dist)
    x_bg_vals = data["x_bg_vals"]

    pdf_kde_noise = data["pdf_kde_noise"]  # noise KDE
    pdf_rayl_noise = data["pdf_rayl_noise"]  # noise Rayleigh
    x_noise_vals = data["x_noise_vals"]

    n_seeds, n_slices, n_points = pdf_kde_bg.shape

    # We'll store per-slice JS for each seed in arrays (n_seeds, n_slices)
    js_kde_values = np.zeros((n_seeds, n_slices), dtype=np.float32)
    js_rayl_values = np.zeros((n_seeds, n_slices), dtype=np.float32)

    # 2) Loop over slices and seeds
    for slice_idx in range(n_slices):
        for s_i in range(n_seeds):
            # Extract the background distribution for this slice, seed=s_i
            x_bg = x_bg_vals[s_i, slice_idx, :]
            bg_kde = pdf_kde_bg[s_i, slice_idx, :]
            bg_rayl = pdf_rayl_bg[s_i, slice_idx, :]

            # Extract the noise distribution for this slice, seed=s_i
            x_n = x_noise_vals[s_i, slice_idx, :]
            n_kde = pdf_kde_noise[s_i, slice_idx, :]
            n_rayl = pdf_rayl_noise[s_i, slice_idx, :]

            # 2a) Unify the domain via interpolation (only if needed).
            # Because min(x_bg) might differ from min(x_n), etc.
            x_min = min(x_bg.min(), x_n.min())
            x_max = max(x_bg.max(), x_n.max())
            # We'll pick the same number of points to unify
            # (Alternatively, you could pick a larger number or keep it at n_points.)
            unified_x = np.linspace(x_min, x_max, n_points)

            f_bg_kde = interp1d(x_bg, bg_kde, bounds_error=False, fill_value=0)
            f_bg_rayl = interp1d(x_bg, bg_rayl, bounds_error=False, fill_value=0)
            f_n_kde = interp1d(x_n, n_kde, bounds_error=False, fill_value=0)
            f_n_rayl = interp1d(x_n, n_rayl, bounds_error=False, fill_value=0)

            bg_kde_res = f_bg_kde(unified_x)
            bg_rayl_res = f_bg_rayl(unified_x)
            noise_kde_res = f_n_kde(unified_x)
            noise_rayl_res = f_n_rayl(unified_x)

            # 2b) Compute JS divergence
            #     Blue => KDE-based (bg_kde vs n_kde)
            js_kde = Metrics.compute_jensen_shannon_divergence_pdfs(
                bg_kde_res, noise_kde_res, unified_x
            )

            #     Red => Rayleigh-based (bg_rayl vs n_rayl)
            js_rayl = Metrics.compute_jensen_shannon_divergence_pdfs(
                bg_rayl_res, noise_rayl_res, unified_x
            )

            js_kde_values[s_i, slice_idx] = js_kde
            js_rayl_values[s_i, slice_idx] = js_rayl

    # 3) Compute mean & std for each slice across seeds
    mean_js_kde = np.mean(js_kde_values, axis=0)
    std_js_kde = np.std(js_kde_values, axis=0)

    mean_js_rayl = np.mean(js_rayl_values, axis=0)
    std_js_rayl = np.std(js_rayl_values, axis=0)

    # 4) Plot
    slices = np.arange(n_slices)
    _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot KDE-based in blue
    plt.errorbar(
        slices,
        mean_js_kde,
        yerr=std_js_kde,
        fmt="-o",
        color="blue",
        ecolor="lightblue",
        capsize=3,
        label="KDE-estimated Divergence",
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
        label="Rayleigh-estimated Divergence",
    )

    ax.set_xlabel("Slice Index")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    out_path = os.path.join(base_results_folder, output_fig_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved JS divergence figure to: {out_path}")


def visualize_slices_with_rayleigh_and_kde(
    npz_path: str,
    slices: list,
    output_filename: str = "multi_slice_visualization.png",
    h_bandwidth: float = 0.5,
):
    """
    Creates an N x 3 figure (where N = len(slices)) to visualize:
      1) volume slice + mask overlay,
      2) Rayleigh-fitted PDFs for background vs. generated noise,
      3) KDE-based PDF + histogram for background vs. generated noise,

    *All subplots in columns 2 and 3 (across all rows) share the same X and Y axes.*
    Column 1 (the volume+mask images) does not share axes with the PDFs.

    For each slice, we also compute and display the JS divergence:
      - JS_Rayleigh: between the two Rayleigh-fitted PDFs,
      - JS_KDE: between the two KDE-based PDFs.

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file containing (nx, ny, nz, 5) data:
         - 0 => original volume
         - 1 => mask
         - 2 => real noise
         - 3 => imaginary noise
         - 4 => final noise
    slices : list of int
        The slice indices to visualize. One row per slice in the figure.
    output_filename : str
        Name of the output image file (e.g., PNG or SVG).
    h_bandwidth : float
        Bandwidth for the KDE estimation in Stats.compute_pdf.

    Example
    -------
    visualize_slices_with_rayleigh_and_kde(
        npz_path="noise_volume_T1_seed123.npz",
        slices=[10, 20, 30],
        output_filename="multi_slice_T1_seed123.png",
        h_bandwidth=0.5
    )
    """

    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found: {npz_path}")
        return

    data_npz = np.load(npz_path)
    data_4d = data_npz["data"]  # shape: (nx, ny, nz, 5)

    volume_3d = data_4d[..., 0]
    mask_3d = data_4d[..., 1].astype(bool)
    noise_3d = data_4d[..., 4]  # final noise

    nx, ny, nz, _ = data_4d.shape

    # Create figure: N rows x 3 columns, initially no global sharing
    N = len(slices)
    fig, axs = plt.subplots(nrows=N, ncols=3, figsize=(14, 4 * N))

    # If there's only one slice, axs is likely a 1D array => make it 2D for indexing
    if N == 1:
        axs = [axs]

    # -------------------------------------------------------------------------
    # MANUALLY ENFORCE that all subplots in columns 2 and 3 share the same X/Y
    # We'll pick axs[0][1] as the "master" for columns (1) and (2).
    # That means everything in columns 2 or 3 uses the same axis scale.
    # (row=0, col=1) is the first row's Rayleigh PDF plot.
    # Also, link row=0, col=2 to the same master:
    axs[0][2].sharex(axs[0][1])
    axs[0][2].sharey(axs[0][1])

    # Now link all subsequent rows in col=1 and col=2 to that same master.
    for row_idx in range(1, N):
        axs[row_idx][1].sharex(axs[0][1])
        axs[row_idx][1].sharey(axs[0][1])

        axs[row_idx][2].sharex(axs[0][1])
        axs[row_idx][2].sharey(axs[0][1])
    # -------------------------------------------------------------------------

    for row_idx, slice_idx in enumerate(slices):
        if slice_idx < 0 or slice_idx >= nz:
            print(f"Warning: slice {slice_idx} out of range [0..{nz - 1}]. Skipping.")
            continue

        vol_slice = volume_3d[:, :, slice_idx]
        mask_slice = mask_3d[:, :, slice_idx]
        gen_slice = noise_3d[:, :, slice_idx]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Column 1: volume + mask overlay
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax_vol = axs[row_idx][0]
        ax_vol.imshow(vol_slice, cmap="gray", origin="lower")
        mask_overlay = np.ma.masked_where(~mask_slice, mask_slice)
        ax_vol.imshow(mask_overlay, cmap="Reds_r", alpha=0.5, origin="lower")
        ax_vol.set_title(f"Slice {slice_idx}\nVolume + Mask", fontsize=11)
        ax_vol.axis("off")

        # Extract background (outside mask) and generated noise
        bg_values = vol_slice[~mask_slice]
        gen_values = gen_slice.ravel()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Column 2: Rayleigh-fitted PDF
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax_rayl = axs[row_idx][1]

        x_bg, _, bg_pdf_rayl, bg_params_str, _ = Stats.compute_pdf(
            bg_values, h=h_bandwidth, dist="rayleigh"
        )
        x_gen, _, gen_pdf_rayl, gen_params_str, _ = Stats.compute_pdf(
            gen_values, h=h_bandwidth, dist="rayleigh"
        )

        # unify x-range
        x_min = min(x_bg.min(), x_gen.min())
        x_max = max(x_bg.max(), x_gen.max())
        n_points = max(len(x_bg), len(x_gen))
        x_common = np.linspace(x_min, x_max, n_points)

        # Interpolate
        f_bg_r = interp1d(x_bg, bg_pdf_rayl, bounds_error=False, fill_value=0)
        f_gen_r = interp1d(x_gen, gen_pdf_rayl, bounds_error=False, fill_value=0)
        bg_pdf_r_res = f_bg_r(x_common)
        gen_pdf_r_res = f_gen_r(x_common)

        # Plot
        ax_rayl.plot(
            x_common,
            bg_pdf_r_res,
            color="red",
            linewidth=2,
            label=f"Background Rayleigh\n{bg_params_str}",
        )
        ax_rayl.plot(
            x_common,
            gen_pdf_r_res,
            color="blue",
            linewidth=2,
            label=f"Generated Rayleigh\n{gen_params_str}",
        )

        # Compute JS divergence (Rayleigh-based PDFs)
        js_rayl = Metrics.compute_jensen_shannon_divergence_pdfs(
            bg_pdf_r_res, gen_pdf_r_res, x_common
        )
        ax_rayl.text(
            0.95,
            0.10,
            f"JS Divergence (Raylaith)={js_rayl:.4f}",
            transform=ax_rayl.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            fontsize=9,
        )

        if row_idx == 0:
            ax_rayl.set_title("Rayleigh-Fitted PDF", fontsize=11)
        if row_idx == 2:
            ax_rayl.set_xlabel("Intensity")
        ax_rayl.set_ylabel("Density")
        ax_rayl.legend(fontsize=8, loc="upper right")
        ax_rayl.spines["top"].set_visible(False)
        ax_rayl.spines["right"].set_visible(False)
        ax_rayl.xaxis.tick_bottom()
        ax_rayl.yaxis.tick_left()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Column 3: KDE-based PDF comparison + histogram
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax_kde = axs[row_idx][2]

        # We'll do a histogram for BG (red) and Gen (blue)
        num_bins = 15
        all_min = min(bg_values.min(), gen_values.min())
        all_max = max(bg_values.max(), gen_values.max())

        bg_hist, bg_edges = np.histogram(
            bg_values, bins=num_bins, range=(all_min, all_max), density=True
        )
        gen_hist, gen_edges = np.histogram(
            gen_values, bins=num_bins, range=(all_min, all_max), density=True
        )
        bg_centers = 0.5 * (bg_edges[:-1] + bg_edges[1:])
        gen_centers = 0.5 * (gen_edges[:-1] + gen_edges[1:])

        ax_kde.bar(
            bg_centers,
            bg_hist,
            width=(bg_edges[1] - bg_edges[0]),
            alpha=0.3,
            color="red",
            label="Background Histogram",
        )
        ax_kde.bar(
            gen_centers,
            gen_hist,
            width=(gen_edges[1] - gen_edges[0]),
            alpha=0.3,
            color="blue",
            label="Generated Histogram",
        )

        # The compute_pdf call returns both the Rayleigh-fit PDF and the KDE
        x_bg2, bg_kde, _, _, _ = Stats.compute_pdf(
            bg_values, h=h_bandwidth, dist="rayleigh"
        )
        x_gen2, gen_kde, _, _, _ = Stats.compute_pdf(
            gen_values, h=h_bandwidth, dist="rayleigh"
        )

        # unify x for KDE interpolation
        x_min2 = min(x_bg2.min(), x_gen2.min())
        x_max2 = max(x_bg2.max(), x_gen2.max())
        n_points2 = max(len(x_bg2), len(x_gen2))
        x_common2 = np.linspace(x_min2, x_max2, n_points2)

        f_bg_k = interp1d(x_bg2, bg_kde, bounds_error=False, fill_value=0)
        f_gen_k = interp1d(x_gen2, gen_kde, bounds_error=False, fill_value=0)

        bg_kde_res = f_bg_k(x_common2)
        gen_kde_res = f_gen_k(x_common2)

        # Plot the KDE lines
        ax_kde.plot(x_common2, bg_kde_res, color="red", lw=2, label="Background KDE")
        ax_kde.plot(x_common2, gen_kde_res, color="blue", lw=2, label="Generated KDE")

        # Compute JS divergence (KDE-based PDFs)
        js_kde_val = Metrics.compute_jensen_shannon_divergence_pdfs(
            bg_kde_res, gen_kde_res, x_common2
        )
        ax_kde.text(
            0.95,
            0.1,
            f"JS Divergence (KDE)={js_kde_val:.4f}",
            transform=ax_kde.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
            fontsize=9,
        )

        if row_idx == 0:
            ax_kde.set_title("KDE Comparison", fontsize=11)
        if row_idx == 2:
            ax_kde.set_xlabel("Intensity")
        ax_kde.legend(fontsize=8, loc="upper right")
        ax_kde.spines["top"].set_visible(False)
        ax_kde.spines["right"].set_visible(False)
        ax_kde.xaxis.tick_bottom()
        ax_kde.yaxis.tick_left()

    # Final layout & save
    plt.tight_layout()
    out_dir = os.path.dirname(npz_path)
    out_path = os.path.join(out_dir, output_filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")


def main() -> None:

    results_folder: str = (
        "/home/mariopasc/Python/Results/Meningioma/noise_modelling/NoiseEstimation_20250210_231309"
    )
    seed: int = 123
    pulse: str = "T1"
    slice: int = 112
    slices_of_interest: List[int] = [82, 112, 152]

    visualize_slices_with_rayleigh_and_kde(
        npz_path=os.path.join(results_folder, f"noise_volume_{pulse}_seed{seed}.npz"),
        slices=slices_of_interest,
        output_filename=f"multi_slice_T1_seed{seed}.svg",
        h_bandwidth=0.4,
    )

    visualize_noise_components_2x3(
        npz_path=os.path.join(results_folder, f"noise_volume_{pulse}_seed{seed}.npz"),
        slice_index=slice,  # or None for middle
        h_bandwidth=0.4,
        output_filename=f"noise_2x3_visualization_T1_seed{seed}.svg",
    )

    plot_js_divergence_per_slice_from_pdf(
        base_results_folder=results_folder,
        pulse="T1",
        seeds=[123, 456, 789],
        output_fig_name="js_divergence_T1.svg",
    )

    generate_abstract_figures(
        base_results_folder=results_folder,
        pulse="T1",
        seed=123,
        slices_of_interest=slices_of_interest,  # for example
        output_folder_name=f"abstract_figures_seed{seed}",
        bandwidth_h=0.4,
        num_bins=100,
    )

    visualize_middle_slices_across_pulses(
        base_results_folder=results_folder,
        pulses=["T1", "T1SIN", "T2", "SUSC"],
        seed=seed,
        h_bandwidth=0.4,
        dist_for_fit="rayleigh",
        output_fig_name=f"middle_slices_seed{seed}.svg",
        fig_title="Comparison of Middle Slice and Noise PDF",
    )


if __name__ == "__main__":
    main()
