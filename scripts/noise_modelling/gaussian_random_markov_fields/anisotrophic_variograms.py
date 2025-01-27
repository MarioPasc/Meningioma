import numpy as np
import matplotlib.pyplot as plt
import gstools as gs

from typing import Tuple, Optional

def directional_variogram_map(
    data: np.ndarray,
    bins: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_directions: int = 180,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
    angles_tol: float = np.pi / 90,  # 2° tolerance
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the directional variogram for angles from 0 to 2π and store
    the results in a 2D array (gamma_map).

    Parameters
    ----------
    data : np.ndarray
        2D image data.
    bins : np.ndarray
        Array of bin edges for distance classes.
    mask : np.ndarray, optional
        Boolean mask to exclude regions. If None, use all pixels.
    num_directions : int
        Number of angle subdivisions between 0 and 2π.
    sampling_size : int
        Number of random pairs sampled to estimate each directional variogram.
    sampling_seed : int
        Random seed for reproducibility.
    angles_tol : float
        Angular tolerance (in radians) for directional variograms.

    Returns
    -------
    angles : np.ndarray
        1D array of angles (length = num_directions) from 0 to 2π.
    bin_centers : np.ndarray
        1D array of distance bin centers.
    gamma_map : np.ndarray
        2D array of shape (num_directions, len(bin_centers)).
        gamma_map[i, :] is the variogram for angle angles[i].
    """
    # Prepare the positions and flattened data
    if mask is not None:
        valid_indices = np.argwhere(mask.flatten()).flatten()
        valid_data = data.flatten()[valid_indices]
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))[:, valid_indices]
    else:
        valid_data = data.flatten()
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))

    # Dynamic sampling size check
    sampling_size = min(sampling_size, len(valid_data))

    # Angle array
    angles = np.linspace(0.0, 2.0 * np.pi, num_directions, endpoint=False)

    # Initialize result storage
    gamma_map = np.zeros((num_directions, len(bins) - 1), dtype=np.float64)
    bin_centers = None

    # Estimate variogram for each direction
    for i, angle in enumerate(angles):
        # Convert angle to a direction vector
        direction_vector = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)

        # Estimate the directional variogram
        bc, gamma = gs.vario_estimate(
            pos=pos_flat,
            field=valid_data,
            bin_edges=bins,
            mesh_type="unstructured",
            direction=[direction_vector],
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )

        # Store results
        if bin_centers is None:
            bin_centers = bc
        gamma_map[i, :] = gamma

    return angles, bin_centers, gamma_map


def plot_variogram_polar(
    angles: np.ndarray,
    distances: np.ndarray,
    gamma_map: np.ndarray,
    cmap: str = "viridis",
):
    """
    Create a 2D polar heatmap of the directional variogram.

    Parameters
    ----------
    angles : np.ndarray
        1D array of angles from 0 to 2π (length = # of directions).
    distances : np.ndarray
        1D array of distance bin centers (length = # of bins).
    gamma_map : np.ndarray
        2D array of shape (num_directions, num_distance_bins),
        where gamma_map[i, j] is the variogram at angles[i] and distances[j].
    cmap : str
        Colormap for the heatmap.
    """
    # We create a 2D mesh for angles and radii
    #   shape => (len(angles), len(distances))
    # Note: Each row i is for angles[i], each column j is for distances[j]
    # We want to pass that to pcolormesh in polar coordinates.

    A, R = np.meshgrid(distances, angles)  # 'xy' indexing: A.shape, R.shape => (#angles, #distances)
    # Here:
    #   R[i, j] = angles[i]
    #   A[i, j] = distances[j]
    # We'll invert that usage for clarity:
    #   R is actually angles, A is distances
    # We can rename them for clarity if we prefer R->Angle, A->Dist, etc.

    # So we'll do a polar plot with THETA = R, radius = A
    THETA = R
    RADII = A

    # gamma_map is shape (#angles, #bins).
    # That matches the shape(THETA) and shape(RADII).

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")

    # pcolormesh arguments:
    #   pcolormesh(theta_vals, radius_vals, data_vals)
    #   Usually expects len(theta_vals.shape) == len(radius_vals.shape) == len(data_vals.shape) == 2
    #
    # We want the color-coded data to be gamma_map.

    c = ax.pcolormesh(THETA, RADII, gamma_map, shading="auto", cmap=cmap)

    # Optionally adjust radial limit
    ax.set_ylim(0, distances[-1])  # up to the maximum distance bin

    # Add colorbar. Note that with a polar subplot,
    # it's often easiest to place the colorbar outside the plot.
    cb = fig.colorbar(c, ax=ax, orientation="vertical", pad=0.1)
    cb.set_label("Gamma")

    ax.set_title("Directional Variogram (Polar Representation)", y=1.08)
    plt.show()


def main():
    # 1) Synthesize or load data. For demonstration, let's create a small random field.
    np.random.seed(42)
    data = np.random.normal(loc=0.0, scale=1.0, size=(100, 100))

    # 2) Define bins (distance range)
    max_dist = 50
    bins = np.linspace(0, max_dist, 51)  # e.g., from 0 to 50 in 51 steps

    # 3) Optional: no mask for this demo
    mask = None

    # 4) Compute directional variogram for angles 0 -> 2π
    angles, bin_centers, gamma_map = directional_variogram_map(
        data=data,
        bins=bins,
        mask=mask,
        num_directions=180,      # e.g., 180 directions
        sampling_size=2000,
        sampling_seed=19920516,
        angles_tol=np.pi / 90,   # 2° tolerance
    )

    # 5) Plot as a polar heatmap
    plot_variogram_polar(
        angles=angles,
        distances=bin_centers,
        gamma_map=gamma_map,
        cmap="viridis",
    )


if __name__ == "__main__":
    main()
