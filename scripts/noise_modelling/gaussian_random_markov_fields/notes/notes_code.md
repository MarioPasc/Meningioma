
```python
def directional_variogram_map(
    data: np.ndarray,
    bins: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_directions: int = 180,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
    angles_tol: float = np.pi / 90,  # 2° tolerance
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...
```

1. **data** (`np.ndarray`):  
   The 2D array of the field you want to analyze (e.g., a slice of your MRI or a synthetic noise map). This is the main input from which the variogram is computed. The **statistical variability** at different distances (and directions) is derived from these pixel intensities.

2. **bins** (`np.ndarray`):  
   An array of distance bin edges (e.g., from 0 to a maximum distance). During the variogram calculation, the distance between every sampled pair of points is categorized (binned) into these intervals. The resulting variogram is reported at the **center** of each bin.  
   - **Effect**:  
     - Determines how finely you sample the distance axis (the “horizontal” axis of a variogram).  
     - A larger number of smaller bins yields a more detailed (but noisier) variogram curve. Fewer, wider bins gives smoother but less detailed curves.

3. **mask** (`Optional[np.ndarray]`):  
   An optional boolean mask, the same shape as `data`, that indicates which pixels you want to **include** (True) or **exclude** (False).  
   - **Effect**:  
     - Allows you to focus on specific regions of the image (e.g., background noise regions, removing anatomy if needed) or remove artifacts.  
     - If `mask` is `None`, all pixels are used.

4. **num_directions** (`int`):  
   The number of evenly spaced angles (from \(0\) to \(2\pi\)) for which you compute a directional variogram. For example, if `num_directions=180`, each angle increment is \(\frac{2\pi}{180} = \frac{\pi}{90}\) (2 degrees).  
   - **Effect**:  
     - More angles yield a finer resolution of angular dependence, but also more computations.  
     - Too few angles might miss some anisotropic behaviors, but too many could increase noise or computational cost.

5. **sampling_size** (`int`):  
   A parameter passed to `gstools.vario_estimate` that controls how many random pairs of points are used to estimate the variogram at each angle. Typically, you do not need all possible pairs (which could be huge); instead, random subsampling is used.  
   - **Effect**:  
     - A larger `sampling_size` generally yields a **less noisy** variogram (better statistics), but increases computation time.  
     - If `sampling_size` is too small, your variogram can be unstable and more prone to sampling noise.

6. **sampling_seed** (`int`):  
   The random seed that ensures reproducibility of the subsampled pairs for the variogram calculation.  
   - **Effect**:  
     - Changing this seed will change the exact pairs sampled and thus slightly change the resulting \(\gamma(h)\). However, if your sampling size is large enough, the effect on the variogram should be small.  
     - Using a fixed seed makes your results consistent from run to run.

7. **angles_tol** (`float`):  
   The angular tolerance in radians used by `gstools.vario_estimate` when determining which point pairs count toward the directional bin for each angle. For instance, if the angle tolerance is \(\pi/90\) (2 degrees), then a point pair is considered part of angle \(\theta\) if its connecting vector is within ±2° of \(\theta\).  
   - **Effect**:  
     - A smaller `angles_tol` yields a **tighter** angular cone around each direction, which might reveal strong anisotropy but reduces the number of pairs for each bin (possibly adding noise).  
     - A larger `angles_tol` means more pairs per angle bin (reducing noise), but angles become less distinct.

Overall, these parameters control **what part of the image** is used, **how distances are grouped**, **how many directions** are analyzed, and **how much random subsampling** occurs. Small changes in any of these can alter the resulting directional variogram (i.e., the shape or smoothness of the curves).