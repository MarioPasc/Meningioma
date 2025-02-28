from typing import Any, Tuple, Dict, List, Optional

import numpy as np
import gstools as gs  # type: ignore


def get_fit_model_3d(
    bin_center: np.ndarray,
    gamma: np.ndarray,
    var: float = 1.0,
    len_scale: float = 10.0,
) -> Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]:
    """
    Fit several theoretical variogram models to the estimated 3D variogram.
    (Some models may not converge; this is reported in the console.)

    Parameters
    ----------
    bin_center : np.ndarray
        Centers of the distance bins.
    gamma : np.ndarray
        Estimated variogram values.
    var : float, optional
        Initial variance guess.
    len_scale : float, optional
        Initial guess for the correlation length scale.

    Returns
    -------
    results : Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]
        Mapping of model names to (fitted model, fit parameters including r^2).
    """
    models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Circular": gs.Circular,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
        "TLPGaussian": gs.TPLGaussian,
        "TLPSTable": gs.TPLStable,
        "TLPSimple": gs.TPLSimple,
    }
    print("Fitting 3D covariance models")
    results: Dict[str, Tuple[gs.CovModel, Dict[str, Any]]] = {}
    for model_name, model_class in models.items():
        try:
            model = model_class(dim=3, var=var, len_scale=len_scale)
            params, pcov, r2 = model.fit_variogram(bin_center, gamma, return_r2=True)
            results[model_name] = (model, {"params": params, "pcov": pcov, "r2": r2})
            print(f"Model {model_name} fitted with r^2 = {r2:.3f}")
        except Exception as e:
            print(f"Model {model_name} failed to fit: {e}")
    return results
