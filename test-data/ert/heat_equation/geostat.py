"""Generate initial reservoir realisations with geostatistical methods.

Based on:
https://github.com/patnr/HistoryMatching/blob/master/tools/geostat.py
"""

import numpy as np
import scipy.linalg as sla


def variogram_gauss(xx, r, n=0, a=1 / 3):
    """Compute the Gaussian variogram for the 1D points xx.

    Params:
    range r, nugget n, and a

    Ref:
    https://en.wikipedia.org/wiki/Variogram#Variogram_models

    Example:
    >>> xx = np.array([0, 1, 2])
    >>> variogram_gauss(xx, 1, n=0.1, a=1)
    array([0.        , 0.6689085 , 0.98351593])
    """
    # Gauss
    gamma = 1 - np.exp(-(xx**2) / r**2 / a)
    # Sill (=1)
    gamma *= 1 - n
    # Nugget
    gamma[xx != 0] += n
    return gamma


def vectorize(*XYZ):
    """Reshape coordinate points.

    Input: `nDim` arrays with equal `shape`.
    Let `nPt = np.prod(shape)`
    Output: array of shape `(nPt, nDim)`.
    """
    return np.stack(XYZ).reshape((len(XYZ), -1)).T


def dist_euclid(X):
    """Compute distances.

    X must be a 2D-array of shape `(nPt, nDim)`.

    Note: not periodic.
    """
    diff = X[:, None, :] - X
    d2 = np.sum(diff**2, axis=-1)
    return np.sqrt(d2)


def gaussian_fields(pts, rng, N=1, r=0.2):
    """Random field generation.

    Uses:
    - Gaussian variogram.
    - Gaussian distributions.
    """
    dists = dist_euclid(vectorize(*pts))
    Cov = 1 - variogram_gauss(dists, r)
    C12 = sla.sqrtm(Cov).real
    fields = rng.standard_normal(size=(N, len(dists))) @ C12.T
    return fields
