from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from .colorimetry import xy_from_xyz, xyz_from_reflectance


def make_quadratic_pu_basis(wl: NDArray[np.float64], k: int = 7, degree: int = 2) -> NDArray[np.float64]:
    """Build a non-negative partition-of-unity B-spline basis on wavelength grid."""
    u0 = float(wl[0])
    u1 = float(wl[-1])

    n_internal = k - degree - 1
    if n_internal > 0:
        internal = np.linspace(u0, u1, n_internal + 2)[1:-1]
    else:
        internal = np.array([], dtype=float)

    knots = np.concatenate([np.repeat(u0, degree + 1), internal, np.repeat(u1, degree + 1)])

    basis = np.zeros((k, len(wl)), dtype=float)
    for idx in range(k):
        coeffs = np.zeros(k, dtype=float)
        coeffs[idx] = 1.0
        spline = BSpline(knots, coeffs, degree, extrapolate=False)
        vals = np.nan_to_num(spline(wl), nan=0.0)
        basis[idx] = np.clip(vals, 0.0, None)

    s = np.sum(basis, axis=0)
    s[s == 0] = 1.0
    basis /= s
    return basis


def compute_basis_xyz_xy_abs(
    basis: NDArray[np.float64],
    illum: NDArray[np.float64],
    xbar: NDArray[np.float64],
    ybar: NDArray[np.float64],
    zbar: NDArray[np.float64],
    dlam: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute XYZ, xy and XYZ-sum features for each basis function under illuminant."""
    k = basis.shape[0]
    bk_xyz = np.zeros((k, 3), dtype=float)
    bk_xy = np.zeros((k, 2), dtype=float)
    bk_abs = np.zeros(k, dtype=float)

    for idx in range(k):
        xyz = xyz_from_reflectance(basis[idx], illum, xbar, ybar, zbar, dlam)
        bk_xyz[idx] = xyz
        bk_xy[idx] = xy_from_xyz(xyz)
        bk_abs[idx] = float(np.sum(xyz))

    return bk_xyz, bk_xy, bk_abs
