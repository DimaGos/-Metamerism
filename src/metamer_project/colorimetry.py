from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def interp_to(target_wl: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Interpolate values onto the target wavelength grid with zero outside bounds."""
    return np.interp(target_wl, x, y, left=0.0, right=0.0)


def get_illuminant(
    wl: NDArray[np.float64],
    illum_wl: NDArray[np.float64],
    illum_values: NDArray[np.float64],
    ybar: NDArray[np.float64],
    dlam: float,
) -> NDArray[np.float64]:
    """Return illuminant SPD normalized so a perfect reflector has Y = 1."""
    illum = interp_to(wl, illum_wl, illum_values)
    k = 1.0 / float(np.sum(illum * ybar * dlam))
    return illum * k


def xyz_from_reflectance(
    r: NDArray[np.float64],
    illum: NDArray[np.float64],
    xbar: NDArray[np.float64],
    ybar: NDArray[np.float64],
    zbar: NDArray[np.float64],
    dlam: float,
) -> NDArray[np.float64]:
    """Integrate reflectance under illuminant and observer CMFs to XYZ."""
    x = float(np.sum(r * illum * xbar) * dlam)
    y = float(np.sum(r * illum * ybar) * dlam)
    z = float(np.sum(r * illum * zbar) * dlam)
    return np.array([x, y, z], dtype=float)


def xy_from_xyz(xyz: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    """Convert XYZ tristimulus values to xy chromaticity."""
    x, y, z = xyz
    s = max(float(x + y + z), eps)
    return np.array([x / s, y / s], dtype=float)


def xyz_image_from_reflectance(
    spectral_img: NDArray[np.float64],
    illum: NDArray[np.float64],
    xbar: NDArray[np.float64],
    ybar: NDArray[np.float64],
    zbar: NDArray[np.float64],
    dlam: float,
) -> NDArray[np.float64]:
    """Integrate spectral image (H, W, C) into XYZ image (H, W, 3)."""
    signal = spectral_img * illum[None, None, :]
    x = np.tensordot(signal, xbar * dlam, axes=([2], [0]))
    y = np.tensordot(signal, ybar * dlam, axes=([2], [0]))
    z = np.tensordot(signal, zbar * dlam, axes=([2], [0]))
    return np.stack([x, y, z], axis=-1)


def rmse(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute root-mean-square error."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute mean absolute error."""
    return float(np.mean(np.abs(a - b)))
