from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .colorimetry import get_illuminant, interp_to


@dataclass
class SceneData:
    """Prepared spectral/colorimetric data loaded from CSV files."""

    base_dir: Path
    illum: pd.DataFrame
    munsell: pd.DataFrame
    xyz: pd.DataFrame
    wl: NDArray[np.float64]
    dlam: float
    reflectances: NDArray[np.float64]
    sample_ids: list[str]
    xbar: NDArray[np.float64]
    ybar: NDArray[np.float64]
    zbar: NDArray[np.float64]

    @property
    def illum_wl(self) -> NDArray[np.float64]:
        return self.illum["wavelength"].to_numpy(dtype=float)


def load_scene_data(base_dir: Path) -> SceneData:
    """Load source CSV files and prepare common arrays used across the pipeline."""
    illum_path = base_dir / "illum.csv"
    munsell_path = base_dir / "munsell.csv"
    xyz_path = base_dir / "xyz_matching_fun.csv"

    illum = pd.read_csv(illum_path)
    munsell = pd.read_csv(munsell_path)
    xyz = pd.read_csv(xyz_path)

    wl = munsell["wavelength"].to_numpy(dtype=float)
    dlam = float(np.mean(np.diff(wl)))

    reflectances = munsell.drop(columns=["wavelength"]).to_numpy(dtype=float).T
    sample_ids = [str(s) for s in munsell.columns[1:]]

    xbar = interp_to(wl, xyz["wavelength"].to_numpy(dtype=float), xyz["X"].to_numpy(dtype=float))
    ybar = interp_to(wl, xyz["wavelength"].to_numpy(dtype=float), xyz["Y"].to_numpy(dtype=float))
    zbar = interp_to(wl, xyz["wavelength"].to_numpy(dtype=float), xyz["Z"].to_numpy(dtype=float))

    return SceneData(
        base_dir=base_dir,
        illum=illum,
        munsell=munsell,
        xyz=xyz,
        wl=wl,
        dlam=dlam,
        reflectances=reflectances,
        sample_ids=sample_ids,
        xbar=xbar,
        ybar=ybar,
        zbar=zbar,
    )


def compute_xyz_responses_table(data: SceneData, illum_1: str, illum_2: str) -> pd.DataFrame:
    """Compute XYZ and xy responses for all samples under two illuminants."""
    i1 = get_illuminant(
        data.wl,
        data.illum_wl,
        data.illum[illum_1].to_numpy(dtype=float),
        data.ybar,
        data.dlam,
    )
    i2 = get_illuminant(
        data.wl,
        data.illum_wl,
        data.illum[illum_2].to_numpy(dtype=float),
        data.ybar,
        data.dlam,
    )

    x1 = np.sum(data.reflectances * (i1 * data.xbar) * data.dlam, axis=1)
    y1 = np.sum(data.reflectances * (i1 * data.ybar) * data.dlam, axis=1)
    z1 = np.sum(data.reflectances * (i1 * data.zbar) * data.dlam, axis=1)

    x2 = np.sum(data.reflectances * (i2 * data.xbar) * data.dlam, axis=1)
    y2 = np.sum(data.reflectances * (i2 * data.ybar) * data.dlam, axis=1)
    z2 = np.sum(data.reflectances * (i2 * data.zbar) * data.dlam, axis=1)

    s1 = np.maximum(x1 + y1 + z1, 1e-12)
    s2 = np.maximum(x2 + y2 + z2, 1e-12)

    return pd.DataFrame(
        {
            "sample": data.sample_ids,
            f"X_{illum_1}": x1,
            f"Y_{illum_1}": y1,
            f"Z_{illum_1}": z1,
            f"x_{illum_1}": x1 / s1,
            f"y_{illum_1}": y1 / s1,
            f"X_{illum_2}": x2,
            f"Y_{illum_2}": y2,
            f"Z_{illum_2}": z2,
            f"x_{illum_2}": x2 / s2,
            f"y_{illum_2}": y2 / s2,
        }
    )
