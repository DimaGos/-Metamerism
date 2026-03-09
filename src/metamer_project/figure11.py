from __future__ import annotations

from pathlib import Path
from typing import Any

import colour
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .colorimetry import get_illuminant, xy_from_xyz, xyz_from_reflectance, xyz_image_from_reflectance
from .io_utils import SceneData
from .metamer import build_metamer_class_for_target, choose_strong_pair


def _load_logo_mask(candidates: list[Path], threshold_quantile: float = 0.5) -> tuple[NDArray[np.float64], Path]:
    """Load grayscale logo mask from file candidates."""
    logo_path: Path | None = None
    for pth in candidates:
        if pth.exists():
            logo_path = pth
            break

    if logo_path is None:
        raise FileNotFoundError(f"Logo not found from cwd={Path.cwd()}")

    logo_raw = np.asarray(plt.imread(logo_path))
    if logo_raw.ndim == 3:
        rgb = logo_raw[..., :3].astype(float)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif logo_raw.ndim == 2:
        gray = logo_raw.astype(float)
        if gray.max() > 1.0:
            gray = gray / 255.0
    else:
        raise RuntimeError(f"Unexpected logo shape: {logo_raw.shape}")

    threshold = float(np.quantile(gray, threshold_quantile))
    logo_mask = (gray < threshold).astype(float)
    if np.all(logo_mask == logo_mask.flat[0]):
        logo_mask = (gray < 0.5).astype(float)
    return logo_mask, logo_path


def run_figure11_like(
    data: SceneData,
    basis: NDArray[np.float64],
    bk1_xy: NDArray[np.float64],
    bk1_abs: NDArray[np.float64],
    by1: NDArray[np.float64],
    base_sample: str,
    output_dir: Path,
    illum_1: str = "D65",
    illum_2: str = "F2",
    n_af_per_tri: int = 1200,
    rng_seed: int = 2028,
    i1_tol_xy: float = 2e-4,
    i1_tol_y: float = 2e-3,
    eps: float = 1e-12,
    logo_candidates: list[Path] | None = None,
) -> dict[str, Any]:
    """Create Figure 11-like hidden pattern example from a strong metamer pair."""
    sample_ids_str = [str(s) for s in data.sample_ids]
    if str(base_sample) not in sample_ids_str:
        raise RuntimeError(f"Sample {base_sample} not found in sample_ids")

    idx = sample_ids_str.index(str(base_sample))
    r_target = data.reflectances[idx].copy()

    i_d65 = get_illuminant(data.wl, data.illum_wl, data.illum[illum_1].to_numpy(dtype=float), data.ybar, data.dlam)
    i_f2 = get_illuminant(data.wl, data.illum_wl, data.illum[illum_2].to_numpy(dtype=float), data.ybar, data.dlam)

    target_xyz_d65 = xyz_from_reflectance(r_target, i_d65, data.xbar, data.ybar, data.zbar, data.dlam)
    target_xy_d65 = xy_from_xyz(target_xyz_d65)
    target_y_d65 = float(target_xyz_d65[1])

    rng = np.random.default_rng(rng_seed)
    meta_out = build_metamer_class_for_target(
        c_target=target_xy_d65,
        fy_target=target_y_d65,
        i1=i_d65,
        i2=i_f2,
        bk1_xy=bk1_xy,
        bk1_abs=bk1_abs,
        by1=by1,
        basis=basis,
        xbar=data.xbar,
        ybar=data.ybar,
        zbar=data.zbar,
        dlam=data.dlam,
        n_af_per_tri=n_af_per_tri,
        rng=rng,
        i1_tol_xy=i1_tol_xy,
        i1_tol_y=i1_tol_y,
        eps=eps,
    )

    if meta_out is None or len(meta_out["items"]) < 2:
        raise RuntimeError("Could not build a usable metamer class for Figure 11-like example")

    meta_items = meta_out["items"]
    idx_a, idx_b, d65_lin, f2_lin, d65_xyz, f2_xyz = choose_strong_pair(meta_items, d65_keep_quantile=0.10)

    r_a = np.asarray(meta_items[idx_a]["r"], dtype=float)
    r_b = np.asarray(meta_items[idx_b]["r"], dtype=float)
    xyz_a_d65 = np.asarray(meta_items[idx_a]["F1"], dtype=float)
    xyz_b_d65 = np.asarray(meta_items[idx_b]["F1"], dtype=float)
    xyz_a_f2 = np.asarray(meta_items[idx_a]["F2"], dtype=float)
    xyz_b_f2 = np.asarray(meta_items[idx_b]["F2"], dtype=float)
    xy_a_d65 = xy_from_xyz(xyz_a_d65)
    xy_b_d65 = xy_from_xyz(xyz_b_d65)
    xy_a_f2 = xy_from_xyz(xyz_a_f2)
    xy_b_f2 = xy_from_xyz(xyz_b_f2)

    if logo_candidates is None:
        logo_candidates = [Path("ippi.png"), Path("../ippi.png"), data.base_dir.parent / "ippi.png"]
    logo_mask, logo_path = _load_logo_mask(logo_candidates, threshold_quantile=0.5)

    if float(xyz_a_f2[1]) <= float(xyz_b_f2[1]):
        r_logo, r_bg = r_a, r_b
        logo_uses = "A"
    else:
        r_logo, r_bg = r_b, r_a
        logo_uses = "B"

    spectral_img = logo_mask[..., None] * r_logo[None, None, :] + (1.0 - logo_mask[..., None]) * r_bg[None, None, :]

    xyz_img_d65 = xyz_image_from_reflectance(spectral_img, i_d65, data.xbar, data.ybar, data.zbar, data.dlam)
    xyz_img_f2 = xyz_image_from_reflectance(spectral_img, i_f2, data.xbar, data.ybar, data.zbar, data.dlam)

    y_stack = np.concatenate([xyz_img_d65[..., 1].ravel(), xyz_img_f2[..., 1].ravel()])
    shared_scale = max(float(np.quantile(y_stack, 0.99)), eps)
    img_d65 = np.clip(colour.XYZ_to_sRGB(xyz_img_d65 / shared_scale, apply_cctf_encoding=True), 0.0, 1.0)
    img_f2 = np.clip(colour.XYZ_to_sRGB(xyz_img_f2 / shared_scale, apply_cctf_encoding=True), 0.0, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    axes[0].imshow(logo_mask, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Mask")
    axes[0].axis("off")
    axes[1].imshow(img_d65)
    axes[1].set_title(f"Rendered under {illum_1}")
    axes[1].axis("off")
    axes[2].imshow(img_f2)
    axes[2].set_title(f"Rendered under {illum_2}")
    axes[2].axis("off")
    plt.tight_layout()

    fig_path = output_dir / f"figure11_like_ippi_sample{base_sample}_strong_legacy_xyz.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / f"figure11_like_spectra_sample{base_sample}_strong_legacy_xyz.csv"
    pd.DataFrame({"wavelength": data.wl, "r_A": r_a, "r_B": r_b}).to_csv(csv_path, index=False)

    return {
        "base_sample": str(base_sample),
        "metamer_class_size": len(meta_items),
        "idx_A": idx_a,
        "idx_B": idx_b,
        "d65_lin": d65_lin,
        "f2_lin": f2_lin,
        "d65_xyz": d65_xyz,
        "f2_xyz": f2_xyz,
        "xy_A_D65": xy_a_d65,
        "xy_B_D65": xy_b_d65,
        "xy_A_F2": xy_a_f2,
        "xy_B_F2": xy_b_f2,
        "logo_uses": logo_uses,
        "logo_path": logo_path,
        "fig_path": fig_path,
        "csv_path": csv_path,
    }
