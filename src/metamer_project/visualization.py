from __future__ import annotations

from pathlib import Path
from typing import Any

import colour
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


def plot_spectra_family(
    wl: NDArray[np.float64],
    recon: dict[str, Any],
    illum_1: str,
    illum_2: str,
    output_dir: Path,
) -> Path:
    """Plot family of generated spectra, highlighting true and selected reconstruction."""
    output_dir.mkdir(parents=True, exist_ok=True)

    family = np.array([c["r_hat"] for c in recon["candidates"]])
    xy2_family = np.array([c["c2_hat"] for c in recon["candidates"]])
    angles = np.arctan2(xy2_family[:, 1] - 1.0 / 3.0, xy2_family[:, 0] - 1.0 / 3.0)
    order = np.argsort(angles)
    family = family[order]

    fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#ececec")

    palette = plt.cm.hsv(np.linspace(0.0, 0.95, len(family)))
    for r, col in zip(family, palette):
        ax.plot(wl, r, color=col, alpha=0.14, linewidth=1.8)

    ax.plot(wl, recon["r_rec"], color="#1b1b1b", linewidth=2.8, alpha=0.94, label="r_rec (best by I2)")
    ax.plot(wl, recon["r_true"], color="#3baa3b", linewidth=4.8, alpha=0.95, label="r_true")

    ax.set_xlim(float(wl.min()), float(wl.max()))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(f"Sample {recon['sample']}: generated spectra family")
    ax.grid(True, color="#b4b4b4", linewidth=1.15, alpha=0.62)
    ax.legend(loc="upper left", frameon=True)

    out_path = output_dir / f"spectra_family_sample_{recon['sample']}_{illum_1}_{illum_2}.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectra_compare(
    wl: NDArray[np.float64],
    recon: dict[str, Any],
    illum_2_name: str,
    illum_2: NDArray[np.float64],
    output_dir: Path,
) -> Path:
    """Plot true vs reconstructed reflectance and spectral signal under I2."""
    output_dir.mkdir(parents=True, exist_ok=True)
    l2_true = recon["r_true"] * illum_2
    l2_rec = recon["r_rec"] * illum_2

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    fig.patch.set_facecolor("white")

    axes[0].set_facecolor("#f1f1f1")
    axes[0].plot(wl, recon["r_true"], color="#2a7fdb", linewidth=2.6, label="r_true")
    axes[0].plot(wl, recon["r_rec"], color="#d94841", linewidth=2.3, label="r_rec")
    axes[0].set_title("Reflectance: true vs reconstructed")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Reflectance")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].set_facecolor("#f1f1f1")
    axes[1].plot(wl, l2_true, color="#2a7fdb", linewidth=2.6, label="r_true * I2")
    axes[1].plot(wl, l2_rec, color="#d94841", linewidth=2.3, label="r_rec * I2")
    axes[1].set_title(f"Spectral signal under {illum_2_name}")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Signal")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="best")

    plt.tight_layout()
    out_path = output_dir / f"spectra_compare_sample_{recon['sample']}_{illum_2_name}.png"
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_xy_span(
    metamer_items: list[dict[str, Any]],
    bk1_xy: NDArray[np.float64],
    bk2_xy: NDArray[np.float64],
    target_xy: NDArray[np.float64],
    sample_name: str,
    illum_1: str,
    illum_2: str,
    output_dir: Path,
    eps: float = 1e-12,
) -> Path:
    """Plot chromaticity span of metamer class together with basis gamuts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmfs_locus = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy()
    cmfs_locus = cmfs_locus.align(colour.SpectralShape(380, 780, 5))
    xyz_locus = cmfs_locus.values
    s_locus = np.maximum(np.sum(xyz_locus, axis=1), eps)
    x_locus = xyz_locus[:, 0] / s_locus
    y_locus = xyz_locus[:, 1] / s_locus

    srgb_xy = colour.RGB_COLOURSPACES["sRGB"].primaries
    srgb_poly = np.vstack([srgb_xy, srgb_xy[0]])

    h1 = ConvexHull(bk1_xy).vertices
    h2 = ConvexHull(bk2_xy).vertices
    metamer_xy2 = np.array([it["c2"] for it in metamer_items])

    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#ececec")

    ax.plot(x_locus, y_locus, color="black", linewidth=2.1)
    ax.plot([x_locus[-1], x_locus[0]], [y_locus[-1], y_locus[0]], color="black", linewidth=1.4, alpha=0.8)

    ax.plot(srgb_poly[:, 0], srgb_poly[:, 1], color="#377eb8", linewidth=2.8, label="sRGB gamut")
    ax.scatter(srgb_xy[:, 0], srgb_xy[:, 1], s=80, color="#377eb8", edgecolors="white", linewidths=0.9, zorder=5)

    ax.plot(
        np.r_[bk1_xy[h1, 0], bk1_xy[h1[0], 0]],
        np.r_[bk1_xy[h1, 1], bk1_xy[h1[0], 1]],
        color="#f28e2b",
        linewidth=2.7,
        label=f"gamut {illum_1}",
    )
    ax.scatter(bk1_xy[:, 0], bk1_xy[:, 1], s=52, color="#f28e2b", edgecolors="white", linewidths=0.7, zorder=4)

    ax.plot(
        np.r_[bk2_xy[h2, 0], bk2_xy[h2[0], 0]],
        np.r_[bk2_xy[h2, 1], bk2_xy[h2[0], 1]],
        color="#4daf4a",
        linewidth=2.7,
        label=f"gamut {illum_2}",
    )
    ax.scatter(bk2_xy[:, 0], bk2_xy[:, 1], s=52, color="#4daf4a", edgecolors="white", linewidths=0.7, zorder=4)

    ax.scatter(
        metamer_xy2[:, 0],
        metamer_xy2[:, 1],
        s=34,
        color="#2b6cb0",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.35,
        zorder=7,
        label=f"metamers under {illum_2}",
    )
    ax.scatter(target_xy[0], target_xy[1], s=110, color="black", marker="o", zorder=8, label=f"target under {illum_1}")

    ax.set_xlim(0.0, 0.88)
    ax.set_ylim(0.0, 0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Span of metamers chromaticity")
    ax.grid(True, color="#b5b5b5", linewidth=1.05, alpha=0.65)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    out_path = output_dir / f"xy_span_metamers_sample_{sample_name}_{illum_1}_{illum_2}.png"
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def xyz_to_srgb01(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert XYZ to clipped sRGB in [0, 1]."""
    rgb = colour.XYZ_to_sRGB(np.asarray(xyz), apply_cctf_encoding=True)
    return np.clip(rgb, 0.0, 1.0)


def choose_items_for_grid(
    meta_items: list[dict[str, Any]],
    n_tiles: int,
    mode: str = "random",
    pick_step: int = 32,
    pick_start_1based: int = 1,
    pick_random_seed: int = 7,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select spectra for swatches using random/stride/linspace sampling."""
    xy2 = np.array([it["c2"] for it in meta_items])
    if len(xy2) == 0:
        raise RuntimeError("metamer_items is empty.")

    angles = np.arctan2(xy2[:, 1] - 1.0 / 3.0, xy2[:, 0] - 1.0 / 3.0)
    order = np.argsort(angles)
    items_sorted = [meta_items[i] for i in order]
    n = len(items_sorted)

    if mode == "random":
        if n < n_tiles:
            raise RuntimeError(f"Random no-repeat needs at least {n_tiles} items, got {n}.")
        rng = np.random.default_rng(pick_random_seed)
        idx = rng.choice(n, size=n_tiles, replace=False).tolist()
        return [items_sorted[i] for i in idx], idx

    if n < n_tiles:
        idx = [i % n for i in range(n_tiles)]
        return [items_sorted[i] for i in idx], idx

    if mode == "linspace":
        idx = np.linspace(0, n - 1, n_tiles).astype(int).tolist()
        return [items_sorted[i] for i in idx], idx

    start = (int(pick_start_1based) - 1) % n
    step = max(1, int(pick_step))

    idx: list[int] = []
    used: set[int] = set()
    k = start
    guard = 0
    guard_lim = 4 * n

    while len(idx) < n_tiles and guard < guard_lim:
        if k not in used:
            used.add(k)
            idx.append(k)
        k = (k + step) % n
        guard += 1

    if len(idx) < n_tiles:
        for j in np.linspace(0, n - 1, n).astype(int):
            jj = int(j)
            if jj not in used:
                used.add(jj)
                idx.append(jj)
                if len(idx) == n_tiles:
                    break

    return [items_sorted[i] for i in idx], idx


def plot_swatches(
    metamer_items: list[dict[str, Any]],
    sample_name: str,
    illum_1: str,
    illum_2: str,
    output_dir: Path,
    n_rows: int = 8,
    n_cols: int = 4,
    mode: str = "random",
    pick_step: int = 32,
    pick_start_1based: int = 1,
    pick_random_seed: int = 7,
) -> tuple[Path, list[int]]:
    """Render swatch grid under I1 and I2 from selected metamer spectra."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_tiles = n_rows * n_cols
    items_pick, picked_idx0 = choose_items_for_grid(
        metamer_items,
        n_tiles=n_tiles,
        mode=mode,
        pick_step=pick_step,
        pick_start_1based=pick_start_1based,
        pick_random_seed=pick_random_seed,
    )

    panel1 = np.zeros((n_rows, n_cols, 3), dtype=float)
    panel2 = np.zeros((n_rows, n_cols, 3), dtype=float)

    for n, it in enumerate(items_pick):
        r = n // n_cols
        c = n % n_cols
        panel1[r, c] = xyz_to_srgb01(np.asarray(it["F1"], dtype=float))
        panel2[r, c] = xyz_to_srgb01(np.asarray(it["F2"], dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 9.5), dpi=190)
    fig.patch.set_facecolor("white")

    for ax, panel, title in [
        (axes[0], panel1, f"Color under {illum_1} (sample {sample_name})"),
        (axes[1], panel2, f"Color under {illum_2}"),
    ]:
        ax.imshow(panel, interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1.4, alpha=0.58)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    out_path = output_dir / f"figure10_swatches_sample_{sample_name}_{illum_1}_{illum_2}.png"
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return out_path, [i + 1 for i in picked_idx0]
