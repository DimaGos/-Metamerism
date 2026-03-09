from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

from .colorimetry import mae, rmse, xy_from_xyz, xyz_from_reflectance


def triangular_barycentrics(
    c: NDArray[np.float64],
    p0: NDArray[np.float64],
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return barycentric coordinates of point c in triangle (p0, p1, p2)."""
    t = np.array([[1.0, 1.0, 1.0], [p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]]], dtype=float)
    rhs = np.array([1.0, c[0], c[1]], dtype=float)
    return np.linalg.solve(t, rhs)


def find_containing_triangles(
    points_xy: NDArray[np.float64],
    c: NDArray[np.float64],
    tol: float = 1e-10,
) -> list[tuple[int, int, int, NDArray[np.float64]]]:
    """Find all vertex triangles containing the target chromaticity c."""
    tris: list[tuple[int, int, int, NDArray[np.float64]]] = []
    for i, j, k in combinations(range(len(points_xy)), 3):
        a = triangular_barycentrics(c, points_xy[i], points_xy[j], points_xy[k])
        if np.all(a >= -tol):
            tris.append((i, j, k, a))
    return tris


def sample_a_f_iterative(
    m: NDArray[np.float64],
    a_t: NDArray[np.float64],
    num_samples: int,
    rng: np.random.Generator,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    """Sample free barycentric coefficients inside interval constraints iteratively."""
    n_free = m.shape[1]
    samples: list[NDArray[np.float64]] = []
    stats = {"invalid_interval": 0, "ok": 0}

    for _ in range(num_samples):
        a_f = np.zeros(n_free, dtype=float)
        feasible = True

        for n in range(n_free):
            prev = m[:, :n] @ a_f[:n] if n > 0 else np.zeros(3, dtype=float)

            if n + 1 < n_free:
                rem = m[:, n + 1 :]
                rem_min = np.sum(np.minimum(rem, 0.0), axis=1)
                rem_max = np.sum(np.maximum(rem, 0.0), axis=1)
            else:
                rem_min = np.zeros(3, dtype=float)
                rem_max = np.zeros(3, dtype=float)

            low = (a_t - 1.0) - prev - rem_max
            high = a_t - prev - rem_min

            lo, hi = 0.0, 1.0
            for i in range(3):
                coeff = m[i, n]
                if abs(coeff) <= eps:
                    if not (low[i] - eps <= 0.0 <= high[i] + eps):
                        feasible = False
                        break
                    continue

                if coeff > 0:
                    li, ui = low[i] / coeff, high[i] / coeff
                else:
                    li, ui = high[i] / coeff, low[i] / coeff

                lo = max(lo, li)
                hi = min(hi, ui)

            if (not feasible) or (lo > hi + eps):
                feasible = False
                break

            lo = max(lo, 0.0)
            hi = min(hi, 1.0)
            a_f[n] = rng.uniform(lo, hi)

        if feasible:
            samples.append(a_f)
            stats["ok"] += 1
        else:
            stats["invalid_interval"] += 1

    if not samples:
        return np.empty((0, n_free), dtype=float), stats
    return np.vstack(samples), stats


def evaluate_candidate(
    a: NDArray[np.float64],
    bk1_abs: NDArray[np.float64],
    by1: NDArray[np.float64],
    fy1_true: float,
    basis: NDArray[np.float64],
    i1: NDArray[np.float64],
    i2: NDArray[np.float64],
    f1_true: NDArray[np.float64],
    f2_true: NDArray[np.float64],
    c1_true: NDArray[np.float64],
    c2_true: NDArray[np.float64],
    xbar: NDArray[np.float64],
    ybar: NDArray[np.float64],
    zbar: NDArray[np.float64],
    dlam: float,
    eps: float = 1e-12,
) -> tuple[dict[str, Any] | None, str]:
    """Validate candidate coefficients and compute reconstruction and errors."""
    if np.any(a < -1e-10) or np.any(a > 1.0 + 1e-10):
        return None, "a_out"

    pivot = int(np.argmax(a))
    a0 = float(a[pivot])
    b0_abs = float(bk1_abs[pivot])
    if a0 <= eps or b0_abs <= eps:
        return None, "w0_unreachable"

    l = np.zeros_like(a)
    l[pivot] = 1.0
    w0_max = 1.0

    for k in range(len(a)):
        if k == pivot:
            continue

        ak = float(a[k])
        if ak <= eps:
            l[k] = 0.0
            continue

        bk_abs = float(bk1_abs[k])
        if bk_abs <= eps:
            return None, "w0_unreachable"

        l[k] = (ak * b0_abs) / (a0 * bk_abs)
        w0_max = min(w0_max, (a0 * bk_abs) / (ak * b0_abs))

    denom = float(np.dot(l, by1))
    if denom <= eps:
        return None, "w0_unreachable"

    w0_star = fy1_true / denom
    if not (0.0 < w0_star <= w0_max + 1e-10):
        return None, "w0_unreachable"

    w = l * w0_star
    if np.any(w < -1e-10) or np.any(w > 1.0 + 1e-10):
        return None, "w_out"

    w = np.clip(w, 0.0, 1.0)
    r_hat = np.sum(w[:, None] * basis, axis=0)

    f1_hat = xyz_from_reflectance(r_hat, i1, xbar, ybar, zbar, dlam)
    f2_hat = xyz_from_reflectance(r_hat, i2, xbar, ybar, zbar, dlam)
    c1_hat = xy_from_xyz(f1_hat)
    c2_hat = xy_from_xyz(f2_hat)

    return (
        {
            "a": a,
            "w": w,
            "r_hat": r_hat,
            "F1_hat": f1_hat,
            "F2_hat": f2_hat,
            "c1_hat": c1_hat,
            "c2_hat": c2_hat,
            "err1_xyz": float(np.linalg.norm(f1_hat - f1_true)),
            "err2_xyz": float(np.linalg.norm(f2_hat - f2_true)),
            "err1_c": float(np.linalg.norm(c1_hat - c1_true)),
            "err2_c": float(np.linalg.norm(c2_hat - c2_true)),
            "err1_y": float(abs(f1_hat[1] - f1_true[1])),
            "err2_y": float(abs(f2_hat[1] - f2_true[1])),
        },
        "ok",
    )


def compute_metrics(
    r_true: NDArray[np.float64],
    r_rec: NDArray[np.float64],
    f1_true: NDArray[np.float64],
    f1_hat: NDArray[np.float64],
    f2_true: NDArray[np.float64],
    f2_hat: NDArray[np.float64],
) -> dict[str, float]:
    """Compute spectral and color reconstruction metrics under both illuminants."""
    c1_t = xy_from_xyz(f1_true)
    c1_h = xy_from_xyz(f1_hat)
    c2_t = xy_from_xyz(f2_true)
    c2_h = xy_from_xyz(f2_hat)

    return {
        "rmse_spectrum": rmse(r_true, r_rec),
        "mae_spectrum": mae(r_true, r_rec),
        "err1_xyz_l2": float(np.linalg.norm(f1_hat - f1_true)),
        "err2_xyz_l2": float(np.linalg.norm(f2_hat - f2_true)),
        "err1_xy_l2": float(np.linalg.norm(c1_h - c1_t)),
        "err2_xy_l2": float(np.linalg.norm(c2_h - c2_t)),
        "err1_Y_abs": float(abs(f1_hat[1] - f1_true[1])),
        "err2_Y_abs": float(abs(f2_hat[1] - f2_true[1])),
    }


def weights_from_a_target_y(
    a: NDArray[np.float64],
    bk_abs: NDArray[np.float64],
    by: NDArray[np.float64],
    fy_target: float,
    eps: float = 1e-12,
) -> NDArray[np.float64] | None:
    """Convert barycentric coefficients to bounded basis weights with fixed Y target."""
    pivot = int(np.argmax(a))
    a0 = float(a[pivot])
    b0_abs = float(bk_abs[pivot])
    if a0 <= eps or b0_abs <= eps:
        return None

    l = np.zeros_like(a)
    l[pivot] = 1.0
    w0_max = 1.0

    for k in range(len(a)):
        if k == pivot:
            continue
        ak = float(a[k])
        if ak <= eps:
            continue
        bk = float(bk_abs[k])
        if bk <= eps:
            return None
        l[k] = (ak * b0_abs) / (a0 * bk)
        w0_max = min(w0_max, (a0 * bk) / (ak * b0_abs))

    denom = float(np.dot(l, by))
    if denom <= eps:
        return None

    w0_star = fy_target / denom
    if not (0.0 < w0_star <= w0_max + 1e-10):
        return None

    w = l * w0_star
    if np.any(w < -1e-10) or np.any(w > 1.0 + 1e-10):
        return None

    return np.clip(w, 0.0, 1.0)


def build_metamer_class_for_target(
    c_target: NDArray[np.float64],
    fy_target: float,
    i1: NDArray[np.float64],
    i2: NDArray[np.float64],
    bk1_xy: NDArray[np.float64],
    bk1_abs: NDArray[np.float64],
    by1: NDArray[np.float64],
    basis: NDArray[np.float64],
    xbar: NDArray[np.float64],
    ybar: NDArray[np.float64],
    zbar: NDArray[np.float64],
    dlam: float,
    n_af_per_tri: int,
    rng: np.random.Generator,
    i1_tol_xy: float,
    i1_tol_y: float,
    eps: float = 1e-12,
) -> dict[str, Any] | None:
    """Build metamer class under I1 and evaluate chromaticity spread under I2."""
    triangles = find_containing_triangles(bk1_xy, c_target)
    if not triangles:
        return None

    k = basis.shape[0]
    cols = np.vstack([np.ones(k), bk1_xy[:, 0], bk1_xy[:, 1]])

    meta: dict[tuple[float, float, float], dict[str, NDArray[np.float64]]] = {}
    i1_err_xy_max = 0.0
    i1_err_y_max = 0.0

    for i0, i1_idx, i2_idx, a_t in triangles:
        tri = [i0, i1_idx, i2_idx]
        free = [idx for idx in range(k) if idx not in tri]

        t_mat = cols[:, tri]
        f_mat = cols[:, free]
        m = np.linalg.solve(t_mat, f_mat)

        af_samples, _ = sample_a_f_iterative(m, a_t, n_af_per_tri, rng, eps=eps)

        for a_f in af_samples:
            a = np.zeros(k, dtype=float)
            a[tri] = a_t - (m @ a_f)
            a[free] = a_f

            if np.any(a < -1e-10) or np.any(a > 1.0 + 1e-10):
                continue

            w = weights_from_a_target_y(a, bk1_abs, by1, fy_target, eps=eps)
            if w is None:
                continue

            r = w @ basis
            f1 = xyz_from_reflectance(r, i1, xbar, ybar, zbar, dlam)
            c1 = xy_from_xyz(f1)

            err_xy = float(np.linalg.norm(c1 - c_target))
            err_y = float(abs(f1[1] - fy_target))
            if err_xy > i1_tol_xy or err_y > i1_tol_y:
                continue

            f2 = xyz_from_reflectance(r, i2, xbar, ybar, zbar, dlam)
            c2 = xy_from_xyz(f2)

            key = tuple(np.round([c2[0], c2[1], f2[1]], 5))
            if key in meta:
                continue

            meta[key] = {"r": r, "w": w, "F1": f1, "F2": f2, "c1": c1, "c2": c2}
            i1_err_xy_max = max(i1_err_xy_max, err_xy)
            i1_err_y_max = max(i1_err_y_max, err_y)

    if not meta:
        return None

    items = list(meta.values())
    xy2 = np.array([it["c2"] for it in items])

    spread_max = float(np.max(pdist(xy2))) if len(xy2) > 1 else 0.0
    spread_mean = float(np.mean(np.linalg.norm(xy2 - xy2.mean(axis=0)[None, :], axis=1)))

    if len(xy2) >= 3:
        try:
            hull_area = float(ConvexHull(xy2).volume)
        except Exception:
            hull_area = 0.0
    else:
        hull_area = 0.0

    return {
        "items": items,
        "n_meta": len(items),
        "xy2": xy2,
        "xy_spread_max": spread_max,
        "xy_spread_mean": spread_mean,
        "xy_hull_area": hull_area,
        "i1_err_xy_max": i1_err_xy_max,
        "i1_err_y_max": i1_err_y_max,
    }


def choose_strong_pair(
    meta_items: list[dict[str, Any]],
    d65_keep_quantile: float = 0.10,
) -> tuple[int, int, float, float, float, float]:
    """Choose pair A/B close under D65 and far apart under F2."""

    def legacy_linear_rgb_from_xyz(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        import colour

        return np.asarray(colour.XYZ_to_sRGB(np.asarray(xyz), apply_cctf_encoding=False), dtype=float)

    f1_all = np.array([it["F1"] for it in meta_items], dtype=float)
    f2_all = np.array([it["F2"] for it in meta_items], dtype=float)
    rgb_d65_all = np.array([legacy_linear_rgb_from_xyz(xyz) for xyz in f1_all], dtype=float)
    rgb_f2_all = np.array([legacy_linear_rgb_from_xyz(xyz) for xyz in f2_all], dtype=float)

    pairs: list[tuple[float, float, float, float, int, int]] = []
    for i in range(len(meta_items)):
        for j in range(i + 1, len(meta_items)):
            d65_lin = float(np.linalg.norm(rgb_d65_all[i] - rgb_d65_all[j]))
            f2_lin = float(np.linalg.norm(rgb_f2_all[i] - rgb_f2_all[j]))
            d65_xyz = float(np.linalg.norm(f1_all[i] - f1_all[j]))
            f2_xyz = float(np.linalg.norm(f2_all[i] - f2_all[j]))
            pairs.append((d65_lin, f2_lin, d65_xyz, f2_xyz, i, j))

    if not pairs:
        raise RuntimeError("No spectrum pairs were generated from the metamer class.")

    d65_vals = np.array([p[0] for p in pairs], dtype=float)
    d65_thr = float(np.quantile(d65_vals, d65_keep_quantile))
    candidate_pairs = [p for p in pairs if p[0] <= d65_thr + 1e-15]

    d65_lin, f2_lin, d65_xyz, f2_xyz, idx_a, idx_b = max(candidate_pairs, key=lambda p: (p[1], -p[0], p[3]))
    return int(idx_a), int(idx_b), d65_lin, f2_lin, d65_xyz, f2_xyz
