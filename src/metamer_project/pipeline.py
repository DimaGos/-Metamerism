from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from time import perf_counter

from .basis import compute_basis_xyz_xy_abs
from .colorimetry import get_illuminant, xy_from_xyz, xyz_from_reflectance
from .io_utils import SceneData
from .metamer import (
    build_metamer_class_for_target,
    compute_metrics,
    evaluate_candidate,
    find_containing_triangles,
    sample_a_f_iterative,
)


@dataclass
class SearchConfig:
    """Configuration for fixed-illuminant sample search."""

    illum_1: str = "D65"
    illum_2: str = "F2"
    n_neutral_candidates: int = 220
    af_samples_per_tri_sample: int = 1800
    seed_sample_search: int = 2028
    min_meta_required: int = 24
    i1_tol_xy: float = 2e-4
    i1_tol_y: float = 2e-3
    force_sample: str | None = None
    top_cache_size: int = 100
    verbose: bool = True
    progress_every: int = 10


@dataclass
class ReconstructionConfig:
    """Configuration for one-to-many reconstruction for a selected sample."""

    n_samples_af: int = 1000
    rng_seed: int = 42
    lambda_smooth: float = 0.03
    lambda_tail: float = 0.12


def search_samples_fixed_illuminants(
    data: SceneData,
    basis: NDArray[np.float64],
    responses_df: pd.DataFrame,
    cfg: SearchConfig,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Search best sample under fixed illuminants by metamer spread under I2."""
    t0 = perf_counter()
    i1 = get_illuminant(
        data.wl,
        data.illum_wl,
        data.illum[cfg.illum_1].to_numpy(dtype=float),
        data.ybar,
        data.dlam,
    )
    i2 = get_illuminant(
        data.wl,
        data.illum_wl,
        data.illum[cfg.illum_2].to_numpy(dtype=float),
        data.ybar,
        data.dlam,
    )

    bk1_xyz, bk1_xy, bk1_abs = compute_basis_xyz_xy_abs(basis, i1, data.xbar, data.ybar, data.zbar, data.dlam)
    bk2_xyz, bk2_xy, _ = compute_basis_xyz_xy_abs(basis, i2, data.xbar, data.ybar, data.zbar, data.dlam)
    by1 = bk1_xyz[:, 1]

    white_i1 = xy_from_xyz(xyz_from_reflectance(np.ones_like(data.wl), i1, data.xbar, data.ybar, data.zbar, data.dlam))

    df_fix = responses_df.copy()
    df_fix["sample"] = df_fix["sample"].astype(str)
    df_fix["dist_to_white_i1"] = np.sqrt(
        (df_fix[f"x_{cfg.illum_1}"] - white_i1[0]) ** 2 + (df_fix[f"y_{cfg.illum_1}"] - white_i1[1]) ** 2
    )

    if cfg.force_sample is None:
        pool = df_fix.sort_values("dist_to_white_i1", ascending=True).head(min(cfg.n_neutral_candidates, len(df_fix)))
        search_mode = "auto_neutral_pool"
    else:
        pool = df_fix[df_fix["sample"] == str(cfg.force_sample)].copy()
        if pool.empty:
            raise RuntimeError(f"force_sample={cfg.force_sample} not found in munsell.csv")
        search_mode = "forced_sample"

    sample_to_idx = {sid: idx for idx, sid in enumerate(data.sample_ids)}
    rng = np.random.default_rng(cfg.seed_sample_search)

    rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    top_payloads: list[dict[str, Any]] = []
    total = len(pool)

    if cfg.verbose:
        print(
            f"[search] Selecting best sample for metamerism under {cfg.illum_1}/{cfg.illum_2} "
            f"from {total} candidates...",
            flush=True,
        )

    for idx_loop, sid in enumerate(pool["sample"].tolist(), start=1):
        sidx = sample_to_idx[str(sid)]
        r_s = data.reflectances[sidx]

        f1_s = xyz_from_reflectance(r_s, i1, data.xbar, data.ybar, data.zbar, data.dlam)
        c1_s = xy_from_xyz(f1_s)
        fy1_s = float(f1_s[1])

        out = build_metamer_class_for_target(
            c_target=c1_s,
            fy_target=fy1_s,
            i1=i1,
            i2=i2,
            bk1_xy=bk1_xy,
            bk1_abs=bk1_abs,
            by1=by1,
            basis=basis,
            xbar=data.xbar,
            ybar=data.ybar,
            zbar=data.zbar,
            dlam=data.dlam,
            n_af_per_tri=cfg.af_samples_per_tri_sample,
            rng=rng,
            i1_tol_xy=cfg.i1_tol_xy,
            i1_tol_y=cfg.i1_tol_y,
            eps=eps,
        )

        dist_white = float(df_fix.loc[df_fix["sample"] == str(sid), "dist_to_white_i1"].iloc[0])

        if out is None:
            rows.append(
                {
                    "sample": str(sid),
                    "status": "no_class",
                    "n_meta": 0,
                    "xy_hull_area": 0.0,
                    "xy_spread_max": 0.0,
                    "xy_spread_mean": 0.0,
                    "score": -1.0,
                    "i1_err_xy_max": np.nan,
                    "i1_err_y_max": np.nan,
                    "dist_to_white_i1": dist_white,
                }
            )
            continue

        score = 120.0 * out["xy_hull_area"] + out["xy_spread_max"] + 0.2 * out["xy_spread_mean"]
        status = "ok" if out["n_meta"] >= cfg.min_meta_required else "few_meta"

        row = {
            "sample": str(sid),
            "status": status,
            "n_meta": out["n_meta"],
            "xy_hull_area": out["xy_hull_area"],
            "xy_spread_max": out["xy_spread_max"],
            "xy_spread_mean": out["xy_spread_mean"],
            "score": score,
            "i1_err_xy_max": out["i1_err_xy_max"],
            "i1_err_y_max": out["i1_err_y_max"],
            "dist_to_white_i1": dist_white,
        }
        rows.append(row)

        if status == "ok":
            payload = {"sample": str(sid), "score": score, **out, "dist_to_white_i1": dist_white}
            if best_payload is None or score > float(best_payload["score"]):
                best_payload = payload

            if len(top_payloads) < cfg.top_cache_size:
                top_payloads.append(payload)
            else:
                j_min = int(np.argmin([float(tp["score"]) for tp in top_payloads]))
                if score > float(top_payloads[j_min]["score"]):
                    top_payloads[j_min] = payload

        if cfg.verbose and (idx_loop % max(1, cfg.progress_every) == 0 or idx_loop == total):
            elapsed = perf_counter() - t0
            print(f"[search] Processed {idx_loop}/{total} samples in {elapsed:.1f}s", flush=True)

    sample_search_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    if best_payload is None:
        raise RuntimeError("No sample passed min_meta_required; relax settings or increase budgets.")

    top_payloads = sorted(top_payloads, key=lambda x: float(x["score"]), reverse=True)
    top_cache = {str(tp["sample"]): tp for tp in top_payloads}

    if cfg.verbose:
        elapsed = perf_counter() - t0
        print(
            f"[search] Done in {elapsed:.1f}s. Best sample: {best_payload['sample']} "
            f"(score={float(best_payload['score']):.6f}, n_meta={int(best_payload['n_meta'])})",
            flush=True,
        )

    return {
        "cfg": cfg,
        "search_mode": search_mode,
        "sample_search_df": sample_search_df,
        "best_payload": best_payload,
        "best_sample": str(best_payload["sample"]),
        "top_cache": top_cache,
        "i1": i1,
        "i2": i2,
        "bk1_xyz": bk1_xyz,
        "bk1_xy": bk1_xy,
        "bk1_abs": bk1_abs,
        "bk2_xyz": bk2_xyz,
        "bk2_xy": bk2_xy,
        "by1": by1,
        "white_i1": white_i1,
    }


def _smoothness_penalty(r: NDArray[np.float64]) -> float:
    """Second-derivative penalty to reduce oscillatory spectra."""
    d2 = np.diff(r, n=2)
    return float(np.mean(d2**2))


def _tail_penalty(r: NDArray[np.float64], n_tail: int = 4) -> float:
    """Penalty for excessive long-wavelength tail peaks."""
    tail = r[-n_tail:]
    body = r[:-n_tail]
    if len(body) == 0:
        return 0.0
    ref = float(np.percentile(body, 95))
    excess = np.maximum(tail - ref, 0.0)
    return float(np.mean(excess**2))


def reconstruct_sample(
    data: SceneData,
    basis: NDArray[np.float64],
    search_result: dict[str, Any],
    sample_name: str,
    cfg: ReconstructionConfig,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Run one-to-many reconstruction and select best candidate by I2-based score."""
    sample_name = str(sample_name)
    if sample_name not in data.sample_ids:
        raise RuntimeError(f"Sample {sample_name} not found in munsell.csv")

    sample_idx = data.sample_ids.index(sample_name)
    r_true = data.reflectances[sample_idx].copy()

    i1 = search_result["i1"]
    i2 = search_result["i2"]

    f1_true = xyz_from_reflectance(r_true, i1, data.xbar, data.ybar, data.zbar, data.dlam)
    f2_true = xyz_from_reflectance(r_true, i2, data.xbar, data.ybar, data.zbar, data.dlam)
    c1_true = xy_from_xyz(f1_true)
    c2_true = xy_from_xyz(f2_true)
    fy1_true = float(f1_true[1])

    bk1_xy = search_result["bk1_xy"]
    bk1_abs = search_result["bk1_abs"]
    by1 = search_result["by1"]

    top_cache = search_result["top_cache"]
    used_cached_class = sample_name in top_cache

    reject_stats = {"invalid_interval": 0, "a_out": 0, "w0_unreachable": 0, "w_out": 0, "ok": 0}
    candidates: list[dict[str, Any]] = []

    if used_cached_class:
        for it in top_cache[sample_name]["items"]:
            f1_hat = np.asarray(it["F1"], dtype=float)
            f2_hat = np.asarray(it["F2"], dtype=float)
            c1_hat = np.asarray(it["c1"], dtype=float)
            c2_hat = np.asarray(it["c2"], dtype=float)
            cand = {
                "a": None,
                "w": np.asarray(it["w"], dtype=float),
                "r_hat": np.asarray(it["r"], dtype=float),
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
            }
            candidates.append(cand)
        reject_stats["ok"] = len(candidates)
        tri_idx: list[int] | list[str] = ["cached_class"]
    else:
        triangles = find_containing_triangles(bk1_xy, c1_true)
        if not triangles:
            raise RuntimeError("Target chromaticity is outside basis gamut under I1.")

        i0, i1_idx, i2_idx, a_t = triangles[0]
        tri_idx = [i0, i1_idx, i2_idx]
        free_idx = [k for k in range(basis.shape[0]) if k not in tri_idx]

        cols = np.vstack([np.ones(basis.shape[0]), bk1_xy[:, 0], bk1_xy[:, 1]])
        t_mat = cols[:, tri_idx]
        f_mat = cols[:, free_idx]
        m = np.linalg.solve(t_mat, f_mat)

        rng = np.random.default_rng(cfg.rng_seed)
        af_samples, sample_stats = sample_a_f_iterative(m, a_t, cfg.n_samples_af, rng, eps=eps)
        reject_stats["invalid_interval"] = int(sample_stats["invalid_interval"])

        for a_f in af_samples:
            a = np.zeros(basis.shape[0], dtype=float)
            a[tri_idx] = a_t - (m @ a_f)
            a[free_idx] = a_f

            cand, status = evaluate_candidate(
                a=a,
                bk1_abs=bk1_abs,
                by1=by1,
                fy1_true=fy1_true,
                basis=basis,
                i1=i1,
                i2=i2,
                f1_true=f1_true,
                f2_true=f2_true,
                c1_true=c1_true,
                c2_true=c2_true,
                xbar=data.xbar,
                ybar=data.ybar,
                zbar=data.zbar,
                dlam=data.dlam,
                eps=eps,
            )
            if status == "ok":
                candidates.append(cand)
                reject_stats["ok"] += 1
            else:
                reject_stats[status] += 1

    if not candidates:
        raise RuntimeError("No valid candidates found for selected sample.")

    for cand in candidates:
        cand["pen_smooth"] = _smoothness_penalty(cand["r_hat"])
        cand["pen_tail"] = _tail_penalty(cand["r_hat"])
        cand["selection_score"] = (
            cand["err2_xyz"] + cfg.lambda_smooth * cand["pen_smooth"] + cfg.lambda_tail * cand["pen_tail"]
        )

    best = min(candidates, key=lambda c: c["selection_score"])
    r_rec = best["r_hat"]

    metrics = compute_metrics(r_true, r_rec, f1_true, best["F1_hat"], f2_true, best["F2_hat"])

    metamer_items = [
        {
            "r": np.asarray(c["r_hat"], dtype=float),
            "w": np.asarray(c["w"], dtype=float),
            "F1": np.asarray(c["F1_hat"], dtype=float),
            "F2": np.asarray(c["F2_hat"], dtype=float),
            "c1": np.asarray(c["c1_hat"], dtype=float),
            "c2": np.asarray(c["c2_hat"], dtype=float),
        }
        for c in candidates
    ]

    return {
        "sample": sample_name,
        "used_cached_class": used_cached_class,
        "tri_idx": tri_idx,
        "candidates": candidates,
        "best": best,
        "r_true": r_true,
        "r_rec": r_rec,
        "f1_true": f1_true,
        "f2_true": f2_true,
        "c1_true": c1_true,
        "c2_true": c2_true,
        "metrics": metrics,
        "reject_stats": reject_stats,
        "metamer_items": metamer_items,
    }


def save_reconstruction_tables(
    output_dir: Path,
    recon: dict[str, Any],
    search_df: pd.DataFrame,
    search_cfg: SearchConfig,
    recon_cfg: ReconstructionConfig,
    wl: NDArray[np.float64],
    k: int | None = None,
    degree: int | None = None,
) -> tuple[Path, Path]:
    """Save reconstructed spectrum and metrics CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = str(recon["sample"])

    rec_path = output_dir / f"reconstructed_sample_{sample}_{search_cfg.illum_1}_{search_cfg.illum_2}.csv"
    pd.DataFrame({"wavelength": wl, "r_true": recon["r_true"], "r_rec": recon["r_rec"]}).to_csv(rec_path, index=False)

    row = search_df[search_df["sample"].astype(str) == sample]
    if len(row):
        sample_search_score = float(row.iloc[0]["score"])
        sample_search_n_meta = int(row.iloc[0]["n_meta"])
    else:
        sample_search_score = float("nan")
        sample_search_n_meta = float("nan")

    summary = {
        "sample": sample,
        "illum_1": search_cfg.illum_1,
        "illum_2": search_cfg.illum_2,
        "K": k if k is not None else np.nan,
        "degree": degree if degree is not None else np.nan,
        "n_samples_af": recon_cfg.n_samples_af,
        "num_candidates": len(recon["candidates"]),
        "used_cached_class": recon["used_cached_class"],
        **recon["reject_stats"],
        **recon["metrics"],
        "best_err2_xyz": recon["best"]["err2_xyz"],
        "best_selection_score": recon["best"]["selection_score"],
        "best_pen_smooth": recon["best"]["pen_smooth"],
        "best_pen_tail": recon["best"]["pen_tail"],
        "best_err2_xy": recon["best"]["err2_c"],
        "best_err2_Y": recon["best"]["err2_y"],
        "sample_search_score": sample_search_score,
        "sample_search_n_meta": sample_search_n_meta,
    }

    met_path = output_dir / f"reconstruction_metrics_sample_{sample}_{search_cfg.illum_1}_{search_cfg.illum_2}.csv"
    pd.DataFrame([summary]).to_csv(met_path, index=False)
    return rec_path, met_path
