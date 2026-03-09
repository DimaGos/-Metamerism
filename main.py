from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from metamer_project.basis import make_quadratic_pu_basis
from metamer_project.figure11 import run_figure11_like
from metamer_project.io_utils import compute_xyz_responses_table, load_scene_data
from metamer_project.pipeline import (
    ReconstructionConfig,
    SearchConfig,
    reconstruct_sample,
    save_reconstruction_tables,
    search_samples_fixed_illuminants,
)
from metamer_project.report import write_run_report
from metamer_project.visualization import plot_spectra_compare, plot_spectra_family, plot_swatches, plot_xy_span


def _search_cache_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Build a stable signature for cached search payload."""
    return {
        "illum_1": args.illum_1,
        "illum_2": args.illum_2,
        "k": args.k,
        "degree": args.degree,
        "n_neutral_candidates": args.n_neutral_candidates,
        "af_samples_per_tri_sample": args.af_samples_per_tri_sample,
        "seed_sample_search": args.seed_sample_search,
        "min_meta_required": args.min_meta_required,
        "i1_tol_xy": args.i1_tol_xy,
        "i1_tol_y": args.i1_tol_y,
        "force_sample": args.force_sample,
        "top_cache_size": args.top_cache_size,
    }


def _search_cache_path(args: argparse.Namespace) -> Path:
    """Return deterministic cache-file path for search stage."""
    signature = _search_cache_signature(args)
    digest = hashlib.sha1(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    fname = f"search_cache_{args.illum_1}_{args.illum_2}_k{args.k}_d{args.degree}_{digest}.pkl"
    return args.cache_dir / fname


def _cached_table_df(search_result: dict[str, Any]) -> Any:
    """Convert cached top payloads to a DataFrame-like table for report."""
    import pandas as pd

    rows = []
    for payload in search_result["top_cache"].values():
        rows.append(
            {
                "sample": str(payload["sample"]),
                "status": "ok",
                "n_meta": int(payload["n_meta"]),
                "xy_hull_area": float(payload["xy_hull_area"]),
                "xy_spread_max": float(payload["xy_spread_max"]),
                "xy_spread_mean": float(payload["xy_spread_mean"]),
                "score": float(payload["score"]),
                "i1_err_xy_max": float(payload["i1_err_xy_max"]),
                "i1_err_y_max": float(payload["i1_err_y_max"]),
                "dist_to_white_i1": float(payload["dist_to_white_i1"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the project pipeline."""
    parser = argparse.ArgumentParser(description="Metamer reconstruction pipeline from notebook as Python project")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="Directory with illum.csv/munsell.csv/xyz_matching_fun.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results", help="Directory for CSV outputs")
    parser.add_argument("--graphics-dir", type=Path, default=ROOT / "graphics", help="Directory for PNG outputs")

    parser.add_argument("--illum-1", type=str, default="D65")
    parser.add_argument("--illum-2", type=str, default="F2")

    parser.add_argument("--k", type=int, default=7, help="Basis size")
    parser.add_argument("--degree", type=int, default=2, help="Spline degree")

    parser.add_argument("--n-neutral-candidates", type=int, default=220)
    parser.add_argument("--af-samples-per-tri-sample", type=int, default=1800)
    parser.add_argument("--seed-sample-search", type=int, default=2028)
    parser.add_argument("--min-meta-required", type=int, default=24)
    parser.add_argument("--i1-tol-xy", type=float, default=2e-4)
    parser.add_argument("--i1-tol-y", type=float, default=2e-3)
    parser.add_argument("--force-sample", type=str, default=None)
    parser.add_argument("--top-cache-size", type=int, default=100)

    parser.add_argument("--sample", type=str, default=None, help="Sample to reconstruct. Defaults to best from search")
    parser.add_argument("--n-samples-af", type=int, default=1000)
    parser.add_argument("--recon-seed", type=int, default=42)
    parser.add_argument("--lambda-smooth", type=float, default=0.03)
    parser.add_argument("--lambda-tail", type=float, default=0.12)

    parser.add_argument("--swatch-mode", type=str, default="random", choices=["random", "stride", "linspace"])
    parser.add_argument("--swatch-step", type=int, default=32)
    parser.add_argument("--swatch-start", type=int, default=1)
    parser.add_argument("--swatch-seed", type=int, default=7)

    parser.add_argument(
        "--figure11-sample",
        type=str,
        default=None,
        help="Base sample for Figure11-like stage. Default: use selected --sample.",
    )
    parser.add_argument("--skip-figure11", action="store_true")

    parser.add_argument("--use-search-cache", action="store_true", help="Load/save search stage cache in --cache-dir")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data", help="Directory for persistent search cache")

    parser.add_argument(
        "--report-table-source",
        type=str,
        default="cached",
        choices=["cached", "search"],
        help="Which ranking table to show in report.",
    )
    parser.add_argument(
        "--report-top-n",
        type=int,
        default=0,
        help="Rows in report table; 0 means all available rows.",
    )
    parser.add_argument("--print-artifacts", action="store_true", help="Print all artifact paths in terminal")
    parser.add_argument(
        "--print-picked-indices",
        action="store_true",
        help="Print swatch source indices (1-based positions inside angle-sorted metamer family).",
    )
    return parser.parse_args()


def main() -> None:
    """Run end-to-end pipeline equivalent to the notebook."""
    args = parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.graphics_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("[stage] Loading data...", flush=True)
    data = load_scene_data(args.data_dir)

    print("[stage] Computing XYZ/xy responses table...", flush=True)
    responses_df = compute_xyz_responses_table(data, args.illum_1, args.illum_2)
    responses_path = args.results_dir / "munsell_xyz_responses.csv"
    responses_df.to_csv(responses_path, index=False)

    print(f"[stage] Building basis (K={args.k}, degree={args.degree})...", flush=True)
    basis = make_quadratic_pu_basis(data.wl, k=args.k, degree=args.degree)

    search_cfg = SearchConfig(
        illum_1=args.illum_1,
        illum_2=args.illum_2,
        n_neutral_candidates=args.n_neutral_candidates,
        af_samples_per_tri_sample=args.af_samples_per_tri_sample,
        seed_sample_search=args.seed_sample_search,
        min_meta_required=args.min_meta_required,
        i1_tol_xy=args.i1_tol_xy,
        i1_tol_y=args.i1_tol_y,
        force_sample=args.force_sample,
        top_cache_size=args.top_cache_size,
        verbose=True,
        progress_every=10,
    )

    search_cache_path = _search_cache_path(args)
    search_result = None
    cache_used = False
    if args.use_search_cache and search_cache_path.exists():
        print(f"[stage] Loading search cache from {search_cache_path} ...", flush=True)
        with search_cache_path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("signature") == _search_cache_signature(args):
            search_result = cached["search_result"]
            cache_used = True
        else:
            print("[stage] Cache signature mismatch, running search again...", flush=True)

    if search_result is None:
        print("[stage] Searching best sample for metamerism...", flush=True)
        search_result = search_samples_fixed_illuminants(data, basis, responses_df, search_cfg)
        if args.use_search_cache:
            with search_cache_path.open("wb") as f:
                pickle.dump({"signature": _search_cache_signature(args), "search_result": search_result}, f)
            print(f"[stage] Saved search cache to {search_cache_path}", flush=True)

    search_df = search_result["sample_search_df"]
    search_csv = args.results_dir / f"metamer_search_samples_fixed_{args.illum_1}_{args.illum_2}.csv"
    search_df.to_csv(search_csv, index=False)

    sample_name = str(args.sample) if args.sample is not None else str(search_result["best_sample"])
    sample_graphics_dir = args.graphics_dir / f"sample_{sample_name}"
    sample_graphics_dir.mkdir(parents=True, exist_ok=True)

    recon_cfg = ReconstructionConfig(
        n_samples_af=args.n_samples_af,
        rng_seed=args.recon_seed,
        lambda_smooth=args.lambda_smooth,
        lambda_tail=args.lambda_tail,
    )

    print(f"[stage] Reconstructing sample {sample_name}...", flush=True)
    recon = reconstruct_sample(data, basis, search_result, sample_name, recon_cfg)
    rec_csv, rec_metrics_csv = save_reconstruction_tables(
        output_dir=args.results_dir,
        recon=recon,
        search_df=search_df,
        search_cfg=search_cfg,
        recon_cfg=recon_cfg,
        wl=data.wl,
        k=args.k,
        degree=args.degree,
    )

    print(f"[stage] Building plots into {sample_graphics_dir} ...", flush=True)
    spectra_family_png = plot_spectra_family(data.wl, recon, args.illum_1, args.illum_2, sample_graphics_dir)
    spectra_compare_png = plot_spectra_compare(data.wl, recon, args.illum_2, search_result["i2"], sample_graphics_dir)
    xy_span_png = plot_xy_span(
        recon["metamer_items"],
        search_result["bk1_xy"],
        search_result["bk2_xy"],
        recon["c1_true"],
        sample_name,
        args.illum_1,
        args.illum_2,
        sample_graphics_dir,
    )
    swatches_png, picked_idx = plot_swatches(
        recon["metamer_items"],
        sample_name,
        args.illum_1,
        args.illum_2,
        sample_graphics_dir,
        mode=args.swatch_mode,
        pick_step=args.swatch_step,
        pick_start_1based=args.swatch_start,
        pick_random_seed=args.swatch_seed,
    )

    fig11_result = None
    if not args.skip_figure11:
        figure11_sample = str(args.figure11_sample) if args.figure11_sample is not None else sample_name
        figure11_graphics_dir = args.graphics_dir / f"sample_{figure11_sample}"
        figure11_graphics_dir.mkdir(parents=True, exist_ok=True)
        print("[stage] Building Figure 11-like example...", flush=True)
        fig11_result = run_figure11_like(
            data=data,
            basis=basis,
            bk1_xy=search_result["bk1_xy"],
            bk1_abs=search_result["bk1_abs"],
            by1=search_result["by1"],
            base_sample=figure11_sample,
            output_dir=figure11_graphics_dir,
            illum_1=args.illum_1,
            illum_2=args.illum_2,
            n_af_per_tri=min(args.af_samples_per_tri_sample, 1200),
            rng_seed=args.seed_sample_search,
            i1_tol_xy=args.i1_tol_xy,
            i1_tol_y=args.i1_tol_y,
            logo_candidates=[ROOT / "data" / "ippi.png", ROOT / "ippi.png", ROOT.parent / "ippi.png"],
        )

    cached_df = _cached_table_df(search_result)
    if args.report_table_source == "cached" and len(cached_df):
        report_df = cached_df
        report_title = f"Cached Ranking (up to top_cache_size={search_cfg.top_cache_size})"
    else:
        report_df = search_df
        report_title = "Search Ranking"

    output_paths = {
        "responses_csv": responses_path,
        "search_csv": search_csv,
        "reconstruction_csv": rec_csv,
        "metrics_csv": rec_metrics_csv,
        "spectra_family_png": spectra_family_png,
        "spectra_compare_png": spectra_compare_png,
        "xy_span_png": xy_span_png,
        "swatches_png": swatches_png,
    }
    if args.use_search_cache:
        output_paths["search_cache"] = search_cache_path
    report_path = write_run_report(
        report_path=args.results_dir / "run_report.md",
        search_cfg=search_cfg,
        recon_cfg=recon_cfg,
        sample_name=sample_name,
        search_df=report_df,
        recon=recon,
        output_paths=output_paths,
        fig11_result=fig11_result,
        ranking_title=report_title,
        ranking_limit=None if args.report_top_n <= 0 else args.report_top_n,
    )

    print("Done.")
    print(f"selected sample: {sample_name}")
    print(f"num candidates: {len(recon['candidates'])}")
    print(f"err2 xyz: {recon['metrics']['err2_xyz_l2']}")
    print(f"search cache used: {cache_used}")
    if args.use_search_cache:
        print(f"search cache file: {search_cache_path}")
    print(f"report: {report_path}")
    print("Open run_report.md to see full outputs and figures.")
    if args.print_picked_indices:
        print(
            "swatch picked indices (first 12, 1-based within angle-sorted metamer family):",
            picked_idx[:12],
        )

    if args.print_artifacts:
        print("responses:", responses_path)
        print("search:", search_csv)
        print("reconstruction:", rec_csv)
        print("metrics:", rec_metrics_csv)
        print("spectra family:", spectra_family_png)
        print("spectra compare:", spectra_compare_png)
        print("xy span:", xy_span_png)
        print("swatches:", swatches_png)
        if fig11_result is not None:
            print("figure11 image:", fig11_result["fig_path"])
            print("figure11 spectra:", fig11_result["csv_path"])


if __name__ == "__main__":
    main()
