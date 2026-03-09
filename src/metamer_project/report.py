from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import ReconstructionConfig, SearchConfig


def _to_kv_lines(items: list[tuple[str, Any]]) -> str:
    """Format key-value list as markdown bullets."""
    return "\n".join([f"- **{k}**: `{v}`" for k, v in items])


def _format_cell(value: Any) -> str:
    """Format a value for markdown-table cell."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e4 or (0 < abs(value) < 1e-4):
            return f"{value:.3e}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to plain GitHub-flavored markdown table."""
    if df.empty:
        return "_No rows._"

    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        cells = [_format_cell(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_run_report(
    report_path: Path,
    search_cfg: SearchConfig,
    recon_cfg: ReconstructionConfig,
    sample_name: str,
    search_df: pd.DataFrame,
    recon: dict[str, Any],
    output_paths: dict[str, Path],
    fig11_result: dict[str, Any] | None,
    ranking_title: str = "Top-10 Search Table",
    ranking_limit: int | None = 10,
) -> Path:
    """Write markdown report with configuration, metrics, and generated artifacts."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if ranking_limit is None:
        ranking_df = search_df.copy()
    else:
        ranking_df = search_df.head(max(0, int(ranking_limit))).copy()

    primary_cols = [
        "sample",
        "status",
        "n_meta",
        "score",
        "xy_spread_max",
        "xy_hull_area",
        "dist_to_white_i1",
    ]
    quality_cols = [
        "xy_spread_mean",
        "i1_err_xy_max",
        "i1_err_y_max",
    ]

    primary_cols = [c for c in primary_cols if c in ranking_df.columns]
    quality_cols = [c for c in quality_cols if c in ranking_df.columns]

    ranking_primary_md = _to_markdown_table(ranking_df[primary_cols] if primary_cols else ranking_df)
    ranking_quality_md = _to_markdown_table(
        ranking_df[["sample", *quality_cols]] if quality_cols and "sample" in ranking_df.columns else pd.DataFrame()
    )

    metrics = recon["metrics"]
    summary_lines = _to_kv_lines(
        [
            ("Selected sample", sample_name),
            ("Used cached class", recon["used_cached_class"]),
            ("Candidates", len(recon["candidates"])),
            ("RMSE spectrum", f"{metrics['rmse_spectrum']:.8f}"),
            ("MAE spectrum", f"{metrics['mae_spectrum']:.8f}"),
            ("err1_xyz_l2", f"{metrics['err1_xyz_l2']:.8e}"),
            ("err2_xyz_l2", f"{metrics['err2_xyz_l2']:.8e}"),
            ("err1_xy_l2", f"{metrics['err1_xy_l2']:.8e}"),
            ("err2_xy_l2", f"{metrics['err2_xy_l2']:.8e}"),
        ]
    )

    cfg_lines = _to_kv_lines(
        [
            ("Illuminants", f"{search_cfg.illum_1}/{search_cfg.illum_2}"),
            ("n_neutral_candidates", search_cfg.n_neutral_candidates),
            ("af_samples_per_tri_sample", search_cfg.af_samples_per_tri_sample),
            ("seed_sample_search", search_cfg.seed_sample_search),
            ("min_meta_required", search_cfg.min_meta_required),
            ("top_cache_size", search_cfg.top_cache_size),
            ("n_samples_af", recon_cfg.n_samples_af),
            ("recon_seed", recon_cfg.rng_seed),
            ("lambda_smooth", recon_cfg.lambda_smooth),
            ("lambda_tail", recon_cfg.lambda_tail),
        ]
    )

    paths_lines = _to_kv_lines([(k, str(v)) for k, v in output_paths.items()])

    def rel(p: Path) -> str:
        return os.path.relpath(p, start=report_path.parent)

    figure_lines: list[str] = []
    for key, title in [
        ("spectra_family_png", "Spectra Family"),
        ("spectra_compare_png", "Spectra Compare"),
        ("xy_span_png", "xy Span"),
        ("swatches_png", "Swatches"),
    ]:
        if key in output_paths:
            p = output_paths[key]
            figure_lines.append(f"### {title}\n![{title}]({rel(p)})")
    figures_block = "\n\n".join(figure_lines)

    fig11_block = ""
    if fig11_result is not None:
        fig11_block = (
            "\n## Figure 11-like\n"
            + _to_kv_lines(
                [
                    ("base_sample", fig11_result["base_sample"]),
                    ("metamer_class_size", fig11_result["metamer_class_size"]),
                    ("pair_A", fig11_result["idx_A"]),
                    ("pair_B", fig11_result["idx_B"]),
                    ("d65_lin", f"{fig11_result['d65_lin']:.8e}"),
                    ("f2_lin", f"{fig11_result['f2_lin']:.8e}"),
                    ("figure", str(fig11_result["fig_path"])),
                    ("spectra_csv", str(fig11_result["csv_path"])),
                ]
            )
            + "\n\n"
            + f"![Figure11]({rel(fig11_result['fig_path'])})\n"
        )

    text = f"""# Metamer Run Report

## Configuration
{cfg_lines}

## Reconstruction Summary
{summary_lines}

## {ranking_title}
Shown rows: `{len(ranking_df)}`.

### Main Columns
{ranking_primary_md}

### Quality Columns
{ranking_quality_md}

## Output Files
{paths_lines}

## Figures
{figures_block}

{fig11_block}
"""

    report_path.write_text(text)
    return report_path
