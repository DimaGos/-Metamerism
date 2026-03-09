# Metamer Reconstruction Project

Article used:
- https://arxiv.org/pdf/2306.11464

## Assignment
Implement the method from the paper and compare reconstructed reflectance spectra against original spectra.

Required steps:
1. Use spectral datasets of reflectances and illuminants.
2. Compute color responses under illuminant 1 and 2 with standard observer sensitivities from `xyz_matching_fun.csv`.
3. Reconstruct spectra that match target color responses under the first and second illuminants (Sections 6.1 and 6.2).
4. Evaluate reconstruction accuracy between original and reconstructed spectra.
5. Build a Figure 11/12-like example (hidden pattern under different illuminants).

## Project Layout

```text
main.py
requirements.txt
.gitignore
README.md
data/
  illum.csv
  munsell.csv
  xyz_matching_fun.csv
  ippi.png
graphics/
  sample_499/
    spectra_family_...
    spectra_compare_...
    xy_span_...
    figure10_swatches_...
    figure11_like_...
  sample_1170/
    ...
src/metamer_project/
  __init__.py
  colorimetry.py
  io_utils.py
  basis.py
  metamer.py
  pipeline.py
  visualization.py
  figure11.py
```

## Pipeline Order

`main.py` runs the same logical order as in the notebook:
1. Compute sample XYZ/xy responses for two illuminants.
2. Build basis and search best sample under fixed illuminants.
3. Reconstruct spectrum for selected sample.
4. Save metrics and reconstructed spectra.
5. Plot spectra family, spectra comparison, xy span, and swatches.
6. Build Figure 11-like hidden pattern rendering under D65 and F2.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Useful options:

```bash
python main.py \
  --illum-1 D65 --illum-2 F2 \
  --k 7 --degree 2 \
  --sample 499 \
  --swatch-mode random --swatch-seed 7 \
  --use-search-cache
```

During long sample search, the script prints progress lines:

```text
[search] Selecting best sample for metamerism under D65/F2 from 220 candidates...
[search] Processed 10/220 samples in 12.3s
...
[search] Done in 143.7s. Best sample: 1170 (score=..., n_meta=...)
```

After run, a markdown summary is saved to:

```text
results/run_report.md
```

It includes configuration, reconstruction metrics, ranking table (search or cached), generated artifact paths, and inline image previews.

By default, Figure11-like stage uses the same sample as reconstruction (`--sample`).
You can override it with `--figure11-sample`.

If you want compact terminal output (recommended), use default behavior and read `results/run_report.md`.
To print all artifact paths in terminal, pass `--print-artifacts`.
All generated images are stored under per-sample folders: `graphics/sample_<sample_id>/`.

Fast rerun with cache:

```bash
python main.py --use-search-cache --sample 504
```

Report table controls:

```bash
python main.py \
  --use-search-cache \
  --report-table-source cached \
  --report-top-n 0
```

How `--report-top-n` works:
- `--report-top-n 10` means show only first 10 rows of selected ranking source.
- `--report-top-n 50` means first 50 rows.
- `--report-top-n 0` means show all rows from selected source.

How `--report-table-source` works:
- `cached`: table is built from cached payload only (at most `--top-cache-size` rows from search stage).
- `search`: table is built from full search dataframe (`metamer_search_samples_fixed_*.csv`).

About swatch indices:
- If you pass `--print-picked-indices`, script prints source indices of selected tiles.
- These are 1-based positions in the angle-sorted metamer family used to fill Figure 10 swatch grid.
- This is mostly a debug/reproducibility detail; for normal runs you can ignore it.
