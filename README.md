# CSAR-Net: Cost-Sensitive Attention Residual Network for Long-Tailed Magnetic-Tile Surface Defect Recognition

This repository accompanies the paper **"CSAR-Net: A Cost-Sensitive Attention Residual Network for Long-Tailed Magnetic Tile Surface Defect Recognition"**. It contains the complete LaTeX source of the manuscript, the training/evaluation code, the deterministic multi-seed results, and the scripts that reproduce every figure.

> **TL;DR.** On the long-tailed Magnetic Tile Surface Defects (MTSD) benchmark, a label-agnostic feature-layer attention module (CBAM, injected at ResNet18 stages 3–4) is the only robust lever: it lifts macro G-mean from **0.7515 → 0.7822 (+3.1 pp)** with the lowest seed-to-seed variance. Loss-layer cost-sensitive calibration (LDAM) *hurts* on this small real dataset (0.6658, −8.6 pp), and stacking it on CBAM dilutes the gain. The paper reframes CSAR-Net as a **decoupled three-module framework** (ResNet18 backbone + CBAM feature attention + LDAM loss calibration) and uses it to answer *where the cost-sensitive signal belongs* — the honest answer being the feature layer, not the loss layer.

## Key results (deterministic, 3 seeds, macro G-mean)

| Configuration | Macro G-mean (mean ± std) |
| --- | --- |
| ResNet18 + CE (baseline) | 0.7515 ± 0.0598 |
| **CBAM + CE** | **0.7822 ± 0.0393** |
| LDAM (margin + DRW) | 0.6658 ± 0.1197 |
| CBAM + LDAM (naive CSAR-Net) | 0.7437 ± 0.0526 |
| CS-CBAM (cost injected into attention) | collapses (~0.0) |

Per-class recall and the full ablation (CBAM injection position, sub-module decomposition, LDAM margin coefficient) are reported in `results_mt/metrics/` and in the manuscript.

## Repository layout

```
csar_net/
├── paper/                     # Manuscript source (Elsevier CAS, single-column)
│   ├── csar_net.tex           #   main document
│   ├── csar_refs.bib          #   bibliography
│   ├── cas-sc.cls / cas-dc.cls / cas-common.sty / model5-names.bst
│   ├── csar_net.pdf           #   compiled version (regenerable)
│   └── figures/               #   all 7 figure PNGs used by the paper
├── ccpr/                      # Python package with the full pipeline
│   ├── models.py              #   ResNet18 + CBAM / CS-CBAM / LDAM losses
│   ├── losses.py              #   LDAM margin loss
│   ├── train.py / evaluate.py #   training loop + macro G-mean / F1 / MCC metrics
│   ├── mt_data.py             #   MTSD loader (grayscale→RGB, resize, cache)
│   └── run_mt*.py             #   experiment entry points (see below)
├── scripts/                   # Figure regeneration (matplotlib, no retraining)
├── results_mt/metrics/        # Reproduced metrics (JSON/npy); weights are NOT tracked
└── README.md
```

## Reproducing the experiments

### 1. Environment

Python 3.9+, with `torch`, `numpy`, `Pillow`, `scikit-learn` (t-SNE), and `matplotlib`.

```bash
pip install torch numpy pillow scikit-learn matplotlib
```

### 2. Data

Download the **Magnetic Tile Surface Defects (MTSD)** dataset — the public benchmark of [Huang, Qiu & Yuan, *The Visual Computer*, 36(1):85–96, 2020](https://doi.org/10.1007/s00371-018-1588-5) (commonly hosted at [`abin24/Magnetic-tile-defect-datasets`](https://github.com/abin24/Magnetic-tile-defect-datasets)). Place the six class folders under `data/magnetic_tile/`:

```
data/magnetic_tile/
├── MT_Blowhole/   (115)
├── MT_Break/      (85)
├── MT_Crack/      (57)
├── MT_Fray/       (32)
├── MT_Free/       (952)   # normal tiles
└── MT_Uneven/     (103)
```

No new data were generated for this study; the dataset is used as released.

### 3. Run

All experiments are launched from the repository root as `python -m ccpr.<module>` (the code uses package-relative imports). Deterministic seeds `{42, 123, 2024}` are fixed inside each runner.

| Command | Output |
| --- | --- |
| `python -m ccpr.run_mt` | Main 5-config × 3-seed experiment → `results_mt/metrics/multiseed.json` (also saves CE/LDAM seed-42 checkpoints for t-SNE) |
| `python -m ccpr.run_mt_attention` | Attention-variant comparison (SE / ECA / CA / CBAM) → `attention_ablation.json` |
| `python -m ccpr.run_mt_ablation_extra` | CBAM position, sub-module decomposition, LDAM margin `C` sweep → `ablation_extra.json` |
| `python -m ccpr.run_mt_diag` | LDAM margin-only (no DRW) and inverse-frequency CS-CE diagnostics → `diag.json` |
| `python -m ccpr.run_mt_figures2` | Confusion matrices, training curves, penultimate features → `confusion_*.json`, `history_*.json`, `feat_*.npy` |
| `python -m ccpr.run_mt_visualize` | t-SNE visualisation of penultimate features |

The macro G-mean is the primary metric throughout (it is the geometric mean of per-class recall and is the standard long-tail metric); accuracy, macro-F1 and MCC are reported alongside it.

### 4. Regenerating the paper figures

The figures in `paper/figures/` are produced offline from the cached metrics — no retraining required:

```bash
python scripts/gen_english_figures.py   # confusion, training, t-SNE, recall bars, ablation bars
python scripts/gen_dataset_figure.py    # dataset sample montage
python scripts/make_longtail_fig.py     # class-distribution (long-tail) figure
```

## Compiling the manuscript

```bash
cd paper
pdflatex csar_net          # 1st pass
bibtex   csar_net          # resolve bibliography
pdflatex csar_net          # 2nd pass
pdflatex csar_net          # 3rd pass (fixes cross-references)
```

The document uses the Elsevier CAS single-column class (`cas-sc`); `cas-dc` (double-column) is also included. All floats use the key-value `[pos=htbp]` placement syntax required by `cas-common.sty`.

## Data and code availability

The source code, training configuration, and figure-generation scripts that reproduce the experiments are available in this repository. The MTSD dataset is a public benchmark introduced by [Huang, Qiu & Yuan (2020)](https://doi.org/10.1007/s00371-018-1588-5) and is used as released; no new data were generated. Trained model checkpoints are **not** distributed in this repository because of their size, and are available from the corresponding author on reasonable request.

## Citation

```bibtex
@article{csarnet2026,
  title   = {CSAR-Net: A Cost-Sensitive Attention Residual Network for
             Long-Tailed Magnetic Tile Surface Defect Recognition},
  author  = {[First author] and Chenyang Xie},
  journal = {Expert Systems with Applications},
  year    = {2026}
}
```

## License

Code: MIT. Data and figures: see the original dataset license (Huang et al., 2020).
