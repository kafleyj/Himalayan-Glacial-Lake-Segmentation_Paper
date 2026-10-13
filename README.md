# Himalayan Glacial Lake Segmentation

## Research project

**Failure Modes and Geographic Evaluation Sensitivity of Deep Learning Models for Himalayan Glacial Lake Segmentation**

This repository contains research code, evaluation metrics, figures,
and reproducibility materials.

## Architectures

- U-Net (ResNet34 encoder)
- DeepLabV3+ (ResNet34 encoder)
- SegFormer-B0

Each architecture was evaluated using three training seeds:
42, 142, and 242.

## Research analyses

1. Nepal-30 glacial lake segmentation
2. Gandaki, Karnali, and Koshi basin-wise comparison
3. Lake-mask connected-component size sensitivity
4. Provisional background false-positive evaluation
5. False-alarm component-size sensitivity
6. Geographic sample-selection sensitivity
7. GLID global transferability analysis

## Repository structure

- `src/` — Model, dataset, training and evaluation code
- `scripts/` — Experimental pipelines and analysis scripts
- `configs/` — Available configurations
- `results/` — Research metrics and analysis tables
- `figures/` — Training curves and qualitative visualizations
- `docs/` — File inventory

## Reproducibility

Original satellite imagery and model checkpoints are not included.
Training requires the GLID dataset and local dataset configuration.

Scripts may contain original HPC-specific locations and need path
configuration before running on another machine.

The original training manifest is excluded because it references
node-local HPC dataset paths.

## Scientific limitations

- GLID training PNGs have no recoverable geographic linkage in
  the available training manifest.
- Nepal evaluation imagery may not be geographically independent
  of the GLID training dataset.
- Some Nepal evaluation patches overlap one another.
- Nepal lake inventory polygons cover multiple years, while
  selected Sentinel-2 images are from 2023.
- Negative-background examples are inventory-negative and
  visually screened, not conclusively verified lake-free.
- Architectures were evaluated at their checkpoint input sizes,
  which differ across models.
- Some figures and analyses are exploratory or sensitivity analyses
  rather than independent benchmark results.

## Publication

IEEE conference manuscript in preparation.

## License

No license is assigned yet. Source data and imagery remain subject
to their original licenses and usage restrictions.
