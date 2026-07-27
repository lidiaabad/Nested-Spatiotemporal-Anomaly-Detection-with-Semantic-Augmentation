# Nested-Spatiotemporal-Anomaly-Detection-with-Semantic-Augmentation
This repository includes the code, model chekpoints and data employed for the development of the paper Nested Spatiotemporal Anomaly Detection with Semantic Augmentation: A Case Study in Heritage Conservation. Developed as part of the **ARGUS** project for structural health
monitoring of cultural heritage sites.

## About ARGUS

ARGUS is a cultural heritage structural health monitoring project spanning five European
pilot sites, each instrumented with heterogeneous environmental and structural sensors
(temperature, humidity, crack meters, tilt, airflow, air quality, etc.):

- **Delos** (Greece)
- **Baltanás** (Spain)
- **Monti Lucretili** (Italy)
- **Sant'Antonio di Ranverso** (Italy)
- **Schenkenberg** (Switzerland)

Sensor data at each site is irregular, multi-cadence, and affected by high missingness — the
preprocessing pipeline in this repo is built to handle exactly that.

## Repository structure

```
.
├── data/                   #Available upon request
├── logs/                   # Training and evaluation run logs
├── models/
│   └── checkpoints/        # Saved models
├── notebooks/               # Exploratory analysis, diagnostics, and figure-generation notebooks
├── src/
│   └── data/                # Data loading & preprocessing (this file's home)
│       ├── pilots_preprocesing.py   # ARGUS pilot-site preprocessing (TimeSeriesData, TSDataset)
│       ├── ek_rules.py              # Definition of EK rules (semantic module)
│       └── dataloaders.py           # creation of the dataset instance and dataloaders.
|       
└── README.md
```

## Core components

### `src/data/pilots_preprocesing.py`

- **`TimeSeriesData`** — base dataset for a single ARGUS pilot site. Reads every sensor CSV in
  a site's folder, merges by sensor/variable name, applies site-specific contextual
  corrections (documented inline, flagged with `# TBC` where a written justification is still
  owed for the paper/appendix), detects frozen periods, resamples to a common frequency,
  converts raw sensor units to physical units, and derives expert-knowledge (EK) anomaly
  labels via a per-site `RuleEngine`.
- **`TSDataset`** — model-ready `torch.utils.data.Dataset` built on top of `TimeSeriesData`.
  Performs the temporal train/val/test split, drops low-coverage features per split, imputes
  and scales (fit on train only, alive-aware), and injects synthetic collective / contextual /
  point anomalies into the test split for controlled evaluation, using a round-robin scheduler
  so no anomaly type starves the others.

### `src/data/pub_ddbb.py`

Loaders and preprocessing utilities for the public MTSAD benchmarks (SMAP, MSL, SMD, WADI,
SWaT) used as baselines, plus a shared forecasting-window / dataloader pipeline
(`PublicTimeSeriesData`) so ARGUS and public-benchmark data can be run through the same
downstream evaluation code.

## Known open items (`# TBC` markers)

Throughout `pilots_preprocesing.py`, manual per-site decisions (sensor drops, frozen-period
date ranges, unit-conversion constants, thresholds) are marked `# TBC` where they still need a
short written justification for reviewers. Search the file for `# TBC` to find the full list
before submission.

## Getting started

```bash
git clone https://github.com/lidiaabad/Nested-Spatiotemporal-Anomaly-Detection-with-Semantic-Augmentation.git
cd Nested-Spatiotemporal-Anomaly-Detection-with-Semantic-Augmentation
pip install -r requirements.txt   # add this file if not already present
```

Example usage:

```python
from data.pilots_preprocesing import TimeSeriesData, TSDataset

ts_data = TimeSeriesData(folder_path="data/delos", freq="15min", ek=True)

dataset = TSDataset(
    mode="train",
    data=ts_data.raw_df_with_alive,
    feature_names=ts_data.feature_names,
    lb=48,
    val_ratio=0.15,
    test_ratio=0.15,
    positive=ts_data.ek_rules[0] if ts_data.ek_rules else None,
)
```

## Citation

If you use this code, please cite:

```bibtex
@article{abad_nstnet,
  title   = {Nested Spatiotemporal Anomaly Detection with Semantic Augmentation: A Case Study in Heritage Conservation},
  author  = {Abad, Lidia and collaborators},
  journal = {Sensors (MDPI)},
  year    = {2026},
  note    = {Under review}
}
```

## License

*(Add a license, e.g. MIT/Apache-2.0, if not already present in the repo.)*
