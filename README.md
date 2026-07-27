# Nested Spatiotemporal Anomaly Detection with Semantic Augmentation

Code, model checkpoints, and data used to develop **NST-Net** (`NestedAD` in code), a nested
spatiotemporal deep learning framework for multivariate time series anomaly detection (MTSAD),
paired with a semantic, expert-knowledge (EK) rule-based module. Developed as part of the
**ARGUS** project for structural health monitoring of cultural heritage sites.

> Paper: *Nested Spatiotemporal Anomaly Detection with Semantic Augmentation: A Case Study in
> Heritage Conservation* (in preparation / under review, MDPI *Sensors*).

## About ARGUS

ARGUS is a cultural heritage structural health monitoring project spanning five European pilot
sites, each instrumented with heterogeneous environmental and structural sensors (temperature,
humidity, crack meters, tilt, airflow, air quality, soil moisture, PPV, GNSS, etc.):

- **Delos** (Greece)
- **Baltanás** (Spain)
- **Monti Lucretili** (Italy)
- **Sant'Antonio di Ranverso** (Italy)
- **Schenkenberg** (Switzerland)

Sensor data at each site is irregular, multi-cadence, and affected by high missingness — the
preprocessing pipeline in this repo is built to handle exactly that, and each site has its own
set of expert-knowledge (EK) anomaly rules and threshold overrides (see `dataloaders.py`).

## Repository structure

```
.
├── data/                       # ARGUS pilot-site data — available upon request
├── logs/                       # Training / evaluation run logs
├── models/
│   └── checkpoints/            # Saved model checkpoints (<model_id>_<dataset>/)
├── notebooks/
│   └── evaluation.ipynb        # Results exploration / figure generation
└── src/
    ├── run.py                  # Main entrypoint: parses args, builds dataset, runs the pipeline
    ├── shrun.sh                # Example SLURM/local batch script running several configs in series
    ├── config/
    │   ├── arguments.py        # All CLI arguments (see below)
    │   └── reporter.py         # Reporter: collects and dumps per-run results to JSON
    ├── data/
    │   ├── pilots_preprocessing.py  # TimeSeriesData (raw ARGUS loading/cleaning) + TSDataset
    │   ├── ekrules.py               # RuleEngine: per-site expert-knowledge anomaly rules
    │   └── dataloaders.py           # Builds the TimeSeriesData entity, TSDataset, and DataLoaders
    ├── models/
    │   └── archs.py             # NestedAD (NST-Net) architecture, RevIN, RMSNorm, SerializableModule registry
    ├── train_test/
    │   └── train_test.py        # Training loop, scoring, thresholding, and metric computation
    └── utils/
        ├── helpers_setup.py     # Seeding, device/experiment setup, ACF/PACF temporal analysis
        ├── helpers_output.py    # Scoring, thresholding (POT/optimized POT), anomaly reporting
        └── helpers_plots.py     # All diagnostic and results plotting
```

## Getting started

```bash
git clone https://github.com/lidiaabad/Nested-Spatiotemporal-Anomaly-Detection-with-Semantic-Augmentation.git
cd Nested-Spatiotemporal-Anomaly-Detection-with-Semantic-Augmentation
pip install -r requirements.txt 
```
` nohup ./src/shrun.sh` to run the overall pipeline 

Add `--eval_only_pt --loaded_path <name>` to re-evaluate an existing checkpoint from
`models/checkpoints/` across 10 seeds instead of training from scratch (see the `eval_only_pt`
branch in `src/run.py`, which reports mean ± std over the 10 runs).

Outputs land in `models/checkpoints/<model_id>_<dataset>/`: the trained checkpoint, a JSON
report (`<model_id>_report.json`, written by `Reporter`) with arguments, training curves, and
injected-anomaly metrics, plus diagnostic plots (feature overview, train/val loss, score
histograms, anomaly-score timelines).

