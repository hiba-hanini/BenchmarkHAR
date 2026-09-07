# BenchmarkHAR & BioPM - Complete Getting Started & Configuration Guide
This guide is designed for researchers and developers setting up the BenchmarkHAR repository for the first time. Because raw dataset files are excluded from the repository due to size constraints and licensing agreements, this document explains the full procedure from scratch: downloading raw data, configuring directory structures, running preprocessing pipelines, training models, and generating benchmark plots.

## Table of Contents
*   [1. Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
*   [2. Dataset Downloading & Raw Data Directory Structure](#2-dataset-downloading--raw-data-directory-structure)
    *   [A. PAMAP2 Dataset](#a-pamap2-dataset)
    *   [B. WISDM Dataset](#b-wisdm-dataset)
    *   [C. Capture24 Dataset](#c-capture24-dataset)
*   [3. Step-by-Step Pipeline Execution (run.py)](#3-step-by-step-pipeline-execution-runpy)
    *   [Option A: Step-by-Step Modular Execution](#option-a-step-by-step-modular-execution)
    *   [Option B: Fully Automated Execution (full)](#option-b-fully-automated-execution-full)
    *   [Option C: BioPM Foundation Model Pipeline](#option-c-biopm-foundation-model-pipeline)
*   [4. Running Full Benchmarks across Multiple Models & Datasets (run_all.py)](#4-running-full-benchmarks-across-multiple-models--datasets-run_allpy)
*   [5. Aggregating & Visualizing Results](#5-aggregating--visualizing-results)
*   [6. Inference & Model Reloading (load_and_predict.py)](#6-inference--model-reloading-load_and_predictpy)
*   [7. Directory Structure Summary](#7-directory-structure-summary)
*   [8. Reproducibility](#8-reproducibility)

## 1. Prerequisites & Environment Setup

### Step 1.1: Clone the Repository and External Repositories & Create a Virtual Environment
We recommend using Python 3.8 to 3.10.

```bash
# Using
git clone https://github.com/PervasiveHealthcareLAB/BenchmarkHAR
cd BenchmarkHAR
git clone git@github.com:teco-kit/ISWC22-HAR.git ISWC22_HAR
git clone git@github.com:Prithvitarale/biopm.git
```

```bash
# Create a virtual environment
python -m venv .venv
```

```bash
# Activate the virtual environment
source .venv/bin/activate      # On Linux / macOS
# .venv\Scripts\activate       # On Windows
```

### Step 1.2: Install BenchmarkHAR Dependencies
Install the required core libraries (PyTorch, TensorFlow/Keras, Scikit-Learn, Pandas, NumPy, etc.):

```bash
pip install -r requirements.txt
```

### Step 1.3: Install the BioPM Submodule in Editable Mode
To enable the BioPM Transformer model, zero-crossing movement element (ME) tokenization, and custom feature extractors:

```bash
pip install -e biopm/
```

## 2. Dataset Downloading & Raw Data Directory Structure
BenchmarkHAR natively supports PAMAP2, WISDM, and Capture24. Path mappings are declared in `configs/datasets/`. You must create the `data/raw/` directories manually and place the uncompressed raw dataset files into the expected locations.

### A. PAMAP2 Dataset
Download the raw files from the UCI PAMAP2 Physical Activity Monitoring Dataset : [link](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)

Extract the subject `.dat` files located in the `Protocol/` folder (e.g., `subject101.dat`, `subject102.dat`, ...).
Place them in the following structure:

```
data/
└── raw/
    └── pamap2/
        └── Protocol/
            ├── subject101.dat
            ├── subject102.dat
            └── ...
```

### B. WISDM Dataset
[link](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
Download the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset.
Extract the raw smartwatch accelerometer and gyroscope text files.
Organize them under `data/raw/wisdm/raw/watch/`:

```
data/
└── raw/
    └── wisdm/
        └── raw/
            └── watch/
                ├── accel/   # Files like data_1600_accel_watch.txt...
                └── gyro/    # Files like data_1600_gyro_watch.txt...
```

### C. Capture24 Dataset
[link](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)
Place the `.csv.gz` subject recordings along with the metadata dictionary files in:

```
data/
└── raw/
    └── capture24/
        ├── metadata.csv
        ├── annotation-label-dictionary.csv
        └── *.csv.gz
```

## 3. Step-by-Step Pipeline Execution (run.py)
The central CLI entry point is `run.py`. It takes the following main arguments:
*   `--dataset`: `pamap2`, `wisdm`, or `capture24`
*   `--step`: `preprocessing`, `feature_extraction`, `dimension_reduction`, `run_model`, or `full`
*   `--model`:
    *   **Classical ML**: `random_forest`, `logistic_regression`
    *   **Deep Learning**: `cnn`, `lstm`, `dnn`, `rnn`, `deepconv_lstm`, `tinyhar`, `astar_har`
    *   **Foundation Model**: `biopm`, `biopm_rf`

### Option A: Step-by-Step Modular Execution
Running individual steps is helpful for debugging or inspecting intermediate data artifacts:

#### Step 1: Preprocessing (Signal Filtering & Windowing)
Applies Butterworth lowpass/bandpass filtering (gravity/body acceleration separation), resamples signal frequencies, cleans missing activity labels, and generates sliding windows (`.npz` files stored in `data/processed/<dataset>/`):

```bash
python run.py --dataset pamap2 --step preprocessing
```

#### Step 2: Feature Extraction (Handcrafted Features for ML)
Extracts 50+ time- and frequency-domain features (RMS, FFT magnitude, Spectral Entropy, inter-axis correlations, etc.) per sliding window, and generates a feature matrix (`.pkl` file stored in `data/features/<dataset>_features_all.pkl`) :

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --step feature_extraction
```

#### Step 3: Dimensionality Reduction (Correlation Filtering)
Filters out highly correlated features (Pearson correlation by default) to reduce collinearity and prevent overfitting, and generates a reduced feature matrix (`.pkl` file stored in `data/features/<dataset>_features_filtered.pkl` :

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --step dimension_reduction
```

#### Step 4: Model Training & Evaluation
Executes subject-aware cross-validation (Leave-One-Subject-Out (LOSO) for ≤10 subjects or 5-fold GroupShuffleSplit for larger cohorts):

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model random_forest --step run_model
```

### Option B: Fully Automated Execution (full)
When passing `--step full`, `run.py` automatically checks for missing preprocessed files or feature matrices, builds them sequentially if necessary, and executes the complete pipeline end-to-end:

```bash
# Example for Deep Learning (CNN):
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model cnn --step full
```

```bash
# Example for Classical Machine Learning (Random Forest):
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model random_forest --step full
```

### Option C: BioPM Foundation Model Pipeline
BioPM uses zero-crossing segmentation to tokenize signals into Movement Elements (MEs):

#### Preprocess data into BioPM format (saved as HDF5 `.h5` files):

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model biopm --step preprocessing
```

(Alternative if standard `.npz` windows are already present):

```bash
PYTHONPATH=. python scripts/run_biopm_windows.py --dataset pamap2
```

#### Extract 1023-d BioPM embeddings & train downstream classifiers:

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model biopm --step run_model
```

## 4. Running Full Benchmarks (run_all.py)
To reproduce paper benchmarks across multiple models and datasets in a single command, use `run_all.py`:

```bash
# Run all available models on PAMAP2 and WISDM:
PYTHONPATH=. python scripts/run_all.py --datasets pamap2 wisdm
```

```bash
# Run specific models on a single dataset:
PYTHONPATH=. python scripts/run_all.py --datasets pamap2 --models random_forest cnn biopm
```

## 8. Reproducibility

BenchmarkHAR seeds each experiment centrally. The default settings live in
`configs/reproducibility.yaml` and are `seed: 42` and `deterministic: true`.
Override the seed for one run without changing YAML:

```bash
PYTHONPATH=. python scripts/run.py --dataset pamap2 --model lstm --step full --seed 42
```

`run_all.py` forwards the same override to every experiment:

```bash
PYTHONPATH=. python scripts/run_all.py --datasets pamap2 wisdm --models random_forest cnn lstm --seed 42
```

The experiment seed controls Python, NumPy, TensorFlow/Keras, PyTorch,
GroupShuffleSplit, stochastic scikit-learn estimators, and TinyHAR training
shuffling. LOSO is deterministic and does not use the seed. Reproducible
results require the same code, data, configuration, seed, and sufficiently
similar software and hardware environment. Different CUDA, TensorFlow, PyTorch,
or GPU implementations can still prevent bitwise-identical results. Also,
`PYTHONHASHSEED` must be set before Python starts to affect the current process;
the utility only makes it available to child processes when called at runtime.
Model checkpoints are stored under `data/models/<dataset>/<model>/seed_<seed>/`
to prevent a run from silently reusing weights trained with a different seed.

## Benchmark Evaluation Protocol

BenchmarkHAR currently contains multiple protocol-specific benchmark tracks rather than a single fully harmonized representation.

- Standard HAR track: classical ML models consume handcrafted window-level features extracted from the standard preprocessing pipeline.
- Standard HAR track: deep learning models consume window-level sensor tensors produced by the same preprocessing pipeline, then standardized per fold.
- BioPM track: BioPM and BioPM-RF use BioPM-specific preprocessing and feature representation. Their input representation is not identical to the standard HAR track and should not be interpreted as an apples-to-apples comparison with standard ML/DL results.

Across all tracks, the outer evaluation remains subject-disjoint. The outer test subjects are never used for feature selection, model selection, checkpoint selection, or early stopping. Canonical labels are defined by dataset configuration and are validated before model fitting.

For the deep-learning track, the outer training subjects are split internally into inner training and inner validation subjects only for checkpoint selection. The best checkpoint is chosen from validation performance, and the outer test fold is evaluated exactly once after selection. This keeps validation-based model choice inside the training portion of the outer fold without contaminating the held-out test subjects.

Results should therefore be interpreted as protocol-specific tracks unless a future benchmark protocol harmonizes preprocessing and input representation across families.

### Benchmark validity / label protocol

The benchmark uses subject-aware CV, and feature-selection is fitted on the
training fold only. The selected feature list is then applied unchanged to the
matching test fold; no test subjects are used to choose the feature set. This
prevents correlation-based leakage from held-out subjects into model training.

Labels are normalized through a dataset-defined canonical vocabulary before
training. The canonical mapping comes from each dataset configuration (`label_mapping`
or `canonical_label_values`), not from a LabelEncoder fit on the full dataset or
held-out test labels. Unknown or unmapped labels raise a clear error instead of
silently expanding the benchmark class set.

### Deterministic experiment identity and manifest checks

The canonical reproducibility configuration is stored in `configs/reproducibility.yaml`:

```yaml
reproducibility:
  seed: 42
  deterministic: true
```

The CLI reads this file and applies any explicit `--seed` or `--deterministic`
override for that run. Every run then computes a stable experiment ID from the
resolved configuration via SHA-256 over a canonical JSON payload. The ID includes
at least the dataset, model, seed, deterministic flag, preprocessing settings,
model parameters, and CV settings. It does not depend on time or a Python process
random seed.

Experiment results and model checkpoints are now written under experiment-specific
paths, for example `data/results/<dataset>/<model>/<experiment_id>/results.pkl`
and `data/models/<dataset>/<model>/<experiment_id>/`. Each artifact directory also
contains a `manifest.json` file with the experiment ID, dataset/model metadata,
seed, deterministic flag, resolved configuration, git commit, artifact path, and
creation timestamp. A manifest is treated as the validity check for reuse: an
artifact is reused only when the artifact exists and the manifest matches the
current experiment identity. If the artifact exists but the manifest is missing or
stale, the run reports the stale cache and rebuilds the artifact without deleting
the old file.

## Scientific evaluation protocol

BenchmarkHAR splits data at the subject level: a subject must never appear in
both sides of a fold. Classical-ML correlation filtering is fit only on each
fold's training subjects, then the saved feature list is applied unchanged to
that fold's test subjects. Fitting it globally would let test-subject feature
distributions choose training features, which is data leakage.

Prediction labels are converted to a configuration-defined canonical vocabulary
before model training; encoders are constructed from that vocabulary rather
than fit on held-out labels. Result pickles record the train/test subject IDs,
CV strategy, canonical label vocabulary, and—for ML—the selected feature names
for every fold.

## 5. Aggregating & Visualizing Results
When model training finishes, individual fold evaluation results are saved in `data/results/*_results.pkl`.

### 1. Generate Summary CSV Table
Consolidates mean Macro F1-Score, Accuracy, standard deviations, and cross-validation strategies across all models into `results_summary.csv`:

```bash
PYTHONPATH=. python scripts/aggregate_results.py
```

### 2. Plot Performance Comparison Figures
Generates performance comparison charts saved as PNG images under `data/plots/`:

```bash
PYTHONPATH=. python scripts/visualize_results.py
```

### 3. Visualize Bio-PM segmentation
Shows the filtered accelerometer signals with shaded regions indicating the detected movement elements (MEs).


```bash
PYTHONPATH=. python scripts/visualize_segmentation.py --dataset pamap2 --subject-id 101 --window-index 50
```


## 6. Inference & Model Reloading (load_and_predict.py)
To evaluate a previously saved model checkpoint on a specific fold without retraining:

```bash
PYTHONPATH=. python scripts/load_and_predict.py --dataset pamap2 --model random_forest --fold-index 0
```

## 7. Directory Structure Summary
After executing the complete workflow, the local folder hierarchy will contain:

```
BenchmarkHAR/
├── configs/                   # Dataset (.yaml) and Model (.yaml) parameters
├── data/
│   ├── raw/                   # Uncompressed raw dataset files
│   ├── processed/             # Sliding window arrays (.npz or .h5)
│   ├── features/              # Feature matrices (.pkl)
│   ├── models/                # Saved model weights (.joblib, .keras, .pt)
│   ├── results/               # Experiment evaluation pickles (*_results.pkl)
│   └── plots/                 # Benchmark visualization plots (.png)
├── biopm/                     # Standalone BioPM Transformer package & checkpoints
├── ISWC22_HAR/                # SOTA Deep Learning modules (TinyHAR, etc.)
├── run.py                     # Main CLI entry point
├── run_all.py                 # Multi-model benchmarking script
├── aggregate_results.py       # Result aggregation script
├── visualize_results.py       # Result plotting script
├── load_and_predict.py        # Checkpoint reloading script
└── results_summary.csv        # Aggregated benchmark summary
```
