# Multimodal Emotion Recognition with EEG-Guided FiLM Fusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/natsumeSama/DEAP-EEG-video-fusion-experiment-pipeline/blob/main/notebooks/colab_main.ipynb)

Research implementation for multimodal emotion recognition from EEG and facial/video information using the **DEAP dataset**.

The project compares EEG-only and video-only emotion recognition against three multimodal fusion strategies:

- Decision-level fusion
- Feature concatenation
- EEG-guided Feature-wise Linear Modulation (FiLM)

The central idea is to use EEG not only as an independent modality, but as a **conditioning signal that modifies the visual representation before classification**.

This repository is the refactored and reproducible implementation associated with the Master's thesis:

> **Multimodal Emotion Recognition Using EEG and Facial/Video Information with EEG-Guided FiLM Fusion**  
> Master's Degree in Computer Engineering, Cybersecurity and Artificial Intelligence  
> University of Cagliari, 2025/2026

---

## 1. Research objective

Emotion can be observed from multiple complementary sources.

Facial/video information represents visible behavioral cues, while EEG provides access to internal physiological activity. Each modality also has important limitations when used independently.

This project studies three main questions:

1. Does multimodal EEG-video fusion outperform unimodal EEG-only and video-only models?
2. Is feature-level interaction more effective than combining predictions only at the decision level?
3. Can EEG guide the visual representation through FiLM while remaining competitive with simple feature concatenation?

The experiments address two binary emotion-recognition tasks:

- **Valence**: low vs high valence
- **Arousal**: low vs high arousal

---

## 2. Overall framework

The same aligned EEG-video trial can follow five experimental routes:

```text
                         ┌───────────────────┐
                         │   Aligned trial   │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                   EEG                          Video
                    │                             │
          EEG preprocessing              sampled face frames
                    │                             │
        EEG Transformer encoder             ResNet18
                    │                             │
                 z_EEG                         z_VIDEO
                    │                             │
        ┌───────────┼─────────────┬──────────────┤
        │           │             │              │
        ▼           ▼             ▼              ▼
    EEG-only    Decision       Concat          FiLM
                Fusion         Fusion          Fusion
        │           │             │              │
        └───────────┴─────────────┴──────────────┘
                              │
                       Emotion prediction
```

All models use the **same aligned subject-trial pairs and the same train/validation splits**.

This is important because differences between fusion methods should come from the models themselves rather than from different data partitions.

---

## 3. Dataset

The experiments are based on the **DEAP dataset**.

The multimodal subset used in this project contains:

| Property | Value |
|---|---:|
| Subjects used | 22 |
| Trials per subject | 40 |
| Candidate multimodal trials | 880 |
| Aligned EEG-video trials | 874 |
| EEG channels used | 32 |
| EEG sampling rate | 128 Hz |
| Baseline duration | 3 s |
| Stimulus duration | 60 s |
| Video frames per trial | 18 |

Only subjects for which the required EEG and facial/video information are available are included in the multimodal experiments.

### Labels

DEAP valence and arousal ratings are converted to binary labels using a threshold of **5**.

```text
rating < 5   -> low
rating >= 5  -> high
```

The classification tasks are run independently:

```python
LABEL_TYPE = 0  # valence
LABEL_TYPE = 1  # arousal
```

The default configuration currently uses:

```python
LABEL_TYPE = 1
```

which corresponds to **arousal**.

---

## 4. EEG preprocessing

The EEG branch uses the first 32 EEG channels from the DEAP preprocessed recordings.

Each 63-second trial is processed as follows:

```text
DEAP EEG trial
      │
      ▼
32 EEG channels
      │
      ▼
Common Average Reference (CAR)
      │
      ▼
4–45 Hz band-pass filtering
      │
      ▼
50 Hz notch filtering
      │
      ▼
3 s baseline + 60 s stimulus separation
      │
      ▼
2 s windows / 1 s stride
      │
      ▼
59 temporal windows
      │
      ▼
Spectral + hemispheric asymmetry features
      │
      ▼
Baseline correction
      │
      ▼
Z-score normalization
      │
      ▼
EEG Transformer
```

### Spectral features

Four frequency bands are used:

| Band | Frequency |
|---|---|
| Theta | 4–8 Hz |
| Alpha | 8–13 Hz |
| Beta | 13–30 Hz |
| Gamma | 30–45 Hz |

Log-bandpower features are computed for every EEG channel.

With:

```text
32 channels × 4 bands = 128 spectral features
```

### Hemispheric asymmetry

Fourteen left-right electrode pairs are also used to compute:

- DASM-like differences
- RASM-like ratios

This adds:

```text
14 pairs × 4 bands × 2 asymmetry measures = 112 features
```

Therefore each temporal window contains:

```text
128 spectral features
+112 asymmetry features
-----------------------
240 EEG features
```

and each trial is represented approximately as:

```text
59 windows × 240 features
```

---

## 5. EEG model

For the primary experiments, EEG uses a Transformer encoder operating on the temporal sequence of EEG feature windows.

The main architecture is:

```text
59 × 240 EEG feature sequence
        │
        ▼
Linear projection
        │
        ▼
Transformer Encoder
        │
        ▼
Mean temporal pooling
        │
        ▼
128-dimensional EEG representation
        │
        ▼
Classification head
```

The implementation also contains additional EEG baselines for other preprocessing modes, but the main experiment uses **mode 2**, corresponding to the window-level EEG feature sequence and Transformer encoder.

---

## 6. Video preprocessing and model

The visual branch represents every trial using sampled facial frames.

The standard experiment uses:

```text
18 frames per trial
```

The processing pipeline is:

```text
Video trial
    │
    ▼
18 sampled frames
    │
    ▼
Face crops
    │
    ▼
ResNet18 preprocessing
    │
    ▼
Pretrained ResNet18 encoder
    │
    ▼
One embedding per frame
    │
    ▼
Mean pooling over frames
    │
    ▼
Trial-level video representation
```

The visual backbone is frozen by default during the unimodal video experiment:

```python
FREEZE_BACKBONE = True
```

The repository also contains utilities for:

- locating raw video files,
- sampling frames,
- detecting faces,
- computing a stable face crop for each trial,
- generating face-cropped images.

These utilities are located in:

```text
src/deap_fusion/data/video.py
```

For the standard training workflow, **precomputed face crops are sufficient**. Raw videos are not required.

---

## 7. Fusion strategies

Five models are compared under the same aligned splits.

### 7.1 EEG-only

Uses only the EEG Transformer representation.

```text
EEG -> Transformer -> classifier
```

### 7.2 Video-only

Uses only the ResNet18 trial-level visual representation.

```text
Video -> ResNet18 -> mean pooling -> classifier
```

### 7.3 Decision-level fusion

The EEG and video models make predictions independently.

Their class probabilities are averaged:

```text
p_final = 0.5 × p_EEG + 0.5 × p_VIDEO
```

No additional fusion model is trained.

This represents a late-fusion baseline.

### 7.4 Concatenation fusion

The learned EEG and video representations are concatenated before classification:

```text
z_EEG ─────┐
           ├── concatenate ── classifier
z_VIDEO ───┘
```

This allows the classifier to operate jointly on both modalities.

### 7.5 EEG-guided FiLM fusion

FiLM uses EEG as a **conditioning modality**.

The EEG representation generates feature-wise modulation parameters for the video representation:

```text
EEG representation
      │
      ▼
FiLM generator
   │       │
   ▼       ▼
 gamma    beta
   │       │
   └───┬───┘
       ▼
video modulation
```

The generated parameters are constrained around an identity transformation:

```python
gamma = 1 + 0.1 * tanh(gamma_raw)
beta  =     0.1 * tanh(beta_raw)
```

The modulated video representation is:

```text
FiLM(video) = gamma * video + beta
```

An EEG-conditioned gate controls the strength of the modulation:

```text
gate = sigmoid(W × z_EEG + b)
```

and the final representation is obtained through residual interpolation:

```text
video_mod =
    video
    + gate * (FiLM(video) - video)
```

The classifier therefore receives an **EEG-conditioned visual representation**.

The FiLM implementation also exposes:

- gamma
- beta
- gate
- modulation magnitude

for later analysis.

---

## 8. Fusion training strategy

FiLM and concatenation fusion use previously trained EEG and video encoders.

Training is performed in two stages.

### Stage 1

The unimodal encoders are frozen.

Only the fusion components and classifier are trained.

Default configuration:

```text
Epochs:        15
Learning rate: 1e-3
```

### Stage 2

Fine-tuning starts from the best Stage 1 checkpoint.

The EEG encoder remains frozen, while only the final visual layers are fine-tuned with a small learning rate.

Default configuration:

```text
Epochs:                    5
Fusion/head learning rate: 5e-5
Encoder learning rate:     1e-6
```

The best available fusion checkpoint is retained for evaluation.

---

## 9. Experimental protocol

### Subject-dependent evaluation

The primary experiment trains a separate model for each subject.

For every subject:

```text
75% training
25% validation
```

The split is stratified whenever the class distribution allows it.

The random seed is:

```python
SEED = 42
```

### Shared splits

The repository creates the train/validation trial pairs once and reuses them across:

- EEG-only
- Video-only
- Decision fusion
- Concatenation fusion
- FiLM fusion

This ensures a fair comparison.

### Four-fold cross-validation

A subject-dependent **4-fold cross-validation** experiment is also included.

For each subject, the same fold definition is activated for every model.

This provides a stability analysis beyond the single 75/25 split.

The codebase also contains support for additional split modes such as LOSO and global random splitting, but the thesis results reported here are based on the subject-dependent protocol.

---

## 10. Evaluation metrics

The project reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score

For trained models, training and validation histories are also retained.

FiLM additionally records trial-level modulation statistics.

---

## 11. Repository structure

```text
DEAP-EEG-video-fusion-experiment-pipeline/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── colab_main.ipynb
│   └── DEAP-EEG-video-fusion-experiment-pipeline.ipynb
│
├── src/
│   └── deap_fusion/
│       │
│       ├── config.py
│       │
│       ├── data/
│       │   ├── cache.py
│       │   ├── datasets.py
│       │   ├── download.py
│       │   ├── eeg.py
│       │   ├── splits.py
│       │   └── video.py
│       │
│       ├── models/
│       │   ├── eeg.py
│       │   ├── video.py
│       │   ├── film.py
│       │   └── concat.py
│       │
│       ├── training/
│       │   ├── common.py
│       │   ├── eeg.py
│       │   ├── video.py
│       │   └── fusion.py
│       │
│       ├── evaluation/
│       │   ├── decision.py
│       │   ├── results.py
│       │   └── plots.py
│       │
│       └── experiments/
│           ├── main.py
│           └── cv.py
│
├── data/
│   └── README.md
│
└── outputs/
```

### Main notebook

The recommended entry point is:

```text
notebooks/colab_main.ipynb
```

The larger notebook:

```text
notebooks/DEAP-EEG-video-fusion-experiment-pipeline.ipynb
```

is retained as the original research notebook and historical implementation reference.

---

# Reproducing the experiments

## 12. Requirements

A CUDA-capable GPU is strongly recommended.

The full experiment trains multiple models for 22 subjects and then repeats the process across four cross-validation folds, so running the complete pipeline can be computationally expensive.

Python dependencies are listed in:

```text
requirements.txt
```

Main dependencies include:

```text
numpy
pandas
scipy
torch
torchvision
scikit-learn
matplotlib
Pillow
opencv-python
tqdm
gdown
```

The pretrained ResNet18 backbone also requires internet access the first time its torchvision weights are downloaded.

---

## 13. Clone the repository

```bash
git clone https://github.com/natsumeSama/DEAP-EEG-video-fusion-experiment-pipeline.git
cd DEAP-EEG-video-fusion-experiment-pipeline
```

---

## 14. Create a Python environment

Using `venv`:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The notebook also checks for missing dependencies and can install them automatically.

---

## 15. Obtain the data

The dataset itself is **not intended to be stored directly in this Git repository**.

Users should obtain the DEAP data through an authorized source and respect the applicable dataset access and redistribution terms.

The normal training workflow requires:

1. DEAP preprocessed EEG files
2. Precomputed facial crops for the aligned video trials

Raw videos are only required if the face crops need to be regenerated.

---

## 16. Option A — use local data

In:

```text
notebooks/colab_main.ipynb
```

set:

```python
DATA_SOURCE = "local"
```

The expected folder structure is:

```text
data/
│
├── eeg/
│   └── data_preprocessed_python/
│       ├── s01.dat
│       ├── s02.dat
│       ├── ...
│       └── s32.dat
│
├── face_crops/
│   ├── s01_trial01/
│   │   ├── ...
│   │   └── image files
│   ├── s01_trial02/
│   ├── ...
│   └── s22_trial40/
│
└── videos/
    └── ...                  # optional
```

Only subjects/trials available in both EEG and face-crop data are used by the multimodal experiment.

Face-crop folders must follow the naming convention:

```text
sXX_trialYY
```

For example:

```text
s01_trial01
s01_trial02
s02_trial01
```

Supported image extensions include:

```text
.jpg
.jpeg
.png
```

---

## 17. Option B — download ZIP archives from Google Drive

The notebook can also download dataset archives using `gdown`.

Set:

```python
DATA_SOURCE = "drive_download"
```

and provide Google Drive **file IDs**:

```python
DEAP_EEG_DRIVE_ID = ""
DEAP_FACE_CROPS_DRIVE_ID = ""
DEAP_VIDEO_DRIVE_ID = ""
```

For normal training, only these two are required:

```python
DEAP_EEG_DRIVE_ID
DEAP_FACE_CROPS_DRIVE_ID
```

Raw videos remain optional:

```python
DOWNLOAD_RAW_VIDEOS = False
```

If raw videos are required:

```python
DOWNLOAD_RAW_VIDEOS = True
DEAP_VIDEO_DRIVE_ID = "..."
```

The downloaded archives must extract into the expected structure.

For EEG:

```text
data/eeg/data_preprocessed_python/*.dat
```

For face crops:

```text
data/face_crops/sXX_trialYY/*.jpg
```

---

## 18. Google Colab

The main notebook can be opened directly in Google Colab:

[Open `colab_main.ipynb` in Colab](https://colab.research.google.com/github/natsumeSama/DEAP-EEG-video-fusion-experiment-pipeline/blob/main/notebooks/colab_main.ipynb)

On a fresh Colab runtime, the notebook:

1. detects the Colab environment,
2. clones this repository,
3. adds `src/` to the Python path,
4. checks dependencies,
5. prepares the selected dataset source.

A GPU runtime is recommended:

```text
Runtime -> Change runtime type -> GPU
```

---

## 19. Configure the experiment

The main experiment defaults are defined in:

```text
src/deap_fusion/config.py
```

Current defaults:

```python
LABEL_TYPE = 1
MAX_SUBJECT_ID = 22
SEED = 42

NUM_FRAMES = 18
FREEZE_BACKBONE = True

BATCH_SIZE = 4
WEIGHT_DECAY = 1e-5
EPOCHS = 20
LR = 1e-3
TRAIN_RATIO = 0.75
NUM_WORKERS = 4
```

To reproduce **valence**:

```python
LABEL_TYPE = 0
```

To reproduce **arousal**:

```python
LABEL_TYPE = 1
```

When changing the target, use a fresh kernel/runtime and rerun the notebook from the beginning so that shared splits and cached experiment state are recreated correctly.

---

## 20. Run the notebook

Execute:

```text
notebooks/colab_main.ipynb
```

from top to bottom.

The notebook is organized into ten main sections:

```text
1. Project Setup & Dependencies
2. Imports
3. User Configuration
4. Dataset Setup & Validation
5. Reproducibility & Shared Split Check
6. Main Experiment - Single 75/25 Shared Split
7. Main Experiment - Analysis & Results
8. 4-Fold Cross-Validation
9. CV Analysis & Visualization
10. Save CV Results
```

### Dataset validation

Before any training starts, the notebook verifies that:

- the EEG directory exists,
- `.dat` files are available,
- face-crop directories exist,
- aligned trial folders can be detected.

If this cell succeeds, the notebook prints:

```text
Dataset setup OK.
```

---

## 21. Main experiment

The main experiment runs models in the following order:

```text
EEG
 ↓
Video
 ↓
FiLM Fusion
 ↓
Concat Fusion
 ↓
Decision Fusion
```

EEG and video are trained first because the fusion models reuse their pretrained checkpoints.

Decision fusion is evaluated last because it directly combines the already-trained unimodal predictions and requires no additional training.

---

## 22. Four-fold cross-validation

After the main 75/25 experiment, the notebook performs subject-dependent four-fold cross-validation.

For each fold:

```text
activate shared fold
      │
      ▼
EEG
      │
      ▼
Video
      │
      ▼
FiLM
      │
      ▼
Concat
      │
      ▼
Decision
```

The active split is reused by every model.

---

## 23. Outputs

### Metrics

Main-experiment tables are written to:

```text
outputs/metrics/main/
```

Cross-validation tables are written to:

```text
outputs/metrics/cv/
```

Generated CSV files include:

- subject-level metrics,
- aggregate summaries,
- paper-ready comparison tables,
- model-comparison tables,
- FiLM modulation statistics.

### Checkpoints

Model checkpoints are isolated by experiment/run.

The main experiment uses directories such as:

```text
models/main_label1_seed42/
```

Cross-validation uses:

```text
models/cv_fold_0/
models/cv_fold_1/
models/cv_fold_2/
models/cv_fold_3/
```

Model checkpoints and generated data are intentionally ignored by Git.

### EEG preprocessing cache

Expensive EEG preprocessing modes are cached on disk.

When the same preprocessing configuration is requested again, the processed EEG representation can be loaded from the cache instead of recomputing every spectral feature.

Cache names include preprocessing settings such as:

- mode,
- label type,
- band-pass range,
- notch configuration,
- window size,
- stride,
- asymmetry setting.

---

# Reference results

## 24. Main subject-dependent split

Best validation accuracy reported in the thesis experiments:

| Model | Valence | Arousal |
|---|---:|---:|
| EEG | 75.91% | 70.00% |
| Video | 85.00% | 74.55% |
| Decision Fusion | 78.64% | 70.00% |
| Concat Fusion | **88.64%** | 75.91% |
| FiLM Fusion | 87.27% | **77.73%** |

The main split suggests different behavior between the two emotion dimensions.

Video is already strong for valence, whereas EEG-guided FiLM produces a larger relative benefit for arousal.

---

## 25. Four-fold cross-validation

Mean best validation accuracy across four folds:

| Model | Valence | Arousal |
|---|---:|---:|
| EEG | 75.20 ± 1.60% | 73.72 ± 0.61% |
| Video | 80.39 ± 4.32% | 72.44 ± 1.98% |
| Decision Fusion | 75.77 ± 2.54% | 69.33 ± 1.82% |
| Concat Fusion | **84.66 ± 2.92%** | **77.82 ± 1.71%** |
| FiLM Fusion | 83.86 ± 3.34% | 77.37 ± 1.84% |

FiLM remains close to concatenation:

```text
Valence difference: 0.80 percentage points
Arousal difference: 0.45 percentage points
```

The purpose of FiLM is therefore not only predictive performance. It provides a structured cross-modal mechanism where EEG explicitly controls how visual features are modified.

---

## 26. Important interpretation of the results

These experiments use a **subject-dependent evaluation protocol**.

Training and validation trials originate from the same subjects.

The reported numbers therefore measure performance under the experimental setup used to compare fusion strategies and **must not be interpreted as evidence of generalization to unseen subjects**.

Subject-independent evaluation, such as LOSO, remains an important direction for future work.

Other limitations include:

- only 22 multimodal subjects,
- 874 aligned EEG-video trials,
- video representation based on 18 sampled frames and mean pooling,
- unidirectional EEG-to-video FiLM conditioning,
- limited comparison with heavier cross-modal architectures.

---

## 27. Reproducibility notes

The repository fixes the main Python, NumPy and PyTorch random seeds.

However, exact bit-for-bit reproduction can still depend on:

- GPU hardware,
- CUDA/cuDNN versions,
- PyTorch/torchvision versions,
- pretrained ResNet18 weights,
- parallel DataLoader execution.

The expected objective is reproduction of the experimental protocol and comparable results rather than guaranteed bit-identical training trajectories across every machine.

---

## 28. Thesis

A complete explanation of the motivation, related work, methodology, experiments, interpretation, and limitations is available in the associated Master's thesis.

If included in this repository:

```text
docs/thesis.pdf
```

> Y. R. Aissani,  
> *Multimodal Emotion Recognition Using EEG and Facial/Video Information with EEG-Guided FiLM Fusion*,  
> Master's Thesis, University of Cagliari, 2026.

---

## 29. Citation

If you use this work, please cite the thesis:

```bibtex
@mastersthesis{aissani2026multimodal,
  author = {Aissani, Yudas Rafik},
  title = {Multimodal Emotion Recognition Using EEG and Facial/Video Information with EEG-Guided FiLM Fusion},
  school = {University of Cagliari},
  year = {2026}
}
```

---

## 30. Data and privacy

The repository should contain source code only and does not require committing raw DEAP data, participant videos, extracted facial images, or generated model checkpoints.

Users are responsible for obtaining and using DEAP data according to the applicable access and redistribution conditions.

In particular, avoid publishing private Google Drive dataset links or derived participant imagery unless redistribution is explicitly permitted.

---

## 31. Future work

Possible extensions include:

- subject-independent / LOSO evaluation,
- deeper analysis of FiLM gamma, beta and gate values,
- bidirectional FiLM conditioning,
- cross-modal attention baselines,
- temporal video modeling instead of simple mean pooling,
- reduced-channel EEG configurations,
- larger multimodal datasets,
- additional interpretability analyses.

---

## Acknowledgment

This repository contains the implementation developed as part of the Master's thesis in Computer Engineering, Cybersecurity and Artificial Intelligence at the University of Cagliari.