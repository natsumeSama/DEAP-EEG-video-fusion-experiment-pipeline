"""
Utilities for downloading and quickly inspecting the DEAP EEG dataset.

What this module is for
-----------------------
This file keeps all "data access" helpers in one place so the rest of your notebook
can focus on modeling/training.

It provides:
- `data_download`: Download + extract the DEAP dataset zip from Google Drive (once).
- `data_info`: Quick sanity-check on one participant file (shapes, label stats, plots).
"""

from pathlib import Path
import os
import pickle
import zipfile

import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_download(file_id: str,dir: str) -> Path:
    """
    Download and extract the DEAP dataset zip from Google Drive.

    The function is **idempotent**:
    - If the dataset folder already exists and is non-empty, it returns immediately.
    - Otherwise, it downloads the zip, extracts it, deletes the zip, and returns the folder.

    Args:
        file_id (str):
            Google Drive file ID for the dataset ZIP archive.

    Returns:
        Path:
            Path to the extracted dataset directory (e.g. `<project_root>/data/deap_dataset`).

    Notes:
        - `gdown` is used because Google Drive links often require special handling.
        - The extracted content depends on how the zip was created (subfolders, file layout, etc.).
    """
    # Resolve the project root (current working directory in Colab / local)
    project_root = Path().resolve()

    # Define where data will live in your project
    data_path = project_root / "data"
    dataset_path = data_path / dir

    # Ensure the parent folder exists (safe to call multiple times)
    data_path.mkdir(parents=True, exist_ok=True)

    # If dataset folder already exists and contains files, do not re-download
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    # Create the dataset directory if missing
    print(f"Creating dataset directory at {dataset_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download the zip inside the dataset directory
    zip_path = dataset_path / "deap_data.zip"
    print("Downloading dataset ...")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", str(zip_path), quiet=False)

    # Extract the zip contents into the dataset directory
    print("Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)

    # Cleanup: remove the zip file after extraction to save space
    zip_path.unlink()

    print(f"Dataset ready at {dataset_path}")
    return dataset_path


def data_info(path: Path) -> None:
    """
    Inspect one participant file from the DEAP dataset and visualize a few signals.

    This is mainly a **sanity-check / exploration** helper. It:
    - Finds participant `.dat` files in `path`.
    - Loads the first file and prints the shape of EEG + labels.
    - Shows descriptive stats + histograms for the 4 emotion ratings.
    - Plots a few EEG channels from the first trial.

    Args:
        path (Path):
            Directory that contains DEAP participant `.dat` files.

    Returns:
        None

    Expected DEAP structure (for each participant `.dat` file):
        - data["data"]:   shape (40, 40, 8064) -> (trials, channels, samples)
        - data["labels"]: shape (40, 4)        -> (trials, {valence, arousal, dominance, liking})

    Notes:
        - DEAP sampling rate is 128 Hz (used here to build the time axis).
        - The "μV" label is conventional for EEG amplitude; depending on preprocessing,
          raw scale might differ. This plot is still useful to confirm signals look reasonable.
    """
    # --- 1) Locate participant files (.dat) ---
    # `path` should be the folder that directly contains s01.dat, s02.dat, ...
    data_path = path
    files = [f for f in os.listdir(data_path) if f.endswith(".dat")]
    print(f"Found {len(files)} participant files in: {data_path}\n")

    # --- 2) Load one participant (first file found) ---
    # DEAP .dat files are pickled dictionaries; latin1 is commonly used for compatibility.
    file_path = data_path / files[0]
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # --- 3) Check core tensors: EEG and labels ---
    eeg = data["data"]       # (trials, channels, samples) e.g. (40, 40, 8064)
    labels = data["labels"]  # (trials, 4) ratings per trial
    print(f"EEG shape: {eeg.shape}  (trials, channels, samples)")
    print(f"Labels shape: {labels.shape}  (trials, 4 ratings)\n")

    # --- 4) Explore label distribution (basic stats + histograms) ---
    # Convert labels to a DataFrame to easily compute summary statistics.
    df = pd.DataFrame(labels, columns=["valence", "arousal", "dominance", "liking"])
    print("--- Label Statistics (first subject) ---")
    print(df.describe())

    # Visual check: are ratings roughly spread, skewed, etc. (scale is 1–9 in DEAP)
    df.hist(figsize=(8, 4), bins=8)
    plt.suptitle("Distribution of Emotion Ratings (1–9 scale)")
    plt.show()

    # --- 5) Plot a few EEG channels for one trial ---
    # Choose one trial and a small set of channels to avoid overcrowding the figure.
    trial_idx = 0
    channels_to_plot = [0, 1, 2, 3, 4, 5]  # first 6 channels

    # Build a time axis in seconds using DEAP sampling frequency (fs = 128 Hz)
    time = np.arange(eeg.shape[2]) / 128

    # Plot each selected channel in its own subplot (same trial)
    plt.figure(figsize=(12, 8))
    for i, ch in enumerate(channels_to_plot):
        plt.subplot(len(channels_to_plot), 1, i + 1)
        plt.plot(time, eeg[trial_idx, ch, :])
        plt.title(f"Trial {trial_idx + 1} – Channel {ch + 1}")
        plt.ylabel("μV")
        plt.tight_layout()
    plt.xlabel("Time (s)")
    plt.show()

    print(" Visualization complete.\n")
