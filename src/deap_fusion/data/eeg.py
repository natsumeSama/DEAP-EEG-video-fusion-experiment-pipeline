"""
Feature extraction utilities for DEAP EEG (preprocessed signals).

This module implements a compact, reusable preprocessing + feature pipeline that:
1) (Optionally) re-references EEG with CAR (common average reference),
2) (Optionally) applies band-pass + notch filtering,
3) splits each 63s DEAP trial into baseline (first 3s) and stimulus (next 60s),
4) extracts DE-like features using log-bandpower (Welch PSD) in standard EEG bands,
5) (Optionally) appends hemispheric asymmetry features (DASM/RASM),
6) (Optionally) baseline-corrects trial features using baseline DE,
7) (Optionally) z-scores features across windows.

All functions are written to work on arrays shaped (C, T):
- C = number of EEG channels
- T = number of time samples
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pickle

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

from .cache import (
    _mode_cache_dir,
    _cache_exists,
    _save_mode_cache,
    _load_mode_cache,
)
# =========================
# Constants & Configuration
# =========================

# DEAP EEG sampling rate (Hz)
FS: int = 128

# Canonical EEG bands (DEAP "preprocessed" EEG often focuses on 4–45 Hz)
# These are used for bandpower/DE features.
BANDS: Dict[str, Tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Left–Right symmetric electrode pairs for asymmetry features.
# Indices are 0-based and MUST match the channel order in your EEG arrays.
# If your channel ordering differs, update this list accordingly.
LR_PAIRS: List[Tuple[int, int]] = [
    (0, 1),    # Fp1, Fp2
    (2, 3),    # AF3, AF4
    (4, 5),    # F3, F4
    (6, 7),    # F7, F8
    (8, 9),    # FC5, FC6
    (10, 11),  # FC1, FC2
    (12, 13),  # C3, C4
    (14, 15),  # T7, T8
    (16, 17),  # CP5, CP6
    (18, 19),  # CP1, CP2
    (20, 21),  # P3, P4
    (22, 23),  # P7, P8
    (24, 25),  # PO3, PO4
    (26, 27),  # O1, O2
]


# ==============
# Core Utilities
# ==============

def butter_bandpass_notch(
    x: np.ndarray,
    fs: int = FS,
    bp: Tuple[float, float] = (4.0, 45.0),
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
) -> np.ndarray:
    """
    Apply a band-pass filter + notch filter to multi-channel EEG.

    This function is designed for signals shaped (C, T) and filters along time (axis=1).

    Args:
        x:
            EEG signal of shape (C, T), where C=channels and T=samples.
        fs:
            Sampling rate in Hz.
        bp:
            (low, high) cutoffs for band-pass filter in Hz.
        notch_freq:
            Notch frequency (commonly 50 Hz in Europe, 60 Hz in US).
        notch_q:
            Notch quality factor (higher => narrower notch).

    Returns:
        Filtered EEG signal with shape (C, T), dtype float32.

    Notes:
        - Uses zero-phase filtering (filtfilt) to avoid phase distortion.
        - 4th-order Butterworth band-pass is a common simple choice for EEG cleaning.
    """
    # Design band-pass (Butterworth) in normalized frequency (Nyquist = fs/2)
    b, a = butter(4, [bp[0] / (fs / 2), bp[1] / (fs / 2)], btype="bandpass")
    y = filtfilt(b, a, x, axis=1)

    # Design and apply a notch filter to suppress powerline noise
    w0 = notch_freq / (fs / 2)
    b2, a2 = iirnotch(w0=w0, Q=notch_q)
    y = filtfilt(b2, a2, y, axis=1)

    return y.astype(np.float32)


def re_reference_car(x: np.ndarray) -> np.ndarray:
    """
    Common Average Reference (CAR).

    CAR subtracts, at each timepoint, the mean over channels:
        x_car[c, t] = x[c, t] - mean_c(x[c, t])

    Args:
        x:
            EEG signal of shape (C, T).

    Returns:
        CAR-referenced EEG of shape (C, T), dtype float32.

    Why it helps:
        CAR can reduce common-mode noise shared across electrodes and sometimes
        stabilizes downstream feature extraction.
    """
    # Subtract the across-channel mean for each time sample
    return (x - x.mean(axis=0, keepdims=True)).astype(np.float32)


def split_baseline_trial(
    x: np.ndarray,
    fs: int = FS,
    baseline_sec: float = 3.0,
    trial_sec: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a DEAP 63-second trial into baseline and stimulus segments.

    DEAP trials are typically:
    - first 3 seconds: baseline (rest)
    - next 60 seconds: stimulus (watching the video / emotion elicitation)

    Args:
        x:
            EEG for a single trial, shape (C, T) where T ≈ 63 * fs.
        fs:
            Sampling rate (Hz).
        baseline_sec:
            Baseline duration in seconds (default 3s).
        trial_sec:
            Stimulus duration in seconds (default 60s).

    Returns:
        x_base:
            Baseline segment, shape (C, baseline_sec * fs).
        x_trial:
            Stimulus segment, shape (C, trial_sec * fs).

    Raises:
        ValueError:
            If the provided signal does not contain enough samples.
    """
    b = int(round(baseline_sec * fs))
    t = int(round(trial_sec * fs))

    # DEAP expects baseline + trial to fit inside the total time series
    if x.shape[1] < (b + t):
        raise ValueError(
            f"Signal too short for expected DEAP trial length: got {x.shape[1]} samples."
        )

    x_base = x[:, :b]
    x_trial = x[:, b : b + t]
    return x_base.astype(np.float32), x_trial.astype(np.float32)


def window_indices(
    T: int,
    fs: int = FS,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Generate sliding-window index ranges [start, end) over a signal length T.

    Args:
        T:
            Total number of samples in the signal.
        fs:
            Sampling rate (Hz).
        win_sec:
            Window length in seconds.
        step_sec:
            Step/stride length in seconds.

    Returns:
        A list of (start, end) sample indices, each representing one window.

    Notes:
        - Uses integer rounding of seconds -> samples.
        - If T is shorter than one window, returns a single clipped window.
    """
    w = max(1, int(round(win_sec * fs)))
    s = max(1, int(round(step_sec * fs)))

    idx: List[Tuple[int, int]] = []
    start = 0
    while start + w <= T:
        idx.append((start, start + w))
        start += s

    # Fallback for very short signals
    if not idx:
        idx = [(0, min(w, T))]

    return idx


def _de_log_bandpower_window(
    x_win: np.ndarray,
    fs: int,
    band: Tuple[float, float],
) -> np.ndarray:
    """
    Compute a DE-like feature for one window via log-bandpower.

    Many DEAP pipelines use "Differential Entropy (DE)" features. A practical proxy
    commonly used is log of bandpower in canonical EEG bands:
        feature = log(mean PSD over band)

    Args:
        x_win:
            EEG window of shape (C, W).
        fs:
            Sampling rate (Hz).
        band:
            (f_low, f_high) frequency band in Hz.

    Returns:
        Per-channel log-bandpower feature, shape (C,), dtype float32.

    Implementation details:
        - Welch PSD estimates power spectral density per channel.
        - Then average PSD inside the target band and apply log().
    """
    # Welch PSD per channel: returns frequencies f and PSD pxx shaped (C, F)
    f, pxx = welch(
        x_win,
        fs=fs,
        nperseg=min(256, x_win.shape[1]),
        axis=1
    )

    f_low, f_high = band
    mask = (f >= f_low) & (f < f_high)

    # Mean band power; epsilon avoids log(0) for numerical safety
    bp = pxx[:, mask].mean(axis=1) + 1e-12
    return np.log(bp).astype(np.float32)


def compute_de_features(
    x: np.ndarray,
    fs: int = FS,
    bands: Dict[str, Tuple[float, float]] = BANDS,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
    include_asym: bool = True,
    lr_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute DE/log-bandpower features over sliding windows, with optional asymmetry.

    Output shape convention:
        features: (F, Nwin)
        - rows are feature dimensions
        - columns are windows over time

    Base (per-band per-channel) features:
        For each window and each band, compute (C,) log-bandpower.
        These are stacked across bands to form (C * nbands,) per window.

    Optional asymmetry features:
        For each band and each left/right pair (L, R):
        - DASM: L - R  (difference)
        - RASM: L / (R + eps) (ratio)
        These are appended after base features.

    Args:
        x:
            EEG of shape (C, T) for the segment you want to featurize.
        fs:
            Sampling rate (Hz).
        bands:
            Dict mapping band_name -> (f_low, f_high).
        win_sec:
            Window length in seconds.
        step_sec:
            Step size in seconds.
        include_asym:
            If True, append DASM and RASM asymmetry features.
        lr_pairs:
            Optional left-right electrode pairs override (0-based indices).

    Returns:
        features:
            Array of shape (F, Nwin), dtype float32.
        info:
            Metadata dict (band names, window indices, and intermediate shapes).
    """
    C, T = x.shape

    # Compute which sample ranges define each temporal window
    idx = window_indices(T, fs, win_sec, step_sec)

    band_list = list(bands.values())
    nbands = len(band_list)

    # Base features: (C * nbands, Nwin)
    base = np.zeros((C * nbands, len(idx)), dtype=np.float32)

    # For each window, compute log-bandpower per band then stack them
    for j, (a, b) in enumerate(idx):
        xw = x[:, a:b]  # (C, W)

        cols = []
        for band in band_list:
            # Each call returns (C,) for this band in this window
            cols.append(_de_log_bandpower_window(xw, fs, band))

        # Stack bands: [band0(C), band1(C), ...] => (C*nbands,)
        base[:, j] = np.concatenate(cols, axis=0)

    # If asymmetry is disabled, we are done
    if not include_asym:
        return base, {
            "bands": list(bands.keys()),
            "windows": idx,
            "base_shape": base.shape
        }

    # Use default LR pairs unless an override is provided
    pairs = LR_PAIRS if lr_pairs is None else lr_pairs
    n_pairs = len(pairs)

    # Asymmetry features: DASM + RASM for each band and each LR pair
    # Shape: (n_pairs * nbands * 2, Nwin)
    asym = np.zeros((n_pairs * nbands * 2, len(idx)), dtype=np.float32)

    for j in range(base.shape[1]):
        # base[:, j] is (C*nbands,), split back into per-band vectors of length C
        band_blocks = np.split(base[:, j], nbands)

        block_cols = []
        for band_vec in band_blocks:
            dasm = []
            rasm = []
            for (l, r) in pairs:
                L = band_vec[l]
                R = band_vec[r]
                dasm.append(L - R)
                rasm.append(L / (R + 1e-6))  # eps avoids divide-by-zero
            block_cols.append(np.array(dasm, dtype=np.float32))
            block_cols.append(np.array(rasm, dtype=np.float32))

        # Concatenate per-band [DASM, RASM] blocks into one asym vector
        asym[:, j] = np.concatenate(block_cols, axis=0)

    # Final feature matrix: base features + asymmetry features
    feats = np.concatenate([base, asym], axis=0)

    info = {
        "bands": list(bands.keys()),
        "windows": idx,
        "base_shape": base.shape,
        "asym_shape": asym.shape,
        "feat_shape": feats.shape,
        "include_asym": include_asym,
        "n_lr_pairs": n_pairs,
    }
    return feats, info

# ===================
# Normalization Tools
# ===================

def baseline_de(
    x_base: np.ndarray,
    fs: int = FS,
    win_sec: float = 1.0,
) -> np.ndarray:
    """
    Compute baseline DE/log-bandpower features from the baseline segment only.

    This uses non-overlapping windows (step_sec == win_sec) so the baseline is
    summarized cleanly.

    Args:
        x_base:
            Baseline EEG segment of shape (C, T_base), e.g. 3 seconds in DEAP.
        fs:
            Sampling rate (Hz).
        win_sec:
            Window length in seconds for baseline windows.

    Returns:
        baseline_vec:
            Mean baseline feature vector of shape (C * nbands,).
            (No asymmetry is computed here by design.)
    """
    F_base, _info = compute_de_features(
        x_base,
        fs=fs,
        win_sec=win_sec,
        step_sec=win_sec,
        include_asym=False
    )

    # Average baseline features across baseline windows -> a single vector
    return F_base.mean(axis=1)  # (C*nbands,)


def baseline_correct_features(
    feat: np.ndarray,
    baseline_vec: np.ndarray,
    include_asym: bool = True,
    nbands: int = 4,
) -> np.ndarray:
    """
    Apply baseline correction by subtracting baseline feature means.

    This function assumes the first `base_len` rows of `feat` correspond to the
    base band features (C * nbands). Any appended asymmetry features (if present)
    are left unchanged because they are already relative quantities.

    Args:
        feat:
            Trial feature matrix of shape (F, Nwin).
        baseline_vec:
            Baseline feature vector of shape (C * nbands,).
        include_asym:
            Present for readability/consistency; asym features are not corrected here.
        nbands:
            Number of frequency bands used for base features.

    Returns:
        corrected:
            Baseline-corrected feature matrix, same shape as `feat`.
    """
    corrected = feat.copy()
    base_len = baseline_vec.shape[0]

    # Subtract baseline per feature dimension for the base band features only
    corrected[:base_len, :] = corrected[:base_len, :] - baseline_vec[:, None]

    # Asymmetry features remain unchanged
    return corrected


def zscore_features(F: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize a feature matrix.

    Default behavior (axis=1):
        Normalize each feature dimension across windows:
            for each row d: (F[d, :] - mean) / std

    Args:
        F:
            Feature matrix of shape (D, N).
        axis:
            Axis along which to compute mean/std.
            - axis=1: per feature across windows (common for time-window features)
            - axis=0: per window across features (less common)
        eps:
            Numerical stability term to avoid divide-by-zero.

    Returns:
        Z-scored feature matrix with the same shape as F.
    """
    mu = F.mean(axis=axis, keepdims=True)
    sd = F.std(axis=axis, keepdims=True) + eps
    return (F - mu) / sd

# ===========================
# High-level Pipeline Wrapper
# ===========================

def build_deap_features_pipeline(
    x_trial_63s: np.ndarray,
    fs: int = FS,
    use_car: bool = True,
    filter_first: bool = True,
    bp: Tuple[float, float] = (4.0, 45.0),
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
    include_asym: bool = True,
    do_baseline: bool = True,
    baseline_win_sec: float = 1.0,
    do_zscore: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    End-to-end DEAP-style preprocessing pipeline returning windowed DE features.

    High-level steps:
        1) cast to float32
        2) optional CAR re-referencing
        3) optional band-pass + notch filtering
        4) split 63s trial into baseline (3s) and stimulus (60s)
        5) extract DE/log-bandpower features on the 60s stimulus
        6) optional baseline correction using baseline DE
        7) optional z-score normalization across windows

    Args:
        x_trial_63s:
            One full DEAP trial shaped (C, T=63*fs).
        fs:
            Sampling rate (Hz).
        use_car:
            If True, apply common average reference.
        filter_first:
            If True, apply band-pass + notch filtering.
        bp, notch_freq, notch_q:
            Filter parameters.
        win_sec, step_sec:
            Sliding window length and step (seconds) for stimulus features.
        include_asym:
            If True, append DASM/RASM features.
        do_baseline:
            If True, compute baseline DE (from 3s) and subtract from base features.
        baseline_win_sec:
            Window length for baseline DE (non-overlapping).
        do_zscore:
            If True, z-score features per feature dimension across windows.

    Returns:
        feats:
            Feature matrix of shape (F, Nwin), float32.
        meta:
            Metadata dictionary describing the pipeline configuration and shapes.
    """
    # Ensure consistent dtype early (helps performance + stability)
    x = x_trial_63s.astype(np.float32)

    # Optional spatial re-reference to reduce common-mode noise
    if use_car:
        x = re_reference_car(x)

    # Optional frequency-domain cleaning (band-pass + notch)
    if filter_first:
        x = butter_bandpass_notch(
            x,
            fs=fs,
            bp=bp,
            notch_freq=notch_freq,
            notch_q=notch_q
        )

    # Separate baseline and stimulus parts according to DEAP trial timing
    x_base, x_stim = split_baseline_trial(
        x,
        fs=fs,
        baseline_sec=3.0,
        trial_sec=60.0
    )

    # Extract features over the stimulus part using sliding windows
    feats, info = compute_de_features(
        x_stim,
        fs=fs,
        win_sec=win_sec,
        step_sec=step_sec,
        include_asym=include_asym
    )

    # Baseline correction only affects the base band features (C*nbands)
    if do_baseline:
        base_vec = baseline_de(
            x_base,
            fs=fs,
            win_sec=baseline_win_sec
        )
        feats = baseline_correct_features(
            feats,
            baseline_vec=base_vec,
            include_asym=include_asym,
            nbands=len(BANDS),
        )

    # Z-score across windows to stabilize learning / reduce scale differences
    if do_zscore:
        feats = zscore_features(feats, axis=1, eps=1e-8)

    # Collect useful metadata for debugging and reproducibility
    meta = {
        "fs": fs,
        "bp": bp,
        "notch_freq": notch_freq,
        "notch_q": notch_q,
        "win_sec": win_sec,
        "step_sec": step_sec,
        "include_asym": include_asym,
        "do_baseline": do_baseline,
        "baseline_win_sec": baseline_win_sec,
        "do_zscore": do_zscore,
        "feat_shape": feats.shape,
        "bands": list(BANDS.keys()),
        "lr_pairs": LR_PAIRS,
        "windows": info.get("windows"),
        "base_shape": info.get("base_shape"),
        "asym_shape": info.get("asym_shape") if include_asym else None,
    }
    return feats.astype(np.float32), meta

# ================
# Quick Smoke Test
# ================

def _smoke_test() -> None:
    """
    Minimal sanity test for the feature pipeline using synthetic EEG.

    What it checks:
    - Output is 2D (F, Nwin)
    - No NaNs or infs appear
    - Prints feature shape and a small subset of meta settings

    This is useful when refactoring or moving code between files/notebooks.
    """
    # Use 32 channels (common EEG setups) and one DEAP-length trial (63s)
    C = 32
    T = 63 * FS

    # Random synthetic EEG-like signal (scaled to a microvolt-ish magnitude)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((C, T)).astype(np.float32) * 5e-6

    # Run the full pipeline with typical settings
    feats, meta = build_deap_features_pipeline(
        x,
        fs=FS,
        use_car=True,
        filter_first=True,
        win_sec=2.0,
        step_sec=1.0,
        include_asym=True,
        do_baseline=True,
        baseline_win_sec=1.0,
        do_zscore=True,
    )

    # Basic assertions for quick failure if something is wrong
    assert feats.ndim == 2
    assert np.isfinite(feats).all(), "NaNs or infs in features"

    print("Smoke test passed. Feature shape:", feats.shape)
    print("Meta:", {k: meta[k] for k in ("feat_shape", "win_sec", "step_sec", "include_asym")})

# =====================
# labels / DEAP loading
# =====================

def make_labels(
    Y_raw: np.ndarray,
    thr: float = 5,
    l: int = 0,
) -> np.ndarray:
    """
    Convert continuous DEAP ratings into binary labels.

    Args:
        Y_raw:
            Raw DEAP labels with shape (40, 4).
        thr:
            Threshold for the binary classes.
        l:
            0 = valence
            1 = arousal

    Returns:
        Binary labels with shape (40,).
    """
    if l == 0:
        y = (Y_raw[:, 0] >= thr).astype(np.int32)
    else:
        y = (Y_raw[:, 1] >= thr).astype(np.int32)

    return y

def get_data(
    root: Path | str,
    mode: int = 0,
    l: int = 0,
    filt_bp: Tuple[float, float] = (4.0, 45.0),
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
    include_asym: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Load DEAP subject files and return data according to the selected mode.

    Mode summary
    ------------
    mode 0:
        Raw DEAP preprocessed EEG
        X -> (40, 32, 8064)
        y -> (40,)

    mode 1:
        CAR + band-pass + notch filtered EEG
        X -> (40, 32, 8064)
        y -> (40,)

    mode 2:
        DE / log-bandpower features kept as one sequence per trial
        X -> (40, Nwin, F)
        y -> (40,)

    mode 3:
        Same DE features as mode 2, but all windows are flattened
        X -> (total_windows, F)
        y -> (total_windows,)
        win_index -> [(trial_idx, window_idx), ...]

    Notes:
        - Modes 1, 2, and 3 use caching because preprocessing is more expensive.
        - Only the first 32 EEG channels are kept from the DEAP files.

    Args:
        root: Root directory containing `data_preprocessed_python/`.
        mode: Data preparation mode (0, 1, 2, or 3).
        l: Label type (0=valence, 1=arousal).
        filt_bp: Band-pass filter range.
        notch_freq: Notch filter frequency.
        notch_q: Notch filter quality factor.
        win_sec: Window length for feature extraction.
        step_sec: Step between windows.
        include_asym: Whether asymmetry features are included.

    Returns:
        Dictionary keyed by subject id (for example "s01"). Each subject contains:
        - "X": mode-dependent data
        - "y": labels
        - "Y_raw": raw DEAP ratings
        - optional fields such as "win_index", "meta_example", and "X_trials_raw"
    """
    root = Path(root)

    # For expensive modes, first try to reuse a previously saved cache.
    if mode in (1, 2, 3):
        _cachedir = _mode_cache_dir(
            root=root,
            mode=mode,
            l=l,
            filt_bp=filt_bp,
            notch_freq=notch_freq,
            notch_q=notch_q,
            win_sec=win_sec,
            step_sec=step_sec,
            include_asym=include_asym,
        )
        if _cache_exists(_cachedir):
            return _load_mode_cache(_cachedir)

    # Folder containing DEAP .dat files.
    datadir = root / "data_preprocessed_python"
    if not datadir.exists():
        raise FileNotFoundError(f"DEAP path not found: {datadir}")

    files = sorted(datadir.glob("*.dat"))
    if not files:
        raise RuntimeError(f"No .dat files under {datadir}")

    subjects: Dict[str, Dict[str, Any]] = {}

    # Process each subject separately.
    for f in files:
        sid = f.stem.lower()  # Example: "s01"

        with open(f, "rb") as fh:
            dat = pickle.load(fh, encoding="latin1")

        # DEAP contains 40 channels, but here only the 32 EEG channels are kept.
        X_trials_raw = dat["data"][:, :32, :].astype(np.float32)  # (40, 32, 8064)
        Y_raw = dat["labels"].astype(np.float32)                  # (40, 4)
        y_trials = make_labels(Y_raw=Y_raw, l=l)                  # (40,)

        # Expected DEAP structure checks.
        assert X_trials_raw.shape == (40, 32, 8064)
        assert Y_raw.shape == (40, 4)
        assert y_trials.shape == (40,)

        # These variables will store the final output for this subject.
        X_out: Any = None
        y_out: Any = None
        extra: Dict[str, Any] = {}

        # ----------------
        # Mode 0: raw EEG
        # ----------------
        if mode == 0:
            X_out = X_trials_raw
            y_out = y_trials

        # -------------------------------------------------
        # Mode 1: filtered EEG (CAR + band-pass + notch)
        # -------------------------------------------------
        elif mode == 1:
            X_filt = np.empty_like(X_trials_raw)

            for t in range(40):
                # Apply re-referencing and filtering trial by trial.
                x = re_reference_car(X_trials_raw[t])
                x = butter_bandpass_notch(
                    x, fs=FS, bp=filt_bp, notch_freq=notch_freq, notch_q=notch_q
                )
                X_filt[t] = x

            X_out = X_filt
            y_out = y_trials

        # ---------------------------------------------------------
        # Mode 2/3: DE features (sequence per trial or flattened)
        # ---------------------------------------------------------
        elif mode in (2, 3):
            # One sequence per trial, each of shape (Nwin, F).
            seq_list: List[np.ndarray] = []

            # Used only for the flattened version in mode 3.
            Xw_list: List[np.ndarray] = []
            yw_list: List[int] = []

            # Keeps the mapping from flattened row index back to trial/window.
            win_index: List[Tuple[int, int]] = []

            # Representative metadata example to keep preprocessing settings.
            meta_example: Dict[str, Any] = {}

            for t in range(40):
                # Full preprocessing pipeline on one trial.
                feats, meta = build_deap_features_pipeline(
                    X_trials_raw[t],
                    fs=FS,
                    use_car=True,
                    filter_first=True,
                    bp=filt_bp,
                    notch_freq=notch_freq,
                    notch_q=notch_q,
                    win_sec=win_sec,
                    step_sec=step_sec,
                    include_asym=include_asym,
                    do_baseline=True,
                    baseline_win_sec=1.0,
                    do_zscore=True,
                )   # feats: (F, Nwin)

                # Convert features to sequence format expected by sequence models.
                seq = feats.T.astype(np.float32)  # (Nwin, F)

                # Keep the sequence trial by trial.
                seq_list.append(seq)

                # Also collect all windows for mode 3.
                Xw_list.append(seq)

                # Repeat the trial label once per window.
                yw_list.extend([int(y_trials[t])] * seq.shape[0])

                # Save the link between flattened window index and original trial/window.
                win_index.extend([(t, w) for w in range(seq.shape[0])])

                # Save any representative metadata.
                meta_example = meta

            # Mode 2 keeps one sequence per trial.
            if mode == 2:
                X_out = np.stack(seq_list, axis=0)   # (40, Nwin, F)
                y_out = y_trials                     # (40,)

            # Mode 3 flattens all windows into a single matrix.
            else:
                X_out = np.vstack(Xw_list).astype(np.float32)  # (40*Nwin, F)
                y_out = np.array(yw_list, dtype=np.int64)      # (40*Nwin,)
                extra["win_index"] = win_index

            extra["meta_example"] = meta_example

        else:
            raise ValueError("mode must be one of {0,1,2,3}")

        # Save this subject's data.
        subjects[sid] = {
            "Y_raw": Y_raw,
            "X": X_out,
            "y": y_out,
            "mode": mode,
            **extra,
            # Raw trials are kept for modes where they may still be useful later.
            "X_trials_raw": X_trials_raw if mode != 0 else None,
        }

    # Save processed data to cache for expensive modes.
    if mode in (1, 2, 3):
        _cachedir = _mode_cache_dir(
            root=root,
            mode=mode,
            l=l,
            filt_bp=filt_bp,
            notch_freq=notch_freq,
            notch_q=notch_q,
            win_sec=win_sec,
            step_sec=step_sec,
            include_asym=include_asym,
        )
        _save_mode_cache(_cachedir, subjects)

    return subjects