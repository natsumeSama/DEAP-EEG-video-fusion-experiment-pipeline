"""
Caching utilities for preprocessed DEAP EEG data.
"""

from pathlib import Path
from typing import Dict, Tuple, Any

import json
import shutil

import numpy as np

def _mode_cache_dir(
    root: Path,
    mode: int,
    l: int,
    filt_bp: Tuple[float, float],
    notch_freq: float,
    notch_q: float,
    win_sec: float,
    step_sec: float,
    include_asym: bool,
) -> Path:
    """
    Build the cache directory path for one exact preprocessing configuration.

    The cache name is made from all important preprocessing settings so that
    changing any of them creates a different cache folder.

    This prevents accidentally reusing old cached data when parameters such as:
    - label type
    - filtering values
    - window length / step
    - asymmetry features

    are changed.

    Args:
        root: Root directory where cache folders are stored.
        mode: Data mode (0, 1, 2, or 3).
        l: Label type (for example valence or arousal).
        filt_bp: Band-pass filter range as (low, high).
        notch_freq: Notch filter frequency.
        notch_q: Notch filter quality factor.
        win_sec: Window length in seconds.
        step_sec: Window step in seconds.
        include_asym: Whether asymmetry features are included.

    Returns:
        Path to the cache directory corresponding to this configuration.
    """
    bp_low, bp_high = filt_bp

    tag = (
        f"data_mode{mode}"
        f"_label{l}"
        f"_bp{bp_low}-{bp_high}"
        f"_notch{notch_freq}_q{notch_q}"
        f"_win{win_sec}_step{step_sec}"
        f"_asym{int(include_asym)}"
    )

    return root / tag

def _cache_manifest_path(mode_dir: Path) -> Path:
    """
    Return the path of the manifest file inside a cache directory.

    The manifest is a small JSON file listing the saved subject files.
    """
    return mode_dir / "manifest.json"

def _cache_exists(mode_dir: Path) -> bool:
    """
    Check whether a cache directory looks usable.

    A cache is considered valid if:
    - the directory exists
    - manifest.json exists
    - at least one subject .npz file exists

    Args:
        mode_dir: Cache directory to check.

    Returns:
        True if the cache looks valid, otherwise False.
    """
    m = _cache_manifest_path(mode_dir)
    return mode_dir.exists() and m.exists() and any(mode_dir.glob("*.npz"))

def _save_mode_cache(mode_dir: Path, subjects: Dict[str, Dict[str, Any]]) -> None:
    """
    Save all subject data for one preprocessing mode into a cache directory.

    Storage design:
    - one compressed .npz file per subject
    - one small manifest.json listing these files

    This format is simple, easy to inspect, and convenient to reload later.

    Args:
        mode_dir: Directory where the cache will be written.
        subjects: Dictionary keyed by subject id (for example "s01").
            Each value contains the processed arrays and optional metadata.
    """
    mode_dir.mkdir(parents=True, exist_ok=True)
    files = []

    for sid, payload in subjects.items():
        # Store mandatory fields for every subject.
        arrs = {
            "mode": np.array(payload.get("mode", -1), dtype=np.int32),
            "X": np.asarray(payload["X"]),
            "y": np.asarray(payload["y"]),
        }

        # Add optional arrays only when they are available.
        if payload.get("Y_raw") is not None:
            arrs["Y_raw"] = np.asarray(payload["Y_raw"])
        if payload.get("X_trials_raw") is not None:
            arrs["X_trials_raw"] = np.asarray(payload["X_trials_raw"])
        if payload.get("win_index") is not None:
            arrs["win_index"] = np.asarray(payload["win_index"], dtype=np.int32)

        # Save example metadata as JSON bytes because .npz does not directly store dicts safely.
        if payload.get("meta_example") is not None:
            meta_json = json.dumps(payload["meta_example"], default=float)
            arrs["meta_json"] = np.frombuffer(meta_json.encode("utf-8"), dtype=np.uint8)

        # Save one compressed file per subject.
        fpath = mode_dir / f"{sid}.npz"
        np.savez_compressed(fpath, **arrs)
        files.append(fpath.name)

    # Save the list of cache files so loading is straightforward later.
    with open(_cache_manifest_path(mode_dir), "w", encoding="utf-8") as fh:
        json.dump({"files": files}, fh, indent=2)

def _load_mode_cache(mode_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all cached subject data from a cache directory.

    This restores the subject dictionary in the same general format used by
    the preprocessing pipeline.

    Args:
        mode_dir: Cache directory to read.

    Returns:
        Dictionary keyed by subject id with arrays and optional metadata.
    """
    with open(_cache_manifest_path(mode_dir), "r", encoding="utf-8") as fh:
        mani = json.load(fh)

    subjects: Dict[str, Dict[str, Any]] = {}
    for fname in mani["files"]:
        # allow_pickle=False keeps loading safer and avoids hidden Python objects.
        D = np.load(mode_dir / fname, allow_pickle=False)
        sid = Path(fname).stem

        payload: Dict[str, Any] = {
            "mode": int(D["mode"]),
            "X": D["X"],
            "y": D["y"],
            "Y_raw": D["Y_raw"] if "Y_raw" in D.files else None,
            "X_trials_raw": D["X_trials_raw"] if "X_trials_raw" in D.files else None,
        }

        # Restore optional fields if they were saved.
        if "win_index" in D.files:
            payload["win_index"] = D["win_index"]
        if "meta_json" in D.files:
            payload["meta_example"] = json.loads(bytes(D["meta_json"]).decode("utf-8"))

        subjects[sid] = payload

    return subjects

def download_preprocessed_data(
    root: Path | str,
    mode: int,
    l: int = 0,
    filt_bp: Tuple[float, float] = (4.0, 45.0),
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    win_sec: float = 2.0,
    step_sec: float = 1.0,
    include_asym: bool = True,
) -> Path:
    """
    Zip an existing cache directory and optionally trigger a Colab download.

    This is useful when preprocessing was already done and you want to save or
    export the prepared data without recomputing it.

    Args:
        root: Project root containing cache folders.
        mode: Data mode to export.
        l: Label type.
        filt_bp: Band-pass filter range.
        notch_freq: Notch filter frequency.
        notch_q: Notch filter quality factor.
        win_sec: Window length in seconds.
        step_sec: Window step in seconds.
        include_asym: Whether asymmetry features are included.

    Returns:
        Path to the generated zip file.

    Raises:
        FileNotFoundError: If no matching cache directory exists.
    """
    root = Path(root)
    mode_dir = _mode_cache_dir(
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

    if not _cache_exists(mode_dir):
        raise FileNotFoundError(f"No cache found under {mode_dir}")

    zip_path = mode_dir.with_suffix(".zip")
    shutil.make_archive(str(mode_dir), "zip", root_dir=mode_dir)

    # If this runs in Google Colab, try to trigger an automatic browser download.
    try:
        from google.colab import files  # type: ignore
        files.download(str(zip_path))
    except Exception:
        pass

    return zip_path
