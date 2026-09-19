"""
Shared trial-pair and train/validation split utilities.

The functions in this module create a common (subject, trial) space
that can be reused by EEG-only, video-only, and multimodal experiments.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import random
import re

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from ..config import (
    LABEL_TYPE,
    MAX_SUBJECT_ID,
    TRAIN_RATIO,
    SEED,
)
from .eeg import get_data

SHARED_SPLITS = {}
SHARED_CV_SPLITS = {}
CURRENT_CV_FOLD = None
CURRENT_RUN_TAG = "single_run"


def _infer_trial_labels_from_entry(entry: Dict[str, Any]) -> np.ndarray:
    """
    Recover trial-level labels from one subject entry.

    This makes the label format consistent across modes:
    - modes 0/1/2 already store one label per trial
    - mode 3 stores one label per window, so trial labels are rebuilt using win_index

    Args:
        entry: One subject dictionary returned by get_data(...).

    Returns:
        Array of trial-level labels with shape (n_trials,).
    """
    y = entry["y"]

    # In mode 3, labels are stored per window.
    if "win_index" in entry:
        win_index: List[Tuple[int, int]] = entry["win_index"]

        # Infer number of trials from the maximum trial index present.
        n_trials = max(t for (t, _) in win_index) + 1
        y_trials = np.full(n_trials, fill_value=-1, dtype=int)

        # Since all windows from the same trial should share the same label,
        # the first observed label for that trial is enough.
        for row_i, (t, _) in enumerate(win_index):
            if y_trials[t] == -1:
                y_trials[t] = int(y[row_i])

        # Fallback: if something is missing, fill it with the majority window label.
        if np.any(y_trials == -1):
            for t in range(n_trials):
                if y_trials[t] == -1:
                    ys = [int(y[i]) for i, (tt, _) in enumerate(win_index) if tt == t]
                    y_trials[t] = max(set(ys), key=ys.count)

        return y_trials

    # In modes 0/1/2, labels are already per trial.
    return np.asarray(y, dtype=int)

def _video_pairs_from_folders(image_root: str | Path) -> Set[Tuple[str, int]]:
    """
    Read available video trial folders and extract (subject, trial) pairs.

    Expected folder naming style:
        s01_trial01  -> ("s01", 0)

    Args:
        image_root: Root folder containing video trial subfolders.

    Returns:
        Set of pairs in the form (sid, trial_idx).
    """
    image_root = Path(image_root)
    pairs: Set[Tuple[str, int]] = set()

    if not image_root.exists():
        return pairs

    for folder in image_root.iterdir():
        if not folder.is_dir():
            continue

        m = re.match(r"^(s\d+)_trial(\d+)$", folder.name, flags=re.IGNORECASE)
        if m is None:
            continue

        sid = m.group(1).lower()
        trial_idx = int(m.group(2)) - 1
        pairs.add((sid, trial_idx))

    return pairs

def build_available_trial_pairs(
    subjects: Dict[str, Dict[str, Any]],
    image_root: str | Path | None = None,
    max_subject_id: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Build the list of valid (sid, trial_idx) pairs.

    Behavior:
    - If image_root is None:
        return EEG trials only
    - If image_root is provided:
        return only trials available in both EEG and video

    This shared pair list is useful when EEG, video, and fusion models must use
    the exact same trial availability.

    Args:
        subjects: Subject dictionary returned by get_data(...).
        image_root: Optional folder containing video trial folders.
        max_subject_id: If given, keep only subjects with id <= this value.

    Returns:
        Sorted list of valid (sid, trial_idx) pairs.
    """
    eeg_pairs: Set[Tuple[str, int]] = set()

    for sid, entry in subjects.items():
        sid_num = int(sid[1:])

        if max_subject_id is not None and sid_num > max_subject_id:
            continue

        y_trials = _infer_trial_labels_from_entry(entry)
        n_trials = len(y_trials)

        for t in range(n_trials):
            eeg_pairs.add((sid, t))

    if image_root is None:
        return sorted(eeg_pairs)

    video_pairs = _video_pairs_from_folders(image_root)
    common_pairs = eeg_pairs & video_pairs
    return sorted(common_pairs)

def _build_pair_label_lookup(
    subjects: Dict[str, Dict[str, Any]],
    max_subject_id: Optional[int] = None,
) -> Dict[Tuple[str, int], int]:
    """
    Build a lookup table from (subject, trial) to label.

    This is useful for creating stratified splits while keeping the trial-pair format.

    Args:
        subjects: Subject dictionary returned by get_data(...), usually from mode 0.
        max_subject_id: Optional upper limit on kept subjects.

    Returns:
        Dictionary mapping:
            (sid, trial_idx) -> binary label
    """
    pair_to_label = {}

    for sid, entry in subjects.items():
        sid_num = int(sid[1:])

        if max_subject_id is not None and sid_num > max_subject_id:
            continue

        y_trials = _infer_trial_labels_from_entry(entry)

        for t, y in enumerate(y_trials):
            pair_to_label[(sid.lower(), int(t))] = int(y)

    return pair_to_label

def _split_list_once(items, train_ratio=TRAIN_RATIO, seed=42):
    """
    Split a list into train and validation parts using plain random shuffling.

    This is a fallback helper used when stratification cannot be applied.

    Args:
        items: List of items to split.
        train_ratio: Fraction to keep for training.
        seed: Random seed.

    Returns:
        train_items, val_items
    """
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)

    if len(items) < 2:
        raise ValueError("Need at least 2 samples to create a train/val split.")

    # Keep at least one sample in each split.
    n_train = int(len(items) * train_ratio)
    n_train = max(1, min(n_train, len(items) - 1))

    train_items = items[:n_train]
    val_items = items[n_train:]

    return train_items, val_items

def _stratified_split_pairs(
    pairs: List[Tuple[str, int]],
    pair_to_label: Dict[Tuple[str, int], int],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
):
    """
    Split (sid, trial_idx) pairs using stratification on their labels.

    If stratification is not possible, this function falls back to a simple
    random split.

    Stratification is considered possible only if:
    - there are at least 2 classes
    - each class has at least 2 samples

    Args:
        pairs: List of (sid, trial_idx) pairs.
        pair_to_label: Mapping from pair to binary label.
        train_ratio: Fraction for the training set.
        seed: Random seed.

    Returns:
        train_pairs, val_pairs
    """
    pairs = list(pairs)

    if len(pairs) < 2:
        raise ValueError("Need at least 2 samples to create a train/val split.")

    labels = [pair_to_label[p] for p in pairs]

    # Count samples per class to decide whether stratification is safe.
    unique, counts = np.unique(labels, return_counts=True)
    class_count = dict(zip(unique.tolist(), counts.tolist()))

    can_stratify = (
        len(class_count) >= 2
        and all(c >= 2 for c in class_count.values())
    )

    if not can_stratify:
        return _split_list_once(pairs, train_ratio=train_ratio, seed=seed)

    test_size = 1.0 - train_ratio

    train_pairs, val_pairs = train_test_split(
        pairs,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    return list(train_pairs), list(val_pairs)

def make_shared_trial_split(
    pairs: List[Tuple[str, int]],
    split_mode: str = "subject_dependent",
    target_sid: Optional[str] = None,
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
    pair_to_label: Optional[Dict[Tuple[str, int], int]] = None,
    stratify: bool = True,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, Any]]:
    """
    Create one shared split over (sid, trial_idx) pairs.

    The goal is to reuse exactly the same train/validation split for:
    - EEG-only experiments
    - video-only experiments
    - fusion experiments

    Supported split modes:
        - "subject_dependent"
        - "random_all"
        - "loso"

    Stratification:
        - used only for "subject_dependent" and "random_all"
        - requires pair_to_label
        - LOSO is left unchanged by design

    Args:
        pairs: List of available (sid, trial_idx) pairs.
        split_mode: Split strategy to use.
        target_sid: Required for "subject_dependent" and "loso".
        train_ratio: Fraction for training when a random split is used.
        seed: Random seed.
        pair_to_label: Optional pair -> label mapping for stratification.
        stratify: Whether to try stratified splitting when possible.

    Returns:
        train_pairs: Training pairs.
        val_pairs: Validation pairs.
        split_info: Dictionary with useful split statistics.
    """
    pairs = sorted(list(pairs))

    if len(pairs) == 0:
        raise ValueError("No available (sid, trial_idx) pairs found.")

    use_stratified = stratify and (pair_to_label is not None)

    if split_mode == "subject_dependent":
        if target_sid is None:
            raise ValueError("target_sid is required for split_mode='subject_dependent'")

        # Keep only pairs from the chosen subject.
        subject_pairs = [p for p in pairs if p[0] == target_sid]

        if len(subject_pairs) == 0:
            raise ValueError(f"No pairs found for subject {target_sid}")

        if use_stratified:
            train_pairs, val_pairs = _stratified_split_pairs(
                subject_pairs,
                pair_to_label=pair_to_label,
                train_ratio=train_ratio,
                seed=seed,
            )
        else:
            train_pairs, val_pairs = _split_list_once(
                subject_pairs,
                train_ratio=train_ratio,
                seed=seed,
            )

    elif split_mode == "random_all":
        # Random split over all available pairs, optionally stratified.
        if use_stratified:
            train_pairs, val_pairs = _stratified_split_pairs(
                pairs,
                pair_to_label=pair_to_label,
                train_ratio=train_ratio,
                seed=seed,
            )
        else:
            train_pairs, val_pairs = _split_list_once(
                pairs,
                train_ratio=train_ratio,
                seed=seed,
            )

    elif split_mode == "loso":
        if target_sid is None:
            raise ValueError("target_sid is required for split_mode='loso'")

        # Leave-one-subject-out:
        # train = all other subjects, val = target subject.
        train_pairs = [p for p in pairs if p[0] != target_sid]
        val_pairs = [p for p in pairs if p[0] == target_sid]

        if len(train_pairs) == 0 or len(val_pairs) == 0:
            raise ValueError(
                f"Bad LOSO split for {target_sid}: "
                f"train={len(train_pairs)}, val={len(val_pairs)}"
            )

    else:
        raise ValueError("split_mode must be one of {'subject_dependent', 'random_all', 'loso'}")

    split_info = {
        "split_mode": split_mode,
        "target_sid": target_sid,
        "n_total_pairs": len(pairs),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "stratified": bool(use_stratified and split_mode in {"subject_dependent", "random_all"}),
    }

    # Add label distribution info when labels are available.
    if pair_to_label is not None:
        train_labels = [pair_to_label[p] for p in train_pairs]
        val_labels = [pair_to_label[p] for p in val_pairs]

        split_info["train_label_counts"] = {
            int(k): int(v) for k, v in zip(*np.unique(train_labels, return_counts=True))
        } if len(train_labels) > 0 else {}

        split_info["val_label_counts"] = {
            int(k): int(v) for k, v in zip(*np.unique(val_labels, return_counts=True))
        } if len(val_labels) > 0 else {}

    return train_pairs, val_pairs, split_info

# =========================================================
# GLOBAL SHARED SPLIT CACHE
# =========================================================

def init_shared_splits(
    eeg_root,
    image_root,
    label_type=LABEL_TYPE,
    max_subject_id=MAX_SUBJECT_ID,
    split_mode="subject_dependent",
    train_ratio=TRAIN_RATIO,
    seed=SEED,
):
    """
    Create all train/validation splits once and store them globally.

    Then every model can reuse the exact same split:
    - EEG
    - Video
    - FiLM fusion
    - Concat fusion
    - Decision fusion
    """
    global SHARED_SPLITS

    subjects = get_data(root=eeg_root, mode=0, l=label_type)

    all_pairs = build_available_trial_pairs(
        subjects=subjects,
        image_root=image_root,
        max_subject_id=max_subject_id,
    )

    pair_to_label = _build_pair_label_lookup(
        subjects=subjects,
        max_subject_id=max_subject_id,
    )

    SHARED_SPLITS = {}

    # One global split over all available pairs
    if split_mode == "random_all":
        train_pairs, val_pairs, split_info = make_shared_trial_split(
            pairs=all_pairs,
            split_mode="random_all",
            target_sid=None,
            train_ratio=train_ratio,
            seed=seed,
            pair_to_label=pair_to_label,
            stratify=True,
        )

        SHARED_SPLITS["global"] = {
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "split_info": split_info,
        }

    # One split per subject: subject_dependent or loso
    else:
        for sid_num in range(1, max_subject_id + 1):
            sid = f"s{sid_num:02d}"

            train_pairs, val_pairs, split_info = make_shared_trial_split(
                pairs=all_pairs,
                split_mode=split_mode,
                target_sid=sid,
                train_ratio=train_ratio,
                seed=seed,
                pair_to_label=pair_to_label,
                stratify=True,
            )

            SHARED_SPLITS[sid] = {
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
                "split_info": split_info,
            }

    print(f"Shared splits initialized for split_mode={split_mode}")
    print(f"Number of stored splits: {len(SHARED_SPLITS)}")

    return SHARED_SPLITS


def get_shared_split(sid=None, split_mode="subject_dependent"):
    """
    Retrieve an already-created split.

    For subject_dependent / loso:
        get_shared_split("s01", "subject_dependent")

    For random_all:
        get_shared_split(split_mode="random_all")
    """
    key = "global" if split_mode == "random_all" else sid

    if key not in SHARED_SPLITS:
        raise ValueError(
            f"No shared split found for key={key}. "
            "Run init_shared_splits(...) before running experiments."
        )

    split_pack = SHARED_SPLITS[key]

    return (
        split_pack["train_pairs"],
        split_pack["val_pairs"],
        split_pack["split_info"],
    )

# =========================================================
# K-FOLD CROSS-VALIDATION SHARED SPLITS
# =========================================================

def _label_counts_from_pairs(pairs, pair_to_label):
    """
    Small helper to count labels inside a list of pairs.
    """
    labels = [pair_to_label[p] for p in pairs]
    if len(labels) == 0:
        return {}

    keys, vals = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(keys, vals)}


def init_shared_cv_splits(
    eeg_root,
    image_root,
    label_type=LABEL_TYPE,
    max_subject_id=MAX_SUBJECT_ID,
    split_mode="subject_dependent",
    n_splits=4,
    seed=SEED,
):
    """
    Create K-fold CV splits once.

    For subject_dependent:
        For each subject, split that subject's trials into n_splits folds.

    For random_all:
        Split all available trials into n_splits global folds.

    Each fold is stored globally, so all models reuse the same train/val pairs.
    """
    global SHARED_CV_SPLITS

    subjects = get_data(root=eeg_root, mode=0, l=label_type)

    all_pairs = build_available_trial_pairs(
        subjects=subjects,
        image_root=image_root,
        max_subject_id=max_subject_id,
    )

    pair_to_label = _build_pair_label_lookup(
        subjects=subjects,
        max_subject_id=max_subject_id,
    )

    SHARED_CV_SPLITS = {fold: {} for fold in range(n_splits)}

    # -----------------------------------------------------
    # Case 1: one global K-fold CV over all trials
    # -----------------------------------------------------
    if split_mode == "random_all":
        pairs = sorted(all_pairs)
        labels = np.array([pair_to_label[p] for p in pairs])

        unique, counts = np.unique(labels, return_counts=True)
        can_stratify = len(unique) >= 2 and counts.min() >= n_splits

        if can_stratify:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(pairs, labels)
        else:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(pairs)

        for fold, (train_idx, val_idx) in enumerate(split_iter):
            train_pairs = [pairs[i] for i in train_idx]
            val_pairs = [pairs[i] for i in val_idx]

            split_info = {
                "cv": True,
                "fold": fold,
                "n_splits": n_splits,
                "split_mode": split_mode,
                "target_sid": None,
                "n_total_pairs": len(pairs),
                "n_train_pairs": len(train_pairs),
                "n_val_pairs": len(val_pairs),
                "stratified": bool(can_stratify),
                "train_label_counts": _label_counts_from_pairs(train_pairs, pair_to_label),
                "val_label_counts": _label_counts_from_pairs(val_pairs, pair_to_label),
            }

            SHARED_CV_SPLITS[fold]["global"] = {
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
                "split_info": split_info,
            }

        print(f"Initialized {n_splits}-fold CV splits for random_all")
        return SHARED_CV_SPLITS

    # -----------------------------------------------------
    # Case 2: subject-dependent K-fold CV
    # -----------------------------------------------------
    for sid_num in range(1, max_subject_id + 1):
        sid = f"s{sid_num:02d}"

        subject_pairs = sorted([p for p in all_pairs if p[0] == sid])

        if len(subject_pairs) < n_splits:
            print(f"Skipping {sid}: not enough trials for {n_splits}-fold CV")
            continue

        labels = np.array([pair_to_label[p] for p in subject_pairs])

        unique, counts = np.unique(labels, return_counts=True)
        can_stratify = len(unique) >= 2 and counts.min() >= n_splits

        if can_stratify:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(subject_pairs, labels)
        else:
            # Fallback if a subject has too few samples in one class
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )
            split_iter = splitter.split(subject_pairs)

        for fold, (train_idx, val_idx) in enumerate(split_iter):
            train_pairs = [subject_pairs[i] for i in train_idx]
            val_pairs = [subject_pairs[i] for i in val_idx]

            split_info = {
                "cv": True,
                "fold": fold,
                "n_splits": n_splits,
                "split_mode": split_mode,
                "target_sid": sid,
                "n_total_pairs": len(subject_pairs),
                "n_train_pairs": len(train_pairs),
                "n_val_pairs": len(val_pairs),
                "stratified": bool(can_stratify),
                "train_label_counts": _label_counts_from_pairs(train_pairs, pair_to_label),
                "val_label_counts": _label_counts_from_pairs(val_pairs, pair_to_label),
            }

            SHARED_CV_SPLITS[fold][sid] = {
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
                "split_info": split_info,
            }

    print(f"Initialized {n_splits}-fold CV splits for split_mode={split_mode}")
    print(f"Available folds: {list(SHARED_CV_SPLITS.keys())}")

    return SHARED_CV_SPLITS


def activate_cv_fold(fold):
    """
    Activate one CV fold by copying its splits into SHARED_SPLITS.

    Your existing run_all_* functions can stay simple:
        they call get_shared_split(...)
    and get_shared_split will now return the active fold split.
    """
    global SHARED_SPLITS
    global CURRENT_CV_FOLD
    global CURRENT_RUN_TAG

    if fold not in SHARED_CV_SPLITS:
        raise ValueError(
            f"Fold {fold} does not exist. "
            "Run init_shared_cv_splits(...) first."
        )

    SHARED_SPLITS = SHARED_CV_SPLITS[fold]
    CURRENT_CV_FOLD = fold
    CURRENT_RUN_TAG = f"cv_fold_{fold}"

    print(f"Activated CV fold {fold}")
    print(f"Number of active splits: {len(SHARED_SPLITS)}")


def ckpt_path(filename):
    """
    Save checkpoints inside a fold-specific folder to avoid overwriting models.
    """
    folder = Path("models") / CURRENT_RUN_TAG
    folder.mkdir(parents=True, exist_ok=True)

    filename = Path(filename).name  # removes any accidental folder prefix

    return str(folder / filename)