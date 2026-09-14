from .eeg import get_data

class DEAPDataset(Dataset):
    """
    PyTorch Dataset wrapper around data returned by `get_data(...)`.

    This dataset supports optional filtering through `allowed_pairs`, where each pair is:
        (sid, trial_idx)

    This is especially useful when you want EEG-only, video-only, and fusion
    pipelines to use exactly the same train/validation split.

    Modes:
        - 0, 1, 2: indexing is trial-based
        - 3: indexing is window-based, but filtering is still done at trial level

    Args:
        path: Root path used by `get_data(...)`.
        mode: Data mode (0, 1, 2, or 3).
        l: Label type (for example valence or arousal).
        allowed_pairs: Optional list of allowed (sid, trial_idx) pairs.
            If provided, only samples belonging to these pairs are kept.
    """
    def __init__(
        self,
        path: Path | str,
        mode: int = 0,
        l: int = 0,
        allowed_pairs: List[Tuple[str, int]] | None = None,
    ):
        assert mode in (0, 1, 2, 3), "mode must be one of {0,1,2,3}"
        self.mode = mode
        self.label_type = l

        # Load all subject data from disk or cache.
        self.data = get_data(path, mode=self.mode, l=l)
        self.sids: List[str] = sorted(self.data.keys())

        # Normalize allowed pairs once so membership checks are fast and consistent.
        # Subject ids are lowercased and trial indices are converted to int.
        allowed_set = None
        if allowed_pairs is not None:
            allowed_set = {(sid.lower(), int(t)) for sid, t in allowed_pairs}

        # Global index used by PyTorch.
        # Each element is (sid, sample_idx), where sample_idx means:
        # - trial index for modes 0/1/2
        # - window index for mode 3
        self.index: List[Tuple[str, int]] = []

        for sid in self.sids:
            entry = self.data[sid]

            # Modes 0/1/2 store one sample per trial,
            # so filtering is done directly with (sid, trial_idx).
            if self.mode in (0, 1, 2):
                n = entry["X"].shape[0]
                for t in range(n):
                    pair = (sid.lower(), int(t))
                    if allowed_set is not None and pair not in allowed_set:
                        continue
                    self.index.append((sid, t))

            # Mode 3 stores one sample per window.
            # Each window belongs to a parent trial, so filtering is done
            # using that parent (sid, trial_idx) pair.
            else:
                n = entry["X"].shape[0]

                if "win_index" not in entry:
                    raise ValueError("mode=3 requires win_index in dataset entry")

                for w in range(n):
                    trial_idx, _win_in_trial = entry["win_index"][w]
                    pair = (sid.lower(), int(trial_idx))
                    if allowed_set is not None and pair not in allowed_set:
                        continue
                    self.index.append((sid, w))

        # Binary class mapping used in the project.
        self.class_to_idx = {"L": 0, "H": 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        """
        Return the total number of samples currently exposed by the dataset.

        Returns:
            Number of indexed samples after optional filtering.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Return one sample from the dataset.

        Returned format:
            x, y, meta

        Where:
            - x is the input tensor
            - y is the class label tensor
            - meta contains identifiers such as subject, trial, and optionally window info

        Args:
            idx: Global dataset index.

        Returns:
            Tuple of (x, y, meta).
        """
        sid, i = self.index[idx]
        entry = self.data[sid]

        # Modes 0/1/2 are trial-based.
        if self.mode in (0, 1, 2):
            x = torch.from_numpy(entry["X"][i]).float()
            y = torch.tensor(entry["y"][i], dtype=torch.long)
            return x, y, {"sid": sid, "trial": i}

        # Mode 3 is window-based.
        else:
            x = torch.from_numpy(entry["X"][i]).float()
            y = torch.tensor(int(entry["y"][i]), dtype=torch.long)

            # Basic metadata always returned for mode 3.
            meta = {"sid": sid, "win": i}

            # Add parent trial and local window index when available.
            if "win_index" in entry:
                tr, w = entry["win_index"][i]
                meta.update({"trial": tr, "win_in_trial": w})

            return x, y, meta

    @property
    def n_subjects(self) -> int:
        """
        Return how many subjects are currently loaded in the dataset.

        Returns:
            Number of subject ids in `self.data`.
        """
        return len(self.sids)

    @property
    def shape_info(self) -> Dict[str, Any]:
        """
        Return a quick summary of shapes for each subject.

        This is useful for debugging and checking that the loaded data
        matches the expected format for the selected mode.

        Returns:
            Dictionary of the form:
                {
                    "s01": {
                        "X_shape": ...,
                        "y_shape": ...,
                        "mode": ...
                    },
                    ...
                }
        """
        info = {}
        for sid in self.sids:
            X = self.data[sid]["X"]
            y = self.data[sid]["y"]
            info[sid] = {
                "X_shape": tuple(X.shape),
                "y_shape": tuple(y.shape),
                "mode": self.mode
            }
        return info


def create_eeg_dataloaders_from_pairs(
    path: str | Path,
    mode: int,
    label_type: int,
    train_pairs,
    val_pairs,
    batch_size: int,
    num_workers: int = 0,
):
    """
    Build EEG train and validation DataLoaders using shared (sid, trial_idx) pairs.

    This function creates two `DEAPDataset` instances:
    - one filtered with `train_pairs`
    - one filtered with `val_pairs`

    Using shared trial pairs ensures consistent splits across different pipelines,
    for example EEG-only, video-only, and fusion models.

    Args:
        path: Root path used by `get_data(...)`.
        mode: Data mode passed to `DEAPDataset`.
        label_type: Label type passed to `DEAPDataset`.
        train_pairs: List of training (sid, trial_idx) pairs.
        val_pairs: List of validation (sid, trial_idx) pairs.
        batch_size: Batch size for the DataLoaders.
        num_workers: Number of DataLoader worker processes.

    Returns:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        train_dataset: Filtered training dataset.
        val_dataset: Filtered validation dataset.
    """

    # Training dataset restricted to the selected training pairs.
    train_dataset = DEAPDataset(
        path=path,
        mode=mode,
        l=label_type,
        allowed_pairs=train_pairs,
    )

    # Validation dataset restricted to the selected validation pairs.
    val_dataset = DEAPDataset(
        path=path,
        mode=mode,
        l=label_type,
        allowed_pairs=val_pairs,
    )

    # Pin memory is useful when training on GPU.
    pin_mem = torch.cuda.is_available()

    # Persistent workers are only valid when num_workers > 0.
    persistent = num_workers > 0

    # Training loader uses shuffle=True.
    # drop_last=True keeps batch sizes consistent during training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=True,
    )

    # Validation loader does not shuffle.
    # drop_last=False ensures all validation samples are evaluated.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset

def build_trial_dict(root_dir):
    """
    Scan the image root directory and build:
    1) a dictionary mapping each (subject, trial) to its image paths
    2) a flat list of available (subject, trial) samples

    Expected folder structure:
        root_dir/
            s01_trial01/
                img1.jpg
                img2.jpg
                ...
            s01_trial02/
            ...

    Args:
        root_dir: Root directory containing trial folders.

    Returns:
        trial_dict:
            {
                "s01": {
                    "trial01": [img_path1, img_path2, ...],
                    ...
                },
                ...
            }

        sample_index:
            [
                ("s01", "trial01"),
                ("s01", "trial02"),
                ...
            ]
    """
    root_dir = Path(root_dir)
    trial_dict = {}
    sample_index = []

    for folder in sorted(root_dir.iterdir()):
        if not folder.is_dir():
            continue

        parts = folder.name.split("_")
        if len(parts) != 2:
            continue

        sid, trial = parts

        # Keep only common image formats.
        image_paths = sorted([
            str(img) for img in folder.iterdir()
            if img.is_file() and img.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(image_paths) == 0:
            continue

        if sid not in trial_dict:
            trial_dict[sid] = {}

        trial_dict[sid][trial] = image_paths
        sample_index.append((sid, trial))

    return trial_dict, sample_index

def create_labels_dict_from_eeg_and_images(subjects, image_root, max_subject_id=22):
    """
    Build a label dictionary for video/image trials using EEG trial labels.

    The image folders are expected to be named like:
        s01_trial01
        s01_trial02
        ...

    For each valid folder, this function matches:
        subject id + trial id -> EEG label

    Args:
        subjects: EEG subject dictionary returned by the EEG pipeline.
        image_root: Root directory containing image trial folders.
        max_subject_id: Ignore subjects with id greater than this value.

    Returns:
        Nested dictionary of the form:
            {
                "s01": {
                    "trial01": 0,
                    "trial02": 1,
                    ...
                },
                ...
            }
    """
    image_root = Path(image_root)
    labels_dict = {}

    for folder in sorted(image_root.iterdir()):
        if not folder.is_dir():
            continue

        parts = folder.name.split("_")
        if len(parts) != 2:
            continue

        sid, trial_key = parts

        # Skip subjects outside the allowed range.
        if int(sid[1:]) > max_subject_id:
            continue

        # Skip if the subject does not exist in EEG labels.
        if sid not in subjects:
            continue

        y = subjects[sid]["y"]
        trial_idx = int(trial_key.replace("trial", "")) - 1

        # Skip invalid trial indices.
        if trial_idx < 0 or trial_idx >= len(y):
            continue

        if sid not in labels_dict:
            labels_dict[sid] = {}

        labels_dict[sid][trial_key] = int(y[trial_idx])

    return labels_dict

def _normalize_pair(pair):
    sid, trial = pair
    return str(sid).lower(), int(trial)


def _normalize_pairs(pairs):
    return [_normalize_pair(pair) for pair in pairs]


def _pair_from_meta(meta):
    return (
        str(meta["sid"]).lower(),
        int(meta["trial"]),
    )


def _build_pair_to_index_map(dataset):
    pair_to_idx = {}

    for idx in range(len(dataset)):
        *_, meta = dataset[idx]
        pair = _pair_from_meta(meta)

        if pair in pair_to_idx:
            raise ValueError(
                f"Duplicate pair found in dataset: {pair}"
            )

        pair_to_idx[pair] = idx

    return pair_to_idx

class DEAPFaceTrialDataset(Dataset):
    """
    PyTorch dataset for video/image trials.

    This dataset can be filtered using a shared list of allowed pairs:
        allowed_pairs = [(sid, trial_idx), ...]

    Internal naming:
        sid   -> "s01"
        trial -> "trial01"

    Shared split format:
        (sid, trial_idx) where trial_idx is 0-based

    Each item returns:
        x: stacked frames of shape [T, C, H, W]
        y: trial label
        meta: dictionary containing subject/trial information

    Args:
        root_dir: Root directory of trial folders.
        labels_dict: Nested dictionary containing labels per subject/trial.
        transform: Optional image transform.
        num_frames: Number of frames to keep per trial.
        allowed_pairs: Optional shared split filter.
    """
    def __init__(
        self,
        root_dir,
        labels_dict,
        transform=None,
        num_frames=18,
        allowed_pairs=None,
    ):
        self.trial_dict, self.sample_index = build_trial_dict(root_dir)
        self.labels_dict = labels_dict
        self.transform = transform
        self.num_frames = num_frames

        # Normalize allowed pairs once for fast membership checking.
        allowed_set = None
        if allowed_pairs is not None:
            allowed_set = {(sid.lower(), int(t)) for sid, t in allowed_pairs}

        filtered = []
        for sid, trial in self.sample_index:
            # Keep only samples that have labels.
            if sid not in labels_dict:
                continue
            if trial not in labels_dict[sid]:
                continue

            # Convert "trial01" -> 0, "trial02" -> 1, ...
            trial_idx = int(trial.replace("trial", "")) - 1
            pair = (sid.lower(), trial_idx)

            # If a shared split is provided, keep only matching trials.
            if allowed_set is not None and pair not in allowed_set:
                continue

            filtered.append((sid, trial))

        self.sample_index = filtered

    def __len__(self):
        """
        Return the number of available filtered samples.
        """
        return len(self.sample_index)

    def __getitem__(self, idx):
        """
        Load one video trial.

        Steps:
        - load up to `num_frames` images from the trial folder
        - apply transform if provided
        - stack frames into one tensor
        - pad by repeating the last frame if there are not enough images

        Args:
            idx: Dataset index.

        Returns:
            x: Tensor of shape [T, C, H, W]
            y: Label tensor
            meta: Dictionary with subject/trial information
        """
        sid, trial = self.sample_index[idx]
        image_paths = self.trial_dict[sid][trial][:self.num_frames]

        frames = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            frames.append(img)

        if len(frames) == 0:
            raise ValueError(f"No images found for sample {sid}_{trial}")

        # If the trial has fewer frames than expected, repeat the last frame.
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        x = torch.stack(frames)  # [T, C, H, W]
        y = torch.tensor(self.labels_dict[sid][trial], dtype=torch.long)

        trial_idx = int(trial.replace("trial", "")) - 1
        meta = {"sid": sid, "trial": trial_idx, "trial_key": trial}

        return x, y, meta

class DEAPFusionTrialDataset(Dataset):
    """
    Fusion dataset that returns matched EEG and video samples for the same trial.

    Each item corresponds to one shared pair:
        (sid, trial_idx)

    Returned values:
        eeg_x   : EEG input for one trial
        video_x : Video input for the same trial
        y       : Single shared label
        meta    : Dictionary containing:
                  - 'sid'
                  - 'trial'
                  - 'eeg_meta'
                  - 'video_meta'

    Important:
    - This dataset does not create train/validation splits by itself.
    - It only consumes already prepared shared pairs.
    - Alignment is enforced strictly so EEG and video always refer to
      the exact same subject and trial.
    """

    def __init__(
        self,
        eeg_root,
        image_root,
        eeg_mode=2,
        label_type=0,
        num_frames=18,
        allowed_pairs=None,
        max_subject_id=22,
        video_transform=None,
        labels_dict=None,
    ):
        """
        Initialize the fusion dataset.

        Args:
            eeg_root: Root path of EEG data
            image_root: Root path of video/image data
            eeg_mode: EEG mode used by DEAPDataset
            label_type: Label type (for example valence or arousal)
            num_frames: Number of frames used per video trial
            allowed_pairs: Shared list of allowed (sid, trial_idx) pairs
            max_subject_id: Maximum allowed subject id
            video_transform: Optional transform for video frames
            labels_dict: Optional precomputed video label dictionary
        """
        super().__init__()

        # Normalize allowed pairs once so alignment checks are consistent.
        self.allowed_pairs = None if allowed_pairs is None else _normalize_pairs(allowed_pairs)

        # -------------------------
        # EEG dataset
        # -------------------------
        self.eeg_dataset = DEAPDataset(
            path=eeg_root,
            mode=eeg_mode,
            l=label_type,
            allowed_pairs=self.allowed_pairs,
        )

        # -------------------------
        # Video labels + transform
        # -------------------------
        # If labels are not already provided, derive them from the EEG side.
        if labels_dict is None:
            subjects = get_data(root=eeg_root, mode=0, l=label_type)
            labels_dict = create_labels_dict_from_eeg_and_images(
                subjects=subjects,
                image_root=image_root,
                max_subject_id=max_subject_id,
            )

        # If no video transform is given, use the default ResNet18 preprocessing.
        if video_transform is None:
            weights = models.ResNet18_Weights.DEFAULT
            video_transform = weights.transforms()

        # -------------------------
        # Video dataset
        # -------------------------
        self.video_dataset = DEAPFaceTrialDataset(
            root_dir=image_root,
            labels_dict=labels_dict,
            transform=video_transform,
            num_frames=num_frames,
            allowed_pairs=self.allowed_pairs,
        )

        # -------------------------
        # Build alignment maps
        # -------------------------
        # Map each (sid, trial) pair to its index inside the EEG and video datasets.
        self.eeg_pair_to_idx = _build_pair_to_index_map(self.eeg_dataset)
        self.video_pair_to_idx = _build_pair_to_index_map(self.video_dataset)

        eeg_pairs = set(self.eeg_pair_to_idx.keys())
        video_pairs = set(self.video_pair_to_idx.keys())
        common_pairs = eeg_pairs & video_pairs

        # If no external pair filter is provided, use every pair that exists in both datasets.
        if self.allowed_pairs is None:
            self.pairs = sorted(common_pairs)
        else:
            allowed_set = set(self.allowed_pairs)

            # Helpful checks to catch missing trials early.
            missing_in_eeg = allowed_set - eeg_pairs
            missing_in_video = allowed_set - video_pairs

            if len(missing_in_eeg) > 0:
                raise ValueError(
                    f"Some allowed_pairs are missing in EEG dataset. "
                    f"Example: {sorted(list(missing_in_eeg))[:5]}"
                )

            if len(missing_in_video) > 0:
                raise ValueError(
                    f"Some allowed_pairs are missing in video dataset. "
                    f"Example: {sorted(list(missing_in_video))[:5]}"
                )

            # Keep exactly the order given by allowed_pairs.
            self.pairs = [p for p in self.allowed_pairs if p in common_pairs]

        if len(self.pairs) == 0:
            raise ValueError("Fusion dataset is empty after alignment.")

    def __len__(self):
        """
        Return the number of aligned fusion samples.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return one aligned EEG + video sample.

        Steps:
        - find the shared (sid, trial) pair
        - fetch the corresponding EEG item
        - fetch the corresponding video item
        - verify that both point to the same pair
        - verify that both labels match

        Args:
            idx: Dataset index

        Returns:
            eeg_x, video_x, eeg_y, meta

        Raises:
            RuntimeError: If alignment or label consistency fails
        """
        pair = self.pairs[idx]

        eeg_idx = self.eeg_pair_to_idx[pair]
        video_idx = self.video_pair_to_idx[pair]

        eeg_x, eeg_y, eeg_meta = self.eeg_dataset[eeg_idx]
        video_x, video_y, video_meta = self.video_dataset[video_idx]

        eeg_pair = _pair_from_meta(eeg_meta)
        video_pair = _pair_from_meta(video_meta)

        # Strict alignment check: both modalities must refer to the same pair.
        if eeg_pair != pair:
            raise RuntimeError(f"EEG alignment error: expected {pair}, got {eeg_pair}")

        if video_pair != pair:
            raise RuntimeError(f"Video alignment error: expected {pair}, got {video_pair}")

        eeg_label = int(eeg_y.item()) if torch.is_tensor(eeg_y) else int(eeg_y)
        video_label = int(video_y.item()) if torch.is_tensor(video_y) else int(video_y)

        # Both modalities must share the exact same label.
        if eeg_label != video_label:
            raise RuntimeError(
                f"Label mismatch for pair {pair}: EEG={eeg_label}, VIDEO={video_label}"
            )

        meta = {
            "sid": pair[0],
            "trial": pair[1],
            "eeg_meta": eeg_meta,
            "video_meta": video_meta,
        }

        return eeg_x, video_x, eeg_y, meta

def fusion_collate_fn(batch):
    """
    Custom collate function for fusion batches.

    Why custom?
    Because we want the metadata to stay as a simple Python list of dicts,
    instead of being automatically merged into a tensor-like structure.

    Args:
        batch: List of dataset items

    Returns:
        eeg_x: Batched EEG tensor
        video_x: Batched video tensor
        y: Batched labels
        meta_list: List of metadata dictionaries
    """
    eeg_x_list, video_x_list, y_list, meta_list = zip(*batch)

    eeg_x = torch.stack(eeg_x_list, dim=0)
    video_x = torch.stack(video_x_list, dim=0)
    y = torch.stack(y_list, dim=0).long()

    return eeg_x, video_x, y, list(meta_list)

def create_fusion_dataloaders_from_pairs(
    eeg_root,
    image_root,
    train_pairs,
    val_pairs,
    eeg_mode=2,
    label_type=0,
    num_frames=18,
    batch_size=4,
    num_workers=NUM_WORKERS,
    max_subject_id=22,
):
    """
    Build train and validation fusion dataloaders from shared trial pairs.

    The important point is that both loaders are built from the exact same
    pair definition used elsewhere in EEG-only and video-only experiments.

    Args:
        eeg_root: EEG root directory
        image_root: Video/image root directory
        train_pairs: Shared training pairs
        val_pairs: Shared validation pairs
        eeg_mode: EEG mode
        label_type: Label type
        num_frames: Number of video frames per trial
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        max_subject_id: Maximum allowed subject id

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """

    # Build labels once and reuse them for both train and validation datasets.
    subjects = get_data(root=eeg_root, mode=0, l=label_type)
    labels_dict = create_labels_dict_from_eeg_and_images(
        subjects=subjects,
        image_root=image_root,
        max_subject_id=max_subject_id,
    )

    weights = models.ResNet18_Weights.DEFAULT
    video_transform = weights.transforms()

    train_dataset = DEAPFusionTrialDataset(
        eeg_root=eeg_root,
        image_root=image_root,
        eeg_mode=eeg_mode,
        label_type=label_type,
        num_frames=num_frames,
        allowed_pairs=train_pairs,
        max_subject_id=max_subject_id,
        video_transform=video_transform,
        labels_dict=labels_dict,
    )

    val_dataset = DEAPFusionTrialDataset(
        eeg_root=eeg_root,
        image_root=image_root,
        eeg_mode=eeg_mode,
        label_type=label_type,
        num_frames=num_frames,
        allowed_pairs=val_pairs,
        max_subject_id=max_subject_id,
        video_transform=video_transform,
        labels_dict=labels_dict,
    )

    pin_mem = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=True,
        collate_fn=fusion_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        drop_last=False,
        collate_fn=fusion_collate_fn,
    )

    return train_loader, val_loader, train_dataset, val_dataset
