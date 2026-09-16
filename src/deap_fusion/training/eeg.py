from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast as amp_autocast, GradScaler
from tqdm.auto import tqdm

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..config import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    WEIGHT_DECAY,
    NUM_WORKERS,
)

from ..data.datasets import create_eeg_dataloaders_from_pairs
from ..data.splits import get_shared_split, ckpt_path
from ..models.eeg import build_model

"""
Training engine utilities for DEAP EEG experiments.

This module provides the main training helpers used in the project:
- `train_step`: runs one full training epoch
- `test_step`: runs one full validation/test pass
- `train`: handles multi-epoch training and keeps the best model in memory
- `train_kfold`: repeats training over several folds for cross-validation

Main design choices:
- Works with batches returned as (X, y, meta) or just (X, y)
- Uses AMP (automatic mixed precision) on CUDA for faster training
- Tracks loss and accuracy for both train and validation sets
- Restores the best model weights at the end of training
"""


def _unpack_xy(batch):
    """
    Extract only (X, y) from a batch.

    Some datasets return:
        (X, y, meta)

    while the training code only needs:
        (X, y)

    Args:
        batch: Batch returned by a DataLoader. Expected formats:
            - (X, y)
            - (X, y, ...)

    Returns:
        X, y

    Raises:
        ValueError: If the batch does not have at least two elements.
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Expected batch to be (X, y) or (X, y, ...).")


def eeg_augmentation(X, eeg_input_type="features", noise_std=0.01, scale_std=0.03):
    """
    Creates a slightly perturbed version of EEG input.

    eeg_input_type:
    - "raw"      for X shape (B, C, T)
    - "features" for X shape (B, Nwin, F)
    """

    noise = torch.randn_like(X) * noise_std

    if X.ndim == 3:
        if eeg_input_type == "raw":
            # raw EEG: scale each channel
            scale = 1.0 + torch.randn(
                X.size(0), X.size(1), 1, device=X.device
            ) * scale_std
        else:
            # DE features: scale each feature dimension
            scale = 1.0 + torch.randn(
                X.size(0), 1, X.size(2), device=X.device
            ) * scale_std

    elif X.ndim == 2:
        scale = 1.0 + torch.randn(
            X.size(0), X.size(1), device=X.device
        ) * scale_std

    else:
        scale = 1.0

    return X * scale + noise


def eeg_consistency_loss(logits_clean, logits_aug):
    """
    Forces the model to give similar predictions for:
    - original EEG
    - slightly perturbed EEG
    """
    p_clean = F.softmax(logits_clean.detach(), dim=1)
    log_p_aug = F.log_softmax(logits_aug, dim=1)

    return F.kl_div(log_p_aug, p_clean, reduction="batchmean")

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    verbose: int = 0,
    lambda_cons: float = 0.0,
    eeg_input_type: str = "features",
) -> Tuple[float, float]:

    model.train()
    train_loss, train_acc, totalseen = 0.0, 0.0, 0

    loop = tqdm(dataloader, desc="Training", leave=False, disable=(verbose < 2))
    use_amp = (device.type == "cuda")

    for batch in loop:
        X, y = _unpack_xy(batch)
        X, y = X.to(device), y.to(device).long()

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device_type=device.type, enabled=use_amp):

            # Normal prediction
            logits_clean = model(X)
            loss_cls = loss_fn(logits_clean, y)

            # Optional consistency regularization
            if lambda_cons > 0:
                X_aug = eeg_augmentation(
                    X,
                    eeg_input_type=eeg_input_type,
                    noise_std=0.01,
                    scale_std=0.03,
                )

                logits_aug = model(X_aug)
                loss_cons = eeg_consistency_loss(logits_clean, logits_aug)

                loss = loss_cls + lambda_cons * loss_cons
            else:
                loss = loss_cls

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bsz = y.size(0)
        train_loss += loss.item() * bsz
        train_acc += (logits_clean.argmax(1) == y).sum().item()
        totalseen += bsz

        if verbose >= 2:
            loop.set_postfix(loss=float(loss.detach().cpu()))

    return train_loss / max(1, totalseen), train_acc / max(1, totalseen)

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    verbose: int = 0,
) -> Tuple[float, float]:
    """
    Evaluate a model on a validation or test DataLoader.

    This function:
    - switches the model to evaluation mode
    - disables gradient computation
    - computes average loss and accuracy over the loader
    - optionally uses AMP on CUDA for faster inference

    Args:
        model: Model to evaluate.
        dataloader: Validation or test DataLoader.
        loss_fn: Loss function.
        device: Device where tensors and model are placed.
        verbose: Controls tqdm display.
            - 0/1: no progress bar
            - 2: show tqdm progress bar

    Returns:
        Tuple:
            - average loss
            - average accuracy
    """
    # Evaluation mode disables training-specific behaviors such as dropout updates.
    model.eval()

    test_loss, correct, total = 0.0, 0, 0
    use_amp = (device.type == "cuda")

    loop = tqdm(dataloader, desc="Evaluating", leave=False, disable=(verbose < 2))

    # inference_mode is faster and uses less memory than full grad-enabled execution.
    with torch.inference_mode():
        for batch in loop:
            X, y = _unpack_xy(batch)
            X, y = X.to(device), y.to(device).long()

            # Forward pass only, no gradients.
            with amp_autocast(device_type=device.type, enabled=use_amp):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

            # Accumulate weighted loss and accuracy.
            bsz = y.size(0)
            test_loss += loss.item() * bsz
            correct += (y_pred.argmax(1) == y).sum().item()
            total += bsz

            if verbose >= 2:
                loop.set_postfix(loss=float(loss.detach().cpu()))

    return test_loss / max(1, total), correct / max(1, total)


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    verbose: int = 0,
    lambda_cons: float = 0.0,
    eeg_input_type: str = "features",
) -> Tuple[torch.nn.Module, float, Dict[str, List[float]]]:
    """
    Train a model for multiple epochs and restore the best checkpoint in memory.

    At each epoch, this function:
    - runs one training epoch
    - runs one validation pass
    - stores train/validation metrics in `history`
    - computes a simple score to decide whether the current model is the best

    Best-model score:
        score = 0.5 * normalized_loss + 0.5 * val_acc

    where:
        normalized_loss = max(0, 1 - min(val_loss, 1))

    This gives a simple balance between lower validation loss and higher validation accuracy.

    Args:
        model: Initialized model.
        train_dataloader: Training DataLoader.
        test_dataloader: Validation DataLoader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        epochs: Number of training epochs.
        device: Device used for training.
        verbose: Logging level.
            - 0: silent
            - 1: print one summary line per epoch
            - 2: show tqdm bars too

    Returns:
        Tuple:
            - model with best weights loaded
            - best_score achieved during training
            - history dictionary with per-epoch metrics
    """
    # History is useful for plots, debugging, and comparing runs later.
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_score = 0.0
    best_model_state = None

    # Create an AMP scaler only when training on CUDA.
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        if verbose >= 1:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training phase for one epoch.
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            verbose=verbose,
            lambda_cons=lambda_cons,
            eeg_input_type=eeg_input_type,
        )

        # Validation phase after the epoch.
        val_loss, val_acc = test_step(
            model, test_dataloader, loss_fn, device, verbose=verbose
        )

        # Convert loss into a simple score where smaller loss becomes better.
        # Then average it with validation accuracy.
        normalized_loss = max(0.0, 1 - min(val_loss, 1))
        score = 0.5 * normalized_loss + 0.5 * val_acc

        # Save metrics for later analysis.
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose >= 1:
            print(f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f} | Score {score:.4f}")

        # Keep the best model weights in memory.
        if score > best_score:
            best_score = score
            best_model_state = deepcopy(model.state_dict())
            if verbose >= 1:
                print(f"↗ New best Score {best_score:.4f}")

    # At the end, restore the best model found during training.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose >= 1:
            print("Loaded best model from memory.")

    return model, best_score, history

def save_model(model: nn.Module, save_path: str | Path) -> None:
    """
    Save a model state_dict to a full checkpoint path.

    Example:
        save_model(model, "models/cv_fold_0/eeg_s01.pth")
    """
    save_path = Path(save_path)

    # Create parent folder automatically
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)

def load_model_checkpoint(
    checkpoint_path: str | Path,
    mode: int,
    feature_dim: Optional[int] = None,
    num_channels: int = 32,
    num_classes: int = 2,
    map_location: Optional[str | torch.device] = None,
) -> nn.Module:
    """
    Rebuild the correct model architecture and load a saved checkpoint.

    This is useful when you saved only the state_dict and want to restore the model
    later for evaluation or inference.

    Args:
        checkpoint_path: Path to the saved .pth file.
        mode: Dataset mode used when the model was trained.
        feature_dim: Required for mode 2 and mode 3 models.
        num_channels: Number of EEG channels for time-domain models (mode 0/1).
        num_classes: Number of output classes.
        map_location: Device mapping used when loading the checkpoint
            (for example "cpu" or "cuda").

    Returns:
        Loaded model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint_path)

    # Recreate the same architecture used during training
    model = build_model(
        mode=mode,
        num_channels=num_channels,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )

    # Load saved weights
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)

    # Put the model in eval mode for inference / validation usage
    model.eval()

    return model

@torch.no_grad()
def evaluate_eeg_detailed(model, loader, criterion, device):
    """
    Evaluate the EEG model on a loader and return detailed classification metrics.

    Returns:
        dict containing:
        - loss
        - acc
        - balanced_acc
        - precision
        - recall
        - f1
        - confusion_matrix
        - y_true
        - y_pred
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_y = []
    all_pred = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())

    acc = float(np.mean(np.array(all_y) == np.array(all_pred))) if total_samples > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(all_y, all_pred)) if len(set(all_y)) > 1 else acc
    precision = float(precision_score(all_y, all_pred, zero_division=0))
    recall = float(recall_score(all_y, all_pred, zero_division=0))
    f1 = float(f1_score(all_y, all_pred, zero_division=0))
    cm = confusion_matrix(all_y, all_pred, labels=[0, 1])

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": acc,
        "balanced_acc": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "y_true": all_y,
        "y_pred": all_pred,
    }


def run_eeg_experiment_shared_split(
    dataset_path: str | Path,
    train_pairs,
    val_pairs,
    model_name: str = "EEG",
    mode: int = 2,
    label_type: int = 0,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    verbose: int = 0,
    save_path: str | None = None,
    lambda_cons: float = 0.0,
    device: torch.device | None = None,
):
    """
    Train one EEG model using a pre-defined shared train/validation split.

    This is useful when you want EEG-only training to use the exact same split as:
    - video-only training
    - fusion training

    The split is given through:
        train_pairs = [(sid, trial_idx), ...]
        val_pairs   = [(sid, trial_idx), ...]

    Args:
        dataset_path: Root path of the EEG dataset.
        train_pairs: Shared training pairs.
        val_pairs: Shared validation pairs.
        model_name: Prefix used for saved model naming.
        mode: EEG data mode.
        label_type: Label type (for example valence or arousal).
        batch_size: Batch size for training and validation.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay used by Adam.
        verbose: Verbosity level passed to the training loop.
        save_path: Optional custom checkpoint filename.

    Returns:
        Dictionary containing:
        - training history
        - best score
        - best / last validation accuracy
        - saved checkpoint path
        - inferred feature_dim
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Build train and validation loaders from the same shared trial pairs
    train_loader, val_loader, train_dataset, val_dataset = create_eeg_dataloaders_from_pairs(
        path=dataset_path,
        mode=mode,
        label_type=label_type,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        batch_size=batch_size,
        num_workers= NUM_WORKERS,
    )

    print("EEG train dataset size:", len(train_dataset))
    print("EEG val dataset size:", len(val_dataset))

    # Infer feature dimension automatically when using feature-based modes
    feature_dim = None
    if mode in (2, 3):
        x0, y0, meta0 = train_dataset[0]
        if mode == 2:
            feature_dim = x0.shape[-1]   # x0 shape: (Nwin, F)
        else:
            feature_dim = x0.shape[0]    # x0 shape: (F,)

    # Build the model matching the selected mode
    model = build_model(
        mode=mode,
        num_channels=32,
        feature_dim=feature_dim,
        num_classes=2,
    ).to(device)

    # Standard classification setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    eeg_input_type = "raw" if mode in (0, 1) else "features"

    # Train the model and keep the best checkpoint in memory
    model, best_score, history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        verbose=verbose,
        lambda_cons=lambda_cons,
        eeg_input_type=eeg_input_type,
    )

    detailed_metrics = evaluate_eeg_detailed(
        model=model,
        loader=val_loader,
        criterion=loss_fn,
        device=device,
    )

    # Default checkpoint filename if the caller did not provide one
    if save_path is None:
        save_path = f"{model_name}_mode{mode}_label{label_type}.pth"

    # Save final restored-best model
    save_model(model, save_path)

    best_val_acc = max(history["val_acc"]) if history.get("val_acc") else None
    last_val_acc = history["val_acc"][-1] if history.get("val_acc") else None

    print("EEG model saved to:", save_path)
    if best_val_acc is not None:
        print(f"EEG best val acc: {best_val_acc:.4f}")
    if last_val_acc is not None:
        print(f"EEG last val acc: {last_val_acc:.4f}")

    return {
        "history": history,
        "best_score": float(best_score),
        "best_val_acc": float(best_val_acc) if best_val_acc is not None else None,
        "last_val_acc": float(last_val_acc) if last_val_acc is not None else None,
        "save_path": str(save_path),
        "feature_dim": feature_dim,
        "detailed_metrics": detailed_metrics,
    }


def run_all_eeg_shared_split(
    dataset_path,
    image_root=None,
    label_type=0,
    mode=2,
    max_subject_id=22,
    split_mode="subject_dependent",
    train_ratio=0.75,
    seed=42,
    model_name="EEG_TR",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    verbose=0,
    lambda_cons=0.0,
    device=None,
):
    """
    Run EEG experiments using shared trial splits.

    This function prepares one common trial space, then creates train/validation splits
    that can also be reused by video or fusion pipelines.

    Supported split modes:
        - "subject_dependent":
            For each subject, split only that subject's trials into train/val
        - "loso":
            Leave-one-subject-out
        - "random_all":
            One global random split across all available pairs

    Args:
        dataset_path: Root path of the EEG dataset.
        image_root: Optional video/image root. If provided, only trials available in
            both EEG and video are kept.
        label_type: Label type used for training.
        mode: EEG mode to train with.
        max_subject_id: Maximum subject id to include.
        split_mode: Split strategy to use.
        train_ratio: Train ratio for random-style splits.
        seed: Random seed.
        model_name: Prefix used for saved model naming.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: Weight decay.
        verbose: Verbosity passed to the training function.

    Returns:
        For random_all:
            {
                "global_result": ...,
                "split_info": ...
            }

        For subject_dependent / loso:
            {
                "subject_results": ...,
                "mean_best_acc": ...,
                "split_mode": ...
            }
    """
    results = {}

    cons_tag = f"cons{str(lambda_cons).replace('.', 'p')}"


    # random_all is one single global run, not one experiment per subject
    if split_mode == "random_all":
        train_pairs, val_pairs, split_info = get_shared_split(
            sid=None,
            split_mode="random_all",
        )

        print("split info:", split_info)

        result = run_eeg_experiment_shared_split(
            dataset_path=dataset_path,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            model_name=model_name,
            mode=mode,
            label_type=label_type,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            save_path=f"{model_name}_random_all_mode{mode}_label{label_type}_{cons_tag}.pth",
            lambda_cons=lambda_cons,
            device=device,
        )

        return {
            "global_result": result,
            "split_info": split_info,
        }

    # For subject_dependent and loso, run one experiment per subject
    for sid_num in range(1, max_subject_id + 1):
        sid = f"s{sid_num:02d}"

        print("\n" + "=" * 60)
        print(f"Running EEG experiment for {sid} | split_mode={split_mode}")
        print("=" * 60)

        try:

            train_pairs, val_pairs, split_info = get_shared_split(
                sid=sid,
                split_mode=split_mode,
            )

            print("split info:", split_info)

            # Train one EEG model using that split
            result = run_eeg_experiment_shared_split(
                dataset_path=dataset_path,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                model_name=model_name,
                mode=mode,
                label_type=label_type,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                verbose=verbose,
                save_path=ckpt_path(
                    f"{model_name}_{split_mode}_{sid}_mode{mode}_label{label_type}.pth"
                ),
                lambda_cons=lambda_cons,
                device=device,
            )

            results[sid] = {
                "best_val_acc": result["best_val_acc"],
                "last_val_acc": result["last_val_acc"],
                "save_path": result["save_path"],
                "history": result["history"],
                "detailed_metrics": result["detailed_metrics"],
                "split_info": split_info,
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
            }

        except Exception as e:
            # Continue even if one subject fails, so the whole sweep is not lost
            print(f"Skipping {sid}: {e}")

    # Compute mean best validation accuracy across valid subject runs
    valid_best = [
        v["best_val_acc"] for v in results.values()
        if v["best_val_acc"] is not None
    ]
    mean_best_acc = sum(valid_best) / len(valid_best) if len(valid_best) > 0 else 0.0

    print("\n" + "#" * 60)
    print("EEG RESULTS")
    print("#" * 60)
    for sid, info in results.items():
        print(f"{sid}: best_val_acc={info['best_val_acc']:.4f}")

    print(f"\nMean best accuracy: {mean_best_acc:.4f}")

    return {
        "subject_results": results,
        "mean_best_acc": mean_best_acc,
        "split_mode": split_mode,
    }
