from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..config import (
    NUM_FRAMES,
    BATCH_SIZE,
    NUM_WORKERS,
    TRAIN_RATIO,
    FREEZE_BACKBONE,
)

from ..data.eeg import get_data
from ..data.datasets import (
    create_labels_dict_from_eeg_and_images,
    DEAPFaceTrialDataset,
)
from ..data.splits import (
    get_shared_split,
    ckpt_path,
)
from ..models.video import ResNet18TrialModel

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU device.

    Returns:
        Tuple:
            - average training loss
            - average training accuracy
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a validation or test loader.

    Args:
        model: Model to evaluate.
        loader: Validation/test DataLoader.
        criterion: Loss function.
        device: CPU or GPU device.

    Returns:
        Tuple:
            - average loss
            - average accuracy
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate_video_detailed(model, loader, criterion, device):
    """
    Evaluate the video model on a loader and return detailed classification metrics.

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
def fit_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):
    """
    Full training loop with validation and best-checkpoint saving.

    At each epoch:
    - train on the training loader
    - evaluate on the validation loader
    - store metrics in `history`
    - save the model if validation accuracy improves

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or GPU device.
        epochs: Number of training epochs.
        save_path: Path where the best checkpoint is saved.

    Returns:
        history: Dictionary of train/validation metrics across epochs.
        best_val_acc: Best validation accuracy reached.
    """
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save only the best validation model.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return history, best_val_acc

# =========================
# MAIN EXPERIMENT
# =========================
def run_experiment(
    deap_root,
    image_root,
    train_pairs=None,
    val_pairs=None,
    split_mode="subject_dependent",
    target_sid=None,
    label_type=0,
    max_subject_id=22,
    num_frames=NUM_FRAMES,
    batch_size=BATCH_SIZE,
    epochs=10,
    lr=1e-3,
    freeze_backbone=FREEZE_BACKBONE,
    train_ratio=TRAIN_RATIO,
    num_workers=NUM_WORKERS,
    seed=42,
    save_path=None,
):
    """
    Run one video-only experiment.

    This function:
    - loads EEG labels
    - matches them to video/image trials
    - builds a shared train/validation split if not provided
    - creates datasets and dataloaders
    - trains a ResNet18 trial classifier
    - saves the best checkpoint

    Args:
        deap_root: Root path for EEG data.
        image_root: Root path for trial image folders.
        train_pairs: Optional externally provided training pairs.
        val_pairs: Optional externally provided validation pairs.
        split_mode: Split mode used when pairs are not provided.
        target_sid: Subject id used for subject-dependent or LOSO split.
        label_type: Label type (for example valence or arousal).
        max_subject_id: Maximum subject id to keep.
        num_frames: Number of frames used per trial.
        batch_size: Batch size.
        epochs: Number of epochs.
        lr: Learning rate.
        freeze_backbone: Whether to freeze the ResNet backbone.
        train_ratio: Train ratio for random-style splits.
        num_workers: Number of DataLoader workers.
        seed: Random seed.
        save_path: Optional custom checkpoint filename.

    Returns:
        Dictionary containing training history, best validation accuracy,
        split information, checkpoint path, and used train/val pairs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # -------------------------
    # Load labels from EEG side
    # -------------------------
    subjects = get_data(root=deap_root, mode=0, l=label_type)

    labels_dict = create_labels_dict_from_eeg_and_images(
        subjects=subjects,
        image_root=image_root,
        max_subject_id=max_subject_id
    )

    # ----------------------------------------
    # If shared pairs are not provided, build them
    # ----------------------------------------
    if train_pairs is None or val_pairs is None:
        train_pairs, val_pairs, split_info = get_shared_split(
            sid=target_sid,
            split_mode=split_mode,
        )
    else:
        split_info = {
            "split_mode": "external_shared_pairs",
            "target_sid": target_sid,
            "n_train_pairs": len(train_pairs),
            "n_val_pairs": len(val_pairs),
            "stratified": True,
        }

    # Use pretrained ResNet18 normalization / resize pipeline.
    weights = models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    # ----------------------------------------
    # Build datasets directly from shared pairs
    # ----------------------------------------
    train_dataset = DEAPFaceTrialDataset(
        root_dir=image_root,
        labels_dict=labels_dict,
        transform=transform,
        num_frames=num_frames,
        allowed_pairs=train_pairs,
    )

    val_dataset = DEAPFaceTrialDataset(
        root_dir=image_root,
        labels_dict=labels_dict,
        transform=transform,
        num_frames=num_frames,
        allowed_pairs=val_pairs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print("split info:", split_info)
    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))

    model = ResNet18TrialModel(
        num_classes=2,
        freeze_backbone=freeze_backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # If the backbone is frozen, train only the classifier layer.
    if freeze_backbone:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if save_path is None:
        save_path = f"best_resnet18_{split_mode}_{target_sid if target_sid else 'all'}_label{label_type}.pth"

    history, best_val_acc = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        save_path=save_path
    )

    model.load_state_dict(torch.load(save_path, map_location=device))

    detailed_metrics = evaluate_video_detailed(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Model saved to: {save_path}")

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "detailed_metrics": detailed_metrics,
        "split_info": split_info,
        "save_path": save_path,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
    }


def run_all_subject_dependent(
    deap_root,
    image_root,
    label_type=0,
    max_subject_id=22,
    train_ratio=TRAIN_RATIO,
    seed=42,
    **kwargs,
):
    """
    Run subject-dependent video experiments for all subjects.

    For each subject:
    - build a shared subject-dependent split
    - run one experiment
    - store the best validation accuracy and checkpoint path

    Args:
        deap_root: Root path for EEG data.
        image_root: Root path for video/image trials.
        label_type: Label type.
        max_subject_id: Maximum subject id to include.
        train_ratio: Train ratio for the subject-dependent split.
        seed: Random seed.
        **kwargs: Extra arguments forwarded to `run_experiment`.

    Returns:
        Dictionary containing:
        - per-subject results
        - mean validation accuracy across subjects
    """
    results = {}



    for sid_num in range(1, max_subject_id + 1):
        sid = f"s{sid_num:02d}"

        print("\n" + "=" * 60)
        print(f"Running subject-dependent experiment for {sid}")
        print("=" * 60)

        try:
            train_pairs, val_pairs, split_info = get_shared_split(
                sid=sid,
                split_mode="subject_dependent",
            )

            print("split info:", split_info)

            result = run_experiment(
                deap_root=deap_root,
                image_root=image_root,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                split_mode="subject_dependent",
                target_sid=sid,
                label_type=label_type,
                max_subject_id=max_subject_id,
                train_ratio=train_ratio,
                seed=seed,
                save_path=ckpt_path(
                    f"best_resnet18_subject_dependent_{sid}_label{label_type}.pth"
                ),
                **kwargs,
            )

            results[sid] = {
                "best_val_acc": result["best_val_acc"],
                "save_path": result["save_path"],
                "split_info": split_info,
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
                "history": result["history"],
                "detailed_metrics": result["detailed_metrics"],
            }

        except Exception as e:
            # Continue even if one subject fails.
            print(f"Skipping {sid}: {e}")

    if len(results) > 0:
        mean_acc = sum(v["best_val_acc"] for v in results.values()) / len(results)
    else:
        mean_acc = 0.0

    print("\n" + "#" * 60)
    print("SUBJECT RESULTS")
    print("#" * 60)
    for sid, info in results.items():
        print(f"{sid}: {info['best_val_acc']:.4f}")

    print(f"\nMean accuracy across subjects: {mean_acc:.4f}")

    return {
        "subject_results": results,
        "mean_acc": mean_acc,
    }

def load_video_model_checkpoint(
    checkpoint_path: str | Path,
    num_classes: int = 2,
    freeze_backbone: bool = False,
    freeze_bn_stats: bool | None = None,
    map_location: str | torch.device | None = None,
) -> nn.Module:
    """
    Rebuild a video model and load its saved state_dict.

    Args:
        checkpoint_path: Path to the saved .pth file.
        num_classes: Number of output classes.
        freeze_backbone: Whether the rebuilt model should freeze the backbone.
        freeze_bn_stats: Whether BatchNorm statistics should stay frozen.
        map_location: Device mapping for loading the checkpoint.

    Returns:
        Loaded video model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint_path)

    model = ResNet18TrialModel(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        freeze_bn_stats=freeze_bn_stats,
    )

    state_dict = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.eval()
    return model