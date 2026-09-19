import numpy as np
import pandas as pd
import torch
from torch import nn


from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..config import NUM_WORKERS, MAX_SUBJECT_ID

from ..data.datasets import create_fusion_dataloaders_from_pairs
from ..data.splits import get_shared_split, ckpt_path
from ..models.film import EEGVideoFiLMFusion
from ..models.concat import EEGVideoConcatFusion
from .eeg import load_model_checkpoint
from .video import load_video_model_checkpoint
from ..config  import TRAIN_RATIO

# =========================
# FUSION TRAIN / EVAL
# =========================

def train_one_epoch_fusion(model, loader, criterion, optimizer, device):
    """
    Train the fusion model for one epoch.

    Args:
        model: Fusion model
        loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device

    Returns:
        Average training loss and accuracy
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for eeg_x, video_x, y, _ in loader:
        eeg_x = eeg_x.to(device)
        video_x = video_x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(eeg_x, video_x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_fusion(model, loader, criterion, device):
    """
    Evaluate the fusion model on a validation loader.

    Args:
        model: Fusion model
        loader: Validation dataloader
        criterion: Loss function
        device: Torch device

    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for eeg_x, video_x, y, _ in loader:
        eeg_x = eeg_x.to(device)
        video_x = video_x.to(device)
        y = y.to(device)

        logits = model(eeg_x, video_x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples



def fit_fusion_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    save_path,
    patience=None,
):
    """
    Train the fusion model with validation tracking and optional early stopping.

    The best checkpoint is selected primarily by validation accuracy, and in case
    of a tie, by lower validation loss.

    Args:
        model: Fusion model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Torch device
        epochs: Number of epochs
        save_path: Path where the best model is saved
        patience: Optional early stopping patience

    Returns:
        history: Training history dictionary
        best_val_acc: Best validation accuracy
        best_val_loss: Best validation loss associated with the selected checkpoint
    """
    best_val_acc = -1.0
    best_val_loss = float("inf")
    wait = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_fusion(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_fusion(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save a new checkpoint if validation performance improves.
        improved = (
            (val_acc > best_val_acc) or
            (val_acc == best_val_acc and val_loss < best_val_loss)
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        # Stop early if patience is used and there is no improvement.
        if patience is not None and wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history, best_val_acc, best_val_loss


@torch.no_grad()
def evaluate_fusion_detailed(model, loader, criterion, device):
    """
    Evaluate the fusion model and return detailed classification metrics.

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

    for eeg_x, video_x, y, _ in loader:
        eeg_x = eeg_x.to(device)
        video_x = video_x.to(device)
        y = y.to(device)

        logits = model(eeg_x, video_x)
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

# =========================================================
# FiLM PARAMETER ANALYSIS
# =========================================================

@torch.no_grad()
def collect_film_modulation_stats(
    model,
    loader,
    criterion,
    device,
    split_name="val",
    selected_stage=None,
):
    """
    Collect gamma, beta, gate, and modulation-strength statistics
    for the EEG-guided FiLM model.

    This uses:
        logits, feats = model(eeg_x, video_x, return_features=True)

    Expected feats:
        gamma, beta, gate, video_z, video_mod
    """
    model.eval()

    rows = []

    for eeg_x, video_x, y, meta_list in loader:
        eeg_x = eeg_x.to(device)
        video_x = video_x.to(device)
        y = y.to(device)

        logits, feats = model(eeg_x, video_x, return_features=True)

        loss = criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        gamma = feats["gamma"].detach()
        beta = feats["beta"].detach()
        gate = feats["gate"].detach()
        video_z = feats["video_z"].detach()
        video_mod = feats["video_mod"].detach()

        modulation_norm = (video_mod - video_z).norm(dim=1)
        video_norm = video_z.norm(dim=1)
        relative_modulation = modulation_norm / (video_norm + 1e-8)

        for i in range(y.size(0)):
            meta = meta_list[i] if meta_list is not None else {}

            row = {
                "split": split_name,
                "selected_stage": selected_stage,

                "sid": meta.get("sid", None),
                "trial": meta.get("trial", None),

                "label": int(y[i].item()),
                "pred": int(preds[i].item()),
                "correct": int(preds[i].item() == y[i].item()),

                "prob_0": float(probs[i, 0].item()),
                "prob_1": float(probs[i, 1].item()),
                "loss": float(loss.item()),

                # Gamma statistics
                "gamma_mean": float(gamma[i].mean().item()),
                "gamma_std": float(gamma[i].std().item()),
                "gamma_min": float(gamma[i].min().item()),
                "gamma_max": float(gamma[i].max().item()),
                "gamma_abs_mean": float(gamma[i].abs().mean().item()),

                # Since your gamma is identity-centered:
                # gamma = 1 + 0.1 * tanh(gamma_raw)
                "gamma_deviation_from_1": float((gamma[i] - 1.0).abs().mean().item()),

                # Beta statistics
                "beta_mean": float(beta[i].mean().item()),
                "beta_std": float(beta[i].std().item()),
                "beta_min": float(beta[i].min().item()),
                "beta_max": float(beta[i].max().item()),
                "beta_abs_mean": float(beta[i].abs().mean().item()),

                # Gate statistics
                "gate_mean": float(gate[i].mean().item()),
                "gate_std": float(gate[i].std().item()),
                "gate_min": float(gate[i].min().item()),
                "gate_max": float(gate[i].max().item()),

                # How much FiLM changed the video embedding
                "video_norm": float(video_norm[i].item()),
                "modulation_norm": float(modulation_norm[i].item()),
                "relative_modulation": float(relative_modulation[i].item()),
            }

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_film_modulation_stats(film_stats_df):
    """
    Compact summary of FiLM behavior.
    """
    if film_stats_df is None or len(film_stats_df) == 0:
        return pd.DataFrame()

    cols = [
        "gamma_mean",
        "gamma_std",
        "gamma_deviation_from_1",
        "beta_mean",
        "beta_std",
        "beta_abs_mean",
        "gate_mean",
        "gate_std",
        "relative_modulation",
        "correct",
    ]

    existing_cols = [c for c in cols if c in film_stats_df.columns]

    summary = film_stats_df[existing_cols].agg(["mean", "std", "min", "max"]).T
    summary = summary.reset_index().rename(columns={"index": "statistic"})

    return summary
# =========================
# FUSION OPTIMIZERS
# =========================

def build_fusion_optimizer_stage1(model, lr=1e-3, weight_decay=1e-5):
    """
    Build the optimizer for fusion stage 1.

    Stage 1 trains only:
        - FiLM generator
        - fusion classifier

    The encoders are expected to already be frozen before this is used.

    Args:
        model: Fusion model
        lr: Learning rate
        weight_decay: Weight decay

    Returns:
        Adam optimizer
    """
    params = (
        list(model.film.parameters()) +
        list(model.gate.parameters()) +
        list(model.classifier.parameters())
    )
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def build_fusion_optimizer_stage2(
    model,
    head_lr=1e-4,
    encoder_lr=1e-5,
    weight_decay=1e-5,
):
    """
    Build the optimizer for stage 2 fine-tuning.

    In this stage the fusion blocks usually keep a larger learning rate,
    while the encoders use a smaller learning rate.

    Args:
        model: Fusion model
        head_lr: Learning rate for fusion head / FiLM parts
        encoder_lr: Learning rate for encoder parameters
        weight_decay: Weight decay

    Returns:
        Adam optimizer with parameter groups
    """
    return torch.optim.Adam(
        [
            {"params": model.film.parameters(), "lr": head_lr},
            {"params": model.gate.parameters(), "lr": head_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
            {"params": model.eeg_encoder.parameters(), "lr": encoder_lr},
            {"params": model.video_encoder.parameters(), "lr": encoder_lr},
        ],
        weight_decay=weight_decay,
    )

# =========================
# RUN FUSION EXPERIMENT
# =========================

def run_fusion_experiment_shared_split(
    eeg_root,
    image_root,
    train_pairs,
    val_pairs,
    eeg_checkpoint_path,
    video_checkpoint_path,
    label_type=0,
    eeg_mode=2,
    num_frames=18,
    batch_size=4,
    stage1_epochs=5,
    stage2_epochs=3,
    stage1_lr=1e-3,
    stage2_head_lr=1e-4,
    stage2_encoder_lr=5e-6,
    weight_decay=1e-5,
    save_path_stage1="fusion_stage1_best.pth",
    save_path_stage2="fusion_stage2_best.pth",
    device=None,
):
    """
    Run one full fusion experiment using a shared train/validation split.

    Workflow:
    1) Build aligned fusion dataloaders from shared pairs
    2) Load pretrained EEG and video encoders
    3) Build the FiLM-based fusion model
    4) Stage 1:
         - freeze encoders
         - train FiLM + classifier only
    5) Stage 2 (optional):
         - start from best stage 1 model
         - keep EEG frozen
         - unfreeze only video layer4
         - fine-tune gently

    Args:
        eeg_root: EEG root directory
        image_root: Video/image root directory
        train_pairs: Shared training pairs
        val_pairs: Shared validation pairs
        eeg_checkpoint_path: Path to pretrained EEG model
        video_checkpoint_path: Path to pretrained video model
        label_type: Label type
        eeg_mode: EEG mode
        num_frames: Number of video frames per trial
        batch_size: Batch size
        stage1_epochs: Number of epochs for stage 1
        stage2_epochs: Number of epochs for stage 2
        stage1_lr: Learning rate for stage 1
        stage2_head_lr: Fusion head learning rate in stage 2
        stage2_encoder_lr: Encoder learning rate in stage 2
        weight_decay: Weight decay
        save_path_stage1: Best checkpoint path for stage 1
        save_path_stage2: Best checkpoint path for stage 2

    Returns:
        Dictionary containing histories, best scores, dimensions, and detailed metrics
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    # -------------------------
    # Build paired fusion loaders
    # -------------------------
    fusion_train_loader, fusion_val_loader, fusion_train_ds, fusion_val_ds = create_fusion_dataloaders_from_pairs(
        eeg_root=eeg_root,
        image_root=image_root,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        eeg_mode=eeg_mode,
        label_type=label_type,
        num_frames=num_frames,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        max_subject_id=MAX_SUBJECT_ID,
    )

    print("Fusion train size:", len(fusion_train_ds))
    print("Fusion val size:", len(fusion_val_ds))

    # -------------------------
    # Infer EEG input feature_dim
    # -------------------------
    eeg_x0, video_x0, y0, meta0 = fusion_train_ds[0]
    eeg_feature_dim = eeg_x0.shape[-1]   # mode 2 -> (Nwin, F)

    # -------------------------
    # Load pretrained encoders
    # -------------------------
    eeg_model = load_model_checkpoint(
        checkpoint_path=eeg_checkpoint_path,
        mode=eeg_mode,
        feature_dim=eeg_feature_dim,
        num_channels=32,
        num_classes=2,
        map_location=device,
    )

    video_model = load_video_model_checkpoint(
        checkpoint_path=video_checkpoint_path,
        num_classes=2,
        freeze_backbone=False,
        freeze_bn_stats=True,
        map_location=device,
    )

    eeg_model = eeg_model.to(device)
    video_model = video_model.to(device)

    # Get embedding dimensions directly from the loaded models.
    if hasattr(eeg_model, "d_model"):
        eeg_dim = eeg_model.d_model
    elif hasattr(eeg_model, "emb_dim"):
        eeg_dim = eeg_model.emb_dim
    else:
        raise ValueError("EEG model must expose d_model or emb_dim")
    video_dim = video_model.feat_dim

    fusion_model = EEGVideoFiLMFusion(
        eeg_encoder=eeg_model,
        video_encoder=video_model,
        eeg_dim=eeg_dim,
        video_dim=video_dim,
        num_classes=2,
        film_hidden_dim=256,
        head_hidden_dim=256,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # Stage 1: freeze encoders, train FiLM + head only
    # ==================================================
    fusion_model.freeze_encoders()

    optimizer_stage1 = build_fusion_optimizer_stage1(
        model=fusion_model,
        lr=stage1_lr,
        weight_decay=weight_decay,
    )

    print("\n" + "=" * 60)
    print("FUSION STAGE 1: train FiLM + classifier only")
    print("=" * 60)

    history_stage1, best_val_acc_stage1, best_val_loss_stage1 = fit_fusion_model(
        model=fusion_model,
        train_loader=fusion_train_loader,
        val_loader=fusion_val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        device=device,
        epochs=stage1_epochs,
        save_path=save_path_stage1,
        patience=None,
    )

    print(f"Best stage1 val acc: {best_val_acc_stage1:.4f}")
    print(f"Saved stage1 model to: {save_path_stage1}")

    # Load the best stage 1 checkpoint before evaluating it in detail.
    fusion_model.load_state_dict(torch.load(save_path_stage1, map_location=device))
    stage1_metrics = evaluate_fusion_detailed(
        fusion_model,
        fusion_val_loader,
        criterion,
        device,
    )
    print("Stage 1 metrics:", stage1_metrics)

    # Load best stage 1 checkpoint again before stage 2 fine-tuning.
    fusion_model.load_state_dict(torch.load(save_path_stage1, map_location=device))

    # ==========================================
    # Stage 2: gentle fine-tuning
    # - keep EEG frozen
    # - unfreeze only video layer4
    # - keep BN stats frozen
    # ==========================================
    history_stage2 = None
    best_val_acc_stage2 = None
    best_val_loss_stage2 = None
    stage2_metrics = None

    if stage2_epochs > 0:
        # Start stage 2 from the best stage 1 checkpoint.
        fusion_model.load_state_dict(torch.load(save_path_stage1, map_location=device))

        # Freeze everything first, then selectively unfreeze only what we want.
        fusion_model.freeze_encoders()

        # Keep video BatchNorm statistics stable during fine-tuning.
        fusion_model.video_encoder.freeze_bn_stats = True

        # Unfreeze only the last visual block for a gentle adaptation.
        for p in fusion_model.video_encoder.backbone.layer4.parameters():
            p.requires_grad = True

        optimizer_stage2 = build_fusion_optimizer_stage2(
            model=fusion_model,
            head_lr=stage2_head_lr,
            encoder_lr=stage2_encoder_lr,
            weight_decay=weight_decay,
        )

        print("\n" + "=" * 60)
        print("FUSION STAGE 2: gentle fine-tuning (video layer4 only)")
        print("=" * 60)

        history_stage2, best_val_acc_stage2, best_val_loss_stage2 = fit_fusion_model(
            model=fusion_model,
            train_loader=fusion_train_loader,
            val_loader=fusion_val_loader,
            criterion=criterion,
            optimizer=optimizer_stage2,
            device=device,
            epochs=stage2_epochs,
            save_path=save_path_stage2,
            patience=None,
        )

        print(f"Best stage2 val acc: {best_val_acc_stage2:.4f}")
        print(f"Saved stage2 model to: {save_path_stage2}")

        # Load best stage 2 checkpoint before final detailed evaluation.
        fusion_model.load_state_dict(torch.load(save_path_stage2, map_location=device))
        stage2_metrics = evaluate_fusion_detailed(
            fusion_model,
            fusion_val_loader,
            criterion,
            device,
        )
        print("Stage 2 metrics:", stage2_metrics)

    # ------------------------------------------
    # Select final stage for standardized output
    # ------------------------------------------
    if best_val_acc_stage2 is not None and best_val_acc_stage2 > best_val_acc_stage1:
        final_stage = "stage2"
        final_history = history_stage2
        final_metrics = stage2_metrics
        final_best_val_acc = best_val_acc_stage2
        final_save_path = save_path_stage2
    else:
        final_stage = "stage1"
        final_history = history_stage1
        final_metrics = stage1_metrics
        final_best_val_acc = best_val_acc_stage1
        final_save_path = save_path_stage1


    # ==================================================
    # FiLM gamma / beta / gate analysis on final model
    # ==================================================
    fusion_model.load_state_dict(torch.load(final_save_path, map_location=device))

    film_stats_df = collect_film_modulation_stats(
        model=fusion_model,
        loader=fusion_val_loader,
        criterion=criterion,
        device=device,
        split_name="val",
        selected_stage=final_stage,
    )

    film_summary_df = summarize_film_modulation_stats(film_stats_df)

    print("\nFiLM modulation summary:")
    print(film_summary_df)



    return {
        # standardized final view
        "history": final_history,
        "detailed_metrics": final_metrics,
        "best_val_acc": final_best_val_acc,
        "save_path": final_save_path,
        "selected_stage": final_stage,

        # FiLM interpretability analysis
        "film_stats": film_stats_df,
        "film_summary": film_summary_df,

        # keep stage-specific info too
        "stage1_history": history_stage1,
        "stage1_best_val_acc": best_val_acc_stage1,
        "stage1_save_path": save_path_stage1,
        "stage1_metrics": stage1_metrics,

        "stage2_history": history_stage2,
        "stage2_best_val_acc": best_val_acc_stage2,
        "stage2_save_path": save_path_stage2 if stage2_epochs > 0 else None,
        "stage2_metrics": stage2_metrics,

        # useful dimensions
        "eeg_feature_dim": eeg_feature_dim,
        "eeg_dim": eeg_dim,
        "video_dim": video_dim,
    }


# =========================
# RUN FUSION FOR ALL SUBJECTS
# =========================

def run_all_fusion_shared_split(
    eeg_root,
    image_root,

    eeg_results=None,
    video_results=None,

    label_type=0,
    eeg_mode=2,
    max_subject_id=22,
    split_mode="subject_dependent",
    train_ratio=TRAIN_RATIO,
    seed=42,

    num_frames=18,
    batch_size=4,

    stage1_epochs=5,
    stage2_epochs=3,
    stage1_lr=1e-3,
    stage2_head_lr=1e-4,
    stage2_encoder_lr=5e-6,
    weight_decay=1e-5,
):
    """
    Run fusion experiments for either:
    - one global random_all split
    - or one experiment per subject for subject_dependent / loso

    Expected inputs:
        - eeg_results: output of EEG shared-split training
        - video_results: output of video shared-split training

    Args:
        eeg_root: EEG root directory
        image_root: Video/image root directory
        eeg_results: EEG experiment output
        video_results: Video experiment output
        label_type: Label type
        eeg_mode: EEG mode
        max_subject_id: Maximum subject id
        split_mode: "subject_dependent", "loso", or "random_all"
        train_ratio: Train ratio for random-style splits
        seed: Random seed
        num_frames: Number of video frames per trial
        batch_size: Batch size
        stage1_epochs: Fusion stage 1 epochs
        stage2_epochs: Fusion stage 2 epochs
        stage1_lr: Learning rate for stage 1
        stage2_head_lr: Head learning rate for stage 2
        stage2_encoder_lr: Encoder learning rate for stage 2
        weight_decay: Weight decay

    Returns:
        Global or per-subject fusion results depending on split_mode
    """
    if eeg_results is None:
        raise ValueError("eeg_results is required (use the output of run_all_eeg_shared_split)")
    if video_results is None:
        raise ValueError("video_results is required (use the output of run_all_subject_dependent)")



    # -------------------------------------------------
    # random_all = one single global fusion experiment
    # -------------------------------------------------
    if split_mode == "random_all":
        if "global_result" not in eeg_results:
            raise ValueError("For split_mode='random_all', eeg_results must come from random_all EEG training.")
        if "global_result" not in video_results:
            raise ValueError("For split_mode='random_all', video_results must come from random_all video training.")

        train_pairs, val_pairs, split_info = get_shared_split(
            sid=None,
            split_mode="random_all",
        )
        fusion_result = run_fusion_experiment_shared_split(
            eeg_root=eeg_root,
            image_root=image_root,
            train_pairs=train_pairs,
            val_pairs=val_pairs,

            eeg_checkpoint_path=eeg_results["global_result"]["save_path"],
            video_checkpoint_path=video_results["global_result"]["save_path"],

            label_type=label_type,
            eeg_mode=eeg_mode,
            num_frames=num_frames,
            batch_size=batch_size,

            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            stage1_lr=stage1_lr,
            stage2_head_lr=stage2_head_lr,
            stage2_encoder_lr=stage2_encoder_lr,
            weight_decay=weight_decay,

            save_path_stage1=f"fusion_random_all_stage1_label{label_type}.pth",
            save_path_stage2=f"fusion_random_all_stage2_label{label_type}.pth",
        )

        return {
            "global_result": fusion_result,
            "split_info": split_info,
        }

    # -------------------------------------------------
    # subject_dependent / loso = one fusion per subject
    # -------------------------------------------------
    results = {}

    for sid_num in range(1, max_subject_id + 1):
        sid = f"s{sid_num:02d}"

        print("\n" + "=" * 60)
        print(f"Running FUSION for {sid} | split_mode={split_mode}")
        print("=" * 60)

        try:
            train_pairs, val_pairs, split_info = get_shared_split(
                sid=sid,
                split_mode=split_mode,
            )

            # Retrieve the subject-specific EEG checkpoint.
            eeg_subject_results = eeg_results.get("subject_results", {})
            if sid not in eeg_subject_results:
                raise ValueError(f"Missing EEG checkpoint info for {sid}")

            eeg_ckpt = eeg_subject_results[sid]["save_path"]

            # Retrieve the subject-specific video checkpoint.
            video_subject_results = video_results.get("subject_results", {})
            if sid not in video_subject_results:
                raise ValueError(f"Missing video checkpoint info for {sid}")

            video_ckpt = video_subject_results[sid]["save_path"]

            fusion_result = run_fusion_experiment_shared_split(
                eeg_root=eeg_root,
                image_root=image_root,
                train_pairs=train_pairs,
                val_pairs=val_pairs,

                eeg_checkpoint_path=eeg_ckpt,
                video_checkpoint_path=video_ckpt,

                label_type=label_type,
                eeg_mode=eeg_mode,
                num_frames=num_frames,
                batch_size=batch_size,

                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                stage1_lr=stage1_lr,
                stage2_head_lr=stage2_head_lr,
                stage2_encoder_lr=stage2_encoder_lr,
                weight_decay=weight_decay,

                save_path_stage1=ckpt_path(
                    f"fusion_{split_mode}_{sid}_stage1_label{label_type}.pth"
                ),
                save_path_stage2=ckpt_path(
                    f"fusion_{split_mode}_{sid}_stage2_label{label_type}.pth"
                ),
            )

            results[sid] = {
                "best_val_acc": fusion_result["best_val_acc"],
                "save_path": fusion_result["save_path"],
                "selected_stage": fusion_result["selected_stage"],
                "history": fusion_result["history"],
                "detailed_metrics": fusion_result["detailed_metrics"],

                "stage1_best_val_acc": fusion_result["stage1_best_val_acc"],
                "stage2_best_val_acc": fusion_result["stage2_best_val_acc"],
                "stage1_history": fusion_result["stage1_history"],
                "stage2_history": fusion_result["stage2_history"],
                "stage1_metrics": fusion_result["stage1_metrics"],
                "stage2_metrics": fusion_result["stage2_metrics"],

                # FiLM interpretability analysis
                "film_stats": fusion_result.get("film_stats"),
                "film_summary": fusion_result.get("film_summary"),

                "split_info": split_info,
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
            }
        except Exception as e:
            # Continue with the next subject even if one run fails.
            print(f"Skipping {sid}: {e}")

    valid_best = [
        v["best_val_acc"] for v in results.values()
        if v["best_val_acc"] is not None
    ]
    mean_best_acc = sum(valid_best) / len(valid_best) if len(valid_best) > 0 else 0.0

    print("\n" + "#" * 60)
    print("FUSION RESULTS")
    print("#" * 60)
    for sid, info in results.items():
        print(f"{sid}: best_val_acc={info['best_val_acc']:.4f}")

    print(f"\nMean best fusion accuracy: {mean_best_acc:.4f}")

    return {
        "subject_results": results,
        "mean_best_acc": mean_best_acc,
        "split_mode": split_mode,
    }

from ..models.concat import EEGVideoConcatFusion
def build_concat_optimizer_stage1(model, lr=1e-3, weight_decay=1e-5):
    """
    Stage 1 optimizer.

    Only trains the concat classifier.
    Encoders are frozen before this optimizer is created.
    """
    return torch.optim.Adam(
        model.classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


def build_concat_optimizer_stage2(
    model,
    head_lr=1e-4,
    encoder_lr=5e-6,
    weight_decay=1e-5,
):
    """
    Stage 2 optimizer.

    Uses different learning rates:
    - classifier head: higher LR
    - unfrozen encoder part: lower LR

    Only parameters with requires_grad=True are optimized.
    """

    head_params = [
        p for p in model.classifier.parameters()
        if p.requires_grad
    ]

    encoder_params = [
        p for p in list(model.eeg_encoder.parameters()) + list(model.video_encoder.parameters())
        if p.requires_grad
    ]

    param_groups = []

    if len(head_params) > 0:
        param_groups.append({
            "params": head_params,
            "lr": head_lr,
        })

    if len(encoder_params) > 0:
        param_groups.append({
            "params": encoder_params,
            "lr": encoder_lr,
        })

    return torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
    )

# =========================================================
# SINGLE CONCAT FUSION EXPERIMENT
# =========================================================

def run_concat_fusion_experiment_shared_split(
    eeg_root,
    image_root,
    train_pairs,
    val_pairs,
    eeg_checkpoint_path,
    video_checkpoint_path,
    label_type=0,
    eeg_mode=2,
    num_frames=18,
    batch_size=4,
    stage1_epochs=15,
    stage2_epochs=5,
    stage1_lr=1e-3,
    stage2_head_lr=1e-4,
    stage2_encoder_lr=5e-6,
    weight_decay=1e-5,
    save_path_stage1="concat_fusion_stage1_best.pth",
    save_path_stage2="concat_fusion_stage2_best.pth",
    device=None,
):
    """
    Run one feature-level concatenation fusion experiment.

    This mirrors run_fusion_experiment_shared_split(...), but replaces FiLM with
    simple feature concatenation.

    For fair comparison:
    - same EEG/video train/val pairs
    - same EEG checkpoint
    - same video checkpoint
    - same two-stage training idea
    - same evaluation function
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # -------------------------
    # Build paired fusion loaders
    # -------------------------
    fusion_train_loader, fusion_val_loader, fusion_train_ds, fusion_val_ds = create_fusion_dataloaders_from_pairs(
        eeg_root=eeg_root,
        image_root=image_root,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        eeg_mode=eeg_mode,
        label_type=label_type,
        num_frames=num_frames,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        max_subject_id=MAX_SUBJECT_ID,
    )

    print("Concat fusion train size:", len(fusion_train_ds))
    print("Concat fusion val size:", len(fusion_val_ds))

    # -------------------------
    # Infer EEG feature dimension
    # -------------------------
    eeg_x0, video_x0, y0, meta0 = fusion_train_ds[0]
    eeg_feature_dim = eeg_x0.shape[-1]

    # -------------------------
    # Load pretrained EEG encoder
    # -------------------------
    eeg_model = load_model_checkpoint(
        checkpoint_path=eeg_checkpoint_path,
        mode=eeg_mode,
        feature_dim=eeg_feature_dim,
        num_channels=32,
        num_classes=2,
        map_location=device,
    )

    # -------------------------
    # Load pretrained video encoder
    # -------------------------
    video_model = load_video_model_checkpoint(
        checkpoint_path=video_checkpoint_path,
        num_classes=2,
        freeze_backbone=False,
        freeze_bn_stats=True,
        map_location=device,
    )

    eeg_model = eeg_model.to(device)
    video_model = video_model.to(device)

    # -------------------------
    # Infer embedding dimensions
    # -------------------------
    if hasattr(eeg_model, "d_model"):
        eeg_dim = eeg_model.d_model
    elif hasattr(eeg_model, "emb_dim"):
        eeg_dim = eeg_model.emb_dim
    else:
        raise ValueError("EEG model must expose d_model or emb_dim")

    if not hasattr(video_model, "feat_dim"):
        raise ValueError("Video model must expose feat_dim")

    video_dim = video_model.feat_dim

    # -------------------------
    # Build concat fusion model
    # -------------------------
    concat_model = EEGVideoConcatFusion(
        eeg_encoder=eeg_model,
        video_encoder=video_model,
        eeg_dim=eeg_dim,
        video_dim=video_dim,
        num_classes=2,
        head_hidden_dim=256,
        dropout=0.3,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # Stage 1: freeze encoders, train concat classifier
    # ==================================================
    concat_model.freeze_encoders()

    optimizer_stage1 = build_concat_optimizer_stage1(
        model=concat_model,
        lr=stage1_lr,
        weight_decay=weight_decay,
    )

    print("\n" + "=" * 60)
    print("CONCAT FUSION STAGE 1: train classifier only")
    print("=" * 60)

    history_stage1, best_val_acc_stage1, best_val_loss_stage1 = fit_fusion_model(
        model=concat_model,
        train_loader=fusion_train_loader,
        val_loader=fusion_val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        device=device,
        epochs=stage1_epochs,
        save_path=save_path_stage1,
        patience=None,
    )

    print(f"Best concat stage1 val acc: {best_val_acc_stage1:.4f}")
    print(f"Saved concat stage1 model to: {save_path_stage1}")

    concat_model.load_state_dict(torch.load(save_path_stage1, map_location=device))

    stage1_metrics = evaluate_fusion_detailed(
        concat_model,
        fusion_val_loader,
        criterion,
        device,
    )

    print("Concat stage 1 metrics:", stage1_metrics)

    # ==================================================
    # Stage 2: gentle fine-tuning
    # Same spirit as FiLM stage 2:
    # keep EEG frozen, unfreeze only video layer4
    # ==================================================
    history_stage2 = None
    best_val_acc_stage2 = None
    best_val_loss_stage2 = None
    stage2_metrics = None

    if stage2_epochs > 0:
        concat_model.load_state_dict(torch.load(save_path_stage1, map_location=device))

        concat_model.freeze_encoders()

        # Keep video BN stable if your video model supports it
        if hasattr(concat_model.video_encoder, "freeze_bn_stats"):
            concat_model.video_encoder.freeze_bn_stats = True

        # Unfreeze only video layer4, like your FiLM implementation
        if hasattr(concat_model.video_encoder, "backbone") and hasattr(concat_model.video_encoder.backbone, "layer4"):
            for p in concat_model.video_encoder.backbone.layer4.parameters():
                p.requires_grad = True
        else:
            print("Warning: video_encoder.backbone.layer4 not found. Stage 2 will train classifier only.")

        optimizer_stage2 = build_concat_optimizer_stage2(
            model=concat_model,
            head_lr=stage2_head_lr,
            encoder_lr=stage2_encoder_lr,
            weight_decay=weight_decay,
        )

        print("\n" + "=" * 60)
        print("CONCAT FUSION STAGE 2: gentle fine-tuning")
        print("=" * 60)

        history_stage2, best_val_acc_stage2, best_val_loss_stage2 = fit_fusion_model(
            model=concat_model,
            train_loader=fusion_train_loader,
            val_loader=fusion_val_loader,
            criterion=criterion,
            optimizer=optimizer_stage2,
            device=device,
            epochs=stage2_epochs,
            save_path=save_path_stage2,
            patience=None,
        )

        print(f"Best concat stage2 val acc: {best_val_acc_stage2:.4f}")
        print(f"Saved concat stage2 model to: {save_path_stage2}")

        concat_model.load_state_dict(torch.load(save_path_stage2, map_location=device))

        stage2_metrics = evaluate_fusion_detailed(
            concat_model,
            fusion_val_loader,
            criterion,
            device,
        )

        print("Concat stage 2 metrics:", stage2_metrics)

    # ==================================================
    # Final stage selection
    # Accuracy first, validation loss as tie-breaker
    # ==================================================
    stage2_better = False

    if best_val_acc_stage2 is not None:
        stage2_better = (
            (best_val_acc_stage2 > best_val_acc_stage1)
            or (
                best_val_acc_stage2 == best_val_acc_stage1
                and best_val_loss_stage2 < best_val_loss_stage1
            )
        )

    if stage2_better:
        final_stage = "stage2"
        final_history = history_stage2
        final_metrics = stage2_metrics
        final_best_val_acc = best_val_acc_stage2
        final_best_val_loss = best_val_loss_stage2
        final_save_path = save_path_stage2
    else:
        final_stage = "stage1"
        final_history = history_stage1
        final_metrics = stage1_metrics
        final_best_val_acc = best_val_acc_stage1
        final_best_val_loss = best_val_loss_stage1
        final_save_path = save_path_stage1

    return {
        "history": final_history,
        "detailed_metrics": final_metrics,
        "best_val_acc": final_best_val_acc,
        "best_val_loss": final_best_val_loss,
        "save_path": final_save_path,
        "selected_stage": final_stage,

        "stage1_history": history_stage1,
        "stage1_best_val_acc": best_val_acc_stage1,
        "stage1_best_val_loss": best_val_loss_stage1,
        "stage1_save_path": save_path_stage1,
        "stage1_metrics": stage1_metrics,

        "stage2_history": history_stage2,
        "stage2_best_val_acc": best_val_acc_stage2,
        "stage2_best_val_loss": best_val_loss_stage2,
        "stage2_save_path": save_path_stage2 if stage2_epochs > 0 else None,
        "stage2_metrics": stage2_metrics,

        "eeg_feature_dim": eeg_feature_dim,
        "eeg_dim": eeg_dim,
        "video_dim": video_dim,
    }

# =========================================================
# ALL-SUBJECT CONCAT FUSION RUNNER
# =========================================================

def run_all_concat_fusion_shared_split(
    eeg_root,
    image_root,
    eeg_results=None,
    video_results=None,
    label_type=0,
    eeg_mode=2,
    max_subject_id=22,
    split_mode="subject_dependent",
    train_ratio=TRAIN_RATIO,
    seed=42,
    num_frames=18,
    batch_size=4,
    stage1_epochs=15,
    stage2_epochs=5,
    stage1_lr=1e-3,
    stage2_head_lr=1e-4,
    stage2_encoder_lr=5e-6,
    weight_decay=1e-5,
):
    """
    Run feature-level concatenation fusion for all subjects.

    Same external structure as run_all_fusion_shared_split(...), so the output
    can be passed directly to standardize_fusion_results(...).
    """

    if eeg_results is None:
        raise ValueError("eeg_results is required.")

    if video_results is None:
        raise ValueError("video_results is required.")

    results = {}



    # ---------------------------------------------------------
    # random_all case
    # ---------------------------------------------------------
    if split_mode == "random_all":
        train_pairs, val_pairs, split_info = get_shared_split(
            sid=None,
            split_mode="random_all",
        )

        eeg_ckpt = eeg_results["global_result"]["save_path"]
        video_ckpt = video_results["global_result"]["save_path"]

        concat_result = run_concat_fusion_experiment_shared_split(
            eeg_root=eeg_root,
            image_root=image_root,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            eeg_checkpoint_path=eeg_ckpt,
            video_checkpoint_path=video_ckpt,
            label_type=label_type,
            eeg_mode=eeg_mode,
            num_frames=num_frames,
            batch_size=batch_size,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
            stage1_lr=stage1_lr,
            stage2_head_lr=stage2_head_lr,
            stage2_encoder_lr=stage2_encoder_lr,
            weight_decay=weight_decay,
            save_path_stage1=ckpt_path(
                f"concat_fusion_random_all_stage1_label{label_type}.pth"
            ),
            save_path_stage2=ckpt_path(
                f"concat_fusion_random_all_stage2_label{label_type}.pth"
            ),
        )

        return {
            "global_result": concat_result,
            "split_info": split_info,
            "split_mode": split_mode,
        }

    # ---------------------------------------------------------
    # subject_dependent / loso case
    # ---------------------------------------------------------
    for sid_num in range(1, max_subject_id + 1):
        sid = f"s{sid_num:02d}"

        print("\n" + "=" * 60)
        print(f"Running CONCAT FUSION for {sid} | split_mode={split_mode}")
        print("=" * 60)

        try:
            train_pairs, val_pairs, split_info = get_shared_split(
                sid=sid,
                split_mode=split_mode,
            )

            eeg_ckpt = eeg_results["subject_results"][sid]["save_path"]
            video_ckpt = video_results["subject_results"][sid]["save_path"]

            concat_result = run_concat_fusion_experiment_shared_split(
                eeg_root=eeg_root,
                image_root=image_root,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                eeg_checkpoint_path=eeg_ckpt,
                video_checkpoint_path=video_ckpt,
                label_type=label_type,
                eeg_mode=eeg_mode,
                num_frames=num_frames,
                batch_size=batch_size,
                stage1_epochs=stage1_epochs,
                stage2_epochs=stage2_epochs,
                stage1_lr=stage1_lr,
                stage2_head_lr=stage2_head_lr,
                stage2_encoder_lr=stage2_encoder_lr,
                weight_decay=weight_decay,
                save_path_stage1=ckpt_path(
                    f"concat_fusion_{split_mode}_{sid}_stage1_label{label_type}.pth"
                ),
                save_path_stage2=ckpt_path(
                    f"concat_fusion_{split_mode}_{sid}_stage2_label{label_type}.pth"
                ),
            )

            results[sid] = {
                "best_val_acc": concat_result["best_val_acc"],
                "best_val_loss": concat_result["best_val_loss"],
                "save_path": concat_result["save_path"],
                "selected_stage": concat_result["selected_stage"],
                "history": concat_result["history"],
                "detailed_metrics": concat_result["detailed_metrics"],

                "stage1_best_val_acc": concat_result["stage1_best_val_acc"],
                "stage1_best_val_loss": concat_result["stage1_best_val_loss"],
                "stage1_history": concat_result["stage1_history"],
                "stage1_metrics": concat_result["stage1_metrics"],

                "stage2_best_val_acc": concat_result["stage2_best_val_acc"],
                "stage2_best_val_loss": concat_result["stage2_best_val_loss"],
                "stage2_history": concat_result["stage2_history"],
                "stage2_metrics": concat_result["stage2_metrics"],

                "split_info": split_info,
                "train_pairs": train_pairs,
                "val_pairs": val_pairs,
            }

        except Exception as e:
            print(f"Skipping {sid}: {e}")

    valid_best = [
        v["best_val_acc"]
        for v in results.values()
        if v["best_val_acc"] is not None
    ]

    mean_best_acc = sum(valid_best) / len(valid_best) if len(valid_best) > 0 else 0.0

    print("\n" + "#" * 60)
    print("CONCAT FUSION RESULTS")
    print("#" * 60)

    for sid, info in results.items():
        print(f"{sid}: best_val_acc={info['best_val_acc']:.4f}")

    print(f"\nMean best concat fusion accuracy: {mean_best_acc:.4f}")

    return {
        "subject_results": results,
        "mean_best_acc": mean_best_acc,
        "split_mode": split_mode,
    }

