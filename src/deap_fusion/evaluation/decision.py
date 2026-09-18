import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ..config import (
    NUM_WORKERS,
    MAX_SUBJECT_ID,
)

from ..data.datasets import (
    create_fusion_dataloaders_from_pairs,
)

from ..data.splits import (
    get_shared_split,
)

from ..models.eeg import (
    build_model,
)

from ..models.video import (
    ResNet18TrialModel,
)

def _extract_logits(model_output):
    """
    Some models return logits directly.
    Some may return (logits, features).
    This helper keeps it safe.
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output

def evaluate_decision_level_fusion(
    eeg_model,
    video_model,
    fusion_loader,
    device,
    eeg_weight=0.5,
    video_weight=0.5,
):
    """
    Decision-level fusion baseline.

    It averages the predicted probabilities of the EEG and video models:

        final_prob = eeg_weight * eeg_prob + video_weight * video_prob

    No training.
    No feature-level fusion.
    No FiLM.
    """

    eeg_model.eval()
    video_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for eeg_x, video_x, y, meta in fusion_loader:
            eeg_x = eeg_x.to(device)
            video_x = video_x.to(device)
            y = y.to(device)

            eeg_logits = _extract_logits(eeg_model(eeg_x))
            video_logits = _extract_logits(video_model(video_x))

            eeg_prob = torch.softmax(eeg_logits, dim=1)
            video_prob = torch.softmax(video_logits, dim=1)

            final_prob = eeg_weight * eeg_prob + video_weight * video_prob
            preds = final_prob.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def run_decision_fusion_experiment_shared_split(
      eeg_root,
      image_root,
      train_pairs,
      val_pairs,
      eeg_ckpt_path,
      video_ckpt_path,
      label_type=0,
      eeg_mode=2,
      num_frames=18,
      batch_size=4,
      eeg_weight=0.5,
      video_weight=0.5,
    ):
      """
      Decision-level fusion for one subject.

      Uses already trained unimodal EEG and video models.
      Does NOT train anything.
      """

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # We only need the val loader for decision fusion.
      _, fusion_val_loader, fusion_train_ds, fusion_val_ds = create_fusion_dataloaders_from_pairs(
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

      print("Decision fusion val size:", len(fusion_val_ds))

      # Infer EEG input feature dim
      eeg_x0, video_x0, y0, meta0 = fusion_train_ds[0]
      eeg_feature_dim = eeg_x0.shape[-1]

      # Build EEG model
      eeg_model = build_model(
          mode=eeg_mode,
          num_channels=32,
          feature_dim=eeg_feature_dim,
          num_classes=2,
      ).to(device)

      eeg_model.load_state_dict(
          torch.load(eeg_ckpt_path, map_location=device)
      )

      # Build video model
      video_model = ResNet18TrialModel(
          num_classes=2,
          freeze_backbone=False,
          freeze_bn_stats=False,
      ).to(device)

      video_model.load_state_dict(
          torch.load(video_ckpt_path, map_location=device)
      )

      metrics = evaluate_decision_level_fusion(
          eeg_model=eeg_model,
          video_model=video_model,
          fusion_loader=fusion_val_loader,
          device=device,
          eeg_weight=eeg_weight,
          video_weight=video_weight,
      )

      print("Decision-level fusion metrics:", metrics)

      return {
          "history": None,
          "best_val_acc": metrics["acc"],
          "last_val_acc": metrics["acc"],
          "best_val_loss": None,
          "detailed_metrics": metrics,
          "selected_stage": "decision_level",
          "save_path": None,
          "train_pairs": train_pairs,
          "val_pairs": val_pairs,
          "eeg_weight": eeg_weight,
          "video_weight": video_weight,
      }






def run_all_decision_fusion_shared_split(
    eeg_root,
    image_root,
    eeg_results,
    video_results,
    label_type=0,
    eeg_mode=2,
    max_subject_id=22,
    split_mode="subject_dependent",
    train_ratio=0.75,
    seed=42,
    num_frames=18,
    batch_size=4,
    eeg_weight=0.5,
    video_weight=0.5,
    ):
      """
      Run decision-level fusion for all subjects.

      Requires:
      - all_eeg_sd
      - all_video_sd

      because it loads the trained unimodal checkpoints.
      """



      results = {}

      for sid_num in range(1, max_subject_id + 1):
          sid = f"s{sid_num:02d}"

          print("\n" + "=" * 80)
          print(f"Running DECISION FUSION for {sid}")
          print("=" * 80)

          try:
              train_pairs, val_pairs, split_info = get_shared_split(
                  sid=sid,
                  split_mode=split_mode,
              )

              eeg_ckpt_path = eeg_results["subject_results"][sid]["save_path"]
              video_ckpt_path = video_results["subject_results"][sid]["save_path"]

              result = run_decision_fusion_experiment_shared_split(
                  eeg_root=eeg_root,
                  image_root=image_root,
                  train_pairs=train_pairs,
                  val_pairs=val_pairs,
                  eeg_ckpt_path=eeg_ckpt_path,
                  video_ckpt_path=video_ckpt_path,
                  label_type=label_type,
                  eeg_mode=eeg_mode,
                  num_frames=num_frames,
                  batch_size=batch_size,
                  eeg_weight=eeg_weight,
                  video_weight=video_weight,
              )


              result["split_info"] = split_info
              result["train_pairs"] = train_pairs
              result["val_pairs"] = val_pairs
              results[sid] = result

          except Exception as e:
              print(f"Skipping {sid}: {e}")

      mean_acc = float(np.mean([
          v["best_val_acc"] for v in results.values()
          if v.get("best_val_acc") is not None
      ])) if len(results) > 0 else 0.0

      print("\n" + "#" * 80)
      print("DECISION FUSION RESULTS")
      print("#" * 80)

      for sid, info in results.items():
          print(f"{sid}: {info['best_val_acc']:.4f}")

      print(f"\nMean decision fusion acc: {mean_acc:.4f}")

      return {
          "subject_results": results,
          "mean_best_acc": mean_acc,
          "split_mode": split_mode,
          "eeg_weight": eeg_weight,
          "video_weight": video_weight,
      }