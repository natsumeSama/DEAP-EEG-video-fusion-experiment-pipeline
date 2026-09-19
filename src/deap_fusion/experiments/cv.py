import gc as _gc

import torch

from ..config import (
    LABEL_TYPE,
    MAX_SUBJECT_ID,
    SEED,
    EPOCHS,
    BATCH_SIZE,
    LR,
    WEIGHT_DECAY,
    FREEZE_BACKBONE,
    NUM_FRAMES,
)

from ..data.splits import (
    init_shared_cv_splits,
    activate_cv_fold,
)

from ..training.common import set_seeds
from ..training.eeg import run_all_eeg_shared_split
from ..training.video import run_all_subject_dependent
from ..training.fusion import (
    run_all_fusion_shared_split,
    run_all_concat_fusion_shared_split,
)
from ..evaluation.decision import (
    run_all_decision_fusion_shared_split,
)

def run_Kfold_cv_all_models(
    eeg_root,
    image_root,

    label_type=LABEL_TYPE,
    split_mode="subject_dependent",
    n_splits=4,
    max_subject_id=MAX_SUBJECT_ID,
    seed=SEED,

    run_eeg=True,
    run_video=True,
    run_film=True,
    run_concat=True,
    run_decision=True,

    eeg_mode=2,
    lambda_cons=0.00,

    eeg_epochs=EPOCHS,
    video_epochs=EPOCHS,
    fusion_stage1_epochs=15,
    fusion_stage2_epochs=5,

    batch_size=BATCH_SIZE,
):
    """
    Run the full K-fold cross-validation pipeline.

    For each fold:
        1. Activate shared fold splits
        2. Train EEG models
        3. Train video models
        4. Train FiLM fusion
        5. Train concat fusion
        6. Run decision fusion
    """

    init_shared_cv_splits(
        eeg_root=eeg_root,
        image_root=image_root,
        label_type=label_type,
        max_subject_id=max_subject_id,
        split_mode=split_mode,
        n_splits=n_splits,
        seed=seed,
    )

    cv_results = {}

    for fold in range(n_splits):
        print("\n" + "=" * 80)
        print(f"STARTING CV FOLD {fold + 1}/{n_splits}")
        print("=" * 80)

        activate_cv_fold(fold)

        # Change the seed per fold for training initialization,
        # while keeping the fold split fixed.
        set_seeds(seed + fold)

        fold_results = {}

        # -------------------------------------------------
        # 1) EEG
        # -------------------------------------------------
        if run_eeg:
            all_eeg = run_all_eeg_shared_split(
                dataset_path=eeg_root,
                image_root=image_root,
                label_type=label_type,
                mode=eeg_mode,
                max_subject_id=max_subject_id,
                split_mode=split_mode,
                epochs=eeg_epochs,
                batch_size=batch_size,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                verbose=1,
                model_name="EEG_TR",
                lambda_cons=lambda_cons,
            )
            fold_results["EEG"] = all_eeg
        else:
            all_eeg = None

        # -------------------------------------------------
        # 2) VIDEO
        # -------------------------------------------------
        if run_video:
            all_video = run_all_subject_dependent(
                deap_root=eeg_root,
                image_root=image_root,
                label_type=label_type,
                max_subject_id=max_subject_id,
                epochs=video_epochs,
                batch_size=batch_size,
                lr=LR,
                freeze_backbone=FREEZE_BACKBONE,
            )
            fold_results["VIDEO"] = all_video
        else:
            all_video = None

        if all_eeg is None or all_video is None:
            print("Skipping fusion models because EEG or VIDEO is missing.")
            cv_results[fold] = fold_results
            continue

        # -------------------------------------------------
        # 3) FiLM FUSION
        # -------------------------------------------------
        if run_film:
            all_fusion = run_all_fusion_shared_split(
                eeg_root=eeg_root,
                image_root=image_root,
                eeg_results=all_eeg,
                video_results=all_video,
                label_type=label_type,
                eeg_mode=eeg_mode,
                max_subject_id=max_subject_id,
                split_mode=split_mode,
                num_frames=NUM_FRAMES,
                batch_size=batch_size,
                stage1_epochs=fusion_stage1_epochs,
                stage2_epochs=fusion_stage2_epochs,
                stage1_lr=1e-3,
                stage2_head_lr=5e-5,
                stage2_encoder_lr=1e-6,
                weight_decay=WEIGHT_DECAY,
            )
            fold_results["FUSION"] = all_fusion

        # -------------------------------------------------
        # 4) CONCAT FUSION
        # -------------------------------------------------
        if run_concat:
            all_concat = run_all_concat_fusion_shared_split(
                eeg_root=eeg_root,
                image_root=image_root,
                eeg_results=all_eeg,
                video_results=all_video,
                label_type=label_type,
                eeg_mode=eeg_mode,
                max_subject_id=max_subject_id,
                split_mode=split_mode,
                num_frames=NUM_FRAMES,
                batch_size=batch_size,
                stage1_epochs=fusion_stage1_epochs,
                stage2_epochs=fusion_stage2_epochs,
                stage1_lr=1e-3,
                stage2_head_lr=5e-5,
                stage2_encoder_lr=1e-6,
                weight_decay=WEIGHT_DECAY,
            )
            fold_results["CONCAT"] = all_concat

        # -------------------------------------------------
        # 5) DECISION FUSION
        # -------------------------------------------------
        if run_decision:
            all_decision = run_all_decision_fusion_shared_split(
                eeg_root=eeg_root,
                image_root=image_root,
                eeg_results=all_eeg,
                video_results=all_video,
                label_type=label_type,
                eeg_mode=eeg_mode,
                max_subject_id=max_subject_id,
                split_mode=split_mode,
                num_frames=NUM_FRAMES,
                batch_size=batch_size,
                eeg_weight=0.5,
                video_weight=0.5,
            )
            fold_results["DECISION"] = all_decision

        cv_results[fold] = fold_results

        # Clean memory between folds
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cv_results