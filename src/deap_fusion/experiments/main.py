from ..config import (
    LABEL_TYPE,
    MAX_SUBJECT_ID,
    NUM_FRAMES,
    BATCH_SIZE,
    WEIGHT_DECAY,
    EPOCHS,
    LR,
    FREEZE_BACKBONE,
    TRAIN_RATIO,
    SEED,
)

from ..data import splits as split_state
from ..data.splits import init_shared_splits

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

def run_main_experiment(
    eeg_root,
    image_root,
    label_type=LABEL_TYPE,
    seed=SEED,
):
    """
    Main single-split subject-dependent experiment:
    EEG -> Video -> FiLM -> Concat -> Decision
    """

    split_state.CURRENT_CV_FOLD = None
    split_state.CURRENT_RUN_TAG = (
        f"main_label{label_type}_seed{seed}"
    )

    set_seeds(seed)

    print("=" * 80)
    print(
        f"MAIN EXPERIMENT | "
        f"LABEL_TYPE={label_type} | SEED={seed}"
    )
    print("=" * 80)

    # One shared 75/25 split per subject
    init_shared_splits(
        eeg_root=eeg_root,
        image_root=image_root,
        label_type=label_type,
        max_subject_id=MAX_SUBJECT_ID,
        split_mode="subject_dependent",
        train_ratio=TRAIN_RATIO,
        seed=seed,
    )

    # 1) EEG
    all_eeg = run_all_eeg_shared_split(
        dataset_path=eeg_root,
        image_root=image_root,
        label_type=label_type,
        mode=2,
        max_subject_id=MAX_SUBJECT_ID,
        split_mode="subject_dependent",
        train_ratio=TRAIN_RATIO,
        seed=seed,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        verbose=1,
        lambda_cons=0.0,
    )

    # 2) Video
    all_video = run_all_subject_dependent(
        deap_root=eeg_root,
        image_root=image_root,
        label_type=label_type,
        max_subject_id=MAX_SUBJECT_ID,
        train_ratio=TRAIN_RATIO,
        seed=seed,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        freeze_backbone=FREEZE_BACKBONE,
    )

    # 3) FiLM
    all_film = run_all_fusion_shared_split(
        eeg_root=eeg_root,
        image_root=image_root,
        eeg_results=all_eeg,
        video_results=all_video,
        label_type=label_type,
        eeg_mode=2,
        max_subject_id=MAX_SUBJECT_ID,
        split_mode="subject_dependent",
        train_ratio=TRAIN_RATIO,
        seed=seed,
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        stage1_epochs=15,
        stage2_epochs=5,
        stage1_lr=1e-3,
        stage2_head_lr=5e-5,
        stage2_encoder_lr=1e-6,
        weight_decay=WEIGHT_DECAY,
    )

    # 4) Concat
    all_concat = run_all_concat_fusion_shared_split(
        eeg_root=eeg_root,
        image_root=image_root,
        eeg_results=all_eeg,
        video_results=all_video,
        label_type=label_type,
        eeg_mode=2,
        max_subject_id=MAX_SUBJECT_ID,
        split_mode="subject_dependent",
        train_ratio=TRAIN_RATIO,
        seed=seed,
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        stage1_epochs=15,
        stage2_epochs=5,
        stage1_lr=1e-3,
        stage2_head_lr=5e-5,
        stage2_encoder_lr=1e-6,
        weight_decay=WEIGHT_DECAY,
    )

    # 5) Decision Fusion
    all_decision = run_all_decision_fusion_shared_split(
        eeg_root=eeg_root,
        image_root=image_root,
        eeg_results=all_eeg,
        video_results=all_video,
        label_type=label_type,
        eeg_mode=2,
        max_subject_id=MAX_SUBJECT_ID,
        split_mode="subject_dependent",
        train_ratio=TRAIN_RATIO,
        seed=seed,
        num_frames=NUM_FRAMES,
        batch_size=BATCH_SIZE,
        eeg_weight=0.5,
        video_weight=0.5,
    )

    return {
        "EEG": all_eeg,
        "VIDEO": all_video,
        "FUSION": all_film,
        "CONCAT": all_concat,
        "DECISION": all_decision,
    }