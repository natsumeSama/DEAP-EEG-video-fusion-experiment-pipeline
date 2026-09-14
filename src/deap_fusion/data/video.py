"""
Video preprocessing utilities for the DEAP facial-video pipeline.

This module handles:

1. locating raw video files;
2. sampling a fixed number of frames from each video;
3. indexing extracted frame folders;
4. detecting facial regions;
5. estimating one stable face crop per trial;
6. saving cropped face images and preprocessing metadata.

It does NOT contain:
- dataset downloading;
- PyTorch Dataset classes;
- ResNet models;
- training code;
- Colab-specific paths.
"""

from pathlib import Path
import re

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def find_video_files(video_root):
    """
    Find all supported video files recursively.

    Args:
        video_root:
            Root directory containing DEAP video files.

    Returns:
        Sorted list of Path objects.
    """
    video_root = Path(video_root)

    return sorted(
        p
        for p in video_root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

def extract_n_frames_from_video(
    video_path,
    output_root,
    num_frames=18,
    overwrite=False,
):
    """
    Extract evenly spaced frames from one video and save them.

    Args:
        video_path:
            Path to one video.
        output_root:
            Root directory where extracted frame folders are created.
        num_frames:
            Number of evenly spaced frames to extract.
        overwrite:
            Whether existing frames should be overwritten.

    Returns:
        Dictionary containing extraction metadata.
    """
    video_path = Path(video_path)
    output_root = Path(output_root)

    video_id = video_path.stem
    save_dir = output_root / video_id
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video_path": str(video_path),
            "video_id": video_id,
            "success": False,
            "reason": "could_not_open",
            "saved_frames": 0,
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()

        return {
            "video_path": str(video_path),
            "video_id": video_id,
            "success": False,
            "reason": "invalid_frame_count",
            "saved_frames": 0,
        }

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    saved_paths = []
    used_indices = []

    for i, frame_idx in enumerate(frame_indices):

        out_path = save_dir / f"{video_id}_frame_{i:02d}.jpg"

        if out_path.exists() and not overwrite:
            saved_paths.append(str(out_path))
            used_indices.append(int(frame_idx))
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        cv2.imwrite(str(out_path), frame)

        saved_paths.append(str(out_path))
        used_indices.append(int(frame_idx))

    cap.release()

    return {
        "video_path": str(video_path),
        "video_id": video_id,
        "success": True,
        "reason": "ok",
        "saved_frames": len(saved_paths),
        "total_frames": total_frames,
        "fps": fps,
        "frame_indices": used_indices,
        "saved_paths": saved_paths,
    }

def extract_frames_from_videos(
    video_paths,
    output_root,
    num_frames=18,
    overwrite=False,
):
    """
    Extract frames from multiple videos.

    Args:
        video_paths:
            Iterable of video paths.
        output_root:
            Root directory where frame folders are stored.
        num_frames:
            Number of frames extracted per video.
        overwrite:
            Whether existing images should be replaced.

    Returns:
        DataFrame containing extraction metadata for all videos.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    video_paths = list(video_paths)

    results = []

    for video_path in tqdm(
        video_paths,
        desc="Extracting video frames",
    ):
        result = extract_n_frames_from_video(
            video_path=video_path,
            output_root=output_root,
            num_frames=num_frames,
            overwrite=overwrite,
        )

        results.append(result)

    return pd.DataFrame(results)

def load_face_cascade():
    """
    Load OpenCV's default frontal-face Haar cascade.

    Returns:
        cv2.CascadeClassifier
    """
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError(
            f"Could not load Haar cascade: {cascade_path}"
        )

    return face_cascade

def build_frame_index(frames_root):
    """
    Build a table describing all extracted frames.

    Expected folder format:
        s01_trial01/
        s01_trial02/
        ...

    Args:
        frames_root:
            Directory containing one folder per video trial.

    Returns:
        pandas.DataFrame with columns:
            subject_id
            trial_id
            trial_name
            frame_order
            frame_path
    """
    frames_root = Path(frames_root)

    rows = []

    trial_folders = sorted(
        p
        for p in frames_root.iterdir()
        if p.is_dir()
    )

    for trial_folder in trial_folders:

        folder_name = trial_folder.name

        match = re.match(
            r"s(\d+)_trial(\d+)",
            folder_name,
            flags=re.IGNORECASE,
        )

        if match is None:
            continue

        subject_id = int(match.group(1))
        trial_id = int(match.group(2))

        frame_paths = sorted(
            p
            for p in trial_folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )

        for frame_order, frame_path in enumerate(frame_paths):

            rows.append({
                "subject_id": subject_id,
                "trial_id": trial_id,
                "trial_name": folder_name,
                "frame_order": frame_order,
                "frame_path": str(frame_path),
            })

    return pd.DataFrame(rows)

def detect_face_box(image_bgr, face_cascade):
    """
    Detect the largest face in an image.

    Returns:
        (x, y, w, h) if a face is detected,
        otherwise None.
    """
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
    )

    if len(faces) == 0:

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        gray_enhanced = clahe.apply(gray)

        faces = face_cascade.detectMultiScale(
            gray_enhanced,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(60, 60),
        )

    if len(faces) == 0:
        return None

    areas = [
        w * h
        for (x, y, w, h) in faces
    ]

    idx = int(np.argmax(areas))

    x, y, w, h = faces[idx]

    return (
        int(x),
        int(y),
        int(w),
        int(h),
    )

def expand_box(
    x,
    y,
    w,
    h,
    img_w,
    img_h,
    margin=0.35,
):
    """
    Expand a detected face bounding box while staying inside
    image boundaries.
    """
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)

    x2 = min(
        img_w,
        x + w + mx,
    )

    y2 = min(
        img_h,
        y + h + my,
    )

    return x1, y1, x2, y2

def fallback_center_box(
    img_w,
    img_h,
):
    """
    Create a center crop when no face could be detected.

    The crop is slightly shifted upward because the face/head is
    generally located above the image center.
    """
    crop_w = int(img_w * 0.55)
    crop_h = int(img_h * 0.72)

    cx = img_w // 2
    cy = int(img_h * 0.42)

    x1 = max(
        0,
        cx - crop_w // 2,
    )

    y1 = max(
        0,
        cy - crop_h // 2,
    )

    x2 = min(
        img_w,
        x1 + crop_w,
    )

    y2 = min(
        img_h,
        y1 + crop_h,
    )

    return x1, y1, x2, y2

def estimate_trial_box(
    frame_paths,
    face_cascade,
    margin=0.35,
):
    """
    Estimate one stable face crop for an entire trial.

    The face detector is applied across the trial frames. If faces
    are detected, the median bounding box is used.

    If no face is detected in any frame, a center crop is used.

    Returns:
        trial_box:
            (x1, y1, x2, y2) or None

        box_source:
            "detected", "fallback_center", or "failed_read"

        n_detected:
            Number of frames where a face was detected.
    """
    raw_boxes = []
    first_img_shape = None

    for frame_path in frame_paths:

        img_bgr = cv2.imread(
            str(frame_path)
        )

        if img_bgr is None:
            continue

        if first_img_shape is None:
            first_img_shape = img_bgr.shape

        box = detect_face_box(
            img_bgr,
            face_cascade,
        )

        if box is not None:
            raw_boxes.append(box)

    if first_img_shape is None:
        return None, "failed_read", 0

    img_h, img_w = first_img_shape[:2]

    if len(raw_boxes) > 0:

        boxes = np.array(raw_boxes)

        x, y, w, h = np.median(
            boxes,
            axis=0,
        ).astype(int)

        x1, y1, x2, y2 = expand_box(
            x,
            y,
            w,
            h,
            img_w,
            img_h,
            margin=margin,
        )

        return (
            (x1, y1, x2, y2),
            "detected",
            len(raw_boxes),
        )

    x1, y1, x2, y2 = fallback_center_box(
        img_w,
        img_h,
    )

    return (
        (x1, y1, x2, y2),
        "fallback_center",
        0,
    )

def create_face_crops(
    frame_df,
    crops_root,
    face_cascade=None,
    margin=0.35,
    save_metadata=True,
):
    """
    Generate trial-consistent face crops from extracted frames.

    One face bounding box is estimated for each trial and then
    applied to all frames belonging to that trial.

    Args:
        frame_df:
            DataFrame produced by build_frame_index().
        crops_root:
            Directory where cropped face frames are saved.
        face_cascade:
            Optional preloaded Haar cascade.
            If None, the default OpenCV cascade is loaded.
        margin:
            Expansion applied around the median detected face box.
        save_metadata:
            Whether to save the resulting metadata CSV.

    Returns:
        pandas.DataFrame describing all generated crops.
    """
    crops_root = Path(crops_root)
    crops_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if face_cascade is None:
        face_cascade = load_face_cascade()

    results = []

    grouped = frame_df.groupby(
        [
            "subject_id",
            "trial_id",
            "trial_name",
        ]
    )

    for (
        subject_id,
        trial_id,
        trial_name,
    ), group in tqdm(
        grouped,
        total=len(grouped),
        desc="Cropping faces",
    ):

        group = group.sort_values(
            "frame_order"
        )

        frame_paths = [
            Path(p)
            for p in group["frame_path"].tolist()
        ]

        trial_box, box_source, n_detected = estimate_trial_box(
            frame_paths=frame_paths,
            face_cascade=face_cascade,
            margin=margin,
        )

        if trial_box is None:

            for _, row in group.iterrows():

                results.append({
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "trial_name": trial_name,
                    "frame_order": row["frame_order"],
                    "frame_path": row["frame_path"],
                    "face_path": None,
                    "box_source": "failed_read",
                    "n_detected_seed_frames": 0,
                })

            continue

        x1, y1, x2, y2 = trial_box

        out_dir = crops_root / trial_name

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for _, row in group.iterrows():

            frame_path = Path(
                row["frame_path"]
            )

            frame_order = row[
                "frame_order"
            ]

            img_bgr = cv2.imread(
                str(frame_path)
            )

            if img_bgr is None:

                results.append({
                    "subject_id": subject_id,
                    "trial_id": trial_id,
                    "trial_name": trial_name,
                    "frame_order": frame_order,
                    "frame_path": str(frame_path),
                    "face_path": None,
                    "box_source": "failed_read",
                    "n_detected_seed_frames": n_detected,
                })

                continue

            crop_bgr = img_bgr[
                y1:y2,
                x1:x2,
            ]

            out_path = (
                out_dir
                / f"face_{frame_order:02d}.jpg"
            )

            cv2.imwrite(
                str(out_path),
                crop_bgr,
            )

            results.append({
                "subject_id": subject_id,
                "trial_id": trial_id,
                "trial_name": trial_name,
                "frame_order": frame_order,
                "frame_path": str(frame_path),
                "face_path": str(out_path),
                "box_source": box_source,
                "n_detected_seed_frames": n_detected,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    face_df = pd.DataFrame(results)

    if save_metadata:

        metadata_path = (
            crops_root
            / "frame_face_metadata_trialbox.csv"
        )

        face_df.to_csv(
            metadata_path,
            index=False,
        )

    return face_df