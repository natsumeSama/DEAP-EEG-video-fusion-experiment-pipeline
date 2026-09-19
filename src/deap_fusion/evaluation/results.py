import numpy as np
import pandas as pd
import os

MODEL_DISPLAY_NAMES = {
    "EEG": "EEG",
    "VIDEO": "Video",
    "FUSION": "FiLM Fusion",
    "CONCAT": "Concat Fusion",
    "DECISION": "Decision Fusion",
}

CV_METRICS = [
    "best_val_acc",
    "last_val_acc",
    "acc",
    "balanced_acc",
    "precision",
    "recall",
    "f1",
    "loss",
]

def _safe_last(x):
    """
    Return the last value of a list safely.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return x[-1]
    return None

def _safe_max(x):
    """
    Return max of a list safely.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return max(x)
    return None

def _safe_get(d, key, default=None):
    """
    Safe dictionary getter.
    """
    if d is None:
        return default
    return d.get(key, default)

def _extract_last_val_acc_from_history(history):
    """
    Extract the last validation accuracy from a history dict.
    """
    if history is None:
        return None

    val_acc = history.get("val_acc")

    if val_acc is None or len(val_acc) == 0:
        return None

    return float(val_acc[-1])

def _standardize_entry(
    entry,
    split_info=None,
    selected_stage=None,
    keep_extra_keys=None,
):
    """
    Convert one raw result entry into the common standardized entry format.
    """
    if keep_extra_keys is None:
        keep_extra_keys = []

    history = entry.get("history")
    detailed_metrics = entry.get("detailed_metrics")

    best_val_acc = entry.get("best_val_acc")
    last_val_acc = entry.get("last_val_acc")

    # If last_val_acc is missing but history exists, infer it.
    if last_val_acc is None:
        last_val_acc = _extract_last_val_acc_from_history(
            history
        )

    standardized = {
        "history": history,
        "detailed_metrics": detailed_metrics,

        "best_val_acc": (
            float(best_val_acc)
            if best_val_acc is not None
            else None
        ),

        "last_val_acc": (
            float(last_val_acc)
            if last_val_acc is not None
            else None
        ),

        "save_path": entry.get("save_path"),

        "split_info": (
            split_info
            if split_info is not None
            else entry.get("split_info")
        ),

        "selected_stage": (
            selected_stage
            if selected_stage is not None
            else entry.get("selected_stage")
        ),

        "extra": {},
    }

    for k in keep_extra_keys:
        if k in entry:
            standardized["extra"][k] = entry[k]

    return standardized

def standardize_eeg_results(
    raw_results,
    model_name="EEG",
):
    """
    Standardize EEG results.

    Supports:
    - per-subject results
    - global random_all results
    """
    out = {
        "model_name": model_name,
        "source_type": "eeg",
        "split_mode": raw_results.get("split_mode"),
        "subject_results": {},
    }

    # Per-subject case
    if "subject_results" in raw_results:

        for sid, entry in raw_results[
            "subject_results"
        ].items():

            out["subject_results"][sid] = (
                _standardize_entry(
                    entry,
                    split_info=entry.get(
                        "split_info"
                    ),
                )
            )

        return out

    # random_all/global case
    if "global_result" in raw_results:

        global_entry = raw_results[
            "global_result"
        ]

        out["split_mode"] = (
            raw_results
            .get("split_info", {})
            .get(
                "split_mode",
                out["split_mode"],
            )
        )

        out["subject_results"]["global"] = (
            _standardize_entry(
                global_entry,
                split_info=raw_results.get(
                    "split_info"
                ),
            )
        )

        return out

    raise ValueError(
        "Could not recognize EEG result structure."
    )

def standardize_video_results(
    raw_results,
    model_name="VIDEO",
):
    """
    Standardize video results.

    Supports:
    - per-subject results
    - optional global_result structure
    """
    out = {
        "model_name": model_name,
        "source_type": "video",

        "split_mode": raw_results.get(
            "split_mode",
            "subject_dependent",
        ),

        "subject_results": {},
    }

    if "subject_results" in raw_results:

        for sid, entry in raw_results[
            "subject_results"
        ].items():

            out["subject_results"][sid] = (
                _standardize_entry(
                    entry,
                    split_info=entry.get(
                        "split_info"
                    ),
                )
            )

        return out

    if "global_result" in raw_results:

        global_entry = raw_results[
            "global_result"
        ]

        out["split_mode"] = (
            raw_results
            .get("split_info", {})
            .get(
                "split_mode",
                out["split_mode"],
            )
        )

        out["subject_results"]["global"] = (
            _standardize_entry(
                global_entry,
                split_info=raw_results.get(
                    "split_info"
                ),
            )
        )

        return out

    raise ValueError(
        "Could not recognize VIDEO result structure."
    )

def standardize_fusion_results(
    raw_results,
    model_name="FUSION",
):
    """
    Standardize fusion-style results.

    Used by:
    - FiLM
    - Concat
    - Decision fusion

    Supports:
    - per-subject results
    - global random_all results
    """
    out = {
        "model_name": model_name,
        "source_type": "fusion",
        "split_mode": raw_results.get(
            "split_mode"
        ),
        "subject_results": {},
    }

    if "subject_results" in raw_results:

        for sid, entry in raw_results[
            "subject_results"
        ].items():

            out["subject_results"][sid] = (
                _standardize_entry(
                    entry,
                    split_info=entry.get(
                        "split_info"
                    ),
                    selected_stage=entry.get(
                        "selected_stage"
                    ),
                    keep_extra_keys=[
                        "stage1_best_val_acc",
                        "stage2_best_val_acc",
                        "stage1_history",
                        "stage2_history",
                        "stage1_metrics",
                        "stage2_metrics",
                    ],
                )
            )

        return out

    if "global_result" in raw_results:

        global_entry = raw_results[
            "global_result"
        ]

        out["split_mode"] = (
            raw_results
            .get("split_info", {})
            .get(
                "split_mode",
                out["split_mode"],
            )
        )

        out["subject_results"]["global"] = (
            _standardize_entry(
                global_entry,
                split_info=raw_results.get(
                    "split_info"
                ),
                selected_stage=global_entry.get(
                    "selected_stage"
                ),
                keep_extra_keys=[
                    "stage1_best_val_acc",
                    "stage2_best_val_acc",
                    "stage1_history",
                    "stage2_history",
                    "stage1_metrics",
                    "stage2_metrics",
                ],
            )
        )

        return out

    raise ValueError(
        "Could not recognize FUSION result structure."
    )

def collect_cv_results(cv_results):
    """
    Convert nested K-fold results into one clean DataFrame.

    One row = one fold + one model + one subject.
    """
    rows = []

    for fold, fold_dict in cv_results.items():
        for model_name, model_results in fold_dict.items():

            if model_results is None:
                continue

            subject_results = model_results.get("subject_results", {})

            for sid, entry in subject_results.items():
                dm = entry.get("detailed_metrics", {}) or {}
                history = entry.get("history", {}) or {}
                split_info = entry.get("split_info", {}) or {}

                best_val_acc = entry.get("best_val_acc")
                last_val_acc = entry.get("last_val_acc")

                if last_val_acc is None:
                    last_val_acc = _safe_last(history.get("val_acc"))

                best_train_acc = entry.get("best_train_acc")
                if best_train_acc is None:
                    best_train_acc = _safe_max(history.get("train_acc"))

                last_train_acc = entry.get("last_train_acc")
                if last_train_acc is None:
                    last_train_acc = _safe_last(history.get("train_acc"))

                row = {
                    "fold": int(fold),
                    "model": model_name,
                    "model_display": MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name,
                    ),
                    "subject": sid,

                    "best_val_acc": best_val_acc,
                    "last_val_acc": last_val_acc,
                    "best_train_acc": best_train_acc,
                    "last_train_acc": last_train_acc,

                    "acc": dm.get("acc"),
                    "balanced_acc": dm.get("balanced_acc"),
                    "precision": dm.get("precision"),
                    "recall": dm.get("recall"),
                    "f1": dm.get("f1"),
                    "loss": dm.get("loss"),

                    "selected_stage": entry.get("selected_stage"),
                    "save_path": entry.get("save_path"),

                    "n_train_pairs": split_info.get("n_train_pairs"),
                    "n_val_pairs": split_info.get("n_val_pairs"),
                    "train_label_counts": split_info.get(
                        "train_label_counts"
                    ),
                    "val_label_counts": split_info.get(
                        "val_label_counts"
                    ),
                }

                rows.append(row)

    df = pd.DataFrame(rows)

    numeric_cols = [
        "best_val_acc",
        "last_val_acc",
        "best_train_acc",
        "last_train_acc",
        "acc",
        "balanced_acc",
        "precision",
        "recall",
        "f1",
        "loss",
        "n_train_pairs",
        "n_val_pairs",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    return df

def build_cv_fold_table(cv_subject_df):
    """
    Average all subjects inside each fold.

    One row = one model + one fold.
    """
    existing_metrics = [
        m for m in CV_METRICS
        if m in cv_subject_df.columns
    ]

    fold_df = (
        cv_subject_df
        .groupby(
            ["model", "model_display", "fold"],
            as_index=False,
        )
        .agg(
            n_subjects=("subject", "nunique"),
            **{
                m: (m, "mean")
                for m in existing_metrics
            }
        )
    )

    return fold_df

def summarize_cv_results(cv_fold_df):
    """
    Final CV table.

    Mean and std are computed across folds.
    """
    existing_metrics = [
        m for m in CV_METRICS
        if m in cv_fold_df.columns
    ]

    rows = []

    for (model, model_display), g in cv_fold_df.groupby(
        ["model", "model_display"]
    ):
        row = {
            "model": model,
            "model_display": model_display,
            "n_folds": int(g["fold"].nunique()),
            "n_subjects_mean": float(
                g["n_subjects"].mean()
            ),
        }

        for m in existing_metrics:
            vals = pd.to_numeric(
                g[m],
                errors="coerce",
            ).dropna()

            row[f"{m}_mean"] = (
                float(vals.mean())
                if len(vals) > 0
                else np.nan
            )

            row[f"{m}_std"] = (
                float(vals.std(ddof=1))
                if len(vals) > 1
                else 0.0
            )

            row[f"{m}_min"] = (
                float(vals.min())
                if len(vals) > 0
                else np.nan
            )

            row[f"{m}_max"] = (
                float(vals.max())
                if len(vals) > 0
                else np.nan
            )

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    if "best_val_acc_mean" in summary_df.columns:
        summary_df = summary_df.sort_values(
            "best_val_acc_mean",
            ascending=False,
        )

    return summary_df

def collect_cv_film_stats(cv_results):
    """
    Collect all per-trial FiLM modulation statistics across:
    - folds
    - subjects
    - validation trials

    Output:
        one row = one validation trial for one subject in one fold
    """
    rows = []

    for fold, fold_dict in cv_results.items():
        if "FUSION" not in fold_dict:
            continue

        fusion_results = fold_dict["FUSION"]
        subject_results = fusion_results.get("subject_results", {})

        for sid, entry in subject_results.items():
            film_stats = entry.get("film_stats")

            if film_stats is None or len(film_stats) == 0:
                continue

            df = film_stats.copy()
            df["fold"] = fold
            df["subject"] = sid
            df["model"] = "FUSION"

            rows.append(df)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

def build_cv_film_summary_table(cv_film_stats_df):
    """
    Build a compact table summarizing FiLM behavior across all folds.
    """
    if cv_film_stats_df is None or len(cv_film_stats_df) == 0:
        return pd.DataFrame()

    metrics = [
        "gamma_mean",
        "gamma_std",
        "gamma_deviation_from_1",
        "beta_mean",
        "beta_std",
        "beta_abs_mean",
        "gate_mean",
        "gate_std",
        "relative_modulation",
    ]

    rows = []

    for metric in metrics:
        vals = pd.to_numeric(cv_film_stats_df[metric], errors="coerce").dropna()

        rows.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })

    return pd.DataFrame(rows)

def make_cv_paper_table(cv_summary_df):
    """
    Create a clean table for thesis/report.

    Values are displayed as percentage mean ± std.
    """
    rows = []

    for _, r in cv_summary_df.iterrows():
        row = {
            "Model": r["model_display"],

            "Best Val Acc": (
                f"{r['best_val_acc_mean']*100:.2f} ± "
                f"{r['best_val_acc_std']*100:.2f}"
            ),

            "Accuracy": (
                f"{r['acc_mean']*100:.2f} ± "
                f"{r['acc_std']*100:.2f}"
                if not pd.isna(r.get("acc_mean"))
                else ""
            ),

            "Balanced Acc": (
                f"{r['balanced_acc_mean']*100:.2f} ± "
                f"{r['balanced_acc_std']*100:.2f}"
                if not pd.isna(r.get("balanced_acc_mean"))
                else ""
            ),

            "F1": (
                f"{r['f1_mean']*100:.2f} ± "
                f"{r['f1_std']*100:.2f}"
                if not pd.isna(r.get("f1_mean"))
                else ""
            ),

            "Precision": (
                f"{r['precision_mean']*100:.2f} ± "
                f"{r['precision_std']*100:.2f}"
                if not pd.isna(r.get("precision_mean"))
                else ""
            ),

            "Recall": (
                f"{r['recall_mean']*100:.2f} ± "
                f"{r['recall_std']*100:.2f}"
                if not pd.isna(r.get("recall_mean"))
                else ""
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)

def compare_cv_models(
    cv_fold_df,
    metric="best_val_acc",
    baseline="VIDEO",
):
    """
    Compare each model against a baseline model fold by fold.

    Example:
        FUSION - CONCAT
        FUSION - VIDEO
        FUSION - EEG
    """
    pivot = cv_fold_df.pivot(
        index="fold",
        columns="model",
        values=metric,
    )

    if baseline not in pivot.columns:
        raise ValueError(
            f"Baseline {baseline} not found. "
            f"Available models: {list(pivot.columns)}"
        )

    rows = []

    for model in pivot.columns:
        if model == baseline:
            continue

        diff = pivot[model] - pivot[baseline]
        diff = diff.dropna()

        rows.append({
            "comparison": f"{model} - {baseline}",
            "metric": metric,
            "mean_diff": (
                float(diff.mean())
                if len(diff) > 0
                else np.nan
            ),
            "std_diff": (
                float(diff.std(ddof=1))
                if len(diff) > 1
                else 0.0
            ),
            "min_diff": (
                float(diff.min())
                if len(diff) > 0
                else np.nan
            ),
            "max_diff": (
                float(diff.max())
                if len(diff) > 0
                else np.nan
            ),
            "wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
            "n_folds": int(len(diff)),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "mean_diff",
            ascending=False,
        )
    )

def compare_two_cv_models(
    cv_fold_df,
    model_a="FUSION",
    model_b="CONCAT",
    metric="best_val_acc",
):
    """
    Direct comparison between two specific models.

    Example:
        FiLM Fusion vs Concat Fusion.
    """
    pivot = cv_fold_df.pivot(
        index="fold",
        columns="model",
        values=metric,
    )

    if model_a not in pivot.columns:
        raise ValueError(f"{model_a} not found.")

    if model_b not in pivot.columns:
        raise ValueError(f"{model_b} not found.")

    diff = (
        pivot[model_a]
        - pivot[model_b]
    ).dropna()

    return pd.DataFrame([{
        "comparison": f"{model_a} - {model_b}",
        "metric": metric,

        "mean_diff": (
            float(diff.mean())
            if len(diff) > 0
            else np.nan
        ),

        "std_diff": (
            float(diff.std(ddof=1))
            if len(diff) > 1
            else 0.0
        ),

        "min_diff": (
            float(diff.min())
            if len(diff) > 0
            else np.nan
        ),

        "max_diff": (
            float(diff.max())
            if len(diff) > 0
            else np.nan
        ),

        "wins": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "losses": int((diff < 0).sum()),
        "n_folds": int(len(diff)),
    }])

def compare_all_metrics(
    cv_fold_df,
    baselines=("VIDEO", "CONCAT", "EEG"),
):
    """
    Compare all models against selected baselines using several metrics.
    """
    metrics = [
        "best_val_acc",
        "acc",
        "balanced_acc",
        "f1",
        "precision",
        "recall",
    ]

    all_rows = []

    for metric in metrics:
        if metric not in cv_fold_df.columns:
            continue

        for baseline in baselines:
            if baseline not in cv_fold_df["model"].unique():
                continue

            comp_df = compare_cv_models(
                cv_fold_df,
                metric=metric,
                baseline=baseline,
            )

            comp_df["baseline"] = baseline
            comp_df["metric"] = metric

            all_rows.append(comp_df)

    if len(all_rows) == 0:
        return pd.DataFrame()

    return pd.concat(
        all_rows,
        ignore_index=True,
    )

def compare_film_vs_concat_all_metrics(cv_fold_df):
    """
    Direct FiLM vs Concat comparison for all useful metrics.
    """
    metrics = [
        "best_val_acc",
        "acc",
        "balanced_acc",
        "f1",
        "precision",
        "recall",
    ]

    rows = []

    for metric in metrics:
        if metric not in cv_fold_df.columns:
            continue

        comp = compare_two_cv_models(
            cv_fold_df,
            model_a="FUSION",
            model_b="CONCAT",
            metric=metric,
        )

        rows.append(comp)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.concat(
        rows,
        ignore_index=True,
    )

# =========================================================
# COLLECT MAIN-SPLIT FiLM GAMMA / BETA / GATE ANALYSIS
# =========================================================

def collect_main_film_stats(all_fusion_sd):
    """
    Collect FiLM statistics from the main subject-dependent run.

    Input:
        all_fusion_sd = output of run_all_fusion_shared_split(...)

    Output:
        one row = one validation trial
    """
    rows = []

    subject_results = all_fusion_sd.get("subject_results", {})

    for sid, entry in subject_results.items():
        film_stats = entry.get("film_stats")

        if film_stats is None or len(film_stats) == 0:
            continue

        df = film_stats.copy()
        df["subject"] = sid
        df["model"] = "FUSION"
        df["run_type"] = "main_subject_dependent"

        rows.append(df)

    if len(rows) == 0:
        print(
            "No FiLM stats found. Did you add film_stats to the "
            "fusion return dictionary and rerun all_fusion_sd?"
        )
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)

def build_main_film_summary_table(main_film_stats_df):
    """
    Compact summary table for the main split FiLM analysis.
    """
    if main_film_stats_df is None or len(main_film_stats_df) == 0:
        return pd.DataFrame()

    metrics = [
        "gamma_mean",
        "gamma_std",
        "gamma_deviation_from_1",
        "beta_mean",
        "beta_std",
        "beta_abs_mean",
        "gate_mean",
        "gate_std",
        "relative_modulation",
    ]

    rows = []

    for metric in metrics:
        vals = pd.to_numeric(
            main_film_stats_df[metric],
            errors="coerce",
        ).dropna()

        rows.append({
            "metric": metric,
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        })

    return pd.DataFrame(rows)

def _extract_training_stats_from_history(history):
    """
    Extract useful training-side summary metrics from a history dict.
    """
    if history is None:
        return {
            "best_train_acc": None,
            "last_train_acc": None,
            "best_train_loss": None,
            "last_train_loss": None,
        }

    train_acc = history.get("train_acc", [])
    train_loss = history.get("train_loss", [])

    return {
        "best_train_acc": float(max(train_acc)) if len(train_acc) > 0 else None,
        "last_train_acc": float(train_acc[-1]) if len(train_acc) > 0 else None,
        "best_train_loss": float(min(train_loss)) if len(train_loss) > 0 else None,
        "last_train_loss": float(train_loss[-1]) if len(train_loss) > 0 else None,
    }

def build_subject_metrics_table(std_results):
    """
    Build one per-subject metrics table from standardized results,
    including training-side summary metrics and train/val gaps.
    """
    rows = []

    for sid, entry in std_results["subject_results"].items():
        dm = entry.get("detailed_metrics") or {}
        history = entry.get("history")

        train_stats = _extract_training_stats_from_history(history)

        best_val_acc = entry.get("best_val_acc")
        last_val_acc = entry.get("last_val_acc")

        best_train_acc = train_stats["best_train_acc"]
        last_train_acc = train_stats["last_train_acc"]
        best_train_loss = train_stats["best_train_loss"]
        last_train_loss = train_stats["last_train_loss"]

        row = {
            "subject": sid,

            # validation-side metrics
            "best_val_acc": best_val_acc,
            "last_val_acc": last_val_acc,
            "acc": dm.get("acc"),
            "balanced_acc": dm.get("balanced_acc"),
            "precision": dm.get("precision"),
            "recall": dm.get("recall"),
            "f1": dm.get("f1"),
            "loss": dm.get("loss"),

            # training-side metrics
            "best_train_acc": best_train_acc,
            "last_train_acc": last_train_acc,
            "best_train_loss": best_train_loss,
            "last_train_loss": last_train_loss,

            # train-vs-val gaps
            "best_gap_acc": (
                best_train_acc - best_val_acc
                if best_train_acc is not None
                and best_val_acc is not None
                else None
            ),

            "last_gap_acc": (
                last_train_acc - last_val_acc
                if last_train_acc is not None
                and last_val_acc is not None
                else None
            ),

            "selected_stage": entry.get("selected_stage"),
            "save_path": entry.get("save_path"),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df

    return df.sort_values("subject").reset_index(drop=True)

def build_summary_table(std_results):
    """
    Build a one-row summary table with mean/std/min/max over subjects.
    """
    df = build_subject_metrics_table(std_results)

    if len(df) == 0:
        return pd.DataFrame()

    metric_cols = [
        # validation-side
        "best_val_acc",
        "last_val_acc",
        "acc",
        "balanced_acc",
        "precision",
        "recall",
        "f1",
        "loss",

        # training-side
        "best_train_acc",
        "last_train_acc",
        "best_train_loss",
        "last_train_loss",

        # gaps
        "best_gap_acc",
        "last_gap_acc",
    ]

    summary = {}

    for col in metric_cols:
        vals = pd.to_numeric(
            df[col],
            errors="coerce",
        ).dropna()

        if len(vals) == 0:
            continue

        summary[f"{col}_mean"] = float(vals.mean())

        summary[f"{col}_std"] = (
            float(vals.std(ddof=1))
            if len(vals) > 1
            else 0.0
        )

        summary[f"{col}_min"] = float(vals.min())
        summary[f"{col}_max"] = float(vals.max())

    summary["model_name"] = std_results.get("model_name")
    summary["source_type"] = std_results.get("source_type")
    summary["split_mode"] = std_results.get("split_mode")

    return pd.DataFrame([summary])

def build_confusion_matrix_table(std_results, sid):
    """
    Return the confusion matrix of one subject as a DataFrame.
    """
    entry = std_results["subject_results"][sid]
    dm = entry.get("detailed_metrics") or {}

    cm = np.array(
        dm.get(
            "confusion_matrix",
            [[0, 0], [0, 0]],
        )
    )

    return pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )

def build_prediction_table(std_results, sid):
    """
    Build a DataFrame containing y_true and y_pred for one subject.
    """
    entry = std_results["subject_results"][sid]
    dm = entry.get("detailed_metrics") or {}

    y_true = dm.get("y_true", [])
    y_pred = dm.get("y_pred", [])

    return pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "correct": [
            int(a == b)
            for a, b in zip(y_true, y_pred)
        ],
    })

def save_tables_to_csv(
    std_results,
    out_dir="metrics_exports",
    prefix=None,
):
    """
    Save the per-subject table and summary table to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    if prefix is None:
        prefix = std_results["model_name"].lower()

    subject_df = build_subject_metrics_table(std_results)
    summary_df = build_summary_table(std_results)

    subject_path = os.path.join(
        out_dir,
        f"{prefix}_subject_metrics.csv",
    )

    summary_path = os.path.join(
        out_dir,
        f"{prefix}_summary_metrics.csv",
    )

    subject_df.to_csv(
        subject_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("Saved:", subject_path)
    print("Saved:", summary_path)

