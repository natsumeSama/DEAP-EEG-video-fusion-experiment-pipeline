import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

from .results import (
    MODEL_DISPLAY_NAMES,
    build_subject_metrics_table,
    build_prediction_table,
)

# =========================================================
# VISUALIZATION 1: FINAL CV BARPLOT
# =========================================================

def plot_cv_summary_bar(cv_summary_df, metric="best_val_acc", title=None):
    """
    Bar plot of final mean ± std across folds.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    df = cv_summary_df.copy()
    df = df.sort_values(mean_col, ascending=False)

    labels = df["model_display"].tolist()
    means = df[mean_col].values
    stds = df[std_col].values

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=stds, capsize=5)

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.ylim(0, 1.05)

    if title is None:
        title = f"4-fold CV performance - {metric}"

    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================================================
# VISUALIZATION 2: FOLD STABILITY
# =========================================================

def plot_cv_fold_lines(cv_fold_df, metric="best_val_acc", title=None):
    """
    Line plot showing model performance across folds.
    """
    plt.figure(figsize=(10, 5))

    for model_display, g in cv_fold_df.groupby("model_display"):
        g = g.sort_values("fold")
        plt.plot(g["fold"], g[metric], marker="o", label=model_display)

    plt.xlabel("Fold")
    plt.ylabel(metric)
    plt.ylim(0, 1.05)

    if title is None:
        title = f"Performance stability across folds - {metric}"

    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# VISUALIZATION 3: SUBJECT HEATMAP
# =========================================================

def plot_subject_heatmap(cv_subject_df, model="FUSION", metric="best_val_acc", title=None):
    """
    Heatmap: subjects x folds for one model.
    """
    df = cv_subject_df[cv_subject_df["model"] == model].copy()

    pivot = df.pivot(index="subject", columns="fold", values=metric)
    pivot = pivot.sort_index()

    plt.figure(figsize=(8, max(5, len(pivot) * 0.3)))
    plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    plt.colorbar(label=metric)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
    plt.yticks(np.arange(len(pivot.index)), pivot.index)

    plt.xlabel("Fold")
    plt.ylabel("Subject")

    if title is None:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        title = f"{display_name} subject performance across folds"

    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================================================
# VISUALIZATION 4: MODEL DIFFERENCE PER FOLD
# =========================================================

def plot_cv_model_difference(cv_fold_df, model_a="FUSION", model_b="CONCAT", metric="best_val_acc"):
    """
    Plot fold-by-fold difference between two models.

    Positive value means model_a is better.
    Negative value means model_b is better.
    """
    pivot = cv_fold_df.pivot(index="fold", columns="model", values=metric)

    if model_a not in pivot.columns:
        raise ValueError(f"{model_a} not found.")
    if model_b not in pivot.columns:
        raise ValueError(f"{model_b} not found.")

    diff = pivot[model_a] - pivot[model_b]

    plt.figure(figsize=(8, 4))
    plt.bar(diff.index.astype(str), diff.values)
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Fold")
    plt.ylabel(f"{model_a} - {model_b}")
    plt.title(f"Fold-wise difference: {model_a} vs {model_b} ({metric})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()



# =========================================================
# FiLM PARAMETER VISUALIZATIONS
# =========================================================

def plot_film_parameter_histograms(cv_film_stats_df):
    """
    Histograms of key FiLM modulation statistics.
    """
    if cv_film_stats_df is None or len(cv_film_stats_df) == 0:
        print("No FiLM stats available.")
        return

    cols = [
        "gamma_mean",
        "gamma_std",
        "gamma_deviation_from_1",
        "beta_mean",
        "beta_std",
        "beta_abs_mean",
        "gate_mean",
        "relative_modulation",
    ]

    existing_cols = [c for c in cols if c in cv_film_stats_df.columns]

    cv_film_stats_df[existing_cols].hist(figsize=(14, 10), bins=25)
    plt.suptitle("FiLM modulation statistics across validation trials")
    plt.tight_layout()
    plt.show()


def plot_film_boxplot_by_label(cv_film_stats_df, column):
    """
    Boxplot of one FiLM statistic grouped by label.
    """
    if cv_film_stats_df is None or len(cv_film_stats_df) == 0:
        print("No FiLM stats available.")
        return

    labels = sorted(cv_film_stats_df["label"].dropna().unique())
    data = [
        cv_film_stats_df[cv_film_stats_df["label"] == lab][column].dropna().values
        for lab in labels
    ]

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[str(lab) for lab in labels])
    plt.xlabel("Label")
    plt.ylabel(column)
    plt.title(f"{column} by label")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_film_boxplot_correct_vs_wrong(cv_film_stats_df, column):
    """
    Boxplot of one FiLM statistic for correct vs wrong predictions.
    """
    if cv_film_stats_df is None or len(cv_film_stats_df) == 0:
        print("No FiLM stats available.")
        return

    data = [
        cv_film_stats_df[cv_film_stats_df["correct"] == 0][column].dropna().values,
        cv_film_stats_df[cv_film_stats_df["correct"] == 1][column].dropna().values,
    ]

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=["Wrong", "Correct"])
    plt.xlabel("Prediction")
    plt.ylabel(column)
    plt.title(f"{column}: correct vs wrong predictions")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def _get_history(std_results, sid):
    """
    Fetch subject history safely.
    """
    entry = std_results["subject_results"][sid]
    history = entry.get("history")

    if history is None:
        raise ValueError(
            f"No history found for subject '{sid}'."
        )

    return history


def _subject_ids(std_results):
    """
    Return sorted subject ids.
    """
    return sorted(
        std_results["subject_results"].keys()
    )

def plot_subject_curves(std_results, sid):
    """
    Plot train/val loss and accuracy for one subject.
    """
    history = _get_history(std_results, sid)

    epochs = np.arange(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="train_loss",
    )
    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="val_loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{std_results['model_name']} - {sid} - Loss"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(
        epochs,
        history["train_acc"],
        marker="o",
        label="train_acc",
    )
    plt.plot(
        epochs,
        history["val_acc"],
        marker="o",
        label="val_acc",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(
        f"{std_results['model_name']} - {sid} - Accuracy"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_subject_confusion_matrix(
    std_results,
    sid,
    normalize=False,
):
    """
    Plot the confusion matrix for one subject.
    """
    entry = std_results["subject_results"][sid]
    dm = entry.get("detailed_metrics") or {}

    cm = np.array(
        dm.get(
            "confusion_matrix",
            [[0, 0], [0, 0]],
        )
    )

    if normalize:
        cm = cm.astype(float)

        row_sums = cm.sum(
            axis=1,
            keepdims=True,
        )
        row_sums[row_sums == 0] = 1.0

        cm = cm / row_sums
        values_format = ".2f"

    else:
        cm = cm.astype(int)
        values_format = "d"

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1],
    )

    disp.plot(
        values_format=values_format
    )

    plt.title(
        f"{std_results['model_name']} - {sid} - Confusion Matrix"
        + (" (Normalized)" if normalize else "")
    )

    plt.show()

def plot_subject_prediction_table(
    std_results,
    sid,
    n_rows=20,
):
    """
    Return the first rows of the prediction table
    for quick inspection.
    """
    return build_prediction_table(
        std_results,
        sid,
    ).head(n_rows)

def plot_metric_bar(
    std_results,
    metric="best_val_acc",
    sort=True,
):
    """
    Plot one metric across all subjects.
    """
    df = build_subject_metrics_table(
        std_results
    )

    if len(df) == 0:
        print("Empty results.")
        return

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. "
            f"Available: {list(df.columns)}"
        )

    plot_df = df[
        ["subject", metric]
    ].copy()

    if sort:
        plot_df = plot_df.sort_values(
            metric,
            ascending=False,
        )

    plt.figure(figsize=(12, 5))

    plt.bar(
        plot_df["subject"],
        plot_df[metric],
    )

    plt.xlabel("Subject")
    plt.ylabel(metric)

    plt.title(
        f"{std_results['model_name']} - "
        f"{metric} across subjects"
    )

    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_mean_curves(std_results):
    """
    Plot mean train/val loss and accuracy across subjects,
    even when histories have different lengths.

    Strategy:
    - collect all available histories
    - pad shorter histories with NaN
    - compute nanmean / nanstd per epoch
    """
    sids = sorted(
        std_results["subject_results"].keys()
    )

    if len(sids) == 0:
        print("Empty results.")
        return

    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    for sid in sids:
        hist = std_results[
            "subject_results"
        ][sid].get("history")

        if hist is None:
            continue

        needed = [
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
        ]

        if not all(
            k in hist
            for k in needed
        ):
            continue

        train_loss_all.append(
            list(hist["train_loss"])
        )
        val_loss_all.append(
            list(hist["val_loss"])
        )
        train_acc_all.append(
            list(hist["train_acc"])
        )
        val_acc_all.append(
            list(hist["val_acc"])
        )

    if len(train_loss_all) == 0:
        print("No usable histories available.")
        return

    max_len = max(
        len(x)
        for x in train_loss_all
    )

    def pad_with_nan(seqs, max_len):
        arr = np.full(
            (len(seqs), max_len),
            np.nan,
            dtype=float,
        )

        for i, seq in enumerate(seqs):
            arr[i, :len(seq)] = seq

        return arr

    train_loss_all = pad_with_nan(
        train_loss_all,
        max_len,
    )
    val_loss_all = pad_with_nan(
        val_loss_all,
        max_len,
    )
    train_acc_all = pad_with_nan(
        train_acc_all,
        max_len,
    )
    val_acc_all = pad_with_nan(
        val_acc_all,
        max_len,
    )

    epochs = np.arange(
        1,
        max_len + 1,
    )

    train_loss_mean = np.nanmean(
        train_loss_all,
        axis=0,
    )
    val_loss_mean = np.nanmean(
        val_loss_all,
        axis=0,
    )

    train_loss_std = np.nanstd(
        train_loss_all,
        axis=0,
    )
    val_loss_std = np.nanstd(
        val_loss_all,
        axis=0,
    )

    train_acc_mean = np.nanmean(
        train_acc_all,
        axis=0,
    )
    val_acc_mean = np.nanmean(
        val_acc_all,
        axis=0,
    )

    train_acc_std = np.nanstd(
        train_acc_all,
        axis=0,
    )
    val_acc_std = np.nanstd(
        val_acc_all,
        axis=0,
    )

    plt.figure(figsize=(12, 4))

    # Mean loss
    plt.subplot(1, 2, 1)

    plt.plot(
        epochs,
        train_loss_mean,
        marker="o",
        label="mean_train_loss",
    )
    plt.plot(
        epochs,
        val_loss_mean,
        marker="o",
        label="mean_val_loss",
    )

    plt.fill_between(
        epochs,
        train_loss_mean - train_loss_std,
        train_loss_mean + train_loss_std,
        alpha=0.2,
    )

    plt.fill_between(
        epochs,
        val_loss_mean - val_loss_std,
        val_loss_mean + val_loss_std,
        alpha=0.2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{std_results['model_name']} - Mean Loss"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Mean accuracy
    plt.subplot(1, 2, 2)

    plt.plot(
        epochs,
        train_acc_mean,
        marker="o",
        label="mean_train_acc",
    )
    plt.plot(
        epochs,
        val_acc_mean,
        marker="o",
        label="mean_val_acc",
    )

    plt.fill_between(
        epochs,
        train_acc_mean - train_acc_std,
        train_acc_mean + train_acc_std,
        alpha=0.2,
    )

    plt.fill_between(
        epochs,
        val_acc_mean - val_acc_std,
        val_acc_mean + val_acc_std,
        alpha=0.2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"{std_results['model_name']} - Mean Accuracy"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_all_subject_confusions(
    std_results,
    normalize=False,
    ncols=4,
):
    """
    Plot confusion matrices for all subjects in a grid.
    """
    sids = sorted(
        std_results["subject_results"].keys()
    )

    if len(sids) == 0:
        print("Empty results.")
        return

    n = len(sids)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            4 * ncols,
            4 * nrows,
        ),
    )

    axes = np.array(
        axes
    ).reshape(-1)

    for ax, sid in zip(
        axes,
        sids,
    ):
        entry = std_results[
            "subject_results"
        ][sid]

        dm = (
            entry.get("detailed_metrics")
            or {}
        )

        cm = np.array(
            dm.get(
                "confusion_matrix",
                [[0, 0], [0, 0]],
            )
        )

        if normalize:
            cm = cm.astype(float)

            row_sums = cm.sum(
                axis=1,
                keepdims=True,
            )

            row_sums[
                row_sums == 0
            ] = 1.0

            cm = cm / row_sums

        else:
            cm = cm.astype(int)

        ax.imshow(cm)
        ax.set_title(sid)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xlabel("Pred")
        ax.set_ylabel("True")

        for i in range(2):
            for j in range(2):

                txt = (
                    f"{cm[i, j]:.2f}"
                    if normalize
                    else f"{int(cm[i, j])}"
                )

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                )

    for ax in axes[len(sids):]:
        ax.axis("off")

    plt.suptitle(
        f"{std_results['model_name']} - Confusion Matrices"
        + (
            " (Normalized)"
            if normalize
            else ""
        )
    )

    plt.tight_layout()
    plt.show()

def plot_metric_box(
    std_results,
    metrics=None,
):
    """
    Plot boxplots for selected metrics across subjects.
    """
    if metrics is None:
        metrics = [
            "best_val_acc",
            "acc",
            "balanced_acc",
            "precision",
            "recall",
            "f1",
        ]

    df = build_subject_metrics_table(
        std_results
    )

    if len(df) == 0:
        print("Empty results.")
        return

    existing = [
        m for m in metrics
        if m in df.columns
    ]

    if len(existing) == 0:
        print("No requested metrics found.")
        return

    plot_data = [
        pd.to_numeric(
            df[m],
            errors="coerce",
        )
        .dropna()
        .values
        for m in existing
    ]

    plt.figure(figsize=(10, 5))

    plt.boxplot(
        plot_data,
        labels=existing,
    )

    plt.ylabel("Value")

    plt.title(
        f"{std_results['model_name']} - Metric Distributions"
    )

    plt.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()
    plt.show()