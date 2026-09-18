import numpy as np
import matplotlib.pyplot as plt

from .results import MODEL_DISPLAY_NAMES

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
