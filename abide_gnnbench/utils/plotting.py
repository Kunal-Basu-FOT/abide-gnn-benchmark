# abide_gnnbench/utils/plotting.py

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, accuracy_score,
    recall_score, f1_score,
)
from sklearn.calibration import calibration_curve

from abide_gnnbench.config import PlotConfig


def _save_fig(fig: plt.Figure, name: str, cfg: PlotConfig) -> None:
    os.makedirs(cfg.save_dir, exist_ok=True)
    for fmt in cfg.formats:
        fig.savefig(
            os.path.join(cfg.save_dir, f"{name}.{fmt}"),
            dpi=cfg.dpi,
            bbox_inches="tight",
        )


def _pad_and_mean(list_of_lists: List[list]) -> np.ndarray:
    # FIX: truncate to the shortest fold instead of padding with the last
    # value of each fold. The old padding approach repeated each
    # early-stopped fold's final (often best) value for all remaining
    # epochs, which dragged the mean curve upward/away from the true
    # pooled performance at later epochs. Truncating to the shortest fold
    # length means every plotted point is a genuine average over the same
    # number of folds, with no padding bias.
    min_len = min(len(l) for l in list_of_lists)
    truncated = [l[:min_len] for l in list_of_lists]
    return np.mean(truncated, axis=0)


def plot_dataset_overview(
    y: np.ndarray,
    site_ids: np.ndarray,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
) -> None:
    plt.style.use(cfg.style)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    class_counts = pd.Series(y).map({1: "ASD", 0: "TD"}).value_counts()
    axes[0].bar(class_counts.index, class_counts.values,
                color=["#e74c3c", "#3498db"], edgecolor="black")
    axes[0].set_title("Overall Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Diagnosis")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 2, str(v), ha="center", fontweight="bold")

    site_counts = pd.Series(site_ids).value_counts().sort_values(ascending=False)
    axes[1].bar(range(len(site_counts)), site_counts.values, color="#2ecc71", edgecolor="black")
    axes[1].set_xticks(range(len(site_counts)))
    axes[1].set_xticklabels(site_counts.index, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Site Distribution", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Site")
    axes[1].set_ylabel("Count")

    plt.suptitle(f"Dataset Overview — {atlas}", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_dataset_overview", cfg)
    plt.show()


def plot_training_curves(
    fold_histories: List[Dict[str, list]],
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    pooled_acc: Optional[float] = None,
    pooled_auroc: Optional[float] = None,
) -> None:
    plt.style.use(cfg.style)

    all_train_loss = [h["train_loss"] for h in fold_histories]
    all_val_loss = [h["val_loss"] for h in fold_histories]
    all_train_acc = [h["train_acc"] for h in fold_histories]
    all_val_acc = [h["val_acc"] for h in fold_histories]
    all_val_auroc = [h["val_auroc"] for h in fold_histories]

    mean_train_loss = _pad_and_mean(all_train_loss)
    mean_val_loss = _pad_and_mean(all_val_loss)
    mean_train_acc = _pad_and_mean(all_train_acc)
    mean_val_acc = _pad_and_mean(all_val_acc)
    mean_val_auroc = _pad_and_mean(all_val_auroc)

    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize_wide)

    ep_loss = np.arange(1, len(mean_train_loss) + 1)
    axes[0, 0].plot(ep_loss, mean_train_loss, color="#3498db", lw=2.5, label="Mean Train Loss")
    axes[0, 0].plot(ep_loss, mean_val_loss, color="#e67e22", lw=2.5, label="Mean Val Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title(f"{atlas} {model_name} — Mean Train vs Val Loss", fontsize=12, fontweight="bold")
    axes[0, 0].legend()

    axes[0, 1].plot(ep_loss, mean_train_loss, color="#3498db", lw=2.5, label="Mean Train Loss")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title(f"{atlas} {model_name} — Mean Training Loss", fontsize=12, fontweight="bold")
    axes[0, 1].legend()

    # NOTE: Mean Val Acc / Mean Val AUROC below are computed on each fold's
    # internal validation split (used for early stopping), then averaged
    # across folds epoch-by-epoch. This is NOT the same data or aggregation
    # as the pooled OOF test metrics printed after CV finishes — those are
    # scored once, by concatenating each fold's held-out TEST predictions.
    # A gap between this curve and the pooled metric is expected and is not
    # a plotting bug. Passing pooled_acc/pooled_auroc draws a reference line
    # so that gap is visible and explained instead of looking inflated.
    ep_acc = np.arange(1, len(mean_train_acc) + 1)
    axes[1, 0].plot(ep_acc, mean_train_acc, color="#3498db", lw=2.5, label="Mean Train Acc")
    axes[1, 0].plot(ep_acc, mean_val_acc, color="#e67e22", lw=2.5, label="Mean Val Acc")
    if pooled_acc is not None:
        axes[1, 0].axhline(pooled_acc, color="black", linestyle="--", lw=1.5,
                            label=f"Pooled OOF Acc={pooled_acc:.4f}")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title(f"{atlas} {model_name} — Mean Train vs Val Accuracy", fontsize=12, fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].set_ylim([0, 1.1])

    ep_auc = np.arange(1, len(mean_val_auroc) + 1)
    axes[1, 1].plot(ep_auc, mean_val_auroc, color="#9b59b6", lw=2.5, label="Mean Val AUROC")
    if pooled_auroc is not None:
        axes[1, 1].axhline(pooled_auroc, color="black", linestyle="--", lw=1.5,
                            label=f"Pooled OOF AUROC={pooled_auroc:.4f}")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("AUROC")
    axes[1, 1].set_title(f"{atlas} {model_name} — Mean Val AUROC", fontsize=12, fontweight="bold")
    axes[1, 1].legend(); axes[1, 1].set_ylim([0, 1.1])

    plt.suptitle(
        f"{atlas} {model_name} — Mean Training Curves ({len(fold_histories)}-Fold CV)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_training_curves", cfg)
    plt.show()



def plot_per_fold_curves(
    fold_histories: List[Dict[str, list]],
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
) -> None:
    plt.style.use(cfg.style)
    n_folds = len(fold_histories)
    palette = plt.cm.tab10(np.linspace(0, 1, n_folds))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, h in enumerate(fold_histories):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["val_loss"], color=palette[i], alpha=0.7, lw=1.2, label=f"Fold {i+1}")
        axes[1].plot(ep, h["val_acc"], color=palette[i], alpha=0.7, lw=1.2)
        axes[2].plot(np.arange(1, len(h["val_auroc"]) + 1), h["val_auroc"],
                     color=palette[i], alpha=0.7, lw=1.2)

    axes[0].set_title("Val Loss per Fold", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(fontsize=7)
    axes[1].set_title("Val Accuracy per Fold", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim([0, 1.1])
    axes[2].set_title("Val AUROC per Fold", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("AUROC"); axes[2].set_ylim([0, 1.1])

    plt.suptitle(f"{atlas} {model_name} — Per-Fold Validation Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_per_fold_curves", cfg)
    plt.show()


def plot_core_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    tag: str = "",
) -> None:
    plt.style.use(cfg.style)
    label = f"{model_name}" + (f" — {tag}" if tag else "")
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize_wide)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    axes[0, 0].plot(fpr, tpr, color="#e74c3c", lw=2, label=f"{label} (AUC = {auc_val:.4f})")
    axes[0, 0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, 0].set_xlabel("False Positive Rate"); axes[0, 0].set_ylabel("True Positive Rate")
    axes[0, 0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[0, 0].legend(loc="lower right"); axes[0, 0].set_xlim([0, 1]); axes[0, 0].set_ylim([0, 1.02])

    # FIX: removed the "Baseline" horizontal line (axhline at y_true.mean())
    # from the Precision-Recall plot, per request.
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[0, 1].plot(rec, prec, color="#3498db", lw=2, label=f"{label} (AP = {ap:.4f})")
    axes[0, 1].set_xlabel("Recall"); axes[0, 1].set_ylabel("Precision")
    axes[0, 1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
    axes[0, 1].legend(loc="upper right")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 0],
                xticklabels=["TD", "ASD"], yticklabels=["TD", "ASD"])
    axes[1, 0].set_title("Confusion Matrix — Pooled OOF", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Predicted"); axes[1, 0].set_ylabel("True")

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[1, 1].plot(mean_pred, fraction_pos, "s-", color="#e67e22", lw=2, label=label)
    axes[1, 1].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[1, 1].set_xlabel("Mean Predicted Probability"); axes[1, 1].set_ylabel("Fraction of Positives")
    axes[1, 1].set_title("Calibration Curve", fontsize=13, fontweight="bold")
    axes[1, 1].legend()

    plt.suptitle(f"{atlas} {model_name} — Core Performance Plots ({tag})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_{tag}_core_eval", cfg)
    plt.show()


def plot_probability_and_foldwise(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold_metrics_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    tag: str = "",
) -> None:
    plt.style.use(cfg.style)
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize_wide)

    for label, color, name in [(0, "#3498db", "TD"), (1, "#e74c3c", "ASD")]:
        mask = y_true == label
        axes[0, 0].hist(y_prob[mask], bins=30, alpha=0.6, color=color,
                        label=name, edgecolor="white")
    axes[0, 0].axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold 0.5")
    axes[0, 0].set_xlabel("Predicted Probability"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Predicted Probability Distribution by Class", fontsize=12, fontweight="bold")
    axes[0, 0].legend()

    folds = fold_metrics_df.index.tolist()
    aurocs = fold_metrics_df["AUROC"].values
    colors_bar = ["#e74c3c" if v == min(aurocs) else ("#2ecc71" if v == max(aurocs) else "#3498db")
                  for v in aurocs]
    axes[0, 1].bar(folds, aurocs, color=colors_bar, edgecolor="black")
    axes[0, 1].axhline(np.mean(aurocs), color="black", linestyle="--", lw=1.5,
                       label=f"Mean={np.mean(aurocs):.4f}")
    axes[0, 1].set_xlabel("Fold"); axes[0, 1].set_ylabel("AUROC")
    axes[0, 1].set_title("Fold-wise AUROC", fontsize=12, fontweight="bold")
    axes[0, 1].legend(); axes[0, 1].set_ylim([0, 1])

    accs = fold_metrics_df["Accuracy"].values
    axes[1, 0].bar(folds, accs, color="#9b59b6", edgecolor="black")
    axes[1, 0].axhline(np.mean(accs), color="black", linestyle="--", lw=1.5,
                       label=f"Mean={np.mean(accs):.4f}")
    axes[1, 0].set_xlabel("Fold"); axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Fold-wise Accuracy", fontsize=12, fontweight="bold")
    axes[1, 0].legend(); axes[1, 0].set_ylim([0, 1])

    f1s = fold_metrics_df["F1"].values
    axes[1, 1].bar(folds, f1s, color="#e67e22", edgecolor="black")
    axes[1, 1].axhline(np.mean(f1s), color="black", linestyle="--", lw=1.5,
                       label=f"Mean={np.mean(f1s):.4f}")
    axes[1, 1].set_xlabel("Fold"); axes[1, 1].set_ylabel("F1-Score")
    axes[1, 1].set_title("Fold-wise F1-Score", fontsize=12, fontweight="bold")
    axes[1, 1].legend(); axes[1, 1].set_ylim([0, 1])

    plt.suptitle(f"{atlas} {model_name} — Fold-wise Performance ({tag})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_{tag}_foldwise", cfg)
    plt.show()


def plot_metric_summary(
    ci: Dict[str, Tuple[float, float, float]],
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    tag: str = "",
) -> None:
    plt.style.use(cfg.style)
    plot_metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision",
                    "F1", "AUROC", "AUPRC", "Balanced_Acc"]
    means = [ci[m][0] for m in plot_metrics]
    lowers = [ci[m][0] - ci[m][1] for m in plot_metrics]
    uppers = [ci[m][2] - ci[m][0] for m in plot_metrics]
    yerr = np.array([lowers, uppers])

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(plot_metrics))
    bars = ax.bar(x, means, yerr=yerr, capsize=5, color="#3498db", edgecolor="black",
                  error_kw=dict(ecolor="black", elinewidth=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Metric Summary with 95% Bootstrap CI — {atlas} {model_name} ({tag})",
        fontsize=13, fontweight="bold",
    )
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_{tag}_metric_summary", cfg)
    plt.show()


def plot_metric_boxplots(
    fold_metrics_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    tag: str = "",
) -> None:
    plt.style.use(cfg.style)
    box_metrics = ["Accuracy", "Sensitivity", "Specificity", "F1", "AUROC", "AUPRC"]
    available = [m for m in box_metrics if m in fold_metrics_df.columns]
    data_to_plot = [fold_metrics_df[m].values for m in available]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data_to_plot, labels=available, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Metric Distributions across Folds — {atlas} {model_name} ({tag})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_{tag}_metric_boxplots", cfg)
    plt.show()


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    tag: str = "",
) -> None:
    plt.style.use(cfg.style)
    thresholds = np.linspace(0.01, 0.99, 200)
    accs_t, sens_t, spec_t, f1s_t = [], [], [], []

    for t in thresholds:
        preds_t = (y_prob >= t).astype(int)
        accs_t.append(accuracy_score(y_true, preds_t))
        sens_t.append(recall_score(y_true, preds_t, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, preds_t).ravel()
        spec_t.append(tn / (tn + fp + 1e-9))
        f1s_t.append(f1_score(y_true, preds_t, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, accs_t, label="Accuracy", lw=2, color="#3498db")
    ax.plot(thresholds, sens_t, label="Sensitivity", lw=2, color="#e74c3c")
    ax.plot(thresholds, spec_t, label="Specificity", lw=2, color="#2ecc71")
    ax.plot(thresholds, f1s_t, label="F1-Score", lw=2, color="#e67e22")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Default (0.5)")
    ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
    ax.set_title(f"Metrics vs Classification Threshold — {atlas} {model_name} ({tag})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_{tag}_threshold_analysis", cfg)
    plt.show()


def plot_fold_class_distribution(
    skf,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
) -> None:
    plt.style.use(cfg.style)
    records = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        records.append({
            "Fold": fold_idx + 1,
            "Train ASD": int(y[train_idx].sum()),
            "Train TD": int((y[train_idx] == 0).sum()),
            "Test ASD": int(y[test_idx].sum()),
            "Test TD": int((y[test_idx] == 0).sum()),
        })
    fcd = pd.DataFrame(records).set_index("Fold")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_fold = np.arange(1, n_splits + 1)
    w = 0.35
    for ax, (asd_col, td_col, title) in zip(axes, [
        ("Train ASD", "Train TD", "Training Set per Fold"),
        ("Test ASD", "Test TD", "Test Set per Fold"),
    ]):
        ax.bar(x_fold - w / 2, fcd[asd_col], width=w, label="ASD", color="#e74c3c", edgecolor="black")
        ax.bar(x_fold + w / 2, fcd[td_col], width=w, label="TD", color="#3498db", edgecolor="black")
        ax.set_xlabel("Fold"); ax.set_ylabel("Count")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x_fold); ax.legend()
    plt.suptitle("Fold-wise Class Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_fold_class_dist", cfg)
    plt.show()


def plot_tuning_results(
    tuning_df: pd.DataFrame,
    model_name: str,
    atlas: str,
    cfg: PlotConfig,
    top_n: int = 20,
) -> None:
    plt.style.use(cfg.style)
    top_n = min(top_n, len(tuning_df))
    top = tuning_df.head(top_n)
    config_labels = [str(i + 1) for i in range(top_n)]

    has_f1 = "mean_F1" in tuning_df.columns
    fig, axes = plt.subplots(1, 2 if has_f1 else 1, figsize=cfg.figsize_medium)
    if not has_f1:
        axes = [axes]

    # Guard against duplicate column names (DataFrame returns 2-D if dupes exist)
    _auroc_col = top["mean_AUROC"]
    if isinstance(_auroc_col, pd.DataFrame):
        _auroc_col = _auroc_col.iloc[:, 0]
    mean_auroc_vals = np.asarray(_auroc_col.values, dtype=float).ravel()
    
    if "std_AUROC" in top.columns:
        xerr = np.nan_to_num(np.asarray(top["std_AUROC"].values, dtype=float), nan=0.0)
    else:
        xerr = np.zeros(top_n, dtype=float)

    axes[0].barh(range(top_n)[::-1], mean_auroc_vals,
                 xerr=xerr, capsize=3, color="#3498db", edgecolor="black")
    axes[0].set_yticks(range(top_n)[::-1])
    axes[0].set_yticklabels(config_labels)
    axes[0].set_xlabel("Mean AUROC")
    axes[0].set_title(f"Top {top_n} Configs by Mean AUROC", fontsize=12, fontweight="bold")

    if has_f1:
        mean_f1_vals = np.asarray(top["mean_F1"].values, dtype=float)
        axes[1].barh(range(top_n)[::-1], mean_f1_vals, color="#e67e22", edgecolor="black")
        axes[1].set_yticks(range(top_n)[::-1])
        axes[1].set_yticklabels(config_labels)
        axes[1].set_xlabel("Mean F1-Score")
        axes[1].set_title(f"Top {top_n} Configs by Mean F1", fontsize=12, fontweight="bold")

    plt.suptitle(f"Hyperparameter Tuning — {atlas} {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, f"{model_name}_{atlas}_tuning", cfg)
    plt.show()


def plot_multi_model_comparison(
    results: Dict[str, Dict[str, float]],
    atlas: str,
    cfg: PlotConfig,
) -> None:
    plt.style.use(cfg.style)
    models = list(results.keys())
    compare_metrics = ["Accuracy", "F1", "AUROC", "AUPRC", "Balanced_Acc"]
    x = np.arange(len(compare_metrics))
    width = 0.12
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (model_name, metrics) in enumerate(results.items()):
        vals = [metrics.get(m, 0.0) for m in compare_metrics]
        ax.bar(x + i * width, vals, width, label=model_name, color=palette[i], edgecolor="black")

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(compare_metrics)
    ax.set_ylabel("Score"); ax.set_ylim([0, 1.05])
    ax.set_title(f"Multi-Model Comparison — {atlas}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_fig(fig, f"multi_model_comparison_{atlas}", cfg)
    plt.show()