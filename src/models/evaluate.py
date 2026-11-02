"""Evaluation metrics, business threshold experimentation, curves, and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def threshold_table(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    proba: Sequence[float] | np.ndarray,
    thresholds: Sequence[float] | np.ndarray | None = None,
    cost_ratio: float = 4.0,
) -> pd.DataFrame:
    """Compute confusion counts, class-1 metrics, and expected business cost per loan at each threshold."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    else:
        thresholds = np.asarray(thresholds, dtype=float)

    flagged = probabilities[None, :] >= thresholds[:, None]

    tp = (flagged & (y == 1)).sum(axis=1)
    fp = (flagged & (y == 0)).sum(axis=1)
    fn = (y == 1).sum() - tp
    tn = (y == 0).sum() - fp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    specificity = tn / np.maximum(tn + fp, 1)

    return pd.DataFrame(
        {
            "Threshold": thresholds,
            "Accuracy": (tp + tn) / len(y),
            "Precision_1": precision,
            "Recall_1": recall,
            "Recall_0": specificity,
            "F1_1": f1,
            "TN": tn.astype(int),
            "FP": fp.astype(int),
            "FN": fn.astype(int),
            "TP": tp.astype(int),
            "Cost": (cost_ratio * fn + fp) / len(y),
        }
    )


def best_threshold_row(
    table: pd.DataFrame,
    min_recall: float = 0.50,
) -> pd.Series:
    """Find the lowest-cost row among thresholds meeting the minimum recall requirement."""
    eligible = table[table["Recall_1"] >= min_recall]
    if eligible.empty:
        # Fallback if no threshold meets min_recall: highest recall row
        return table.loc[table["Recall_1"].idxmax()]
    return eligible.loc[eligible["Cost"].idxmin()]


def calculate_comprehensive_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    proba: Sequence[float] | np.ndarray,
    threshold: float,
    cost_ratio: float = 4.0,
    prefix: str = "valid",
) -> dict[str, float | int]:
    """Calculate all standard and business metrics for a given dataset and threshold."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(proba, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    default_rate = float(y.mean())
    cost_per_loan = float((fp + cost_ratio * fn) / len(y))
    approve_all_cost = float(cost_ratio * default_rate)

    roc_auc = float(roc_auc_score(y, probabilities))
    pr_auc = float(average_precision_score(y, probabilities))

    prec_0 = float(precision_score(y, y_pred, pos_label=0, zero_division=0))
    prec_1 = float(precision_score(y, y_pred, pos_label=1, zero_division=0))
    rec_0 = float(recall_score(y, y_pred, pos_label=0, zero_division=0))
    rec_1 = float(recall_score(y, y_pred, pos_label=1, zero_division=0))
    f1_0 = float(f1_score(y, y_pred, pos_label=0, zero_division=0))
    f1_1 = float(f1_score(y, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y, y_pred))

    return {
        f"{prefix}_roc_auc": roc_auc,
        f"{prefix}_pr_auc": pr_auc,
        f"{prefix}_accuracy": acc,
        f"{prefix}_precision_0": prec_0,
        f"{prefix}_precision_1": prec_1,
        f"{prefix}_recall_0": rec_0,
        f"{prefix}_recall_1": rec_1,
        f"{prefix}_f1_0": f1_0,
        f"{prefix}_f1_1": f1_1,
        f"{prefix}_tn": int(tn),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
        f"{prefix}_tp": int(tp),
        f"{prefix}_cost": cost_per_loan,
        f"{prefix}_approve_all_cost": approve_all_cost,
        f"{prefix}_default_rate": default_rate,
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_cost_ratio": float(cost_ratio),
    }


def compute_precision_at_recalls(
    y_true: Sequence[int] | np.ndarray,
    proba: Sequence[float] | np.ndarray,
    recall_levels: Sequence[float] = (0.3, 0.4, 0.5, 0.6),
) -> dict[str, float]:
    """Calculate maximum reachable precision at specified minimum recall targets."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    result = {}
    for level in recall_levels:
        mask = recall >= level
        val = float(precision[mask].max()) if np.any(mask) else 0.0
        result[f"precision_at_recall_{int(level * 100)}"] = val
    return result


def cost_ratio_sensitivity(
    y_true: Sequence[int] | np.ndarray,
    proba: Sequence[float] | np.ndarray,
    cost_ratios: Sequence[int | float] = (1, 2, 3, 4, 5, 8),
    thresholds: Sequence[float] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Analyze how the optimal decision threshold shifts when FN:FP cost ratio changes."""
    sensitivity = []
    for ratio in sorted(set(cost_ratios)):
        t = threshold_table(y_true, proba, thresholds=thresholds, cost_ratio=float(ratio))
        best = t.loc[t["Cost"].idxmin()]
        sensitivity.append(
            {
                "Cost_ratio_FN_to_FP": ratio,
                "Optimal_Threshold": float(best["Threshold"]),
                "Precision_1": float(best["Precision_1"]),
                "Recall_1": float(best["Recall_1"]),
                "Recall_0": float(best["Recall_0"]),
                "F1_1": float(best["F1_1"]),
                "Expected_Cost_per_Loan": float(best["Cost"]),
                "Approve_All_Cost": float(ratio * np.mean(y_true)),
            }
        )
    return pd.DataFrame(sensitivity)


# --- Plotting Functions ---


def plot_roc_curve(
    y_true: Sequence[int] | np.ndarray,
    proba: Sequence[float] | np.ndarray,
    model_name: str,
    output_path: Path | str,
) -> None:
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc_score = roc_auc_score(y_true, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"{model_name} (ROC-AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title(f"Receiver Operating Characteristic - {model_name}", fontsize=12, pad=10)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: Sequence[int] | np.ndarray,
    proba: Sequence[float] | np.ndarray,
    model_name: str,
    output_path: Path | str,
) -> None:
    """Generate and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#ff7f0e", lw=2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline, color="grey", lw=1.5, linestyle="--", label=f"Default Rate ({baseline:.4f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Class 1)", fontsize=11)
    ax.set_ylabel("Precision (Class 1)", fontsize=11)
    ax.set_title(f"Precision-Recall Curve - {model_name}", fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    model_name: str,
    output_path: Path | str,
) -> None:
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Non-Default (0)", "Default (1)"],
        yticklabels=["Non-Default (0)", "Default (1)"],
        title=f"Confusion Matrix - {model_name}",
        ylabel="True label",
        xlabel="Predicted label",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_business_cost_curve(
    table: pd.DataFrame,
    cost_ratio: float,
    approve_all_cost: float,
    selected_threshold: float,
    output_path: Path | str,
) -> None:
    """Plot business cost vs decision threshold with reference approve-all line and selected threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["Threshold"], table["Cost"], color="#2ca02c", lw=2, label=f"Expected Cost (FN={cost_ratio}x FP)")
    ax.axhline(approve_all_cost, color="red", linestyle="--", lw=1.5, label=f"Approve All ({approve_all_cost:.4f})")
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Selected Threshold ({selected_threshold:.2f})",
    )

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Expected Cost per Loan", fontsize=11)
    ax.set_title("Expected Business Cost vs. Decision Threshold", fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_vs_threshold(
    table: pd.DataFrame,
    selected_threshold: float,
    min_recall: float,
    output_path: Path | str,
) -> None:
    """Plot Precision, Recall, and F1 across threshold grid."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["Threshold"], table["Precision_1"], label="Precision (Class 1)", lw=2, color="#1f77b4")
    ax.plot(table["Threshold"], table["Recall_1"], label="Recall (Class 1)", lw=2, color="#ff7f0e")
    ax.plot(table["Threshold"], table["F1_1"], label="F1 Score (Class 1)", lw=2, color="#2ca02c")
    ax.axhline(min_recall, color="grey", linestyle="--", lw=1.2, label=f"Min Recall Floor ({min_recall:.0%})")
    ax.axvline(
        selected_threshold,
        color="black",
        linestyle=":",
        lw=1.5,
        label=f"Selected Threshold ({selected_threshold:.2f})",
    )

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Precision, Recall, and F1 vs. Decision Threshold", fontsize=12, pad=10)
    ax.legend(loc="center right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_classification_report_file(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    output_path: Path | str,
) -> None:
    """Write text classification report to file."""
    rep = classification_report(y_true, y_pred, target_names=["Non-Default (0)", "Default (1)"], digits=4)
    Path(output_path).write_text(rep, encoding="utf-8")
