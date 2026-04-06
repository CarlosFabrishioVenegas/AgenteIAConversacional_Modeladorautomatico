"""
src/visualization/model_plots.py
Gráficos de evaluación del modelo: ROC, PR curve, Confusion Matrix.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
)


def plot_roc_curve(y_test, y_prob, model_name: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"{output_dir}/roc_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 ROC curve guardada: {path}")


def plot_precision_recall(y_test, y_prob, model_name: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#e74c3c", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    plt.tight_layout()
    path = f"{output_dir}/pr_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 PR curve guardada: {path}")


def plot_confusion_matrix(y_test, y_pred, model_name: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Compra", "Compra"],
                yticklabels=["No Compra", "Compra"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusión — {model_name}")
    plt.tight_layout()
    path = f"{output_dir}/confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 Confusion matrix guardada: {path}")


def plot_all(y_test, y_prob, y_pred, model_name: str, output_dir: str):
    plot_roc_curve(y_test, y_prob, model_name, output_dir)
    plot_precision_recall(y_test, y_prob, model_name, output_dir)
    plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
