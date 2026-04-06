"""
src/visualization/eda.py
Gráficos exploratorios para el EDA.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


def plot_distributions(
    df: pd.DataFrame,
    columnas: list,
    hue_col: str = None,
    output_dir: str = "outputs/reports/images",
    filename: str = "distribuciones.png",
):
    """Histogramas y boxplots para variables continuas."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    n = len(columnas)
    fig, axes = plt.subplots(
        nrows=2, ncols=n, figsize=(5 * n, 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, col in enumerate(columnas):
        sns.histplot(data=df, x=col, hue=hue_col, kde=True, ax=axes[0, idx])
        axes[0, idx].set_title(f"Distribución — {col}")
        if hue_col:
            sns.boxplot(data=df, x=col, hue=hue_col, ax=axes[1, idx])
        else:
            sns.boxplot(x=df[col], ax=axes[1, idx])
        axes[1, idx].set_title(f"Boxplot — {col}")

    plt.tight_layout()
    path = f"{output_dir}/{filename}"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 Gráfico guardado: {path}")


def plot_target_distribution(df: pd.DataFrame, target: str, output_dir: str = "outputs/reports/images"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Distribución del Target (incidencia_compra)")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Frecuencia")
    for i, v in enumerate(counts):
        ax.text(i, v + 100, f"{v:,} ({v/len(df):.1%})", ha="center", fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/target_distribution.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 Distribución target guardada: {path}")


def plot_correlation_matrix(
    df: pd.DataFrame,
    output_dir: str = "outputs/reports/images",
    max_cols: int = 20,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=np.number).iloc[:, :max_cols]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, fmt=".2f")
    ax.set_title("Matriz de Correlación")
    plt.tight_layout()
    path = f"{output_dir}/correlation_matrix.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  📊 Matriz de correlación guardada: {path}")
