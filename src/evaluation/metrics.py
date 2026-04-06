"""
src/evaluation/metrics.py
Cálculo y exportación de métricas de evaluación.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)


def compute_metrics(
    y_test,
    y_prob,
    y_pred,
    model_name: str,
    threshold: float,
    config: dict,
) -> dict:
    """
    Calcula métricas completas y las guarda en JSON.
    """
    metrics = {
        "model": model_name,
        "threshold": round(threshold, 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "n_test": int(len(y_test)),
        "n_positivos_real": int(y_test.sum()),
        "n_positivos_pred": int(y_pred.sum()),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    metrics["classification_report"] = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    print("\n  📊 MÉTRICAS FINALES")
    print(f"     AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"     F1        : {metrics['f1']:.4f}")
    print(f"     Precision : {metrics['precision']:.4f}")
    print(f"     Recall    : {metrics['recall']:.4f}")
    print(f"     Threshold : {metrics['threshold']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    metrics_dir = Path(config["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / "metrics.json"

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✅ Métricas guardadas: {output_path}")

    return metrics
