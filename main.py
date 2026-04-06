"""
main.py — Pipeline principal de entrenamiento
Uso: python main.py

Ejecuta de punta a punta:
  1. Carga de datos
  2. Preprocesamiento y outliers
  3. Clustering K-Means
  4. Feature Engineering
  5. Entrenamiento y selección de modelo
  6. Fine-tuning (opcional)
  7. Evaluación + métricas
  8. Explicabilidad SHAP
  9. Generación de predicciones
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── Paths ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.ingestion       import load_config, load_raw_data
from src.data.preprocessing   import preprocess
from src.models.clustering    import run_clustering
from src.features.feature_engineering import build_features
from src.features.feature_selection   import get_feature_cols
from src.models.train         import train_all_models, find_optimal_threshold, save_model
from src.models.tuning        import tune_model
from src.models.predict       import generate_predictions
from src.evaluation.metrics   import compute_metrics
from src.evaluation.explainability import compute_shap
from src.visualization.model_plots import plot_all


def separador(titulo: str):
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"{'═'*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Propensión de Compra — Ferreycorp")
    parser.add_argument("--data",    type=str, default=None, help="Ruta al CSV de datos (override config)")
    parser.add_argument("--tune",    action="store_true",    help="Activar fine-tuning con RandomizedSearchCV")
    parser.add_argument("--no-shap", action="store_true",    help="Omitir cálculo de SHAP values")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 0. Config ──────────────────────────────────────────────────────────────
    separador("0/8  CONFIGURACIÓN")
    config = load_config()
    random_state = config["project"]["random_state"]
    target       = config["data"]["target"]
    print(f"  Proyecto  : {config['project']['name']}")
    print(f"  Versión   : {config['project']['version']}")
    print(f"  Seed      : {random_state}")

    # ── 1. Carga ───────────────────────────────────────────────────────────────
    separador("1/8  CARGA DE DATOS")
    df = load_raw_data(path=args.data, config=config)

    # ── 2. Preprocesamiento ────────────────────────────────────────────────────
    separador("2/8  PREPROCESAMIENTO")
    df = preprocess(df, config)

    # ── 3. Clustering ──────────────────────────────────────────────────────────
    separador("3/8  CLUSTERING")
    df = run_clustering(df, config)

    # ── 4. Feature Engineering ─────────────────────────────────────────────────
    separador("4/8  FEATURE ENGINEERING")
    df = build_features(df)

    # ── 5. Split ───────────────────────────────────────────────────────────────
    separador("5/8  SPLIT TRAIN / TEST")
    feature_cols = get_feature_cols(df, config)
    X = df[feature_cols].fillna(0)
    y = df[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X, y,
        test_size=config["data"]["val_size"],
        stratify=y,
        random_state=random_state,
    )

    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}, |  Val: {len(X_val):,}")
    print(f"  Features: {len(feature_cols)}")

    # ── 6. Entrenamiento ───────────────────────────────────────────────────────
    separador("6/8  ENTRENAMIENTO")

    from src.models.train import _load_model_params
    params = _load_model_params()


    best_pipeline, best_name, results_df = train_all_models(
    X_train, X_val, y_train, y_val, config, params)

    print(f"\n  Resumen comparativo:")
    print(results_df.to_string(index=False))

    # ── Fine-tuning (opcional) ─────────────────────────────────────────────────
    if args.tune:
        separador("6b/8  FINE-TUNING")

        best_pipeline = tune_model(
            best_pipeline, best_name,
            X_train, y_train,
            n_iter=20, cv=5,
            random_state=random_state,
        )

    best_pipeline.fit(X_train, y_train)

    # ── 7. Threshold + Métricas ────────────────────────────────────────────────
    # separador("7/8  EVALUACIÓN")

    # y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    # threshold = find_optimal_threshold(y_test, y_prob, strategy=params["threshold_strategy"])
    # print(f"  Threshold óptimo ({params['threshold_strategy']}): {threshold:.4f}")

    
    separador("7/8  THRESHOLD OPTIMIZATION")

    # y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]

    # threshold = find_optimal_threshold(
    #     y_val,
    #     y_prob_val,
    #     strategy=params["threshold_strategy"]
    # )

    # print(f"  Threshold óptimo ({params['threshold_strategy']}): {threshold:.4f}")

    # y_pred = (y_prob_val >= threshold).astype(int)

    # metrics = compute_metrics(y_test, y_prob_val, y_pred, best_name, threshold, config)

    # images_dir = config["paths"]["images_dir"]
    # plot_all(y_test, y_prob_val, y_pred, best_name, images_dir)




    ################

    # ── 7. Threshold + Métricas ────────────────────────────────────────────────
    separador("7/8 THRESHOLD OPTIMIZATION")

    # 1. Calcular threshold en VALIDACIÓN
    y_prob_val = best_pipeline.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, y_prob_val, strategy=params["threshold_strategy"])
    print(f" Threshold óptimo ({params['threshold_strategy']}): {threshold:.4f}")

    # 2. Aplicar threshold en TEST
    y_prob_test = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold).astype(int)

    # 3. Calcular métricas con TEST
    metrics = compute_metrics(y_test, y_prob_test, y_pred_test, best_name, threshold, config)

    # 4. Graficar resultados con TEST
    images_dir = config["paths"]["images_dir"]
    plot_all(y_test, y_prob_test, y_pred_test, best_name, images_dir)

    ################

    # ── 8. SHAP ────────────────────────────────────────────────────────────────
    shap_df = None
    if not args.no_shap:
        separador("8/8  EXPLICABILIDAD SHAP")
        shap_sample = params.get("shap", {}).get("sample_size", 500)
        shap_df = compute_shap(
            best_pipeline, X_test, feature_cols, config,
            sample_size=shap_sample, random_state=random_state,
        )

    # ── 9. Predicciones + Guardado ─────────────────────────────────────────────
    separador("9/8  PREDICCIONES Y GUARDADO")

    save_model(best_pipeline, feature_cols, best_name, config)

    pred_df = generate_predictions(
        df, best_pipeline, feature_cols, shap_df, config, threshold=threshold
    )

    print(f"\n{'═'*60}")
    print(f"  ✅  Pipeline completado exitosamente")
    print(f"  Modelo: {best_name}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}  |  F1: {metrics['f1']:.4f}  |  Recall: {metrics['recall']:.4f}")
    print(f"\n  Para lanzar el agente:")
    print(f"     python src/agent/agent.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
