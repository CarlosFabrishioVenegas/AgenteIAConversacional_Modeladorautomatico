"""
src/evaluation/explainability.py
SHAP values: importancia global y gráficos.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# def compute_shap(
#     pipeline,
#     X_test: pd.DataFrame,
#     feature_cols: list,
#     config: dict,
#     sample_size: int = 500,
#     random_state: int = 42,
# ) -> pd.DataFrame:
#     """
#     Calcula SHAP values y guarda gráficos.
#     Retorna DataFrame con importancias ordenadas.
#     """
#     if not HAS_SHAP:
#         print("  ⚠️  shap no instalado — omitiendo explicabilidad.")
#         return pd.DataFrame({"feature": feature_cols, "shap": np.zeros(len(feature_cols))})

#     images_dir = Path(config["paths"]["images_dir"])
#     images_dir.mkdir(parents=True, exist_ok=True)

#     base_model = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline
#     sample = X_test.sample(min(sample_size, len(X_test)), random_state=random_state)

#     print("\n  Calculando SHAP values...")

#     tree_models = ("RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier")
#     linear_models = ("LogisticRegression", "SGDClassifier")
#     class_name = base_model.__class__.__name__

#     try:
#         if class_name in tree_models:
#             explainer = shap.TreeExplainer(base_model)
#             shap_vals = explainer.shap_values(sample)
#             if isinstance(shap_vals, list):
#                 shap_vals = shap_vals[1]
#         elif class_name in linear_models:
#             explainer = shap.LinearExplainer(base_model, sample)
#             shap_vals = explainer.shap_values(sample)
#         else:
#             explainer = shap.Explainer(base_model, sample)
#             shap_vals = explainer(sample).values

#         # Importancia global
#         mean_shap = np.abs(shap_vals).mean(axis=0)
#         shap_df = pd.DataFrame({"feature": feature_cols, "shap": mean_shap})
#         shap_df.sort_values("shap", ascending=False, inplace=True)
#         shap_df.reset_index(drop=True, inplace=True)

#         print("\n  Top 10 — SHAP Importance:")
#         print(f"  {'Feature':<35} {'SHAP':>10}")
#         print(f"  {'─'*50}")
#         for _, row in shap_df.head(10).iterrows():
#             bar = "█" * int(row["shap"] * 200)
#             print(f"  {row['feature']:<35} {row['shap']:>10.5f} {bar}")

#         # Summary plot
#         plt.figure(figsize=(10, 6))
#         shap.summary_plot(shap_vals, sample, show=False)
#         plt.tight_layout()
#         plt.savefig(images_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
#         plt.close()

#         # Bar plot
#         plt.figure(figsize=(10, 6))
#         shap.summary_plot(shap_vals, sample, plot_type="bar", show=False)
#         plt.tight_layout()
#         plt.savefig(images_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
#         plt.close()

#         # Dependence top feature
#         top_feat = shap_df.iloc[0]["feature"]
#         plt.figure(figsize=(8, 5))
#         shap.dependence_plot(top_feat, shap_vals, sample, show=False)
#         plt.tight_layout()
#         plt.savefig(images_dir / f"shap_dependence_{top_feat}.png", bbox_inches="tight", dpi=150)
#         plt.close()

#         print(f"\n  ✅ Gráficos SHAP guardados en: {images_dir}")
#         return shap_df

#     except Exception as e:
#         print(f"  ❌ Error en SHAP: {e}")
#         return pd.DataFrame({"feature": feature_cols, "shap": np.zeros(len(feature_cols))})
def compute_shap(
    pipeline,
    X_test: pd.DataFrame,
    feature_cols: list,
    config: dict,
    sample_size: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calcula SHAP values y guarda gráficos.
    Retorna DataFrame con importancias ordenadas.
    """
    if not HAS_SHAP:
        print("  ⚠️  shap no instalado — omitiendo explicabilidad.")
        return pd.DataFrame({"feature": feature_cols, "shap": np.zeros(len(feature_cols))})

    images_dir = Path(config["paths"]["images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── Modelo base ─────────────────────────────────────────
    base_model = pipeline.named_steps["clf"] if hasattr(pipeline, "named_steps") else pipeline

    # ── Sample ──────────────────────────────────────────────
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=random_state)

    print("\n  Calculando SHAP values...")

    # ── IMPORTANTE: transformar si hay pipeline ─────────────
    if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
        X_input = pipeline.named_steps["preprocessor"].transform(sample)

        # Obtener nombres reales de features transformadas
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        except:
            feature_names = [f"f_{i}" for i in range(X_input.shape[1])]
    else:
        X_input = sample
        feature_names = feature_cols

    # ── Selección de explainer ──────────────────────────────
    tree_models = ("RandomForestClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier")
    linear_models = ("LogisticRegression", "SGDClassifier")
    class_name = base_model.__class__.__name__

    try:
        if class_name in tree_models:
            explainer = shap.TreeExplainer(base_model)
            shap_vals = explainer.shap_values(X_input)

            # Caso lista (versiones antiguas)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # Caso array 3D → seleccionar clase positiva
            if hasattr(shap_vals, "shape") and len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]

        elif class_name in linear_models:
            explainer = shap.LinearExplainer(base_model, X_input)
            shap_vals = explainer.shap_values(X_input)

        else:
            explainer = shap.Explainer(base_model, X_input)
            shap_vals = explainer(X_input).values

            if len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]

        # ── IMPORTANCIA GLOBAL ───────────────────────────────
        mean_shap = np.abs(shap_vals).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap": mean_shap
        })

        shap_df.sort_values("shap", ascending=False, inplace=True)
        shap_df.reset_index(drop=True, inplace=True)

        # ── PRINT TOP FEATURES ───────────────────────────────
        print("\n  Top 10 — SHAP Importance:")
        print(f"  {'Feature':<40} {'SHAP':>10}")
        print(f"  {'─'*55}")
        for _, row in shap_df.head(10).iterrows():
            bar = "█" * int(row["shap"] * 200)
            print(f"  {row['feature']:<40} {row['shap']:>10.5f} {bar}")

        # ── SUMMARY PLOT ─────────────────────────────────────
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_input, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(images_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()

        # ── BAR PLOT ─────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_vals, X_input, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(images_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
        plt.close()

        # ── DEPENDENCE PLOT ──────────────────────────────────
        top_feat = shap_df.iloc[0]["feature"]

        if top_feat in feature_names:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(top_feat, shap_vals, X_input, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(images_dir / f"shap_dependence_{top_feat}.png", bbox_inches="tight", dpi=150)
            plt.close()

        print(f"\n  ✅ Gráficos SHAP guardados en: {images_dir}")

        shap_df.to_csv(config["paths"]["feactures_importance"], index=False)
        return shap_df

    except Exception as e:
        print(f"  ❌ Error en SHAP: {e}")
        return pd.DataFrame({
            "feature": feature_cols,
            "shap": np.zeros(len(feature_cols))
        })