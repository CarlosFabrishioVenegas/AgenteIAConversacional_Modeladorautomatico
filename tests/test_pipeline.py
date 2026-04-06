"""
tests/test_pipeline.py
Pruebas unitarias básicas del pipeline.
Uso: pytest tests/
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame sintético que imita la estructura del dataset real."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "id":                    range(n),
        "incidencia_compra":     np.random.randint(0, 2, n),
        "edad":                  np.random.randint(18, 65, n),
        "ingreso_anual":         np.random.randint(20000, 120000, n),
        "genero":                np.random.randint(0, 2, n),
        "estado_civil":          np.random.randint(0, 3, n),
        "nivel_educacion":       np.random.randint(0, 4, n),
        "ocupacion":             np.random.randint(0, 5, n),
        "ultima_marca_comprada": np.random.randint(0, 6, n),
        "ultima_cantidad_comprada": np.random.uniform(1, 10, n),
        "cantidad":              np.random.uniform(1, 5, n),
        "precio_marca_1":        np.random.uniform(5, 20, n),
        "precio_marca_2":        np.random.uniform(5, 20, n),
        "precio_marca_3":        np.random.uniform(5, 20, n),
        "precio_marca_4":        np.random.uniform(5, 20, n),
        "precio_marca_5":        np.random.uniform(5, 20, n),
        "promo_marca_1":         np.random.randint(0, 2, n),
        "promo_marca_2":         np.random.randint(0, 2, n),
        "promo_marca_3":         np.random.randint(0, 2, n),
        "promo_marca_4":         np.random.randint(0, 2, n),
        "promo_marca_5":         np.random.randint(0, 2, n),
        "id_marca":              np.random.randint(1, 6, n),
    })


@pytest.fixture
def config():
    import yaml
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_price_features(self, sample_df):
        from src.features.feature_engineering import add_price_features
        df = add_price_features(sample_df)
        assert "precio_min" in df.columns
        assert "precio_max" in df.columns
        assert "precio_rango" in df.columns
        assert (df["precio_rango"] >= 0).all(), "precio_rango no puede ser negativo"

    def test_promo_features(self, sample_df):
        from src.features.feature_engineering import add_promo_features
        df = add_promo_features(sample_df)
        assert "n_promos_activas" in df.columns
        assert "hay_promo" in df.columns
        assert df["hay_promo"].isin([0, 1]).all()

    def test_behavioral_features(self, sample_df):
        from src.features.feature_engineering import add_behavioral_features
        df = add_behavioral_features(sample_df)
        assert "tiene_historial" in df.columns
        assert df["tiene_historial"].isin([0, 1]).all()

    def test_build_features_no_cluster(self, sample_df):
        from src.features.feature_engineering import build_features
        # Sin cluster no deben crearse interacciones — no debe lanzar error
        df = build_features(sample_df)
        assert len(df) == len(sample_df)


class TestFeatureSelection:
    def test_get_feature_cols_excludes_target(self, sample_df, config):
        from src.features.feature_engineering import build_features
        from src.features.feature_selection import get_feature_cols
        df = build_features(sample_df)
        cols = get_feature_cols(df, config)
        assert config["data"]["target"] not in cols

    def test_get_feature_cols_excludes_id(self, sample_df, config):
        from src.features.feature_engineering import build_features
        from src.features.feature_selection import get_feature_cols
        df = build_features(sample_df)
        cols = get_feature_cols(df, config)
        assert "id" not in cols


class TestThreshold:
    def test_threshold_range(self):
        from src.models.train import find_optimal_threshold
        y_test = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.4, 0.6])
        t = find_optimal_threshold(y_test, y_prob, strategy="recall")
        assert 0.1 <= t <= 0.9


class TestPredictions:
    def test_prediction_columns(self, sample_df, config):
        """Verifica que generate_predictions crea las columnas esperadas."""
        from src.features.feature_engineering import build_features
        from src.features.feature_selection import get_feature_cols
        from src.models.predict import generate_predictions
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        import tempfile, os

        # Pipeline mínimo
        df = build_features(sample_df)
        df["cluster"] = 0
        feature_cols = get_feature_cols(df, config)
        X = df[feature_cols].fillna(0)
        y = df[config["data"]["target"]]

        pipe = Pipeline([("clf", RandomForestClassifier(n_estimators=10, random_state=42))])
        pipe.fit(X, y)

        # Override path temporal
        config_tmp = config.copy()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_tmp["paths"] = config["paths"].copy()
            config_tmp["paths"]["predictions"] = f"{tmpdir}/pred.parquet"
            pred = generate_predictions(df, pipe, feature_cols, None, config_tmp, threshold=0.5)

        expected_cols = {"id", "cluster", "prob_compra", "prediccion", "score_decil",
                         "edad", "ingreso_anual", "genero"}
        assert expected_cols.issubset(set(pred.columns))
        assert pred["prob_compra"].between(0, 1).all()
        assert pred["prediccion"].isin([0, 1]).all()
