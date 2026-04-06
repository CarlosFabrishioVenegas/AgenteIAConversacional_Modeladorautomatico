"""
src/features/feature_engineering.py
Creación de features derivadas a partir del dataset limpio.
"""

import numpy as np
import pandas as pd


def _separador(titulo: str):
    print(f"\n{'─'*60}\n  {titulo}\n{'─'*60}")


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features agregadas sobre precios de marcas."""
    precio_cols = [c for c in df.columns if c.startswith("precio_marca_")]
    if not precio_cols:
        return df
    df = df.copy()
    df["precio_min"]   = df[precio_cols].min(axis=1)
    df["precio_max"]   = df[precio_cols].max(axis=1)
    df["precio_rango"] = df["precio_max"] - df["precio_min"]
    df["precio_std"]   = df[precio_cols].std(axis=1)
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de actividad promocional."""
    promo_cols = [c for c in df.columns if c.startswith("promo_marca_")]
    if not promo_cols:
        return df
    df = df.copy()
    df["n_promos_activas"] = df[promo_cols].sum(axis=1)
    df["hay_promo"]        = (df["n_promos_activas"] > 0).astype(int)
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de comportamiento histórico."""
    df = df.copy()
    if "ultima_marca_comprada" in df.columns:
        df["tiene_historial"] = (df["ultima_marca_comprada"] > 0).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interacciones entre cluster y variables de negocio."""
    df = df.copy()
    if "cluster" in df.columns:
        if "precio_min" in df.columns:
            df["cluster_x_precio"] = df["cluster"] * df["precio_min"]
        if "n_promos_activas" in df.columns:
            df["cluster_x_promos"] = df["cluster"] * df["n_promos_activas"]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo de feature engineering.
    """
    _separador("FEATURE ENGINEERING")

    df = add_price_features(df)
    df = add_promo_features(df)
    df = add_behavioral_features(df)
    df = add_interaction_features(df)

    nuevas = [
        "precio_min", "precio_max", "precio_rango", "precio_std",
        "n_promos_activas", "hay_promo", "tiene_historial",
        "cluster_x_precio", "cluster_x_promos",
    ]
    creadas = [c for c in nuevas if c in df.columns]
    print(f"  Variables creadas ({len(creadas)}): {', '.join(creadas)}")
    return df
