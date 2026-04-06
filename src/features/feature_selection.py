"""
src/features/feature_selection.py
Selección de features para entrenamiento.
"""

import pandas as pd


def get_feature_cols(df: pd.DataFrame, config: dict) -> list:
    """
    Retorna la lista de columnas de features, excluyendo target, id y columnas no útiles.
    """
    data_cfg = config["data"]
    drop = {
        data_cfg["target"],
        data_cfg.get("id_col", "id"),
        "id_marca",
        "cantidad",
        "tamanio_ciudad",
        "tamanio_ciudad ",     # con espacio (typo del dataset original)
        "outlier_if",
    }
    return [c for c in df.columns if c not in drop]


def describe_features(df: pd.DataFrame, feature_cols: list):
    print(f"\n  Features seleccionadas: {len(feature_cols)}")
    for c in feature_cols:
        dtype = df[c].dtype
        nuniq = df[c].nunique()
        print(f"    {c:<40} dtype={dtype}  nunique={nuniq}")
