"""Preprocesamiento sin fuga: cada vista se ajusta solo con train.

Dos vistas sobre los mismos datos:
- tabpfn_view: NaN nativos y categóricas a ordinal.
- aux_view: imputación, escalado y one-hot para modelos clásicos y distancias.

"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Se resuelven en fit, así que se adaptan a cualquier esquema de columnas.
NUM = selector(dtype_include="number")
CAT = selector(dtype_exclude="number")


def tabpfn_view() -> ColumnTransformer:
    """Vista para TabPFN: numéricas intactas (con NaN) y categóricas a ordinal.

    Las categorías no vistas en train se codifican como NaN.
    """
    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    ct = ColumnTransformer(
        [("num", "passthrough", NUM), ("cat", ordinal, CAT)],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def aux_view() -> ColumnTransformer:
    """Vista para modelos clásicos y métricas de distancia.

    Numéricas: imputa con la mediana de train y estandariza.
    Categóricas: one-hot (NaN es una categoría) las no vistas quedan en 0.
    """
    num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ct = ColumnTransformer(
        [("num", num, NUM), ("cat", onehot, CAT)],
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def cat_indices(view: ColumnTransformer) -> list[int]:
    """Índices de las categóricas en la salida de tabpfn_view.

    Asume la vista ya ajustada y el orden numéricas-primero-categóricas.
    """
    n_num = len(view.transformers_[0][2])
    n_total = len(view.get_feature_names_out())
    return list(range(n_num, n_total))
